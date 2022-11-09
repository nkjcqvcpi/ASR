# encoding: utf-8
import torch
import time

from tqdm import tqdm, trange
from utils.reid_metric import r1_mAP_mINP
from utils.meter import AverageMeter
from tester import Tester


class Trainer:
    epoch = 0
    iters = 0

    def __init__(self, cfg, model, optimizer, loss_fn, num_query, start_epoch, n):
        self.n = n
        self.device = cfg.MODEL.DEVICE
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.loss_meter = {k: AverageMeter() for k in list(loss_fn.keys()) + ['all']}
        self.acc_meter = AverageMeter()

        self.metrics = r1_mAP_mINP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, re_ranking=cfg.TEST.RE_RANKING)
        self.scaler = torch.cuda.amp.GradScaler()

        self.evaluator = Tester(cfg, model, num_query)

        if cfg.MODEL_DIR and False:
            # checkpointer = ModelCheckpoint(cfg.MODEL_DIR, cfg.MODEL.NAME, n_saved=1, require_empty=False, atomic=False)
            save_param = {'model': model, 'optimizer': optimizer['model']}
            if cfg.MODEL.CENTER_LOSS:
                save_param += {'center_param': loss_fn['center'], 'optimizer_center': optimizer['center']}
            # trainer.add_event_handler(Events.EPOCH_COMPLETED(every=cfg.SOLVER.CHECKPOINT_PERIOD), checkpointer, save_param)

    def __call__(self, logger, dataloader, scheduler):
        self.logger = logger
        logger.info('Start training')
        pbar = tqdm(total=len(dataloader['train']),
                    bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]")
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.model.train()
        for self.epoch in range(1, self.cfg.SOLVER.MAX_EPOCHS + 1):
            pbar.set_description(f"{'Iteration' if self.cfg.SOLVER.MAX_EPOCHS == 1 else 'Epoch'} "
                                 f"[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]")
            start_time = time.time()
            for m in self.loss_meter.values():
                m.reset()
            self.acc_meter.reset()
            self.metrics.reset()

            if self.cfg.SOLVER.LR_SCHEDULER == 'warmup':
                lr = self.scheduler.get_lr()[0]
            else:
                lr = self.scheduler.get_lr(self.epoch)[0]

            for batch in self.dataloader['train']:
                # if self.epoch % self.cfg.SOLVER.LOG_PERIOD == 0 and iter == 0:
                #     self.log_training_batch()

                self.iters += 1
                loss, acc = self.train_step(batch)
                self.log_traing_status(loss, acc)
                pbar.set_postfix({'loss': loss['all'], 'acc': acc, 'lr': lr})
                pbar.update((self.iters - 1) % pbar.total + 1 - pbar.n)

            end_time = time.time()

            self.epoch_complete(lr, start_time - end_time)
            if self.epoch % self.cfg.SOLVER.EVAL_PERIOD == 0:
                self.log_validation_results()

        pbar.close()

    def train_step(self, batch):
        img, target, target_cam, target_view = batch
        self.optimizer['model'].zero_grad()

        if 'center' in self.optimizer.keys():
            self.optimizer['center'].zero_grad()

        if self.device == 'cuda':
            img = img.cuda()
            target = target.cuda()
            if self.cfg.MODEL.BACKBONE == 'transformer':
                target_cam = target_cam.cuda()
                target_view = target_view.cuda()
        elif self.device == 'cpu':
            img = img.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast():
            score, feat = self.model(img, target, cam_label=target_cam, view_label=target_view)
            if self.cfg.MODEL.BACKBONE == 'cnn':
                loss = {'ce': self.loss_fn['ce'](score, target),
                        'tri': self.loss_fn['tri'](feat, target)[0]}
                loss['all'] = torch.add(loss['ce'], loss['tri'])
                if self.cfg.MODEL.CENTER_LOSS:
                    loss['center'] = self.cfg.SOLVER.CENTER_LOSS_WEIGHT * self.loss_fn['center'](feat, target)
                    loss['all'].add_(loss['center'])
            elif self.cfg.MODEL.BACKBONE == 'transformer':
                if 'ce' in self.loss_fn.keys():
                    loss = {'ce': self.loss_fn['ce'](score, target)}
                    loss['all'] = loss['ce']
                else:
                    loss = {'id': self.loss_fn['id'](score, target),
                            'tri': self.loss_fn['tri'](feat, target)}
                    loss['all'] = torch.add(self.cfg.MODEL.ID_LOSS_WEIGHT * loss['id'],
                                            self.cfg.MODEL.TRIPLET_LOSS_WEIGHT * loss['tri'])
            else:
                raise SyntaxWarning

        self.scaler.scale(loss['all']).backward()
        self.scaler.step(self.optimizer['model'])

        if 'center' in self.optimizer.keys():
            for param in self.loss_fn['center'].parameters():
                param.grad.data *= (1. / self.cfg.SOLVER.CENTER_LOSS_WEIGHT)
            self.scaler.step(self.optimizer['center'])

        self.scaler.update()

        # compute acc
        if isinstance(score, list):
            acc = (score[0].max(1)[1] == target).float().mean()
        else:
            acc = (score.max(1)[1] == target).float().mean()

        if self.device == 'cuda':
            torch.cuda.synchronize()

        for k, v in loss.items():
            self.loss_meter[k].update(v, img.shape[0])
        self.acc_meter.update(acc, 1)

        return loss, acc.item()

    def log_training_batch(self, batch):
        img, target = batch
        self.logger.writer.add_images(f'training_batch/{target}', img, self.iters)

    def log_traing_status(self, loss, acc):
        for k, v in loss.items():
            self.logger.writer.add_scalar(f'train_{self.n}/{k}', v, self.iters)
        self.logger.writer.add_scalar(f'train_{self.n}/acc', acc, self.iters)

    def epoch_complete(self, lr, t):
        self.logger.writer.add_scalar(f'train_{self.n}/lr', lr, self.epoch)
        if self.cfg.SOLVER.LR_SCHEDULER == 'warmup':
            self.scheduler.step()
        elif self.cfg.SOLVER.LR_SCHEDULER == 'cos':
            self.scheduler.step(self.epoch)
        self.logger.writer.add_scalar(f'train_{self.n}/Time per epoch', t, self.epoch)
        self.logger.info('Epoch {} done. Time per epoch: {:.3f}[s]'.format(self.epoch, t))
        self.logger.info('-' * 10)

    def log_validation_results(self):
        cmc, mAP, mINP = self.evaluator(self.logger, self.dataloader['eval'])
        self.logger.info(f"Validation Results - Epoch: {self.epoch}, mINP: {mINP:.1%}, mAP: {mAP:.1%}; "
                         f"Rank-1:{cmc[0]:.1%}, Rank-5:{cmc[4]:.1%}, Rank-10:{cmc[9]:.1%}")
        self.logger.writer.add_scalar('eval_{}/mINP'.format(self.n), mINP, self.epoch)
        self.logger.writer.add_scalar('eval_{}/mAP'.format(self.n), mAP, self.epoch)
        self.logger.add_ranks(self.n, cmc, self.epoch)
        if self.cfg.MODEL.DEVICE == 'cuda':
            torch.cuda.empty_cache()
