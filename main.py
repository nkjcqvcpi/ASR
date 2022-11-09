# encoding: utf-8

import os
import time
import torch
import random
import numpy as np

import argparse

from data import make_dataloader
from modeling import make_model, get_criterion
from solver import create_scheduler, make_optimizer
from utils.logger import Logger
from trainer import Trainer
from config import get_cfg_defaults
from tester import Tester


class ASR:
    def __init__(self, seed, device):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.g = torch.Generator()
        self.g.manual_seed(seed)
        if device == 'cuda':
            from torch.backends import cudnn
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

    def __call__(self):
        if cfg.TEST.EVAL_ONLY:
            self.build()
            self.test()
        elif cfg.DATALOADER.FOLD > 1:
            for i in range(cfg.DATALOADER.FOLD):
                logger.info('==========Running ' + str(i) + ' as Test Set==========')
                self.build(i)
                self.train()
                break
        else:
            self.build()
            self.train()
        logger.writer.close()

    def build(self, k=0, record=False):
        self.dataloaders, self.num_query, self.num_classes, camera_num, view_num = make_dataloader(cfg, logger,
                                                                                                   self.g, k)
        self.model = make_model(cfg, logger, self.num_classes, camera_num=camera_num, view_num=view_num)

        if record:
            images, labels, _, _ = next(iter(self.dataloaders['train']))
            logger.writer.add_images('images_{}'.format(k), images, 0)
            logger.writer.add_graph(self.model, images)

        if cfg.MODEL.DEVICE == 'cuda':
            self.model.cuda()
        elif cfg.MODEL.DEVICE == 'cpu':
            self.model = self.model.to(memory_format=torch.channels_last)

    def test(self):
        self.model.base.load_param(cfg.MODEL.PRETRAIN_PATH)  # cfg.TEST.WEIGHT
        logger.info('Loading finetune model......')
        tester = Tester(cfg, self.model, self.num_query)
        tester(logger, self.dataloaders['eval'])

    def train(self, k=0):
        criterion = get_criterion(cfg, self.num_classes, logger)
        optimizer = make_optimizer(cfg, self.model, criterion, logger)
        start_epoch = 0

        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            logger.info('Start epoch: {}' % start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            logger.info('Path to the checkpoint of optimizer: {}' % path_to_optimizer)
            path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
            logger.info('Path to the checkpoint of center_param: {}' % path_to_center_param)
            path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
            logger.info('Path to the checkpoint of optimizer_center: {}' % path_to_optimizer_center)
            self.model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer['model'].load_state_dict(torch.load(path_to_optimizer))
            criterion['center'].load_state_dict(torch.load(path_to_center_param))
            optimizer['center'].load_state_dict(torch.load(path_to_optimizer_center))
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.model.base.load_param(cfg.MODEL.PRETRAIN_PATH)
            logger.info('Loading pretrained ImageNet model......')
            start_epoch = -1

        scheduler = create_scheduler(cfg, optimizer['model'], logger, start_epoch=start_epoch)
        trainer = Trainer(cfg, self.model, optimizer, criterion, self.num_query, start_epoch, k)
        trainer(logger, self.dataloaders, scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR')
    parser.add_argument('config_file', metavar='cfg', type=str)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join('configs', args.config_file))
    opts = ["TENSORBOARD_PATH", os.path.join(cfg.TENSORBOARD_PATH, time.strftime("%b%d_%H%M%S", time.localtime())),
            'MODEL.DEVICE', 'cuda' if torch.cuda.is_available() and cfg.MODEL.DEVICE == 'cuda' else 'cpu']
    cfg.merge_from_list(opts)
    cfg.freeze()

    logger = Logger(cfg, "asr")
    logger.info("Running with config:\n{}".format(cfg))

    asr = ASR(0, cfg.MODEL.DEVICE)
    asr()
