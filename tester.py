# encoding: utf-8
import torch

from utils.reid_metric import r1_mAP_mINP


class Tester:
    def __init__(self, cfg, model, num_query):
        self.model = model
        self.cfg = cfg

        self.metrics = r1_mAP_mINP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, re_ranking=cfg.TEST.RE_RANKING)

    def __call__(self, logger, val_loader, train=False):
        self.metrics.reset()
        for i in val_loader:
            feat, pid, camid = self.validation_step(i)
            self.metrics.update(feat, pid, camid)

        cmc, mAP, mINP = self.metrics.compute()

        if train:
            return cmc, mAP, mINP
        else:
            logger.info(f"Validation Results - mINP: {mINP:.1%}, mAP: {mAP:.1%}; "
                        f"Rank-1:{cmc[0]:.1%}, Rank-5:{cmc[4]:.1%}, Rank-10:{cmc[9]:.1%}")

    def validation_step(self, batch):
        self.model.eval()
        img, pids, camid, camids, target_view, _ = batch
        with torch.no_grad():
            if self.cfg.MODEL.DEVICE == 'cuda':
                img = img.cuda()
                if self.cfg.MODEL.BACKBONE == 'transformer':
                    camids = camids.cuda()
                    target_view = target_view.cuda()
            elif self.cfg.MODEL.DEVICE == 'cpu':
                img = img.to(memory_format=torch.channels_last)

            feat = self.model(img, cam_label=camids, view_label=target_view)

        return feat, pids, camids
