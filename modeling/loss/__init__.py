# encoding: utf-8

import torch.nn.functional as F

from .arcface import ArcFace
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss, WeightedRegularizedTriplet
from .center_loss import CenterLoss
from .circle_loss import CircleLoss
from .metric_learning import ContrastiveLoss, AMSoftmax, Cosface


def get_criterion(cfg, num_classes, logger):    # modified by gu
    loss_func = {}  # new add by luo

    if cfg.MODEL.CENTER_LOSS:
        loss_func['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM,
                                         use_gpu=cfg.MODEL.DEVICE == 'cuda')

    if cfg.MODEL.BACKBONE == 'transformer':
        sampler = cfg.DATALOADER.SAMPLER

        if sampler == 'softmax':
            def ce_loss(score, target):
                return F.cross_entropy(score, target)
            loss_func['ce'] = ce_loss
            return loss_func

        elif sampler == 'softmax_triplet' and cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            if cfg.MODEL.IF_LABELSMOOTH:
                _loss = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=cfg.MODEL.DEVICE == 'cuda')
                logger.info("label smooth on, num of classes:" + str(num_classes))
            else:
                _loss = F.cross_entropy

            def id_loss(score, target):
                if isinstance(score, list):
                    ID_LOSS = [_loss(scor, target) for scor in score[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    return 0.5 * ID_LOSS + 0.5 * _loss(score[0], target)
                else:
                    return _loss(score, target)

            loss_func['id'] = id_loss

            if cfg.MODEL.NO_MARGIN:
                triplet = TripletLoss()
                logger.info("using soft triplet loss for training")
            else:
                triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
                logger.info("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))

            def tri_loss(feat, target):
                if isinstance(feat, list):
                    TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                    TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    return 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                else:
                    return triplet(feat, target)[0]

            loss_func['tri'] = tri_loss
        else:
            raise SyntaxWarning(f'expected sampler should be softmax, triplet, softmax_triplet or '
                                f'softmax_triplet_center but got {cfg.DATALOADER.SAMPLER} or expected METRIC_LOSS_TYPE '
                                f'should be triplet but got {cfg.MODEL.METRIC_LOSS_TYPE}')

    elif cfg.MODEL.BACKBONE == 'cnn':
        loss_func['ce'] = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=cfg.MODEL.DEVICE == 'cuda')

        print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
        if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET:
            loss_func['tri'] = WeightedRegularizedTriplet()
        else:
            loss_func['tri'] = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

    return loss_func
