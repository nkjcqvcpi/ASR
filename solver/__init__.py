# encoding: utf-8

import torch
from torch.optim import SGD, AdamW

from .WarmupMultiStepLR import WarmupMultiStepLR
from .CosineLRScheduler import CosineLRScheduler


def make_optimizer(cfg, model, criterion, logger):
    optimizer = {}
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR and ("classifier" in key or "arcface" in key):
            lr = cfg.SOLVER.BASE_LR * 2
            logger.info('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer['model'] = SGD(params, momentum=cfg.SOLVER.MOMENTUM, lr=cfg.SOLVER.CENTER_LR)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer['model'] = AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    if cfg.MODEL.CENTER_LOSS:
        optimizer['center'] = SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer


def create_scheduler(cfg, optimizer, logger, start_epoch=0):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    # type 1
    # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # type 2
    lr_min = 0.002 * cfg.SOLVER.BASE_LR
    warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    noise_range = None

    if cfg.SOLVER.LR_SCHEDULER == 'warmup':
        lr_scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                         cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)

    elif cfg.SOLVER.LR_SCHEDULER == 'cos':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            num_epochs, logger,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )
    else:
        raise SyntaxWarning

    return lr_scheduler

