# encoding: utf-8

import logging
import os
import sys

from .iotools import mkdir_if_missing
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, cfg, name):
        self.log_step = 0

        self.p_logger = logging.getLogger(name)
        self.p_logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        self.p_logger.addHandler(ch)

        if cfg.LOG_DIR:
            mkdir_if_missing(cfg.LOG_DIR)
            fn = "test_log.txt" if cfg.TEST.EVAL_ONLY else "train_log.txt"
            fh = logging.FileHandler(os.path.join(cfg.LOG_DIR, fn), mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.p_logger.addHandler(fh)

        self.writer = SummaryWriter(cfg.TENSORBOARD_PATH)

    def info(self, info):
        self.p_logger.info(info)
        self.writer.add_text('main', info, self.log_step)
        self.log_step += 1

    def warning(self, warning):
        self.p_logger.warning(warning)
        self.writer.add_text('main', warning, self.log_step)
        self.log_step += 1

    def error(self, error):
        self.p_logger.error(error)
        self.writer.add_text('main', error, self.log_step)
        self.log_step += 1

    def add_ranks(self, cross, ranks, epoch):
        if epoch == 120:
            self.writer.add_text('ranks_{}'.format(cross), str(ranks), self.log_step)
        ranks = {str(k): v for k, v in enumerate(ranks)}
        self.writer.add_scalars('eval_{}/Ranks'.format(cross), ranks, epoch)
