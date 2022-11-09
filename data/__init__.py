# encoding: utf-8
import torch
from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset
from .triplet_sampler import RandomIdentitySampler

from .transforms import build_transforms
from .ASRD import ASRD, KCrossASRD

__factory = {
    'ASRD': ASRD,
    'KCrossASRD': KCrossASRD
}


def init_dataset(cfg, *args, **kwargs):
    if cfg.DATASETS.NAMES == 'ASRD':
        if cfg.DATALOADER.FOLD:
            return KCrossASRD(cfg.DATALOADER.FOLD - 1, cfg.DATASETS.ROOT_DIR)
        else:
            return ASRD(cfg.DATASETS.ROOT_DIR)
    elif cfg.DATASETS.NAMES not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(cfg.DATASETS.NAMES))
    return __factory[cfg.DATASETS.NAMES](*args, **kwargs)


def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg, logger, g, n=0):
    transforms = build_transforms(cfg)
    dataset = init_dataset(cfg)

    num_classes = dataset.num_train_pids
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = ImageDataset(dataset.train, transforms['train'])
    train_set_normal = ImageDataset(dataset.train, transforms['eval'])
    data_loader = {}

    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        data_loader['train'] = DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, generator=g)
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        logger.info('using softmax sampler')
        data_loader['train'] = DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
                                          num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True,
                                          generator=g)
    else:
        raise SyntaxWarning('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    data_loader['train_normal'] = DataLoader(train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
                                             num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True,
                                             generator=g)

    eval_set = ImageDataset(dataset.query + dataset.gallery, transforms['eval'])

    data_loader['eval'] = DataLoader(eval_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
                                     num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True, generator=g)

    return data_loader, len(dataset.query), num_classes, cam_num, view_num
