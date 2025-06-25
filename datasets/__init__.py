# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from .shha import SHHA

def build_loader(config, mode):
    data_path = config.DATA_PATH
    label_prob = config.LABEL_PERCENT
    protc_path = config.LABEL_PROTOCOL
    batch_size = config.BATCH_SIZE
    num_workers = config.NUM_WORKERS
    Dataset = {
        'shha': SHHA
    }[config.DATASET.lower()]

    data_set = Dataset(data_path, mode, label_prob, protc_path)
    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    # sampler = DistributedSampler(
    #         train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )

    return DataLoader(
        data_set,
        batch_size = batch_size if (mode == 'train') else 1,
        #sampler=sampler,
        num_workers = num_workers,
        pin_memory=config.PIN_MEMORY,
        shuffle = (mode == 'train'),
        collate_fn=Dataset.collate_fn
    )

def build_normal_loader(config, mode):
    data_path = config.DATA_PATH
    batch_size = config.BATCH_SIZE
    
    Dataset = {
        'shha': SHHA
    }[config.DATASET.lower()]
    
    data_set = Dataset(data_path, mode)

    return DataLoader(
        data_set,
        batch_size = batch_size,
        num_workers = 4,
        pin_memory=config.PIN_MEMORY,
        shuffle = False,
        #drop_last = mode=='Train'
        collate_fn=Dataset.collate_fn
    )