"""create dataset and dataloader"""
from loguru import logger
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
        batch_size = dataset_opt['batch_size']
        shuffle = dataset_opt['use_shuffle']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler,
                                           pin_memory=True)
    else:
        batch_size = dataset_opt['batch_size']
        shuffle = False
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                                           pin_memory=False, drop_last=False)


def create_dataset(opt, dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'UEN_train':
        from data.SIEN_dataset import DatasetFromFolder as D

        dataset = D(upscale_factor=opt['scale'], data_augmentation=dataset_opt['augment'],
                    group_file=dataset_opt['filelist'],
                    patch_size=dataset_opt['IN_size'], black_edges_crop=False, hflip=True, rot=True)

    elif mode == 'UEN_val':
        from data.SIEN_dataset import DatasetFromFolder as D
        dataset = D(upscale_factor=opt['scale'], data_augmentation=False,
                    group_file=dataset_opt['filelist'],
                    patch_size=dataset_opt['IN_size'], black_edges_crop=False, hflip=False, rot=False)

    elif mode == 'UEN_test':
        from data.SIEN_dataset import DatasetFromFolderSingle as D
        dataset = D(upscale_factor=opt['scale'], data_augmentation=False,
                    group_file=dataset_opt['filelist'],
                    patch_size=dataset_opt['IN_size'], black_edges_crop=False, hflip=False, rot=False,
                    mask=dataset_opt['mask'])

    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    logger.info(f'Dataset [{dataset.__class__.__name__} - {dataset_opt["name"]:s}] is created.')
    return dataset
