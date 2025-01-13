import os
import os.path as osp
import math
import argparse
import random
from loguru import logger
import torch
import numpy as np
from tqdm import tqdm
import options.base_options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from collections import OrderedDict
import sys
import shutil

def main():
    
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, default=None, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--verbose', type=util.str2bool, default=False, help='Print metrics for each image during testing.')
    parser.add_argument('--save_img', type=util.str2bool, default=False, help='if save predicted images during testing.')

    
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    verbose = args.verbose
    save_img = args.save_img

    #### distributed training settings
    opt['dist'] = False
    rank = 1
    logger.info('Disabled distributed training.')
    
    logger.info(f'python {" ".join(sys.argv)}')

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info(f'Random seed: {seed}')
    util.set_random_seed(seed)
    torch.backends.cudnn.benchmark = True

    #### create test dataloader
    # dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = create_dataset(opt, dataset_opt)
            test_loader = create_dataloader(test_set, dataset_opt, opt, None)
            logger.info(f'Number of test images in [{dataset_opt["name"]}][{dataset_opt["filelist"]}]: {len(test_set)}')

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info(f"Resuming training from epoch: {resume_state['epoch']:06d}, iter: {resume_state['iter']:06d}.")
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    
    img_dir = osp.join(opt['path']['log'], 'test_images')
    util.mkdir(img_dir)

    if opt['datasets'].get('test', None):
        avg_psnr_exp1 = []
        avg_psnr_exp2 = []
        avg_psnr_exp3 = []
        avg_psnr_exp4 = []
        avg_psnr_exp5 = []
        avg_ssim_exp1 = []
        avg_ssim_exp2 = []
        avg_ssim_exp3 = []
        avg_ssim_exp4 = []
        avg_ssim_exp5 = []
        for test_data in tqdm(test_loader):

            model.feed_data(test_data)
            model.test()

            out_dict = OrderedDict()
            out_dict['LQs'] = model.var_L.detach().float().cpu()
            out_dict['rlts'] = model.fake_H.detach().float().cpu()
            out_dict['GTs'] = model.real_H.detach().float().cpu()

            # Save SR images for reference
            for i in range(len(test_data['LQ_path'])):
                img_path = test_data['LQ_path'][i]
                # get pure name without .jpg and so on
                img_name = '.'.join(os.path.basename(img_path).split(".")[:-1])

                gt_img = util.tensor2img(out_dict['GTs'][i])
                en_img = util.tensor2img(out_dict['rlts'][i])

                # calculate metrics
                psnr_inst = util.calculate_psnr(en_img, gt_img)
                ssim_inst = util.calculate_ssim(en_img, gt_img)
                if math.isinf(psnr_inst) or math.isnan(psnr_inst) or \
                        math.isinf(ssim_inst) or math.isnan(ssim_inst):
                    psnr_inst = 0
                    ssim_inst = 0
                    logger.warning(f'Inf or Nan occurred in calculating PSNR and SSIM for {osp.basename(img_path)}.')
                if verbose:
                    verbose_path = "/".join(img_path.split("/")[-6:])
                    logger.info(f'[Verbose] {verbose_path} - PSNR: {psnr_inst:.4f}, SSIM: {ssim_inst:.4f}.')
                

                suffix = img_name.split('_')[-1]
                if suffix == 'N1.5':
                    avg_psnr_exp1.append(psnr_inst)
                    avg_ssim_exp1.append(ssim_inst)
                elif suffix == 'N1':
                    avg_psnr_exp2.append(psnr_inst)
                    avg_ssim_exp2.append(ssim_inst)
                elif suffix == '0':
                    avg_psnr_exp3.append(psnr_inst)
                    avg_ssim_exp3.append(ssim_inst)
                elif suffix == 'P1':
                    avg_psnr_exp4.append(psnr_inst)
                    avg_ssim_exp4.append(ssim_inst)
                elif suffix == 'P1.5':
                    avg_psnr_exp5.append(psnr_inst)
                    avg_ssim_exp5.append(ssim_inst)
                else:
                    raise FileNotFoundError("File is not found.......")
                
                if save_img:
                    # out_input_dir = osp.join(img_dir, 'input')
                    # out_gt_dir = osp.join(img_dir, 'gt')
                    out_en_dir = osp.join(img_dir, 'enhanced')
                    # util.mkdir(out_input_dir)
                    # util.mkdir(out_gt_dir)
                    util.mkdir(out_en_dir)
                    # shutil.copy(img_path, osp.join(out_input_dir, osp.basename(img_path)))
                    # util.save_img(gt_img, osp.join(out_gt_dir, f'{img_name}.png'), mode='BGR')
                    util.save_img(en_img, osp.join(out_en_dir, f'{img_name}.png'), mode='BGR')

        avg_psnr_all = sum(avg_psnr_exp1) + sum(avg_psnr_exp2) + sum(avg_psnr_exp3) \
                       + sum(avg_psnr_exp4) + sum(avg_psnr_exp5)
        avg_ssim_all = sum(avg_ssim_exp1) + sum(avg_ssim_exp2) + sum(avg_ssim_exp3) \
                       + sum(avg_ssim_exp4) + sum(avg_ssim_exp5)
        count = len(avg_psnr_exp1) + len(avg_psnr_exp2) + len(avg_psnr_exp3) \
                + len(avg_psnr_exp4) + len(avg_psnr_exp5)

        logger.info(f'# Test # Average PSNR: {(avg_psnr_all / count):.4f}, Average SSIM: {(avg_ssim_all / count):.4f}.')        
        
        logger.info(f'# Test # PSNR N1.5: {np.mean(avg_psnr_exp1):.4f}, PSNR N1: {np.mean(avg_psnr_exp2):.4f}, PSNR 0: {np.mean(avg_psnr_exp3):.4f}, PSNR P1: {np.mean(avg_psnr_exp4):.4f}, PSNR P1.5: {np.mean(avg_psnr_exp5):.4f}.')

        logger.info(f'# Test # SSIM N1.5: {np.mean(avg_ssim_exp1):.4f}, SSIM N1: {np.mean(avg_ssim_exp2):.4f}, SSIM 0: {np.mean(avg_ssim_exp3):.4f}, SSIM P1: {np.mean(avg_ssim_exp4):.4f}, SSIM P1.5: {np.mean(avg_ssim_exp5):.4f}.')

    logger.info('End of testing.')


if __name__ == '__main__':
    main()
