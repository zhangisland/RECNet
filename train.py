import os
import math
import argparse
import random
from loguru import logger

import torch
from tqdm import tqdm
import options.base_options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import time
import os.path as osp
timestr = time.strftime('%Y%m%d-%H%M%S')
import wandb
os.environ["WANDB_MODE"] = "offline"


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=None, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    logger.info('Disabled distributed training.')
    
    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    opt = option.dict_to_nonedict(opt)
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info(f'Random seed: {seed}')
    util.set_random_seed(seed)
    torch.backends.cudnn.benchmark = True

    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MultiExposure",

        # track hyperparameters and run metadata
        config={
        "expid": opt['expid'],
        "name": opt['name'],
        "model": opt['model'],
        "architecture": opt['network_G']["which_model_G"],
        "IN_size": opt['datasets']['train']['IN_size'],
        "train_dataset": osp.basename(opt['datasets']['train']['filelist']),
        "val_dataset": osp.basename(opt['datasets']['val']['filelist']),
        "test_dataset": osp.basename(opt['datasets']['test']['filelist']),
        "niter": opt['train']['niter'],
        "batch_size": opt['datasets']['train']['batch_size'],
        "lr_G": opt['train']['lr_G'],
        "lr_steps": opt['train']['lr_steps'],
        "lr_scheme": opt['train']['lr_scheme'],        
        "lr_gamma": opt['train']['lr_gamma'],
        "eta_min": opt['train']['eta_min'],
        "pixel_criterion": opt['train']['pixel_criterion'],
        "pixel_weight": opt['train']['pixel_weight'],
        "ssim_weight": opt['train']['ssim_weight'],
        "exc_weight": opt['train']['exc_weight'],
        "mask_weight": opt['train']['mask_weight'],
        "color_weight": opt['train']['color_weight'],
        "val_epoch": opt['train']['val_epoch'],
        "manual_seed": opt['train']['manual_seed'],
        }
    )

    #### create train and val dataloader
    # dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(opt, dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))

            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(f'Number of train images: {len(train_set):06d}, iters: {train_size:06d}')
                logger.info(f'Total epochs needed: {total_epochs:06d} for iters {total_iters:06d}')
        elif phase == 'val':
            val_set = create_dataset(opt, dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(f'Number of val images in [{dataset_opt["name"]}]: {len(val_set):06d}')

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info(f'Resuming training from epoch: {resume_state["epoch"]}, iter: {resume_state["iter"]}.')

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    best_psnr_avg = 0
    best_step_psnr = 0

    #### training
    logger.info(f'Start training from epoch: {start_epoch:06d}, iter: {current_step:06d}')

    for epoch in range(start_epoch, total_epochs + 2):

        total_loss = 0
        print_iter = 0

        if opt['train']['istraining'] == True:
            start_step = current_step
            for batch_step, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > total_iters + start_step:
                    break
                #### update learning rate
                model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

                #### training
                model.feed_data(train_data)
                model.optimize_parameters(current_step)
                #### logs
                if current_step % opt['logger']['print_freq'] == 0:
                    print_iter += 1
                    logs = model.get_current_log()
                    message = f'epoch:{epoch:06d}/{total_epochs:06d}, iter:{current_step:06d}/{total_iters:06d}, lr:'
                    for v in model.get_current_learning_rate():
                        message += f'{v:.4e}, '

                    total_loss += logs['l_total']
                    mean_total = total_loss / print_iter
                    message += f'mean_total_loss: {mean_total:.4e}'
                    # tensorboard logger
                    if rank <= 0:
                        wandb.log({"mean_loss": mean_total})
                        logger.info(message)

        ##### valid test
        if opt['datasets'].get('val', None) and epoch % opt['train']['val_epoch'] == 0:
            avg_psnr_exp1 = 0.
            avg_psnr_exp2 = 0.
            avg_psnr_exp3 = 0.
            avg_psnr_exp4 = 0.
            avg_psnr_exp5 = 0.
            idx = 0
            for val_data in tqdm(val_loader):
                idx += 1
                img_name = val_data['LQ_path'][0]

                model.feed_data(val_data)
                model.test()

                visuals = model.get_current_visuals()
                en_img = util.tensor2img(visuals['rlt'])  # uint8
                gt_img = util.tensor2img(visuals['GT'])  # uint8

                psnr_inst = util.calculate_psnr(en_img, gt_img)
                if math.isinf(psnr_inst) or math.isnan(psnr_inst):
                    psnr_inst = 0
                    idx -= 1

                suffix = img_name.split('_')[-1][:-4]
                if suffix == '0':
                    avg_psnr_exp1 = avg_psnr_exp1 + psnr_inst
                elif suffix == 'N1':
                    avg_psnr_exp2 = avg_psnr_exp2 + psnr_inst
                elif suffix == 'N1.5':
                    avg_psnr_exp3 = avg_psnr_exp3 + psnr_inst
                elif suffix == 'P1':
                    avg_psnr_exp4 = avg_psnr_exp4 + psnr_inst
                elif suffix == 'P1.5':
                    avg_psnr_exp5 = avg_psnr_exp5 + psnr_inst
                else:
                    raise FileNotFoundError("File is not found......")

            avg_psnr_all = avg_psnr_exp1 + avg_psnr_exp2 + avg_psnr_exp3 + avg_psnr_exp4 + avg_psnr_exp5
            # log
            logger.info(f'# Validation # Epoch: {epoch:06d}/{total_epochs:06d}, PSNR: Exp1 {(avg_psnr_exp1 / 150.0):.4f}, Exp2 {(avg_psnr_exp2 / 150.0):.4f}, Exp3 {(avg_psnr_exp3 / 150.0):.4f}, Exp4 {(avg_psnr_exp4 / 150.0):.4f}, Exp5 {(avg_psnr_exp5 / 150.0):.4f}')
            wandb.log({'avg_psnr_exp1': avg_psnr_exp1 / 150.0, 'avg_psnr_exp2': avg_psnr_exp2 / 150.0, 'avg_psnr_exp3': avg_psnr_exp3 / 150.0, 'avg_psnr_exp4': avg_psnr_exp4 / 150.0, 'avg_psnr_exp5': avg_psnr_exp5 / 150.0})
            
            logger.info(
                f'# Validation # Epoch: {epoch:06d}/{total_epochs:06d}, Average PSNR: {(avg_psnr_all / idx):.4f}, Previous best Average PSNR: {best_psnr_avg:.4f}, Previous best Average step: {best_step_psnr}')
            
            wandb.log({'avg_psnr': avg_psnr_all / idx})

            if avg_psnr_all / idx > best_psnr_avg:
                if rank <= 0:
                    best_psnr_avg = avg_psnr_all / idx
                    best_step_psnr = current_step
                    logger.info(f'Saving best average models!!!!!!!The best psnr is:{best_psnr_avg:4f}')
                    model.save_best('avg_psnr')

        if epoch % opt['logger']['save_checkpoint_epoch'] == 0 and epoch >= 1:
            if rank <= 0:
                logger.info('Saving models and training states.')
                model.save(epoch)
                model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
