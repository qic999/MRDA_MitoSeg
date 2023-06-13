"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import imp
from multiprocessing.spawn import import_main_path
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import pdb
import random
import numpy as np
import torch
import os
from torchsummary import summary
import tensorboardX

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_seeds(seed=0, cuda_deterministic=True):
    print(f'seed:{seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['PL_GLOBAL_SEED'] = str(seed)
    torch.backends.cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_ratio_foregrond(input):
    mask = input['A']
    mask[mask>0] = 1
    fore_pix_num = np.count_nonzero(mask)
    all_pix_num = 32 * 256 * 256
    ratio = 100*fore_pix_num / (all_pix_num*1.0)
    # print(mask.shape)
    # print('fore_pix_num',fore_pix_num)
    # print('all_pix_num',all_pix_num)
    # print('ratio',ratio)
    return ratio, fore_pix_num

def record_ratio_foregrond(writer, ratio, fore_pix_num, step, epoch, ratio_list):
    writer.add_scalar('ratio', ratio, step+(epoch-1)*4000)
    writer.add_scalar('fore_pix_num', fore_pix_num, step+(epoch-1)*4000)
    ratio_list.append(ratio)
    if (step+1) == 4000:
        mean = np.mean(ratio_list)
        std = np.std(ratio_list)
        writer.add_scalar('every_epoch_ratio_mean', mean, epoch)
        writer.add_scalar('every_epoch_ratio_std', std, epoch)
        ratio_list = []
    return ratio_list


if __name__ == '__main__':
    # init_seed(0)
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(dataset)    # get the number of images in the dataset.
    dataset_size = 4000
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    ratio_sum = 0
    ratio_list = []
    log_save_path = 'records/'+opt.name
    writer = tensorboardX.SummaryWriter(log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        print('epoch',epoch)
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for iter_num, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            ratio, fore_pix_num = compute_ratio_foregrond(data)
            ratio_list = record_ratio_foregrond(writer, ratio, fore_pix_num, iter_num, epoch, ratio_list)

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print('losses',losses)
                print('t_comp',t_comp)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()
            if (iter_num+1) == dataset_size:
                break
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        # if epoch == 10:
        #     break
