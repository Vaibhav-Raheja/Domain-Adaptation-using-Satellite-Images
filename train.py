import time
import torch
import ntpath

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import os
from options.test_options import TestOptions
from util.visualizer import Visualizer
from util.visualizer import save_images
from util import html
import util.util as util

from datetime import datetime

# from synthetic_data_modeling.data_processing.FID_eval.pytorch_fid.src.pytorch_fid import fid_score
from pytorch_fid import fid_score

from os.path import join




def print_current_losses(epoch, iters, losses, t_comp, t_data):
    """print current losses on console; also save the losses to the disk
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message

def evaluate_model(opt):

    # hard-code some parameters for test
    opt.preprocess = "r1"#resize to half
    opt.crop_size = 0 # Since preprocess is None, this parameter is not used
    opt.load_size = 0 # Since preprocess is None, this parameter is not used

    opt.results_dir = join(opt.checkpoints_dir, "val_results")
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.phase = 'test'
    opt.isTrain = False

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    labels = ["real_A", "fake_B", "real_B"]
    return web_dir
    
def evaluate_fid(web_dir, total_iters, epoch):
    # print("======================================",web_dir)
    real_B_dir = join(web_dir,"images")#inria
    fake_B_dir = join(web_dir, "WHU")#whu
    paths = [real_B_dir, fake_B_dir]
    print("======================================",paths)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    fid = fid_score.calculate_fid_given_paths(paths,
          batch_size=1,
        #   resize_input=True,
          device="cuda:0",
          dims=2048,
          num_workers=8)

    fid_log = {
        "fid": fid,
        "step": total_iters,
        "epoch": epoch
    }
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''
    command = "python fid_score.py  \
    {} {} --batch-size {} --device {} --resize_input".format(real_B_dir, fake_B_dir, 1, "cuda:1")
    os.system(command)
    '''

if __name__ == '__main__':


    opt = TrainOptions().parse()   # get training options
    opt.isTrain = True
    
    cfg = vars(opt)
    

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    total_iters = 0                # the total number of training iterations
    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            #print(data['A'].size(), data['B'].size())
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            losses = model.get_current_losses()
            #losses is a dict. Add step information to this dict
            losses["step"] = total_iters
            losses["epoch"] = epoch
            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
            if total_iters % opt.log_freq == 0:    # print training losses and save logging information to the disk
                print(losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

                modified_opt = util.copyconf(opt)
                web_dir = evaluate_model(modified_opt)
                evaluate_fid(web_dir, total_iters, epoch)
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        if epoch % opt.eval_epoch_freq == 0:
            modified_opt = util.copyconf(opt)
            web_dir = evaluate_model(modified_opt)
            evaluate_fid(web_dir, total_iters, epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
