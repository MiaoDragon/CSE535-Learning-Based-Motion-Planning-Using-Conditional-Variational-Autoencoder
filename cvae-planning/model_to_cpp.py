"""
This implements the Kinodynamic Planning using MPNet, by using MPNet
to generate random samples, that will guide the SST algorithm.
"""
import sys
#sys.path.append('deps/sparse_rrt')
sys.path.append('.')
import torch
import torch.nn as nn
import importlib
#from model import ae_s2d
#from model import cvae_s2d_model1 as cvae_s2d

from model.SMPNet import SMPNet
from model.SMPPathNet import SMPPathNet
from model.SMPPathWithPriorNet import SMPPathWithPriorNet
from model.SMPPathSimpleNet import SMPPathSimpleNet
from model.TSMPPathWithPriorNet import TSMPPathWithPriorNet
#from tools import data_loader
from tools.utility import *
#from plan_utility import cart_pole, cart_pole_obs, pendulum, acrobot_obs, car_obs
import argparse
import numpy as np
import random
import os
from tqdm import tqdm, trange


def main(args):
    #global hl
    #if torch.cuda.is_available():
    if False:
        torch.cuda.set_device(args.device)
    # environment setting
    cae = None
    mlp = None

    # load net
    # load previously trained model if start epoch > 0

    # dynamically import model
    if args.model_type != "TSMPPathWithPriorNet":
        ae_module = importlib.import_module('model.ae_%s_model_%d' % (args.env_name, args.model_id))
        cvae_module = importlib.import_module('model.cvae_%s_model_%d' % (args.env_name, args.model_id))
        e_net = ae_module.Encoder(input_size=args.e_net_input_size, output_size=args.e_net_output_size)
        cvae = cvae_module.CVAE(input_size=args.input_size, latent_size=args.latent_size, cond_size=args.cond_size)
    else:
        cvae_module = importlib.import_module('model.cvae_%s_model_%d' % (args.env_name, args.model_id))        
        cvae = cvae_module.CVAE(args.x_size, args.s_size, args.x_latent_size, args.s_latent_size, args.c_latent_size, args.cond_size)

    data_loader = importlib.import_module('tools.data_loader_%s' % (args.env_type))
    plan_util = importlib.import_module('plan_util.%s' % (args.env_type))
    normalize = plan_util.normalize
    if args.model_type == "SMPNet":
        smpnet = SMPNet(e_net, cvae)
    elif args.model_type == "SMPPathNet":
        smpnet = SMPPathNet(e_net, cvae)
    elif args.model_type == "SMPPathWithPriorNet":
        smpnet = SMPPathWithPriorNet(e_net, cvae)
    elif args.model_type == "SMPPathSimpleNet":
        smpnet = SMPPathSimpleNet(e_net, cvae)
    elif args.model_type == "TSMPPathWithPriorNet":
        smpnet = TSMPPathWithPriorNet(cvae)

    model_dir = args.model_dir + '%s/%s/model_%d/param_%s/' % (args.env_name, args.model_type, args.model_id, args.param_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path='smpnet_epoch_%d.pkl' %(args.start_epoch)
    torch_seed, np_seed, py_seed = 0, 0, 0
    if args.start_epoch > 0:
        load_net_state(smpnet, os.path.join(model_dir, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(model_dir, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)

    #if torch.cuda.is_available():
    if False: 
        smpnet.cuda()
        smpnet.cvae.cuda()
        if hasattr(smpnet, 'e_net'):
            smpnet.e_net.cuda()

        if args.opt == 'Adagrad':
            smpnet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            smpnet.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            smpnet.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
        elif args.opt == 'ASGD':
            smpnet.set_opt(torch.optim.ASGD, lr=args.learning_rate)
    if args.start_epoch > 0:
        load_opt_state(smpnet, os.path.join(model_dir, model_path))

    # load train and test data
    print('loading...')
    val_sp = int(args.no_motion_paths * (1-args.val_ratio))
    data_folder = args.data_folder+args.env_name+'/'
    print('data_folder: ')
    print(data_folder)
    if args.model_type == "SMPPathNet":
        load_dis_ratio = True
    elif args.model_type == "SMPPathWithPriorNet":
        load_dis_ratio = True
    elif args.model_type == "SMPPathSimpleNet":
        load_dis_ratio = True
    elif args.model_type == "SMPNet":
        load_dis_ratio = False
    elif args.model_type == "TSMPPathWithPriorNet":
        load_dis_ratio = True
    waypoint_dataset, cond_dataset, obs, env_indices = data_loader.load_train_dataset(N=args.no_env, NP=1, s=0, sp=0,
                                                                   data_folder=data_folder, load_dis_ratio=load_dis_ratio)
    cond_dataset = np.array(cond_dataset)
    # randomize the dataset before training
    data=list(zip(waypoint_dataset, cond_dataset, env_indices))
    random.shuffle(data)
    waypoint_dataset,cond_dataset,env_indices=list(zip(*data))
    waypoint_dataset = list(waypoint_dataset)
    cond_dataset = list(cond_dataset)
    env_indices = list(env_indices)
    waypoint_dataset = np.array(waypoint_dataset)
    cond_dataset = np.array(cond_dataset)
    env_indices = np.array(env_indices)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(smpnet.opt, mode='min', \
                                        factor=args.lr_schedule_gamma, patience=args.lr_schedule_patience, verbose=True)
    epoch_early_stop_checker = EarlyStopChecker(1, 10)
    early_stop_checker = EarlyStopChecker(args.early_stop_freq, args.early_stop_patience)
    # Train the Models

    record_i = 0
    val_record_i = 0
    loss_avg_i = 0
    val_loss_avg_i = 0

    loss_avg = 0.
    loss_gen_avg = 0.
    loss_kl_avg = 0.

    val_loss_avg = 0.
    val_loss_gen_avg = 0.
    val_loss_kl_avg = 0.
    loss_steps = args.log_steps
    save_steps = args.save_steps
    prev_epoch_val_loss = None

    i = 0
    smpnet.eval()
    waypoint_dataset_i = waypoint_dataset[i:i+args.batch_size]
    cond_dataset_i = cond_dataset[i:i+args.batch_size]
    env_indices_i = env_indices[i:i+args.batch_size]
    # record
    bi = waypoint_dataset_i
    bi = torch.FloatTensor(bi)
    #print('mpnet input before normalization:')
    #print(bi)
    bi = normalize(bi, args.world_size)
    bi=to_var(bi)
    
    cond_dataset_i = torch.FloatTensor(cond_dataset_i)
    cond_dataset_i = normalize(cond_dataset_i, args.world_size)  # assume the first Sx2 are start and goal
    cond_dataset_i = to_var(cond_dataset_i)

    bobs = obs[env_indices_i].astype(np.float32)
    bobs = torch.FloatTensor(bobs)
    bobs = to_var(bobs)

    cond_dataset_i = cond_dataset_i.cpu()

    # trace the model
    traced_gen_f = torch.jit.trace_module(smpnet.cvae, {'forward': cond_dataset_i})
    traced_gen_f.save("c++/env_%s_param_%s_epoch_%d.pt" % (args.env_name, args.param_name, args.start_epoch))


##############################################################################
import yaml
# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper

parser = argparse.ArgumentParser()
# for training
parser.add_argument('--param_path', type=str, default='param/train/',help='path for loading training param')
parser.add_argument('--param_name', type=str, default="cvae_tamp_param2.yaml")

# parse the parameter file
args = parser.parse_args()
print(args)
param_f = open(args.param_path+args.param_name, 'r')
param = yaml.load(param_f)
param = DictDot(param)
main(param)