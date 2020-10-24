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
#from tools import data_loader
from tools.utility import *
#from plan_utility import cart_pole, cart_pole_obs, pendulum, acrobot_obs, car_obs
import argparse
import numpy as np
import random
import os
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

def main(args):
    #global hl
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    # environment setting
    cae = None
    mlp = None

    # load net
    # load previously trained model if start epoch > 0

    # dynamically import model
    ae_module = importlib.import_module('model.ae_%s_model_%d' % (args.env_type, args.model_id))
    cvae_module = importlib.import_module('model.cvae_%s_model_%d' % (args.env_type, args.model_id))
    e_net = ae_module.Encoder(input_size=args.e_net_input_size, output_size=args.e_net_output_size)
    cvae = cvae_module.CVAE(input_size=args.input_size, latent_size=args.latent_size, cond_size=args.cond_size)

    data_loader = importlib.import_module('tools.data_loader_%s' % (args.env_type))
    plan_util = importlib.import_module('plan_util.%s' % (args.env_type))
    normalize = plan_util.normalize
    
    smpnet = SMPNet(e_net, cvae)
    model_dir = args.model_dir + "%s/model_%d/lr_%f_lrgamma_%f_lrpatience_%d_opt_%s_beta_%f/" % \
                                (args.env_type, args.model_id, args.learning_rate, \
                                 args.lr_schedule_gamma, args.lr_schedule_patience, args.opt, args.beta)
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

    if torch.cuda.is_available():
        smpnet.cuda()
        smpnet.cvae.cuda()
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
    data_folder = args.data_folder+args.env_type+'/'
    waypoint_dataset, start_dataset, goal_dataset, obs, start_indices, goal_indices, env_indices \
             = data_loader.load_train_dataset(N=args.no_env, NP=args.no_motion_paths,
                                                data_folder=data_folder)
    start_dataset, goal_dataset = np.array(start_dataset), np.array(goal_dataset)

    # randomize the dataset before training
    data=list(zip(waypoint_dataset, start_indices, goal_indices, env_indices))
    random.shuffle(data)
    waypoint_dataset,start_indices,goal_indices,env_indices=list(zip(*data))
    waypoint_dataset = list(waypoint_dataset)
    start_indices = list(start_indices)
    goal_indices = list(goal_indices)
    env_indices = list(env_indices)
    waypoint_dataset = np.array(waypoint_dataset)
    start_indices, goal_indices, env_indices = np.array(start_indices), np.array(goal_indices), np.array(env_indices)

    # use 5% as validation dataset
    val_len = int(len(waypoint_dataset) * 0.05)
    val_waypoint_dataset = waypoint_dataset[-val_len:]
    val_start_indices = start_indices[-val_len:]
    val_goal_indices = goal_indices[-val_len:]
    val_env_indices = env_indices[-val_len:]

    waypoint_dataset = waypoint_dataset[:-val_len]
    start_indices, goal_indices, env_indices = start_indices[:-val_len], goal_indices[:-val_len], env_indices[:-val_len]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(smpnet.opt, mode='min', \
                                        factor=args.lr_schedule_gamma, patience=args.lr_schedule_patience, verbose=True)

    # Train the Models
    print('training...')

    writer_fname = 'env_%s_model_%d_lr_%f_lrgamma_%f_lrpatience_%d_opt_%s_beta_%f' % \
                    (args.env_type, args.model_id, args.learning_rate, \
                     args.lr_schedule_gamma, args.lr_schedule_patience, args.opt, args.beta)

    writer = SummaryWriter('./train_stats/'+writer_fname)
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
    loss_steps = 100  # record every 100 loss
    prev_epoch_val_loss = None
    for epoch in trange(args.start_epoch+1,args.num_epochs+1):
        print('epoch' + str(epoch))
        val_i = 0
        smpnet.train()
        for i in trange(0,len(waypoint_dataset),args.batch_size):
            print('epoch: %d, training... path: %d' % (epoch, i+1))
            waypoint_dataset_i = waypoint_dataset[i:i+args.batch_size]
            start_indices_i, goal_indices_i, env_indices_i = start_indices[i:i+args.batch_size], goal_indices[i:i+args.batch_size], env_indices[i:i+args.batch_size]
            # record
            bi = waypoint_dataset_i
            bi = torch.FloatTensor(bi)
            #print('mpnet input before normalization:')
            #print(bi)
            bi = normalize(bi, args.world_size)
            bi=to_var(bi)
            y_start_i = start_dataset[start_indices_i]
            y_goal_i = goal_dataset[goal_indices_i]
            y_start_i, y_goal_i = torch.FloatTensor(y_start_i), torch.FloatTensor(y_goal_i)
            y_start_i, y_goal_i = normalize(y_start_i, args.world_size), normalize(y_goal_i, args.world_size)
            y_start_i, y_goal_i = to_var(y_start_i), to_var(y_goal_i)

            bobs = obs[env_indices_i].astype(np.float32)
            bobs = torch.FloatTensor(bobs)
            bobs = to_var(bobs)

            print('before training losses:')
            print(smpnet.loss(bi, smpnet.train_forward(bi, y_start_i, y_goal_i, bobs), beta=args.beta).mean())
            smpnet.step(bi, y_start_i, y_goal_i, bobs, beta=args.beta)
            print('after training losses:')
            print(smpnet.loss(bi, smpnet.train_forward(bi, y_start_i, y_goal_i, bobs), beta=args.beta).mean())
            loss = smpnet.loss(bi, smpnet.train_forward(bi, y_start_i, y_goal_i, bobs), beta=args.beta)
            loss = torch.mean(loss, dim=0)
            #update_line(hl, ax, [i//args.batch_size, loss.data.numpy()])
            loss_gen = smpnet.generation_loss(bi, smpnet.train_forward(bi, y_start_i, y_goal_i, bobs)).mean(dim=0)
            loss_kl = smpnet.kl_divergence(bi, smpnet.train_forward(bi, y_start_i, y_goal_i, bobs)).mean()
            loss_avg += loss.cpu().data
            loss_gen_avg += loss_gen.cpu().data  # K
            loss_kl_avg += loss_kl.cpu().data

            loss_avg_i += 1
            if loss_avg_i >= loss_steps:
                loss_avg = loss_avg / loss_avg_i
                loss_gen_avg = loss_gen_avg / loss_avg_i
                loss_kl_avg = loss_kl_avg / loss_avg_i
                writer.add_scalar('train_loss', torch.mean(loss_avg), record_i)
                writer.add_scalar('train_loss_generation', torch.mean(loss_gen_avg), record_i)
                writer.add_scalar('train_loss_kl', torch.mean(loss_kl_avg), record_i)

                for loss_i in range(len(loss_avg)):                    
                    writer.add_scalar('train_loss_%d' % (loss_i), loss_avg[loss_i], record_i)
                    writer.add_scalar('train_loss_generation_%d' % (loss_i), loss_gen_avg[loss_i], record_i)

                record_i += 1
                loss_avg = 0.
                loss_gen_avg = 0.
                loss_kl_avg = 0.
                loss_avg_i = 0

            # validation
            # calculate the corresponding batch in val_dataset
            waypoint_dataset_i = val_waypoint_dataset[val_i:val_i+args.batch_size]
            start_indices_i, goal_indices_i, env_indices_i = val_start_indices[val_i:val_i+args.batch_size], \
                                                            val_goal_indices[val_i:val_i+args.batch_size], \
                                                            val_env_indices[val_i:val_i+args.batch_size]
            val_i = val_i + args.batch_size
            if val_i > val_len:
                val_i = 0
            # record
            bi = waypoint_dataset_i
            bi = torch.FloatTensor(bi)
            bi = normalize(bi, args.world_size)
            bi=to_var(bi)
            
            y_start_i = start_dataset[start_indices_i]
            y_goal_i = goal_dataset[goal_indices_i]
            y_start_i, y_goal_i = torch.FloatTensor(y_start_i), torch.FloatTensor(y_goal_i)
            y_start_i, y_goal_i = normalize(y_start_i, args.world_size), normalize(y_goal_i, args.world_size)
            y_start_i, y_goal_i = to_var(y_start_i), to_var(y_goal_i)

            bobs = obs[env_indices_i].astype(np.float32)
            bobs = torch.FloatTensor(bobs)
            bobs = to_var(bobs)
            loss = smpnet.loss(bi, smpnet.train_forward(bi, y_start_i, y_goal_i, bobs), beta=args.beta)
            loss = torch.mean(loss, dim=0)
            print('validation loss: ', loss.cpu().data)
            loss_gen = smpnet.generation_loss(bi, smpnet.train_forward(bi, y_start_i, y_goal_i, bobs)).mean(dim=0)
            loss_kl = smpnet.kl_divergence(bi, smpnet.train_forward(bi, y_start_i, y_goal_i, bobs)).mean()
            val_loss_avg += loss.cpu().data
            val_loss_gen_avg += loss_gen.cpu().data  # K
            val_loss_kl_avg += loss_kl.cpu().data

            val_loss_avg_i += 1
            if val_loss_avg_i >= loss_steps:
                val_loss_avg = val_loss_avg / val_loss_avg_i
                val_loss_gen_avg = val_loss_gen_avg / val_loss_avg_i
                val_loss_kl_avg = val_loss_kl_avg / val_loss_avg_i
                writer.add_scalar('val_loss', torch.mean(val_loss_avg), val_record_i)
                writer.add_scalar('val_loss_generation', torch.mean(val_loss_gen_avg), val_record_i)
                writer.add_scalar('val_loss_kl', torch.mean(val_loss_kl_avg), val_record_i)
                for val_loss_i in range(len(val_loss_avg)):                    
                    writer.add_scalar('val_loss_%d' % (val_loss_i), val_loss_avg[val_loss_i], val_record_i)
                    writer.add_scalar('val_loss_generation_%d' % (val_loss_i), val_loss_gen_avg[val_loss_i], val_record_i)

                val_record_i += 1
                val_loss_avg = 0.
                val_loss_gen_avg = 0.
                val_loss_kl_avg = 0.
                val_loss_avg_i = 0

        # do a validation test using 2048 samples
        smpnet.eval()
        epoch_val_num = 2048
        waypoint_dataset_i = val_waypoint_dataset[:epoch_val_num]
        start_indices_i, goal_indices_i, env_indices_i = val_start_indices[:epoch_val_num], \
                                                        val_goal_indices[:epoch_val_num], \
                                                        val_env_indices[:epoch_val_num]
        bi = waypoint_dataset_i
        bi = torch.FloatTensor(bi)
        bi = normalize(bi, args.world_size)
        bi=to_var(bi)
        y_start_i = start_dataset[start_indices_i]
        y_goal_i = goal_dataset[goal_indices_i]
        y_start_i, y_goal_i = torch.FloatTensor(y_start_i), torch.FloatTensor(y_goal_i)
        y_start_i, y_goal_i = normalize(y_start_i, args.world_size), normalize(y_goal_i, args.world_size)
        y_start_i, y_goal_i = to_var(y_start_i), to_var(y_goal_i)
        bobs = obs[env_indices_i].astype(np.float32)
        bobs = torch.FloatTensor(bobs)
        bobs = to_var(bobs)
        loss = smpnet.loss(bi, smpnet.train_forward(bi, y_start_i, y_goal_i, bobs), beta=args.beta)
        loss = torch.mean(loss)
        scheduler.step(loss)
        if prev_epoch_val_loss is not None and (loss.cpu().item() - prev_epoch_val_loss) / prev_epoch_val_loss >= .5:
            # when the loss increases too much that the difference is larger than ratio * prev_epoch_val_loss
            # we early stop without saving for this epoch
            print('early stop with epoch vaildation loss: %f, previous loss: %f' % (loss.cpu().item(), prev_epoch_val_loss))
            break
        prev_epoch_val_loss = loss.cpu().item()
        
        # Save the models
        if epoch > 0 and epoch % 1 == 0:
            model_path='smpnet_epoch_%d.pkl' %(epoch)
            save_state(smpnet, torch_seed, np_seed, py_seed, os.path.join(model_dir,model_path))
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_dir', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/',help='path for saving trained models')
parser.add_argument('--no_env', type=int, default=100,help='directory for obstacle images')
parser.add_argument('--no_motion_paths', type=int,default=4000,help='number of optimal paths in each environment')
# Model parameters
parser.add_argument('--e_net_input_size', type=int, default=32, help='dimension of environment encoding network input')
parser.add_argument('--e_net_output_size', type=int, default=28, help='dimension of environment encoding network output')
parser.add_argument('--model_id', type=int, default=0, help='id of the model that we are using')

parser.add_argument('--input_size', type=int, default=3, help='dimension of input to be reconstructed')
parser.add_argument('--latent_size', type=int, default=28, help='dimension of latent variable')
parser.add_argument('--cond_size', type=int, default=6+28, help='dimension of condition variable (start, goal, environment encoding)')

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=1.0, help='scale the KL divergence loss relative to reconstruction loss')

parser.add_argument('--lr_schedule_gamma', type=float, default=0.8)
parser.add_argument('--lr_schedule_patience', type=int, default=2)

parser.add_argument('--device', type=int, default=0, help='cuda device')

parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=100, help='rehersal on how many data (not path)')
parser.add_argument('--data_folder', type=str, default='../data/simple/')

parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--env_type', type=str, default='s2d', help='environment')
parser.add_argument('--world_size', nargs='+', type=float, default=20., help='boundary of world')
parser.add_argument('--opt', type=str, default='Adam')

args = parser.parse_args()
print(args)
main(args)