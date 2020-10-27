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
    if args.model_type == "SMPNet":
        smpnet = SMPNet(e_net, cvae)
    elif args.model_type == "SMPPathNet":
        smpnet = SMPPathNet(e_net, cvae)
    elif args.model_type == "SMPPathWithPriorNet":
        smpnet = SMPPathWithPriorNet(e_net, cvae)
    elif args.model_type == "SMPPathSimpleNet":
        smpnet = SMPPathSimpleNet(e_net, cvae)


    model_dir = args.model_dir + '%s/%s/model_%d/param_%s/' % (args.env_type, args.model_type, args.model_id, args.param_name)

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
    val_sp = int(args.no_motion_paths * (1-args.val_ratio))
    data_folder = args.data_folder+args.env_type+'/'
    if args.model_type == "SMPPathNet":
        load_dis_ratio = True
    elif args.model_type == "SMPPathWithPriorNet":
        load_dis_ratio = True
    elif args.model_type == "SMPPathSimpleNet":
        load_dis_ratio = True
    elif args.model_type == "SMPNet":
        load_dis_ratio = False
    waypoint_dataset, cond_dataset, obs, env_indices = data_loader.load_train_dataset(N=args.no_env, NP=val_sp, s=0, sp=0,
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

    # validation dataset
    val_waypoint_dataset, val_cond_dataset, val_obs, val_env_indices \
             = data_loader.load_train_dataset(N=args.no_env, NP=args.no_motion_paths-val_sp+1, s=0, sp=val_sp,
                                              data_folder=data_folder, load_dis_ratio=load_dis_ratio)
    
    # randomize the dataset before training
    data=list(zip(val_waypoint_dataset, val_cond_dataset, val_env_indices))
    random.shuffle(data)
    val_waypoint_dataset,val_cond_dataset,val_env_indices=list(zip(*data))
    val_waypoint_dataset = list(val_waypoint_dataset)
    val_cond_dataset = list(val_cond_dataset)
    val_env_indices = list(val_env_indices)
    val_waypoint_dataset = np.array(val_waypoint_dataset)
    val_cond_dataset = np.array(val_cond_dataset)
    val_env_indices = np.array(val_env_indices)    
    val_len = len(val_waypoint_dataset)



    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(smpnet.opt, mode='min', \
                                        factor=args.lr_schedule_gamma, patience=args.lr_schedule_patience, verbose=True)
    epoch_early_stop_checker = EarlyStopChecker(1, 10)
    early_stop_checker = EarlyStopChecker(args.early_stop_freq, args.early_stop_patience)
    # Train the Models
    print('training...')

    writer_fname = 'env_%s_param_%s' % (args.env_type, args.param_name)

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
    loss_steps = args.log_steps
    save_steps = args.save_steps
    prev_epoch_val_loss = None
    for epoch in trange(args.start_epoch+1,args.num_epochs+1):
        print('epoch' + str(epoch))
        val_i = 0
        for i in trange(0,len(waypoint_dataset),args.batch_size):
            smpnet.train()
            print('epoch: %d, training... path: %d' % (epoch, i+1))
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

            print('before training losses:')
            print(smpnet.loss(bi, smpnet.train_forward(bi, cond_dataset_i, bobs), beta=args.beta).mean())
            smpnet.step(bi, cond_dataset_i, bobs, beta=args.beta)
            print('after training losses:')
            print(smpnet.loss(bi, smpnet.train_forward(bi, cond_dataset_i, bobs), beta=args.beta).mean())
            loss = smpnet.loss(bi, smpnet.train_forward(bi, cond_dataset_i, bobs), beta=args.beta)
            loss = torch.mean(loss, dim=0)
            #update_line(hl, ax, [i//args.batch_size, loss.data.numpy()])
            loss_gen = smpnet.generation_loss(bi, smpnet.train_forward(bi, cond_dataset_i, bobs)).mean(dim=0)
            loss_kl = smpnet.kl_divergence(bi, smpnet.train_forward(bi, cond_dataset_i, bobs)).mean()
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
            smpnet.eval()
            waypoint_dataset_i = val_waypoint_dataset[val_i:val_i+args.batch_size]
            cond_dataset_i = val_cond_dataset[val_i:val_i+args.batch_size]
            env_indices_i = val_env_indices[val_i:val_i+args.batch_size]
            val_i = val_i + args.batch_size
            if val_i > val_len:
                val_i = 0
            # record
            bi = waypoint_dataset_i
            bi = torch.FloatTensor(bi)
            bi = normalize(bi, args.world_size)
            bi=to_var(bi)
            
            cond_dataset_i = torch.FloatTensor(cond_dataset_i)
            cond_dataset_i = normalize(cond_dataset_i, args.world_size)  # assume the first Sx2 are start and goal
            cond_dataset_i = to_var(cond_dataset_i)

            bobs = val_obs[env_indices_i].astype(np.float32)
            bobs = torch.FloatTensor(bobs)
            bobs = to_var(bobs)
            loss = smpnet.loss(bi, smpnet.train_forward(bi, cond_dataset_i, bobs), beta=args.beta)
            loss = torch.mean(loss, dim=0)
            print('validation loss: ', loss.cpu().data)
            loss_gen = smpnet.generation_loss(bi, smpnet.train_forward(bi, cond_dataset_i, bobs)).mean(dim=0)
            loss_kl = smpnet.kl_divergence(bi, smpnet.train_forward(bi, cond_dataset_i, bobs)).mean()
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

                # check early stop using average loss
                early_stopping = early_stop_checker.early_stop_check(torch.mean(val_loss_avg).item())
                if early_stopping:
                    # store model and just exit
                    # Save the models
                    if epoch > 0 and epoch % 1 == 0:
                        model_path='smpnet_epoch_%d_iter_%d.pkl' %(epoch, int(i/args.batch_size))
                        save_state(smpnet, torch_seed, np_seed, py_seed, os.path.join(model_dir,model_path))
                    writer.export_scalars_to_json("./all_scalars.json")
                    writer.close()          
                    return          

                val_record_i += 1
                val_loss_avg = 0.
                val_loss_gen_avg = 0.
                val_loss_kl_avg = 0.
                val_loss_avg_i = 0

            # save model after certain steps
            if int(i/args.batch_size) % save_steps == 0:
                model_path='smpnet_epoch_%d_iter_%d.pkl' %(epoch, int(i/args.batch_size))
                save_state(smpnet, torch_seed, np_seed, py_seed, os.path.join(model_dir,model_path))


        # do a validation test using 2048 samples
        smpnet.eval()
        epoch_val_num = 2048
        waypoint_dataset_i = val_waypoint_dataset[:epoch_val_num]
        cond_dataset_i = val_cond_dataset[:epoch_val_num]
        env_indices_i = val_env_indices[:epoch_val_num]
        bi = waypoint_dataset_i
        bi = torch.FloatTensor(bi)
        bi = normalize(bi, args.world_size)
        bi=to_var(bi)
        cond_dataset_i = torch.FloatTensor(cond_dataset_i)
        cond_dataset_i = normalize(cond_dataset_i, args.world_size)  # assume the first Sx2 are start and goal
        cond_dataset_i = to_var(cond_dataset_i)

        bobs = val_obs[env_indices_i].astype(np.float32)
        bobs = torch.FloatTensor(bobs)
        bobs = to_var(bobs)
        loss = smpnet.loss(bi, smpnet.train_forward(bi, cond_dataset_i, bobs), beta=args.beta)
        loss = torch.mean(loss)
        writer.add_scalar('epoch_val_loss', loss, epoch)
        scheduler.step(loss)
        if epoch_early_stop_checker.early_stop_check(loss.cpu().item()):
            #if prev_epoch_val_loss is not None and (loss.cpu().item() - prev_epoch_val_loss) / prev_epoch_val_loss > 0.:
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



##############################################################################
import yaml
# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper

parser = argparse.ArgumentParser()
# for training
parser.add_argument('--param_path', type=str, default='param/train/',help='path for loading training param')
parser.add_argument('--param_name', type=str, default="cvae_s2d_param1.yaml")

# parse the parameter file
args = parser.parse_args()
print(args)
param_f = open(args.param_path+args.param_name, 'r')
param = yaml.load(param_f)
param = DictDot(param)
main(param)