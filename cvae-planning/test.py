"""
This implements the Kinodynamic Planning using MPNet, by using MPNet
to generate random samples, that will guide the SST algorithm.
"""
import sys
sys.path.append('../FasterRobusterMotionPlanningLibrary/python')
#sys.path.append('/freespace/local/ym420/course/cse535/CSE535-Learning-Based-Motion-Planning-Using-Conditional-Variational-Autoencoder/FasterRobusterMotionPlanningLibrary/python')
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
import time
# planner library
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
    unnormalize = plan_util.unnormalize

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
    data_folder = args.data_folder+args.env_type+'/'
    obs_repre,obs_pcd,paths,path_lengths = data_loader.load_test_dataset(N=args.no_env,NP=args.no_motion_paths, \
                                                    s=args.s,sp=args.sp, folder=data_folder)
    # obs_repre: obstacle representation for collision checking
    #     
    # test
    # setup planner
    planner_module = importlib.import_module('frmpl.planners.%s' % (args.planner_type))
    nearest_computer_module = importlib.import_module('frmpl.nearest_computer.%s' % (args.nearest_neighbor))
    plan_utility = importlib.import_module('frmpl.env.%s_utility' % (args.env_type))
    plan_struct_module = importlib.import_module('frmpl.planner_structure.%s' % (args.plan_struct_type))

    nearest_computer = nearest_computer_module.NearestComputer()
    plan_env = plan_utility.Environment()
    collision_checker = plan_utility.CollisionChecker()
    metrics = plan_utility.Metrics()
    plan_struct = plan_struct_module.PlanStructure()

    # setup sampler: uniform or SMPNet
    class SMPSampler():
        def __init__(self, smpnet, obs_pcd=None):
            self.smpnet = smpnet
            self.smpnet.eval()
            self.obs_pcd = obs_pcd
        def set_input(self, x_start, x_goal, obs_pcd, env):
            self.x_start = x_start
            self.x_goal = x_goal
            self.obs_pcd = obs_pcd
            self.env = env
            self.path_len = 4  # init
            self.disc_num = 10
            self.samples = self.sample_batch(x_start, x_goal, obs_pcd, self.path_len, self.disc_num)
            self.sample_idx = 0
            self.smp_sample_ratio = 0.9
        def sample(self):
            # extract samples stored in sample_batch
            # generate more if used up
            
            # has 90% chance of using learned sampler
            if np.random.rand() > self.smp_sample_ratio:
                # use uniform samplling
                return np.random.uniform(low=self.env.x_lower_bound, high=self.env.x_upper_bound)

            if self.sample_idx == len(self.samples):
                # generate batch sample
                # dynamically increase the path_len
                self.path_len = self.path_len * 2
                self.disc_num = max(int(self.disc_num / 2), 1)
                print('increasing sample length')
                self.samples = self.sample_batch(self.x_start, self.x_goal, self.obs_pcd, self.path_len, self.disc_num)
                self.sample_idx = 0
            res = self.samples[self.sample_idx]
            self.sample_idx += 1
            return res

        def sample_batch(self, start_i, goal_i, obs_i, path_len, disc_num):
            # path_len: number of nodes on the path
            # disc_num: number of node to draw for each specific discete position
            dist = np.linspace(1./path_len, 1., num=path_len)
            dist = np.tile(dist, [disc_num, 1]).reshape(-1,1)
            num_sample = path_len * disc_num

            start_i = np.tile(start_i, [num_sample, 1])
            goal_i = np.tile(goal_i, [num_sample, 1])
            cond_dataset_i = np.concatenate([start_i, goal_i, dist], axis=1)
            cond_dataset_i = torch.FloatTensor(cond_dataset_i)
            cond_dataset_i = normalize(cond_dataset_i, args.world_size)  # assume the first Sx2 are start and goal
            cond_dataset_i = to_var(cond_dataset_i)
            
            #print([num_sample]+list(obs_i.shape))
            bobs = torch.FloatTensor(obs_i).unsqueeze(0).repeat([num_sample, 1, 1, 1])
            bobs = to_var(bobs)
            #print(bobs.size())
            # generating
            samples = self.smpnet.gen_forward(cond_dataset_i, obs=bobs, obs_z=None)
            samples = samples.cpu().data
            samples = unnormalize(samples, args.world_size).numpy()
            return samples

    # decide which sample to use
    if args.sample_type == "smpnet":
        sampler = SMPSampler(smpnet)
    else:
        sample_module = importlib.import_module('frmpl.sample.%s' % (args.sample_type))
        sampler = sample_module.Sampler(plan_env)
    
    planner = planner_module.Planner(sampler, nearest_computer, metrics, collision_checker, plan_struct)

    if args.env_type == "s2d":
        plan_env_data = 5.0 # obs_width

    if args.visual:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from tools.s2d_plan_visual import plot_and_save, make_video

    # log results
    plan_results = {}
    plan_results['init_sol_time'] = [[] for i in range(args.no_env)]
    plan_results['final_sol_time'] = [[] for i in range(args.no_env)]
    plan_results['init_sol_iter'] = [[] for i in range(args.no_env)]
    plan_results['final_sol_iter'] = [[] for i in range(args.no_env)]
    plan_results['init_sol_cost'] = [[] for i in range(args.no_env)]
    plan_results['final_sol_cost'] = [[] for i in range(args.no_env)]
    #plan_results['init_sol_state'] = [[] for i in range(args.no_env)]
    plan_results['sol_state'] = [[] for i in range(args.no_env)]  # final path
    plan_results['success'] = [[] for i in range(args.no_env)]  # 0 or 1

    result_folder = args.result_folder+"%s/%s/%s/param_%d/" % \
                        (args.env_type, args.planner_type, args.sample_type, args.param_name)

    os.makedirs(result_folder, exist_ok=True)

    running_stats = {}
    running_stats['success'] = []
    running_stats['final_time'] = []
    running_stats['final_cost'] = []
    #hard instance: [(0, 28), (0, 15)]
    for obs_idx in trange(args.no_env):
        for path_idx in trange(args.no_motion_paths):
            print('obs_idx: %d, path_idx: %d' % (obs_idx, path_idx))
            path_i = paths[obs_idx][path_idx]
            path_length_i = path_lengths[obs_idx][path_idx]
            path_i = path_i[:path_length_i]
            if path_length_i <=1:
                plan_results['success'][obs_idx].append(None)
                plan_results['init_sol_time'][obs_idx].append(None)
                plan_results['final_sol_time'][obs_idx].append(None)
                plan_results['init_sol_iter'][obs_idx].append(None)
                plan_results['final_sol_iter'][obs_idx].append(None)
                plan_results['init_sol_cost'][obs_idx].append(None)
                plan_results['final_sol_cost'][obs_idx].append(None)
                plan_results['sol_state'][obs_idx].append(None)
                continue
            plan_env.set_obs(obs_repre[obs_idx], plan_env_data)
            collision_checker.set_env(plan_env)
            x_start = path_i[0]
            x_goal = path_i[-1]
            planner.setup(x_start, x_goal, args.d_goal, plan_env, args.eps, args.plan_param)
            if args.sample_type == "smpnet":
                sampler.set_input(x_start, x_goal, obs_pcd[obs_idx], plan_env)
            # obtain expert solution cost
            data_cost = np.linalg.norm(path_i[1:]-path_i[:-1], axis=1).sum()

            # visualization
            if args.visual:
                plot_path = "plots/planner/%s/%s/%s/param_%d/e_%d_p_%d/" % \
                            (args.env_type, args.planner_type, args.sample_type, args.param_name, obs_idx+args.s, path_idx+args.sp)
                video_path = "video/planner/%s/%s/%s/param_%d/e_%d_p_%d/" % \
                            (args.env_type, args.planner_type, args.sample_type, args.param_name, obs_idx+args.s, path_idx+args.sp)
                os.makedirs(plot_path, exist_ok=True)
                os.makedirs(video_path, exist_ok=True)

            start_time = time.time()
            last_status = False
            init_time = -1
            final_time = -1
            init_iter = -1
            final_iter = -1
            init_cost = -1
            final_cost = -1
            
            for plan_iter in trange(args.max_iter):
                #print('plan iteration %d...' % (plan_iter))
                status = planner.step()
                if args.visual:
                    fig = plot_and_save(planner, obs_repre[obs_idx], plan_env_data, path_i, plan_struct, path=plot_path)
                    plt.savefig(plot_path+'plot_%d.png' % (plan_iter))
                    plt.close(fig)

                # record initial solution time
                if last_status == False and status:
                    init_time = time.time() - start_time
                    init_iter = plan_iter
                    init_cost = planner.node_sol.cost
                # record optimal solution time
                if status and planner.node_sol.cost <= data_cost * 1.10:
                    break
                last_status = status
            final_time = time.time() - start_time
            final_iter = plan_iter
            if status:
                final_cost = planner.node_sol.cost
            # print('initial solution time: %f' % (init_time))
            # print('final solution time: %f' % (opt_time))
            # print('solution:')
            # print(solution)
            solution = planner.get_solution()
            if solution is None:
                success = 0
            else:
                success = 1
                solution = np.array(solution).tolist()

            print('success: %d' % (success))
            if success:
                print('init_iter: %d, init_time: %f, init_cost: %f (%.2f)' % (init_iter, init_time, init_cost, init_cost / data_cost))
                print('final_iter: %d, final_time: %f, final_cost: %f (%.2f)' % (final_iter, final_time, final_cost, final_cost / data_cost))

            # store 
            plan_results['success'][obs_idx].append(success)
            plan_results['init_sol_time'][obs_idx].append(init_time)
            plan_results['final_sol_time'][obs_idx].append(final_time)
            plan_results['init_sol_iter'][obs_idx].append(init_iter)
            plan_results['final_sol_iter'][obs_idx].append(final_iter)
            plan_results['init_sol_cost'][obs_idx].append(init_cost)
            plan_results['final_sol_cost'][obs_idx].append(final_cost)
            plan_results['sol_state'][obs_idx].append(solution)

            # running stats
            running_stats['success'].append(success)
            if success:
                running_stats['final_time'].append(final_time)
                running_stats['final_cost'].append(final_cost)
            print('current success rate: %f' % (np.mean(running_stats['success'])))
            print('current mean time: %f, mean cost: %f' % (np.mean(running_stats['final_time']), np.mean(running_stats['final_cost'])))


            if args.visual:
                make_video(plot_path, video_path)
        # save the current plan_results after one environment
        import json

        f = open(result_folder+"N_%d_NP_%d_s_%d_sp_%d.json" % (args.no_env, args.no_motion_paths, args.s, args.sp), 'w')
        json.dump(plan_results, f)


##############################################################################
import yaml
# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper

parser = argparse.ArgumentParser()
# for training
parser.add_argument('--param_path', type=str, default='param/planning/',help='path for loading training param')
parser.add_argument('--param_name', type=str, default="cvae_s2d_param1.yaml")

# parse the parameter file
args = parser.parse_args()
print(args)
param_f = open(args.param_path+args.param_name, 'r')
param = yaml.load(param_f)
param = DictDot(param)
main(param)