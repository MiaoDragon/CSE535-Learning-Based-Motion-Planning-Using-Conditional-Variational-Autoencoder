"""
using SST* to generate near-optimal paths in specified environment
"""
import sys
sys.path.append('../../FasterRobusterMotionPlanningLibrary/python')
import argparse
import numpy as np
import time
import pickle
import os
import gc
from multiprocessing import Process, Queue
import importlib
# environment parameters are stored in param

def main(args, env_param):
    # set up the environment

    ####################################################################################
    def plan_one_path(planner, out_queue, path_file, control_file, cost_file, time_file):
        # generate a path by using SST to plan for some maximal iterations
        time0 = time.time()
        #print('obs: %d, path: %d' % (i, j))
        for iter in range(args.max_iter):
            #print('iteration: %d' % (iter))
            planner.step(env, min_time_steps, max_time_steps, integration_step)

        solution = planner.get_solution()
        plan_time = time.time() - time0
        if solution is None:
            out_queue.put(0)
        else:
            print('path succeeded.')
            path, controls, cost = solution
            print(path)
            path = np.array(path)
            controls = np.array(controls)
            cost = np.array(cost)

            file = open(path_file, 'wb')
            pickle.dump(path, file)
            file.close()
            file = open(control_file, 'wb')
            pickle.dump(controls, file)
            file.close()
            file = open(cost_file, 'wb')
            pickle.dump(cost, file)
            file.close()
            file = open(time_file, 'wb')
            pickle.dump(plan_time, file)
            file.close()
            out_queue.put(1)
    ####################################################################################
    queue = Queue(1)
    id_list = []

    env_obs_gen = importlib.import_module('%s_obs_gen' % (args.env_name))
    if env_param['gen_obs']:
        # generate obstacle here
        pass
        env_obs_gen.obs_gen(env_param['obs_params'], args.obs_folder)
    obs = env_obs_gen.obs_load(args.obs_folder, args.N, args.s)

    # load sg gen code
    env_sg_gen = importlib.import_module('%s_sg_gen' % (args.env_name))

    # set up planner
    planner_module = importlib.import_module('frmpl.planners.%s' % (env_param['planner_type']))
    nearest_computer_module = importlib.import_module('frmpl.nearest_computer.%s' % (env_param['nearest_neighbor']))
    plan_utility = importlib.import_module('frmpl.env.%s_utility' % (args.env_name))
    plan_struct_module = importlib.import_module('frmpl.planner_structure.%s' % (env_param['plan_struct_type']))

    nearest_computer = nearest_computer_module.NearestComputer()

    low = plan_param['x_lower_bound']
    high = plan_param['x_upper_bound']
    low = np.array(low), high = np.array(high)

    plan_env = plan_utility.Environment(low, high)

    collision_checker = plan_utility.CollisionChecker()
    metrics = plan_utility.Metrics()
    plan_struct = plan_struct_module.PlanStructure()

    sample_module = importlib.import_module('frmpl.sample.%s' % (env_param['sample_type']))
    sampler = sample_module.Sampler(plan_env)


    planner = planner_module.Planner(sampler, nearest_computer, metrics, collision_checker, plan_struct)

    plan_env_data = env_param['plan_env_data']  # representing obs


    for i in range(args.N):
        obs_i = obs[i]
        plan_env.set_obs(obs_i, plan_env_data)
        collision_checker.set_env(plan_env)

        paths = []
        actions = []
        costs = []
        times = []
        suc_n = 0

        for j in range(args.NP):
            plan_start = time.time()
            while True:
                print('env_id: %d, path_id: %d' % (i, j))
                # randomly sample collision-free start and goal
                #start = np.random.uniform(low=low, high=high)
                #end = np.random.uniform(low=low, high=high)


                # generate start and goal
                start, end = env_sg_gen.start_goal_gen(low, high, obs_i)
                
                # for checking if start-goal pair appeared before
                start_str = []
                for si in range(len(start)):
                    start_str.append(round(start[si], round_digit))
                end_str = []
                for si in range(len(start)):
                    end_str.append(round(end[si], round_digit))
                id_new = str(start_str + end_str)
                #print('start end id: %s' % (id_new))
                if id_new in id_list:
                    print('same start goal!')
                    continue

                x_start = path_i[0]
                x_goal = path_i[-1]
                planner.setup(x_start, x_goal, env_param['d_goal'], plan_env, env_param['eps'], env_param['plan_param'])


                dir = args.path_folder+str(i+args.s)+'/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                path_file = dir+'state'+'_%d'%(j+args.sp) + ".pkl"
                control_file = 'control'+'_%d'%(j+args.sp) + ".pkl"
                cost_file = 'cost'+'_%d'%(j+args.sp) + ".pkl"
                time_file = 'time'+'_%d'%(j+args.sp) + ".pkl"
                sg_file = dir+'start_goal'+'_%d'%(j+args.sp)+".pkl"
                p = Process(target=plan_one_path, args=(planner, queue, path_file, control_file, cost_file, time_file))
                p.start()
                p.join()
                res = queue.get()
                print('obtained result:')
                print(res)
                if res:
                    # plan successful
                    file = open(sg_file, 'wb')
                    sg = [start, end]
                    pickle.dump(sg, file)
                    file.close()
                    
                    # add to the list of ids
                    id_list.append(id_new)
                    break
            print('path planning time: %f' % (time.time() - plan_start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='cartpole')

    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--s', type=int, default=0)
    parser.add_argument('--sp', type=int, default=0)
    parser.add_argument('--NP', type=int, default=1)
    # parser.add_argument('--max_iter', type=int, default=10000)
    # parser.add_argument('--path_folder', type=str, default='./data/cartpole/')
    # parser.add_argument('--path_file', type=str, default='path')
    # parser.add_argument('--control_file', type=str, default='control')
    # parser.add_argument('--cost_file', type=str, default='cost')
    # parser.add_argument('--time_file', type=str, default='time')
    # parser.add_argument('--sg_file', type=str, default='start_goal')
    # parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
    # parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')
    args = parser.parse_args()

    # load yaml file
    import yaml
    env_param_f = open('param/%s.yaml' % (args.env_name), 'r')
    env_param = yaml.load(env_param_f)
    main(args, env_param)
