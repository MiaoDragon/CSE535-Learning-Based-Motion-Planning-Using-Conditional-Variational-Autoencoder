"""
using SST* to generate near-optimal paths in specified environment
"""
import sys
sys.path.append('../../FasterRobusterMotionPlanningLibrary/python')
sys.path.append('../../')

import argparse
import numpy as np
import time
import pickle
import os
import gc
from multiprocessing import Process, Queue
import importlib
import numpy as np
from tqdm import tqdm, trange

# environment parameters are stored in param

def main(args, env_param):
    # set up the environment

    id_list = []

    env_obs_gen = importlib.import_module('%s_obs_gen' % (env_param['env_name']))
    if env_param['gen_obs']:
        # generate obstacle here
        pass
        env_obs_gen.obs_gen(env_param['obs_params'], env_param['obs_folder'])
    obs = env_obs_gen.obs_load(env_param['obs_folder'], args.N, args.s)

    # load sg gen code
    env_sg_gen = importlib.import_module('%s_sg_gen' % (env_param['env_name']))

    # set up planner
    planner_module = importlib.import_module('frmpl.planners.%s' % (env_param['planner_type']))
    nearest_computer_module = importlib.import_module('frmpl.nearest_computer.%s' % (env_param['nearest_neighbor']))
    plan_utility = importlib.import_module('frmpl.env.%s_utility' % (env_param['env_name']))
    plan_struct_module = importlib.import_module('frmpl.planner_structure.%s' % (env_param['plan_struct_type']))

    nearest_computer = nearest_computer_module.NearestComputer()

    low = env_param['x_low']
    high = env_param['x_high']
    low = np.array(low)
    high = np.array(high)

    plan_env = plan_utility.Environment(low, high)

    collision_checker = plan_utility.CollisionChecker()
    metrics = plan_utility.Metrics()
    plan_struct = plan_struct_module.PlanStructure()

    sample_module = importlib.import_module('frmpl.sample.%s' % (env_param['sample_type']))
    sampler = sample_module.Sampler(plan_env)


    planner = planner_module.Planner(sampler, nearest_computer, metrics, collision_checker, plan_struct)

    plan_env_data = env_param['plan_env_data']  # representing obs

    if env_param['visual']:
        import matplotlib.pyplot as plt
        visual_module = importlib.import_module('cvae-planning.tools.%s_plan_visual' % (env_param['env_name']))
        plot_and_save = visual_module.plot_and_save
        make_video = visual_module.make_video


    # for checking if start and goal already exists
    round_digit = env_param['round_digit']

    for i in trange(args.N):
        obs_i = obs[i]
        plan_env.set_obs(obs_i, plan_env_data)
        collision_checker.set_env(plan_env)

        paths = []
        actions = []
        costs = []
        times = []
        suc_n = 0

        for j in trange(args.NP):
            plan_start = time.time()
            trial = -1

            dir = env_param['path_folder']+str(i+args.s)+'/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            path_file = dir+'state'+'_%d'%(j+args.sp) + ".pkl"
            control_file = dir+'control'+'_%d'%(j+args.sp) + ".pkl"
            cost_file = dir+'cost'+'_%d'%(j+args.sp) + ".pkl"
            time_file = dir+'time'+'_%d'%(j+args.sp) + ".pkl"
            sg_file = dir+'start_goal'+'_%d'%(j+args.sp)+".pkl"

            while True:
                trial += 1
                #print('env_id: %d, path_id: %d' % (i, j))
                # generate start and goal
                start, end = env_sg_gen.start_goal_gen(low, high, collision_checker, env_param['eps'])
                
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

                x_start = start
                x_goal = end

                if env_param['visual']:
                    plot_path = "plots/planner/%s/%s/e_%d_p_%d/trial_%d/" % \
                                (args.env_name, env_param['planner_type'], i+args.s, j+args.sp, trial)
                    video_path = "video/planner/%s/%s/e_%d_p_%d/trial_%d/" % \
                                (args.env_name, env_param['planner_type'], i+args.s, j+args.sp, trial)
                    os.makedirs(plot_path, exist_ok=True)
                    os.makedirs(video_path, exist_ok=True)

                planner.setup(x_start, x_goal, env_param['d_goal'], plan_env, env_param['eps'], env_param['plan_param'])


                ############Planning##########################3
                time0 = time.time()
                #print('obs: %d, path: %d' % (i, j))
                for iter in range(env_param['max_iter']):
                    #print('iteration: %d' % (iter))
                    planner.step()
                    if env_param['visual']:
                        fig = plot_and_save(planner, obs_i, plan_env_data, [start, end], plan_struct, path=plot_path)
                        plt.savefig(plot_path+'plot_%d.png' % (iter))
                        plt.close(fig)

                if env_param['visual']:
                    make_video(plot_path, video_path)

                path_x = planner.get_solution()
                plan_time = time.time() - time0
                if path_x is None:
                    res = 0
                else:
                    print('path succeeded.')
                    path = path_x
                    #path, controls, cost = solution  # later implement this part when control is involved
                    print(path)
                    path = np.array(path)
                    #controls = np.array(controls)
                    #cost = np.array(cost)
                    cost = np.linalg.norm(path[1:] - path[:-1], axis=1)

                    file = open(path_file, 'wb')
                    pickle.dump(path, file)
                    file.close()
                    #file = open(control_file, 'wb')
                    #pickle.dump(controls, file)
                    #file.close()
                    file = open(cost_file, 'wb')
                    pickle.dump(cost, file)
                    file.close()
                    file = open(time_file, 'wb')
                    pickle.dump(plan_time, file)
                    file.close()
                    res = 1
                print('obtained result:')
                print(res)

                ################################################3
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

    args = parser.parse_args()

    # load yaml file
    import yaml
    import copy
    env_param_f = open('param/%s.yaml' % (args.env_name), 'r')
    env_param = yaml.load(env_param_f)


    #args_list = []
    ps = []
    if args.N > 1:
        for i in range(args.N):
            args_i = copy.deepcopy(args)
            args_i.N = 1
            args_i.s = i + args.s  # offset
            env_param_i = copy.deepcopy(env_param)


            for j in range(args.NP):
                # check if file already exists, stop at the last existing file or 0
                dir = env_param['path_folder']+str(i+args.s)+'/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                path_file = dir+'state'+'_%d'%(j+args.sp) + ".pkl"
                control_file = dir+'control'+'_%d'%(j+args.sp) + ".pkl"
                cost_file = dir+'cost'+'_%d'%(j+args.sp) + ".pkl"
                time_file = dir+'time'+'_%d'%(j+args.sp) + ".pkl"
                sg_file = dir+'start_goal'+'_%d'%(j+args.sp)+".pkl"

                if not os.path.exists(path_file):
                    break
            sp = max(args.sp, j-2)  # new sp starts from the last existing file
            args_i.sp = sp  # update sp to this process
            p = Process(target=main, args=(args_i, env_param_i))
            ps.append(p)
            p.start()
        print('finished starting processes.')
        try:
            for p in ps:
                p.join()
        except:
            print('terminating child processes...')
            for i in range(args.N):
                ps[i].terminate()
                ps[i].join()
            print('finished terminate.')            

    else:
        main(args, env_param)
