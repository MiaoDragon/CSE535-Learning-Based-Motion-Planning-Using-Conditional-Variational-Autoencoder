import sys
sys.path.insert(1, '../FasterRobusterMotionPlanningLibrary/python')
import numpy as np

from frmpl.env.s2d_utility import Environment, CollisionChecker, Metrics
from frmpl.nearest_computer.naive_nearest_neighbor import NearestComputer
from frmpl.planner_structure.plan_tree import PlanStructure

from tools.data_loader_s2d import load_test_dataset
import importlib
import argparse
def main(args):
    #data_path = '/freespace/local/ym420/course/cse535/data/s2d/''
    data_path = '/home/yinglong/Documents/research/learning_motion_planning/data/s2d/'
    obs_center,obs_pcd,paths,path_lengths = load_test_dataset(N=10,NP=200, s=0,sp=4000, folder=data_path)

    # find one path with longer length
    def path_length_calculate(path):
        return np.linalg.norm(path[1:]-path[:-1], axis=1).sum()

    obs_i = -1
    path_i = -1
    for i in range(len(paths)):
        for j in range(len(paths[i])):
            if path_length_calculate(paths[i][j][:path_lengths[i][j]]) >= 10.:
                # interesting path
                obs_i = i
                path_i = j
                break
        if obs_i != -1:
            break

    obs_width = 5.0
    print('obstacle center:')
    print(obs_center.shape)

    obs_center = obs_center[obs_i]
    paths = paths[obs_i][path_i]
    path_lengths = path_lengths[obs_i][path_i]
    paths = paths[:path_lengths]
    # create environment
    env = Environment(obs_center, obs_width)
    collision_checker = CollisionChecker(env)
    d_metrics = Metrics()
    nearest_computer = NearestComputer()
    planner_structure = PlanStructure()
    class Sampler():
        def sample(self):
            # uniformly sample in the state space
            x = np.random.uniform(low=env.x_lower_bound, high=env.x_upper_bound)
            print('drawing sample: ', x)
            return x


    planner_name = args.planner_name
    planner_module = importlib.import_module('frmpl.planners.%s' % (planner_name))


    sampler = Sampler()
    planner = planner_module.Planner(sampler, nearest_computer, d_metrics, collision_checker, planner_structure)
    eps = 0.01
    radius = 20.
    goal_bias = 0.1

    plan_params = {'radius': radius, 'goal_bias': goal_bias}
    planner.setup(paths[0], paths[-1], 0.1, env, eps, plan_params)
    #planner.step()

    # visualization
    obs_center_i = obs_center
    path_i = paths

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from frmpl.visual.visualize_planner import visualize_tree_2d
    import imageio
    import os

    def plot_and_save(path='plots/planner/%s/s2d/' % (planner_name)):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # visualize obstacles
        ax.set_xlim(-22, 22)
        ax.set_ylim(-22, 22)
        #print(obs_center_i.shape)

        for i in range(len(obs_center_i)):
            x, y = obs_center_i[i][0], obs_center_i[i][1]
            obs_patch_i = patches.Rectangle((x-obs_width/2,y-obs_width/2),\
                                            obs_width,obs_width,\
                                            linewidth=0.0, facecolor='black')
            ax.add_patch(obs_patch_i)

        # if informedrrt, then visualize the sample
        if args.planner_name == "informedrrtstar":
            samples = planner.informed_samples
            if len(samples) > 0:
                # visualize the sample
                ax.scatter(samples[:,0], samples[:,1], c='palegreen', alpha=0.7)
        # show start and goal
        ax.scatter([path_i[0][0]], [path_i[0][1]], c='green', s=100.0)
        ax.scatter([path_i[-1][0]], [path_i[-1][1]], c='red', s=100.0, marker='*')


        visualize_tree_2d(planner_structure, ax)
        # visualize the path if it exists
        if planner.node_sol is not None:
            opt_path = planner.get_solution()
            opt_path = np.array(opt_path)
            ax.scatter(opt_path[:,0], opt_path[:,1], c='orangered')
            ax.plot(opt_path[:,0], opt_path[:,1], c='orangered')
        return fig


    def make_video():
        def tryint(s):
            try:
                return int(s)
            except ValueError:
                return s
        def str2int(v_str):
            idx = v_str.split('_')[1]
            idx = int(idx[:-4])
            return idx

        def sort_humanly(v_list):
            return sorted(v_list, key=str2int)
        image_folder = 'plots/planner/%s/s2d/' % (planner_name)
        video_folder = 'video/planner/%s/s2d/' % (planner_name)   
        os.makedirs(video_folder, exist_ok=True)
        video_name = video_folder+'tree.gif'
        images = [img for img in os.listdir(image_folder) \
                if 'plot_' in img]
        images = sort_humanly(images)
        imgs = []
        for filename in images:
    #         print('./'+image_folder+'/'+filename)
            imgs.append(imageio.imread('./'+image_folder+'/'+filename))
        imageio.mimsave(video_name, imgs)

    max_step = 200
    plot_path='plots/planner/%s/s2d/' % (planner_name)
    for i in range(max_step):
        fig = plot_and_save(plot_path)
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(plot_path+'plot_%d.png' % (i))
        plt.close(fig)
        res = planner.step()
        print('step %d...' % (i))
        print('step result: ', res)
        if res:
            print('current solution cost: %f' % (planner.node_sol.cost))
        #if res:
        #    break
    make_video()






parser = argparse.ArgumentParser()
parser.add_argument('--planner_name', type=str, default='rrtstar')

args = parser.parse_args()
main(args)