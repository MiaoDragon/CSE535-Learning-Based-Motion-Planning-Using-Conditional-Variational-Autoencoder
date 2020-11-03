import sys
sys.path.append('/freespace/local/ym420/course/cse535/CSE535-Learning-Based-Motion-Planning-Using-Conditional-Variational-Autoencoder/FasterRobusterMotionPlanningLibrary/python')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from frmpl.visual.visualize_planner import visualize_tree_2d
import imageio
import os
import numpy as np

def plot_and_save(planner, obs, plan_env_data, path_i, planner_structure, path='plots/planner/rrt/s2d/'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # visualize obstacles
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    #print(obs_center_i.shape)
    
    # when plotting figure, matplotlib will assume image has origin at the upper-left corner
    # and the x-axis is vertical, while y-axis is horizontal (which matches the matrix form)
    # hence we have to flip the axis when plotting
    obs_img = np.zeros(list(obs.T.shape) + [3]).astype(int)
    obs_img[:,:,0] = obs.T
    obs_img[:,:,1] = obs.T
    obs_img[:,:,2] = obs.T
    origin = 'lower'
    extent = (-10, 10., -10., 10.)
    ax.imshow(obs_img, origin=origin, extent=extent)

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


def make_video(image_folder, video_folder):
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
    #image_folder = 'plots/planner/rrt/s2d/'
    #video_folder = 'video/planner/rrt/s2d/'    
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
