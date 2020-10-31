import sys
sys.path.append('/freespace/local/ym420/course/cse535/CSE535-Learning-Based-Motion-Planning-Using-Conditional-Variational-Autoencoder/FasterRobusterMotionPlanningLibrary/python')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from frmpl.visual.visualize_planner import visualize_tree_2d
import imageio
import os

def plot_and_save(obs_center_i, obs_width, path_i, planner_structure, path='plots/planner/rrt/s2d/'):
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

    # show start and goal
    ax.scatter([path_i[0][0]], [path_i[0][1]], c='green', s=100.0)
    ax.scatter([path_i[-1][0]], [path_i[-1][1]], c='red', s=100.0, marker='*')


    visualize_tree_2d(planner_structure, ax)
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
