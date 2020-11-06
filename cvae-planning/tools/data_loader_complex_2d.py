"""
This implements data loader for both training and testing procedures.
"""
import pickle
import numpy as np
import sys
import os
import imageio

def sample_pcd(img, num_pts=4000, save_path='motion_planning_datasets/forest/train/pcd/', filename='0.npy'):
    # given the pixel img, sample point clouds representing the obs (0 in pixel values)
    # we assume the size of the workspace is [-20,20]^2
    x_low = -10.
    x_high = 10.
    y_low = -10.
    y_high = 10.
    # obtain the location for each mesh
    x_step = (x_high - x_low) / img.shape[0]
    y_step = (y_high - y_low) / img.shape[1]
    xy_low = np.array([x_low, y_low])
    xy_step = np.array([x_step, y_step])
    xy_max = np.array([img.shape[0], img.shape[1]])
    # mapping from location to pixel
    def loc_to_pixel(pt):
        # assume pt: Bx2
        pixel_xy = np.floor((pt - xy_low) / xy_step)
        pixel_xy = pixel_xy.astype(int)
        xs = pixel_xy[:,0]
        xs[xs==img.shape[0]] = xs[xs==img.shape[0]] - 1
        pixel_xy[:,0] = xs
        ys = pixel_xy[:,1]
        ys[ys==img.shape[1]] = ys[ys==img.shape[1]] - 1
        pixel_xy[:,1] = ys
        return pixel_xy
    samples = []
    current_size = 0
    while True:
        # uniformly sample points, and then obtain fixed number of pcd representing the obs
        new_samples = np.random.uniform(low=[x_low, y_low], high=[x_high, y_high], size=(num_pts*2,2))
        # check if in collision
        pixel_xy = loc_to_pixel(new_samples)
        selected_pixels = img[pixel_xy[:,0], pixel_xy[:,1]] # has shape: N
        new_samples = new_samples[selected_pixels == 0,:] # select the ones that are in collision
        if current_size + len(new_samples) >= num_pts:
            samples.append(new_samples[num_pts-current_size,:])
            break
        samples.append(new_samples)
        current_size += len(new_samples)
    samples = np.concatenate(samples, axis=0)
    # store
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path+filename, samples)
    return samples


def load_train_dataset(N, NP, s, sp, data_folder='motion_planning_datasets/forest/', load_dis_ratio=False):
    #  data_folder: motion_planning_datasets/forest
    path_folder =  data_folder + 'path/'  # folder containing all imitation trajectories
    obs_folder = data_folder + 'train/'  # we use the training folder as environment
    pcd_folder = data_folder + 'train/pcd/'  # folder storing the pcd
    # check if the pcd exists, if not, create a new one

    obs_repre = []
    obs_voxel = []
    obs_pcd = []
    for i in range(N):
        obs_repre_i = imageio.imread(obs_folder+'%d.png' % (i+s))
        obs_repre.append(obs_repre_i)
        # can directly use the image as "voxel"
        obs_voxel_i = np.zeros([3] + list(obs_repre_i.shape)).astype(int)  # first channel: 0,1 map  # second channel: row loc  # third channel: col loc
        
        obs_voxel_i[0] = obs_repre_i
        obs_voxel_i[0,obs_voxel_i[0]==0] = 1  # obstacle
        obs_voxel_i[0,obs_voxel_i[0]==255] = 0  # non-obstacle
        # unit testing here: visualize the loaded voxel

        img_indices = np.indices(obs_repre_i.shape)  # return: 2ximgshape
        obs_voxel_i[1] = img_indices[0]
        obs_voxel_i[2] = img_indices[1]
        obs_voxel_i = obs_voxel_i.astype(float)  # result: 3x201x201
        # scale indices
        obs_voxel_i[1] = obs_voxel_i[1] / obs_voxel_i[1].max()
        obs_voxel_i[2] = obs_voxel_i[2] / obs_voxel_i[2].max()

        obs_voxel.append(np.array(obs_voxel_i))

        # below is useful if using pcd

        # pcd_filename = pcd_folder + '%d.npy' % (i+s)
        # if os.path.exists(pcd_filename):
        #     # directly load if already there
        #     obs_pcd.append(np.load(pcd_filename))
        # else:
        #     # sample and save to disk
        #     obs_pcd_i = sample_pcd(obs_repre_i, num_pts=4000, save_path=pcd_folder, filename='%d.npy' % (i+s))
        #     obs_pcd.append(obs_pcd_i)

    obs_voxel = np.array(obs_voxel)
    obs_repre = np.array(obs_repre)

    # obtain path
    waypoint_dataset = []
    cond_dataset = []
    env_indices = []
    for i in range(N):
        for j in range(NP):
            path_filename = path_folder+'%d/state_%d.pkl' % (i+s, j+sp)
            f = open(path_filename, 'rb')
            path = pickle.load(f)
            path = np.array(path)
            #path = np.load(path_filename) # Nx2 shape
            # generate training data
            if load_dis_ratio:
                # calculate distance ratio
                total_d = 0.
                dist_list = []
                for k in range(len(path)-1):
                    dist_list.append(total_d)
                    total_d += np.linalg.norm(path[k+1] - path[k])
                dist_list.append(total_d)
                dist_list = np.array(dist_list) / total_d  # normalize to 0 - 1
                dist_list = dist_list.reshape(-1,1)
            for m in range(0, len(path)):
                data = np.array(path[m])
                waypoint_dataset.append(data)
                # add the recently added one
                if load_dis_ratio:
                    # add here
                    cond = np.concatenate([path[0], path[-1], dist_list[m]])
                else:
                    cond = np.concatenate([path[0], path[-1]])
                cond_dataset.append(cond)
                # start_indices.append(len(start_dataset)-1)
                # goal_indices.append(len(goal_dataset)-1)
                env_indices.append(i)

    return waypoint_dataset, cond_dataset, obs_voxel, env_indices

    
def load_test_dataset(N=100,NP=200, s=0,sp=4000, folder='../data/s2d/'):
    #  data_folder: motion_planning_datasets/forest
    path_folder =  folder + 'path/'  # folder containing all imitation trajectories
    obs_folder = folder + 'train/'  # we use the training folder as environment
    pcd_folder = folder + 'train/pcd/'  # folder storing the pcd
    # check if the pcd exists, if not, create a new one

    obs_repre = []
    obs_voxel = []
    obs_pcd = []
    for i in range(N):
        obs_repre_i = imageio.imread(obs_folder+'%d.png' % (i+s))
        obs_repre.append(obs_repre_i)
        # can directly use the image as "voxel"
        obs_voxel_i = np.zeros([3] + list(obs_repre_i.shape)).astype(int)  # first channel: 0,1 map  # second channel: row loc  # third channel: col loc
        
        obs_voxel_i[0] = obs_repre_i
        obs_voxel_i[0,obs_voxel_i[0]==0] = 1  # obstacle
        obs_voxel_i[0,obs_voxel_i[0]==255] = 0  # non-obstacle
        # unit testing here: visualize the loaded voxel

        img_indices = np.indices(obs_repre_i.shape)  # return: 2ximgshape
        obs_voxel_i[1] = img_indices[0]
        obs_voxel_i[2] = img_indices[1]
        obs_voxel_i = obs_voxel_i.astype(float)  # result: 3x201x201
        # scale indices
        obs_voxel_i[1] = obs_voxel_i[1] / obs_voxel_i[1].max()
        obs_voxel_i[2] = obs_voxel_i[2] / obs_voxel_i[2].max()

        obs_voxel.append(np.array(obs_voxel_i))
    obs_repre = np.array(obs_repre)
    obs_voxel = np.array(obs_voxel)

    paths = []
    path_lengths = np.zeros((N,NP)).astype(int)
    for i in range(0,N):
        paths_env = []
        for j in range(0,NP):
            path_filename = path_folder+'%d/state_%d.pkl' % (i+s, j+sp)
            f = open(path_filename, 'rb')
            path = pickle.load(f)
            path = np.array(path)
            paths_env.append(path)
            path_lengths[i][j] = len(path)
        paths.append(paths_env)
    # obc: obstacle center
    # obs: obstacle point cloud
    return obs_repre,obs_voxel,paths,path_lengths

