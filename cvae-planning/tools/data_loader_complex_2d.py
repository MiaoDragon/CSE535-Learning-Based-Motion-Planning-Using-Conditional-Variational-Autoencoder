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
        ys[ys==img_shape[1]] = ys[ys==img.shape[1]] - 1
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
        obs_repre_i = imageio.imread(obs_path+'%d.png' % (i+s))
        obs_repre.append(obs_repre_i)
        # can directly use the image as "voxel"
        obs_voxel_i = np.array(obs_repre_i)
        obs_voxel_i[obs_voxel_i==0] = 1  # obstacle
        obs_voxel_i[obs_voxel_i==255] = 0  # non-obstacle
        obs_voxel.append(np.array([obs_repre_i]))  # add one more dimension

        # below is useful if using pcd

        # pcd_filename = pcd_folder + '%d.npy' % (i+s)
        # if os.path.exists(pcd_filename):
        #     # directly load if already there
        #     obs_pcd.append(np.load(pcd_filename))
        # else:
        #     # sample and save to disk
        #     obs_pcd_i = sample_pcd(obs_repre_i, num_pts=4000, save_path=pcd_folder, filename='%d.npy' % (i+s))
        #     obs_pcd.append(obs_pcd_i)


    # obtain path
    waypoint_dataset = []
    cond_dataset = []
    env_indices = []
    for i in range(N):
        for j in range(NP):
            path_filename = path_folder+'%d/%d.npy' % (i+s, j+sp)
            path = np.load(path_filename) # Nx2 shape
            # generate training data
            if load_dis_ratio:
                # calculate distance ratio
                total_d = 0.
                dist_list = []
                for k in range(len(path-1)):
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
    obc=np.zeros((N,7,2),dtype=np.float32)
    temp=np.fromfile(folder+'obs.dat')
    obs=temp.reshape(len(temp)//2,2)

    temp=np.fromfile(folder+'obs_perm2.dat',np.int32)
    perm=temp.reshape(77520,7)

    ## loading obstacles
    for i in range(0,N):
        for j in range(0,7):
            for k in range(0,2):
                obc[i][j][k]=obs[perm[i+s][j]][k]
    obs = []
    k=0
    for i in range(s,s+N):
        temp=np.fromfile(folder+'obs_cloud/obc'+str(i)+'.dat')
        obs.append(temp)
    obs = np.array(obs).reshape(len(obs),-1,2)
    obs = pcd_to_voxel2d(obs, voxel_size=[32,32]).reshape(-1,1,32,32)

    ## calculating length of the longest trajectory
    max_length=0
    path_lengths=np.zeros((N,NP),dtype=np.int8)
    for i in range(0,N):
        for j in range(0,NP):
            fname=folder+'e'+str(i+s)+'/path'+str(j+sp)+'.dat'
            if os.path.isfile(fname):
                path=np.fromfile(fname)
                path=path.reshape(len(path)//2,2)
                path_lengths[i][j]=len(path)
                if len(path)> max_length:
                    max_length=len(path)


    paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

    for i in range(0,N):
        for j in range(0,NP):
            fname=folder+'e'+str(i+s)+'/path'+str(j+sp)+'.dat'
            if os.path.isfile(fname):
                path=np.fromfile(fname)
                path=path.reshape(len(path)//2,2)
                for k in range(0,len(path)):
                    paths[i][j][k]=path[k]

    # obc: obstacle center
    # obs: obstacle point cloud
    return obc,obs,paths,path_lengths



def pcd_to_voxel2d(points, voxel_size=(24, 24), padding_size=(32, 32)):
    voxels = [voxelize2d(points[i], voxel_size, padding_size) for i in range(len(points))]
    # return size: BxV*V*V
    return np.array(voxels)

def voxelize2d(points, voxel_size=(24, 24), padding_size=(32, 32), resolution=0.05):
    """
    Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
    `padding_to_size`. The outside part is cut, rather than scaling the points.
    Args:
    `points`: pointcloud in 3D numpy.ndarray (shape: N * 3)
    `voxel_size`: the centerlized voxel size, default (24,24,24)
    `padding_to_size`: the size after zero-padding, default (32,32,32)
    `resolution`: the resolution of voxel, in meters
    Ret:
    `voxel`:32*32*32 voxel occupany grid
    `inside_box_points`:pointcloud inside voxel grid
    """
    # calculate resolution based on boundary
    if abs(resolution) < sys.float_info.epsilon:
        print('error input, resolution should not be zero')
        return None, None

    """
    here the point cloud is centerized, and each dimension uses a different resolution
    """
    OCCUPIED = 1
    FREE = 0
    resolution = [(points[:,i].max() - points[:,i].min()) / voxel_size[i] for i in range(2)]
    resolution = np.array(resolution)
    #resolution = np.max(res)
    # remove all non-numeric elements of the said array
    points = points[np.logical_not(np.isnan(points).any(axis=1))]

    # filter outside voxel_box by using passthrough filter
    # TODO Origin, better use centroid?
    origin = (np.min(points[:, 0]), np.min(points[:, 1]))
    # set the nearest point as (0,0,0)
    points[:, 0] -= origin[0]
    points[:, 1] -= origin[1]
    #points[:, 2] -= origin[2]
    # logical condition index
    x_logical = np.logical_and((points[:, 0] < voxel_size[0] * resolution[0]), (points[:, 0] >= 0))
    y_logical = np.logical_and((points[:, 1] < voxel_size[1] * resolution[1]), (points[:, 1] >= 0))
    #z_logical = np.logical_and((points[:, 2] < voxel_size[2] * resolution[2]), (points[:, 2] >= 0))
    xy_logical = np.logical_and(x_logical, y_logical)
    #xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    #inside_box_points = points[xyz_logical]
    inside_box_points = points[xy_logical]
    # init voxel grid with zero padding_to_size=(32*32*32) and set the occupany grid
    voxels = np.zeros(padding_size)
    # centerlize to padding box
    center_points = inside_box_points + (padding_size[0] - voxel_size[0]) * resolution / 2
    # TODO currently just use the binary hit grid
    x_idx = (center_points[:, 0] / resolution[0]).astype(int)
    y_idx = (center_points[:, 1] / resolution[1]).astype(int)
    #z_idx = (center_points[:, 2] / resolution[2]).astype(int)
    #voxels[x_idx, y_idx, z_idx] = OCCUPIED
    voxels[x_idx, y_idx] = OCCUPIED
    return voxels
    #return voxels, inside_box_points


