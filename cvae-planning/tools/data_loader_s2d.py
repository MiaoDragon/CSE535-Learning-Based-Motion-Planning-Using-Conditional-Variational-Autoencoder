"""
This implements data loader for both training and testing procedures.
"""
import pickle
import numpy as np
import sys
import os
def preprocess(data_path, data_control, data_cost, dynamics, enforce_bounds, system, step_sz, num_steps):
    #print('inside preprocess to generate detailed trajectory...')
    p_start = data_path[0]
    detail_paths = [p_start]
    detail_controls = []
    detail_costs = []
    state = [p_start]
    control = []
    cost = []
    for k in range(len(data_control)):
        #state_i.append(len(detail_paths)-1)
        max_steps = int(np.round(data_cost[k]/step_sz))
        accum_cost = 0.
        ## modify it because of small difference between data and actual propagation
        #p_start = data_path[k]
        #state[-1] = data_path[k]
        for step in range(1,max_steps+1):
            p_start = dynamics(p_start, data_control[k], step_sz)
            accum_cost += step_sz
            if (step % 1 == 0) or (step == max_steps):
                state.append(p_start)
                control.append(data_control[k])
                cost.append(accum_cost)
                accum_cost = 0.
    
    # new method: don't care if intermediate nodes have the same control or not
    # take every num_steps after the entire path is stored
    remaining_states = (len(state)-1) % num_steps

    state = state[::num_steps]
    
    # if last node is not the same as goal, then add goal
    if np.linalg.norm(np.array(state[-1]) - np.array(data_path[-1])) > 1e-3:
        state.append(data_path[-1])

    #last_cost = cost[-remaining_states:].sum()
    # this cost is correct. But the intermediate controls might not be the same each time
    cost = np.array(cost)
    cost = [cost[i:i+num_steps].sum() for i in range(0, len(cost), num_steps)]    

    #control = control[::num_steps]  # this data is wrong
    #cost = cost[::num_steps]   # this data is wrong
    
    #state[-1] = data_path[-1]
    return state, control, cost

def load_train_dataset(N, NP, s, sp, data_folder, load_dis_ratio=False):
    folder = data_folder
    obs = []
    # add start s
    for i in range(0,N):
        #load obstacle point cloud
        temp=np.fromfile(folder+'obs_cloud/obc'+str(i+s)+'.dat')
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

    paths=np.zeros((N,NP,max_length,2))   ## padded paths
    for i in range(0,N):
        for j in range(0,NP):
            fname=folder+'e'+str(i+s)+'/path'+str(j+sp)+'.dat'
            if os.path.isfile(fname):
                path=np.fromfile(fname)
                path=path.reshape(len(path)//2,2)
                for k in range(0,len(path)):
                    paths[i][j][k]=path[k]

    waypoint_dataset=[]
    # start_indices = []
    # goal_indices = []
    # start_dataset = []
    # goal_dataset = []
    cond_dataset = []
    env_indices=[]
    for i in range(0,N):
        for j in range(0,NP):
            if path_lengths[i][j]>0:
                # start_dataset.append(paths[i][j][0])
                # goal_dataset.append(paths[i][j][path_lengths[i][j]-1])
                if load_dis_ratio:
                    # calculate distance ratio
                    total_d = 0.
                    dist_list = []
                    for k in range(path_lengths[i][j]-1):
                        dist_list.append(total_d)
                        total_d += np.linalg.norm(paths[i][j][k+1] - paths[i][j][k])
                    dist_list.append(total_d)
                    dist_list = np.array(dist_list) / total_d  # normalize to 0 - 1
                    dist_list = dist_list.reshape(-1,1)
                for m in range(0, path_lengths[i][j]):
                    data = np.array(paths[i][j][m])
                    waypoint_dataset.append(data)
                    # add the recently added one
                    if load_dis_ratio:
                        # add here
                        cond = np.concatenate([paths[i][j][0], paths[i][j][path_lengths[i][j]-1], dist_list[m]])
                    else:
                        cond = np.concatenate([paths[i][j][0], paths[i][j][path_lengths[i][j]-1]])
                    cond_dataset.append(cond)
                    # start_indices.append(len(start_dataset)-1)
                    # goal_indices.append(len(goal_dataset)-1)
                    env_indices.append(i)
    return waypoint_dataset, cond_dataset, obs, env_indices
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


