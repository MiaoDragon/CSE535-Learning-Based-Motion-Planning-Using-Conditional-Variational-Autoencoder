"""
This implements data loader for both training and testing procedures.
"""
import pickle
import numpy as np
import sys
import os

symbolic_idx = 46

def load_train_dataset(N, NP, s, sp, data_folder, load_dis_ratio=False):
    folder = data_folder
    obs = []
    # add start s
    for i in range(0,N):
        #load obstacle point cloud
        obs.append(None)
    obs = np.array(obs).reshape(len(obs), 1)

    # quaternion: [21 - 24], [28, 31], [35, 38], [42, 45]
    # robot configuration: the start 18 values
    robot_conf_num = 18

    waypoint_dataset=[]
    # start_indices = []
    # goal_indices = []
    # start_dataset = []
    # goal_dataset = []
    cond_dataset = []
    env_indices=[]
    for i in range(0,N):
        for j in range(0,NP):
            path=np.loadtxt(folder+'%d/clutter_ml_cont_state.txt' % (j))
            disc_path=np.loadtxt(folder+'%d/clutter_ml_disc_state.txt' % (j))
            combined_path = np.concatenate([path, disc_path], axis=1)
            if load_dis_ratio:
                # calculate distance ratio
                total_d = 0.
                dist_list = []
                for k in range(len(path)-1):
                    dist_list.append(total_d)
                    total_d += np.linalg.norm(path[k+1,:robot_conf_num] - path[k, :robot_conf_num])
                dist_list.append(total_d)
                dist_list = np.array(dist_list) / total_d  # normalize to 0 - 1
                dist_list = dist_list.reshape(-1,1)
            for m in range(0, len(path)):
                # we concatenate the continuous state and the discrete state for waypoint
                waypoint_dataset.append(combined_path[m])
                #print('waypoint shape: ')
                #print(combined_path[m].shape)
                #print('waypoint:')
                #print(combined_path[m])
                # add the recently added one
                if load_dis_ratio:
                    # add here
                    cond = np.concatenate([combined_path[0], combined_path[-1,symbolic_idx:], dist_list[m]])
                else:
                    cond = np.concatenate([combined_path[0], combined_path[-1,symbolic_idx:]])
                #print('cond shape: ')
                #print(cond.shape)
                #print('cond:')
                #print(cond)
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

