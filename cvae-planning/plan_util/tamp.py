import numpy as np

import torch
from torch.autograd import Variable
import copy

# bound:
# [x, y, z] X [-3.14 ~ 3.14]^15 X [obj_x, obj_y, obj_z, qx, qy, qz, qw]^4
# x: [-2.75, 2.75]
# y: [-2.75, 2.75]
# z: [0, 5.54]

quat_idx = [21, 28, 35, 42]
symbolic_idx = 46
symbolic_len = 26
def normalize(x, bound):
    # normalize to -1 ~ 1
    if type(x) is not np.ndarray:
        bound = torch.FloatTensor(bound)
        x_shape = x.size()
    else:
        bound = np.array(bound)
        x_shape = x.shape
    if len(x_shape) > 1:
        # batch version
        if len(x[0]) != len(bound):
            # preceding are start, goal
            x[:,:len(bound)] = x[:,:len(bound)] / bound
            # goal only has symbolic part available
            x[:,len(bound):len(bound)+symbolic_len] = x[:,len(bound):len(bound)+symbolic_len] / bound[symbolic_idx:]
            # normalize the quarternion part
            for i in quat_idx:
                x[:,i:i+4] = torch.norm(x[:,i:i+4], dim=1, keepdim=True)
                #x[:,len(bound)+i:len(bound)+i+4] = torch.norm(x[:,len(bound)+i:len(bound)+i+4], dim=1, keepdim=True)
        else:
            x = x / bound
            # normalize the quaternion part
            for i in quat_idx:
                x[:,i:i+4] = torch.norm(x[:,i:i+4], dim=1, keepdim=True)

    else:
        #print('before normalizing...')
        #print(x)
        if len(x[0]) != len(bound):
            # preceding are start, goal
            x[:len(bound)] = x[:len(bound)] / bound
            x[len(bound):len(bound)+symbolic_len] = x[len(bound):len(bound)+symbolic_len] / bound[symbolic_idx:]
            # normalize the quarternion part
            for i in quat_idx:
                x[i:i+4] = torch.norm(x[i:i+4], dim=1, keepdim=True)
                #x[len(bound)+i:len(bound)+i+4] = torch.norm(x[len(bound)+i:len(bound)+i+4], dim=1, keepdim=True)
        else:
            x = x / bound
            # normalize the quaternion part
            for i in quat_idx:
                x[i:i+4] = torch.norm(x[i:i+4], dim=1, keepdim=True)

    return x
def unnormalize(x, bound):
    if type(x) is not np.ndarray:
        bound = torch.FloatTensor(bound)
        x_shape = x.size()
    else:
        bound = np.array(bound)
        x_shape = x.shape
    if len(x_shape) > 1:
        # batch version
        if len(x[0]) != len(bound):
            # preceding are start, goal
            x[:,:len(bound)] = x[:,:len(bound)] * bound
            x[:,len(bound):len(bound)+symbolic_len] = x[:,len(bound):len(bound)+symbolic_len] * bound[symbolic_idx:]
            # normalize the quarternion part
            for i in quat_idx:
                x[:,i:i+4] = torch.norm(x[:,i:i+4], dim=1, keepdim=True)
                #x[:,len(bound)+i:len(bound)+i+4] = torch.norm(x[:,len(bound)+i:len(bound)+i+4], dim=1, keepdim=True)

        else:
            x = x * bound
            # normalize the quaternion part
            for i in quat_idx:
                x[:,i:i+4] = torch.norm(x[:,i:i+4], dim=1, keepdim=True)


        #print('after normalizing...')
        #print(x[:,-2*len(bound):])
    else:
        #print('before normalizing...')
        #print(x)
        if len(x[0]) != len(bound):
            # preceding are start, goal
            x[:len(bound)] = x[:len(bound)] * bound
            x[len(bound):len(bound)+symbolic_len] = x[len(bound):len(bound)+symbolic_len] * bound[symbolic_idx:]
            # normalize the quarternion part
            for i in quat_idx:
                x[i:i+4] = torch.norm(x[i:i+4], dim=1, keepdim=True)
                #x[len(bound)+i:len(bound)+i+4] = torch.norm(x[len(bound)+i:len(bound)+i+4], dim=1, keepdim=True)

        else:
            x = x * bound
            # normalize the quaternion part
            for i in quat_idx:
                x[i:i+4] = torch.norm(x[i:i+4], dim=1, keepdim=True)
    return x