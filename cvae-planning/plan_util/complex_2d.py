import numpy as np

def IsInCollision(x,obc):
    pass
import torch
from torch.autograd import Variable
import copy
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
            x[:,len(bound):2*len(bound)] = x[:,len(bound):2*len(bound)] / bound
            # normalize the quarternion part
        else:
            x = x / bound

        #print('after normalizing...')
        #print(x[:,-2*len(bound):])
    else:
        #print('before normalizing...')
        #print(x)
        if len(x[0]) != len(bound):
            # preceding are start, goal
            x[:len(bound)] = x[:len(bound)] / bound
            x[len(bound):2*len(bound)] = x[len(bound):2*len(bound)] / bound
            # normalize the quarternion part
        else:
            x = x / bound

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
            x[:,len(bound):2*len(bound)] = x[:,len(bound):2*len(bound)] * bound
        else:
            x = x * bound

        #print('after normalizing...')
        #print(x[:,-2*len(bound):])
    else:
        #print('before normalizing...')
        #print(x)
        if len(x[0]) != len(bound):
            # preceding are start, goal
            x[:len(bound)] = x[:len(bound)] * bound
            x[len(bound):2*len(bound)] = x[len(bound):2*len(bound)] * bound
        else:
            x = x * bound
    return x