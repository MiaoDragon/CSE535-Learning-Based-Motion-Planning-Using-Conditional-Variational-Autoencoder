import numpy as np

def IsInCollision(x,obc):
    size = 5.0
    s=np.zeros(2,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    for i in range(0,7):
        cf=True
        for j in range(0,2):
            if abs(obc[i][j] - s[j]) > size/2.0 and s[j]<20.0 and s[j]>-20:
                cf=False
                break
        if cf==True:
            return True
    return False


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