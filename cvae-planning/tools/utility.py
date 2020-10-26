import torch
from torch.autograd import Variable
import copy

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def save_state(net, torch_seed, np_seed, py_seed, fname):
    # save both model state and optimizer state
    states = {
        'state_dict': net.state_dict(),
        'optimizer': net.opt.state_dict(),
        'torch_seed': torch_seed,
        'np_seed': np_seed,
        'py_seed': py_seed
    }
    torch.save(states, fname)

def load_net_state(net, fname):
    if torch.cuda.is_available():
        checkpoint = torch.load(fname, map_location='cuda:%d' % (torch.cuda.current_device()))
    else:
        checkpoint = torch.load(fname, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])

def load_opt_state(net, fname):
    if torch.cuda.is_available():
        checkpoint = torch.load(fname, map_location='cuda:%d' % (torch.cuda.current_device()))
    else:
        checkpoint = torch.load(fname, map_location='cpu')
    net.opt.load_state_dict(checkpoint['optimizer'])

def load_seed(fname):
    # load both torch random seed, and numpy random seed
    if torch.cuda.is_available():
        checkpoint = torch.load(fname, map_location='cuda:%d' % (torch.cuda.current_device()))
    else:
        checkpoint = torch.load(fname, map_location='cpu')
    return checkpoint['torch_seed'], checkpoint['np_seed'], checkpoint['py_seed']


class EarlyStopChecker():
    def __init__(self, early_stop_freq, early_stop_patience):
        self.early_stop_freq = early_stop_freq
        self.early_stop_patience = early_stop_patience
        self.freq_counter = 0
        self.patience_counter = 0  # count how many times loss goes up
        self.prev_loss = None
    def early_stop_check(self, loss):
        # input: need to be scalar
        # output: True if early stop
        self.freq_counter += 1
        if self.freq_counter % self.early_stop_freq != 0:
            # not yet to check early stop
            return False
        self.freq_counter = 0  # clear counter
        if self.prev_loss is None:
            self.prev_loss = loss
            return False
        res = True
        if loss > self.prev_loss:
            # increase counter
            self.patience_counter += 1
            if self.patience_counter > self.early_stop_patience:
                # early stop
                res = True
            else:
                # ignoring
                res = False
        else:
            # clear counter
            self.patience_counter = 0
            res = False
        self.prev_loss = loss
        return res


class DictDot(dict):
    # useful tool to convert from dictionary to object that can dot
    # reference: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(DictDot, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DictDot, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DictDot, self).__delitem__(key)
        del self.__dict__[key]