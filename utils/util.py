import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch.nn.functional as F


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
    
    
"""
visualize activation map
author: Kim Namtae
"""
def visualize_activation(model, x, p=3, eps=1e-6):
    feature_net = model.features
    lwhiten_net = model.lwhiten
    pool_net=  model.pool
    
    f = feature_net(x)
    s = f.size()
    f = f.permute(0,2,3,1).contiguous().view(-1,s[1])
    f = lwhiten_net(f)
    f = f.view(s[0],s[2],s[3],lwhiten_net.out_features).permute(0,3,1,2)
    pow_value = f.clamp(min=eps).pow(p)
    size = f.size(-2)*f.size(-1)
    activation_map = (pow_value/size).pow(1./p)
    activation_map = torch.einsum('ijkw->ikw', activation_map)
    activation_map = activation_map/torch.max(activation_map)
    return activation_map

    
        


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
