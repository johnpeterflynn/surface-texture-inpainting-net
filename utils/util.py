import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict, defaultdict
from time import time
import torch

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

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, write=True):
        if write and self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self, write=False):
        data_averages = dict(self._data.average)
        if write:
            for key, value in data_averages.items():
                self.writer.add_scalar(key, value)

        return data_averages


class SyncedTimer:
    def __init__(self, num_drop=0):
        self.reset()
        self.num_drop = num_drop

    def reset(self):
        self.times = {}
        self.total_times = defaultdict(list)

    def _synced_time(self):
        torch.cuda.synchronize()
        return time.perf_counter()

    def start(self, name):
        self.times[name] = self._synced_time()

    def end(self, name):
        time_diff = self._synced_time() - self.times[name]
        #print('end {}:'.format(name), time_diff)
        del self.times[name]
        self.total_times[name].append(time_diff)

    def print(self, reset=False):
        for key, values in self.total_times.items():
            values_kept = values[self.num_drop:]
            #print(key, 'values:', values_kept)
            print('total {}:'.format(key), sum(values_kept) / len(values_kept))
        if reset:
            self.reset()
