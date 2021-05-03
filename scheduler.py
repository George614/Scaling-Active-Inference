import numpy as np
import math


class Schedule(object):
    def __init__(self, start=0.0, stop=1.0):
        self.start = start
        self.stop = stop

class Linear_schedule(Schedule):
    def __init__(self, start=0.0, stop=1.0, duration=300):
        super().__init__(start, stop)
        self.duration = duration

    def __call__(self, idx):
        val = self.start + (self.stop - self.start) * np.minimum(idx / self.duration, 1.0)
        return val


class Exponential_schedule(Schedule):
    def __init__(self, start=0.0, stop=1.0, decay=300):
        super().__init__(start, stop)
        self.decay = decay

    def __call__(self, idx):
        val = self.stop + (self.start - self.stop) * math.exp(-1.0 * idx / self.decay)
        return val


class Frange_cycle_linear(Schedule):
    '''
    Cyclical Annealing Schedule 
    A Simple Approach to Mitigating KL-Vanishing. Fu etal NAACL 2019
    Ref: https://github.com/haofuml/cyclical_annealing
    '''
    def __init__(self, n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
        super().__init__(start, stop)
        self.n_cycle = n_cycle
        self.ratio = ratio
        self.n_iter = n_iter

    def __call__(self):
        L = np.ones(self.n_iter) * self.stop
        period = self.n_iter / self.n_cycle
        step = (self.stop - self.start) / (period * self.ratio) # linear schedule

        for c in range(self.n_cycle):
            v, i = self.start, 0
            while v <= self.stop and (int(i + c * period) < self.n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L


class SWA_schedule(Schedule):
    '''
    Advanced / composite learning schedule used with SWA.
    Ref: https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
    start: starting learning rate of the entire schedule
    stop: final / lowest learning rate of the schedule
    swa_lr: learning rate to start with druing SWA
    freq: cycle duration of SWA
    ratio_bf_start: ratio of duration before starting SWA
    '''
    def __init__(self, n_iter, start=1e-3, stop=1e-4, swa_lr=5e-4, freq=10, ratio_bf_start=0.75):
        super().__init__(start, stop)
        self.n_iter = n_iter
        self.cycle_len = freq
        self.start_swa = int(np.floor(ratio_bf_start * n_iter))
        self.start_decay = self.start_swa // 2
        self.swa_lr = swa_lr
        n_cycle = (self.n_iter - self.start_swa) // self.cycle_len
        lr_list = np.ones(self.n_iter) * self.start
        self.swa_point = np.zeros(self.n_iter, dtype=np.int32)
        decay_step = (self.stop - self.start) / self.start_decay
        
        for i in range(self.start_decay, self.start_swa):
            lr_list[i] = self.start + (i - self.start_decay) * decay_step
        
        self.swa_point[self.start_swa-1] = 1
        lr_list[self.start_swa:] = self.swa_lr
        swa_decay_step = (self.stop - self.swa_lr) / (self.cycle_len - 1)
        
        for c in range(n_cycle):
            lr = self.swa_lr
            for idx in range(1, self.cycle_len):
                lr += swa_decay_step
                lr_list[int(self.start_swa + idx + c * self.cycle_len)] = lr
            self.swa_point[self.start_swa + idx + c * self.cycle_len] = 1

        self.lr_list = lr_list

    def __call__(self, idx):
        return self.lr_list[idx], self.swa_point[idx]