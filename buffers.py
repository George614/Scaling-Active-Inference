from collections import deque
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, capacity, seed=None):
        self.buffer = deque(maxlen=capacity)
        ramdom.seed(seed)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state = np.asarray(np.concatenate(state), dtype=np.float32)
        next_state = np.asarray(np.concatenate(next_state), dtype=np.float32)
        action = np.asarray(action, dtype=np.int32)
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=bool)
        return (state, action, reward, next_state, done)
    
    def __len__(self):
        return len(self.buffer)