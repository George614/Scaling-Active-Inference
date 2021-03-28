import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from mlagents_envs.environment import UnityEnvironment
import mlagents_envs.base_env

class InterceptionEnv(gym.Env):
    def __init__(self, file_name=None):
        self.env = UnityEnvironment(file_name=file_name)

        self.action_space = spaces.Discrete(6)
        self.action_speed_mappings = [2.0, 4.0, 8.0, 10.0, 12.0, 14.0]

    def seed(self, seed=None):
        pass

    def reset(self):
        self.env.reset()
        self.behavior = self.env.get_behavior_names()[0]
        dsteps, tsteps = self.env.get_steps(self.behavior)
        return dsteps.obs[0]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        subject_speed = self.action_speed_mappings[action]
        self.env.set_actions(self.behavior, np.array([[subject_speed]]))
        self.env.step()

        dsteps, tsteps = self.env.get_steps(self.behavior)
        if len(tsteps) > 0:
            steps = tsteps
            done = True
        else:
            steps = dsteps
            done = False
            
        return steps.obs[0][0], steps.reward[0], done, {}

    def close(self):
        self.env.close()

if __name__ == '__main__':
    print('waiting to connect')
    e = InterceptionEnv('unitybuild/InterceptionAgent')
    e.reset()

    state, reward, done, _ = e.step(1)
    while not done:
        state, reward, done, _ = e.step(1)
    print(reward)

    input('space to stop')
    e.close()
