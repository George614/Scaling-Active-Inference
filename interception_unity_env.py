import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from mlagents_envs.environment import UnityEnvironment
import mlagents_envs.base_env
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import json

class InterceptionEnv(gym.Env):
    def __init__(self, settings_file=None, file_name=None, graphics=True):
        self.channel = EnvironmentParametersChannel()
        self.env = UnityEnvironment(file_name=file_name, no_graphics=not graphics, side_channels=[self.channel])

        self.action_space = spaces.Discrete(6)
        self.action_speed_mappings = [2.0, 4.0, 8.0, 10.0, 12.0, 14.0]

        if settings_file is not None:
            with open(settings_file) as f:
                self.settings = json.load(f)
        else:
            self.settings = None

        self.seed()

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, target_init_speed_idx=None, approach_angle_idx=None):
        '''
        target_init_speed: index of the starting target speed to use, or None for randomly assigned valid value
        approach_angle: index of the approaching angle to use, or None for randomly assigned valid value
        '''
        
        # Send all the parameters that are controlled per experiment or from the RNG seeded in this class.
        # Otherwise if nothing is passed, let the environment handle randomness
        if self.settings is not None:
            if target_init_speed_idx is None:
                target_init_speed = self.rng.choice(self.settings['targetInitSpeeds'])
            else:
                target_init_speed = self.settings['targetInitSpeeds'][target_init_speed_idx]

            if approach_angle_idx is None:
                approach_angle = self.rng.choice(self.settings['approachAngles'])
            else:
                approach_angle = self.settings['approachAngles'][approach_angle_idx]

            self.channel.set_float_parameter('approachAngle', approach_angle)
            self.channel.set_float_parameter('subjectInitDistance', 
                        self.rng.uniform(low=self.settings['subjectInitDistanceMin'], high=self.settings['subjectInitDistanceMax']))
            self.channel.set_float_parameter('targetInitSpeed', target_init_speed)
            self.channel.set_float_parameter('timeToChangeSpeed',
                        self.rng.uniform(low=self.settings['timeToChangeSpeedMin'], high=self.settings['timeToChangeSpeedMax']))
            self.channel.set_float_parameter(
                'targetFinalSpeed',
                np.clip(
                    self.rng.normal(loc=self.settings['targetSpeedMean'], scale=self.settings['targetSpeedStdDev']), 
                    self.settings['targetSpeedMin'], self.settings['targetSpeedMax']
                )
            )
        self.env.reset()

        self.behavior = self.env.get_behavior_names()[0]

        dsteps, _ = self.env.get_steps(self.behavior)
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

# If this script is run as main, test the environment
if __name__ == '__main__':
    print('connecting to unity environment')
    e = InterceptionEnv('interception-unity-build/InterceptionTask_Data/StreamingAssets/interception_task.json',
                            'interception-unity-build/InterceptionTask', False)
    print()

    print('connected, running first simulation')
    e.reset(approach_angle_idx=0)
    state, reward, done, _ = e.step(4)
    while not done:
        print('state:', state)
        state, reward, done, _ = e.step(4)
    print('state:', state)
    print('final reward:', reward)
    print()

    print('running second simulation')
    e.reset(approach_angle_idx=2)
    state, reward, done, _ = e.step(4)
    while not done:
        print('state:', state)
        state, reward, done, _ = e.step(4)
    print('state:', state)
    print('final reward:', reward)
    print()

    print('done, closing environment')
    e.close()
