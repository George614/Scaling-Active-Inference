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
    """Description:
        The agent needs to intercept a target that moves along a predictable
        (straight-line) trajectory, with a sudden acceleration after X ms.
        The new speed is selected from a distribution. For any given state
        the agent may choose a paddle position which affects the travel
        speed with a log.
    Source:
        Diaz, G. J., Phillips, F., & Fajen, B. R. (2009). Intercepting moving
        targets: a little foresight helps a lot. Experimental brain research,
        195(3), 345-360.
    Observation:
        Type: Box(2)
        Num    Observation                  Min         Max
        0      Target distance              0.0         45.0
        1      Target velocity              8.18        20.0
        2       Subject distance            0.0         30.0
        3        Subject velocity           0.0         14.0
        4       Whether the target has       0           1
               changed speed (0 or 1)
    Actions:
        Type: Discrete(6)
        Num    Action
        0      Change the paddle positon to be 1 (0 means no accerleration)
        1      Change the paddle positon to be 2
        2      Change the paddle positon to be 3
        3       Change the paddle positon to be 4
        4       Change the paddle positon to be 5
        5       Change the paddle positon to be 6
        Note: Paddle position at one of N positions and changes are instantaneous.
        Change in speed determined by the difference between the current traveling
        speed and the new pedal position  V_dot =  K * ( Vp - Vs)
    Reward:
         Reward of 0 is awarded if the agent intercepts the target (position = 0.5)
         Reward of -1 is awarded everywhere else.
    Starting State:
         The simulated target approached the unmarked interception point from an
         initial distance of 45 m.
         Initial approach angle of 135, 140, or 145 degree from the subject’s
         path of motion.
         The starting velocity of the target is one from 11.25, 9.47, 8.18 m/s,
         which corresponds to initial first-order time-to-contact values of 4,
         4.75, and 5.5 seconds.
         Subject's initial distance is sampled from a uniform distribution
         between 25 and 30 meters.
    Episode Termination:
         The target position is at 0 (along target trajectory).
         The subject position is at 0 (along subject trajectory).
         Episode length is greater than 6 seconds (180 steps @ 30FPS).
    Notes:
        The settings are set from the JSON file packaged in interception-unity-build.zip, 
        in the 'InteceptionTask_Data/StreamingAssets/interception_task.json' file. This is
        done so that it uses exactly the same settings as is used by the Unity testing environment
        for gathering human data.
    """

    def __init__(self, settings_file=None, file_name=None, graphics=True):
        """Creates an Interception Gym Environment that connects to the executable for the Unity build.

        Args:
            settings_file (string, optional): Filename that the settings JSON file is located at. Defaults to None.
            file_name (string, optional): Filename that the Unity Executable is located at. Defaults to None.
            graphics (bool, optional): Whether the graphical interface is started or not. Defaults to True.
        """
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
        """Seeds the environment with a given seed, or randomly seeded if None is given.

        Args:
            seed (int, optional): The seed to use for this environment, or None for a random seed. Defaults to None.

        Returns:
            [int]: The seed in a one dimensional array.
        """
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, target_init_speed_idx=None, approach_angle_idx=None):
        """Resets the environment with the given settings.

        Args:
            target_init_speed_idx (int, optional): The initial speed for the target, an index from the 
                self.settings['targetInitSpeeds'] array, or None for a random value. Defaults to None.
            approach_angle_idx (int, optional): The approach angle between the subject and the target, which should
                have no effect on the outcome. An index from the self.settings['approachAngles'] array, or None 
                for a random value. Defaults to None.

        Returns:
            np.ndarray[5,]: The state space after the environment has been reset. The values
                are (target_distance, target_speed, has_changed_speed, subject_distance, subject_speed).
        """
        
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
        return dsteps.obs[0][0]

    def step(self, action):
        """Takes a step in the environment with the given action, and reports the next state of the environment.

        Args:
            action (int): a valid index in the action space, which is then mapped to the subject speed given in self.action_speed_mappings

        Returns:
            tuple(np.ndarray[5,], float, bool, dict): the state space, reward, done state, and empty
                diagnostic dictionary after the step is taken with the given action. The state space is the same as reset(),
                or (target_distance, target_speed, has_changed_speed, subject_distance, subject_speed).
                Reward values will be -1 for non-intercepted steps, and 100 for intercepted steps.
        """
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
        """Close the environment- this is necessary to clean up the Unity program that is running
        """
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
    print(type(state))
    print('state:', state)
    print('final reward:', reward)
    print()

    print('done, closing environment')
    e.close()
