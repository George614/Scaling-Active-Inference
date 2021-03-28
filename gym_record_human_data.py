import gym
import numpy as np
import os, logging, time
from gym.utils import play
from gym_recording.wrappers import TraceRecordingWrapper
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
	### below is the code for recording human data ###
	env = gym.make('MountainCar-v0')  #'CartPole-v0'
	env = TraceRecordingWrapper(env)
	print("\nDirectory of recording: ", env.directory, "\n")
	# the play API lets the human player control the game
	# note that empty input is mapped to going left to exclude the "stay" action
	play.play(env, fps=30, keys_to_action={(): 0, (ord('a'),): 0, (ord('d'),): 2})
	env.env.close()
