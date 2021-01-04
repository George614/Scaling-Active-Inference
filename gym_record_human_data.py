import gym
from gym.utils import play
from gym_recording.wrappers import TraceRecordingWrapper
from gym_recording import test_recording

if __name__ == '__main__':
	env = gym.make('MountainCar-v0')  #'CartPole-v0'
	env = TraceRecordingWrapper(env)
	print("\nDirectory of recording: ", env.directory, "\n")
	play.play(env, fps=30, keys_to_action={(): 1, (ord('a'),): 0, (ord('d'),): 2})
	env.env.close()

	# test_recording.test_trace_recording()