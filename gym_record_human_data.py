import gym
import numpy as np
import os, logging, time
from gym.utils import play
from gym_recording import test_recording
from gym_recording.wrappers import TraceRecordingWrapper
from gym_recording.playback import scan_recorded_traces
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def record_random_agent():
    env = gym.make('MountainCar-v0')  #'CartPole-v0'
    env = TraceRecordingWrapper(env)
    recdir = env.directory
    print("\nDirectory of recording: ", recdir, "\n")
    agent = lambda ob: env.action_space.sample()

    for epi in range(1024):
        ob = env.reset()
        a = None
        for _ in range(200):
            assert env.observation_space.contains(ob)
            switch = np.random.uniform(low=0.0, high=1.0) < 0.1
            # a = agent(ob)
            if switch or a is None:
                left = np.random.uniform(low=0.0, high=1.0) < 0.5
                a = 0 if left else 2
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done: break
    print("has recording" if env.recording is not None else "no recording")
    env.close()

    counts = [0, 0]
    def handle_ep(observations, actions, rewards, all_data, idx):
    	length = len(observations)
    	all_data[idx, :length, :2] = observations[:, :]
    	all_data[idx, 1:length, 2] = actions[:]
    	all_data[idx, 1:length, 3] = rewards[:]
        counts[0] += 1
        counts[1] += observations.shape[0]
        print('Observations.shape={}, actions.shape={}, rewards.shape={}', observations.shape, actions.shape, rewards.shape)

    all_batch_data = scan_recorded_traces(recdir, handle_ep, max_episodes=1024)
    np.save(recdir + "/all_data.npy", all_batch_data, allow_pickle=True)
    assert counts[0] == 1024
    assert counts[1] > 100

if __name__ == '__main__':
	### below is the code for recording human data ###
	# env = gym.make('MountainCar-v0')  #'CartPole-v0'
	# env = TraceRecordingWrapper(env)
	# print("\nDirectory of recording: ", env.directory, "\n")
	# play.play(env, fps=30, keys_to_action={(): 1, (ord('a'),): 0, (ord('d'),): 2})
	# env.env.close()

	# record batches of data from a random agent
	record_random_agent()