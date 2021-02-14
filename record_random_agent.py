import gym
import numpy as np
import os, logging, time
from gym_recording import test_recording
from gym_recording.wrappers import TraceRecordingWrapper
from gym_recording.playback import scan_recorded_traces
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

max_episodes = 2048
episode_len = 300
continuous = True

def record_random_agent():
    if continuous:
        env = gym.make('MountainCarContinuous-v0')
    else:
        env = gym.make('MountainCar-v0').env  #'CartPole-v0'
    env = TraceRecordingWrapper(env)
    recdir = env.directory
    print("\nDirectory of recording: ", recdir, "\n")

    for epi in range(max_episodes):
        ob = env.reset()
        a = None
        for _ in range(episode_len):
            assert env.observation_space.contains(ob)
            switch = np.random.uniform(low=0.0, high=1.0) < 0.1
            if switch or a is None:
                if continuous:
                    a = [np.random.uniform(low=-1.0, high=1.0)]
                else:
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
        if continuous:
            actions = np.asarray(actions)
            all_data[idx, 1:length, 2] = actions[:, 0]
        else:
            all_data[idx, 1:length, 2] = actions[:]
        all_data[idx, 1:length, 3] = rewards[:]
        counts[0] += 1
        counts[1] += observations.shape[0]
        print("Observations.shape={}, rewards.shape={}".format(observations.shape, rewards.shape))

    all_batch_data = scan_recorded_traces(recdir, handle_ep, max_episodes=max_episodes)
    np.save(recdir + "/all_random_data.npy", all_batch_data, allow_pickle=True)
    assert counts[0] == max_episodes
    assert counts[1] > 100

if __name__ == '__main__':
    # record batches of data from a random agent
    record_random_agent()