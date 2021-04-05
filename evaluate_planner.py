import numpy as np
import gym
import os
import logging
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from utils import PARSER
from planner import Planner

AUTOTUNE = tf.data.experimental.AUTOTUNE
args = PARSER.parse_args()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    # # read in human data and load AI stateModel
    model_path_name = "results/{}/{}/vae_ai".format(args.exp_name, args.env_name)
    loadedModel = tf.saved_model.load(model_path_name)
    human_data = np.load(args.human_data_path + "all_human_data.npy", allow_pickle=True)
    
    a_idx_right = human_data[:, :, 2] == 2
    human_data[a_idx_right, 2] = 1
    all_actions = human_data[:, :, 2].reshape(-1)
    all_actions_onehot = tf.keras.utils.to_categorical(all_actions)
    all_actions_onehot = np.reshape(all_actions_onehot, (human_data.shape[0], human_data.shape[1], args.a_width))
    human_data = np.concatenate((human_data[:, :, 0:1], all_actions_onehot), axis=-1)

    human_data = tf.convert_to_tensor(human_data[1:, 1:, :], dtype=tf.float32)
    n_episode = tf.shape(human_data)[0]
    len_episode = tf.shape(human_data)[1]
    state_post_end = np.zeros((n_episode, args.z_size), dtype=np.float32)
    
    # run human data through the posterior model and save the goal states
    for epi in range(n_episode):
        for time_step in range(len_episode):
            o_cur, a_prev = human_data[epi:epi+1, time_step, 0:1], human_data[epi:epi+1, time_step, 1:]
            if time_step == 0:
                # batch_samples = loadedModel.posterior.mu + loadedModel.posterior.std * tf.random.normal(tf.shape(loadedModel.posterior.mu))
                # batch_mean = tf.reduce_mean(batch_samples, axis=0)
                # batch_std = tf.math.reduce_std(batch_samples, axis=0)
                # initial_states = tf.random.normal((1, args.z_size), mean=batch_mean, stddev=batch_std)
                initial_states = tf.zeros((1, args.z_size))
                state_post, mu_post, std_post = loadedModel.posterior(tf.concat([initial_states, a_prev, o_cur], axis=-1))
            else:
                if tf.squeeze(o_cur) == 0:
                    # state_post_end[epi, :] = state_post.numpy().squeeze()  # use a sample of posterior states
                    state_post_end[epi, :] = mu_post.numpy().squeeze()  # use the mean of states from posterior
                    break
                else:
                    state_post, mu_post, std_post = loadedModel.posterior(tf.concat([state_post, a_prev, o_cur], axis=-1))
    
    # save the mean and std of the human goal state
    end_state_mean = np.mean(state_post_end, axis=0)
    end_state_std = np.std(state_post_end, axis=0)

    #%%
    #  create planner using the trained AI state model and human prior preference
    # planner = Planner(loadedModel, args, end_state_mean, s_std_prefer=end_state_std)
    planner = Planner(loadedModel, args, end_state_mean)
    env = gym.make('MountainCar-v0').env
    for i_episode in range(20):
        print("starting new episode...")
        observation = env.reset()
        for t in range(300):
            env.render()
            print("Observation: ", observation)
            # only give the planner the position as observation
            action = planner(tf.convert_to_tensor(observation[0:1], dtype=tf.float32))
            if action == 1:
                action = 2
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

