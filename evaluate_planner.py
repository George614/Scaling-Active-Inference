import numpy as np
import gym
import os
import logging
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from utils import PARSER
from AI_model import StateModel, Planner

AUTOTUNE = tf.data.experimental.AUTOTUNE
args = PARSER.parse_args()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    # # read in human data and load AI stateModel
    model_path_name = "results/{}/{}/vae_ai".format(args.exp_name, args.env_name)
    loadedModel = tf.saved_model.load(model_path_name)
    human_data = np.load("D:\\Projects\\TF2_ML\\openai.gym.human3\\all_data.npy", allow_pickle=True)
    human_data = tf.convert_to_tensor(human_data[1:], dtype=tf.float32)
    n_episode = tf.shape(human_data)[0]
    len_episode = tf.shape(human_data)[1]
    state_post_end = np.zeros((n_episode, args.z_size))
    
    # run human data through the posterior model and save the goal states
    for epi in range(n_episode):
        for time_step in range(len_episode):
            o_cur, a_prev = human_data[epi:epi+1, time_step, 0:1], human_data[epi:epi+1, time_step, 2:3]
            if time_step == 0:
                batch_samples = loadedModel.posterior.mu + loadedModel.posterior.std * tf.random.normal(tf.shape(loadedModel.posterior.mu))
                batch_mean = tf.reduce_mean(batch_samples, axis=0)
                batch_std = tf.math.reduce_std(batch_samples, axis=0)
                initial_states = tf.random.normal((1, args.z_size), mean=batch_mean, stddev=batch_std)
                _, _, state_post = loadedModel.serve(initial_states, a_prev, o_cur)
            else:
                if tf.squeeze(o_cur) == 0:
                    state_post_end[epi, :] = state_post.numpy().squeeze()
                    break
                else:
                    _, _, state_post = loadedModel.serve(state_post, a_prev, o_cur)
    
    # save the mean and std of the human goal state
    end_state_mean = np.mean(state_post_end, axis=0)
    end_state_std = np.std(state_post_end, axis=0)
    # s_prefer_mean = np.array([-1.18251, -1.16918, -0.715479, 0.872501], dtype=np.float32)
    # s_prefer_std = np.array([0.0498461, 0.131107, 0.0632425, 0.0814742], dtype=np.float32)

    #%%
    #  create planner using the trained AI state model and human prior preference
    # planner = Planner(loadedModel, args, s_prefer_mean, s_std_prefer=s_prefer_std)
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
            action = 0 if action.numpy() == 0 else 2
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

