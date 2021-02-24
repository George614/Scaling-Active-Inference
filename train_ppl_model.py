import logging
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import math
import gym
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from datetime import datetime
from utils import PARSER
from AI_model import PPLModel

AUTOTUNE = tf.data.experimental.AUTOTUNE
args = PARSER.parse_args()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# @tf.function
def train_step(model, optimizer, o_cur):
    with tf.GradientTape() as tape:
        total_loss, loss_model, loss_efe = model(o_cur)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, loss_model, loss_efe


def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    num_frames = 10000

    # use tensorboard for monitoring training if needed
    now = datetime.now()
    model_save_path = "results/ppl_model/{}/".format(args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard', now.strftime("%b-%d-%Y %H-%M-%S"))
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    opt = tf.keras.optimizers.get(args.vae_optimizer)
    opt.__setattr__('learning_rate', args.vae_learning_rate)

    # load the prior preference model
    prior_model_save_path = "results/prior_model/{}/".format(args.env_name)
    priorModel = tf.saved_model.load(prior_model_save_path)
    # initial our model using parameters in the config file
    pplModel = PPLModel(priorModel, args=args)

    losses = []
    all_rewards = []
    episode_reward = 0
    
    env = gym.make('MountainCar-v0')
    observation = env.reset()

    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        pplModel.epsilon.assign(epsilon)
        
        obv = tf.convert_to_tensor(observation, dtype=tf.float32)
        obv = tf.expand_dims(obv, axis=0)
        if pplModel.obv_t is not None:
            total_loss, loss_model, loss_efe = train_step(pplModel, opt, obv)
            action = tf.argmax(pplModel.a_t, axis=-1).numpy().squeeze()

            observation, reward, done, info = env.step(action)
            episode_reward += reward
            losses.append(total_loss.numpy())
        else:
            pplModel(obv)

        if frame_idx > 1 and done:
            observation = env.reset()
            pplModel.obv_t = None
            all_rewards.append(episode_reward)
            episode_reward = 0
            done = False

        if frame_idx % 200 == 0:
            plot(frame_idx, all_rewards, losses)
            
    env.close()