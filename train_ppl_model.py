import logging
import numpy as np
from IPython.display import clear_output
from collections import deque
import matplotlib.pyplot as plt
import random
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


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state = np.asarray(np.concatenate(state), dtype=np.float32)
        next_state = np.asarray(np.concatenate(next_state), dtype=np.float32)
        action = np.asarray(action, dtype=np.int32)
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=bool)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    num_frames = 100000
    target_update_freq = 500
    buffer_size = 2000
    batch_size = 32
    grad_norm_clip = 1.0

    # use tensorboard for monitoring training if needed
    now = datetime.now()
    model_save_path = "results/ppl_model/{}/".format(args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard', now.strftime("%b-%d-%Y %H-%M-%S"))
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    #TODO: try RMSProp optimizer, change opt epsilon to 0.01
    opt = tf.keras.optimizers.get(args.vae_optimizer)
    opt.__setattr__('learning_rate', args.vae_learning_rate)
    opt.__setattr__('epsilon', 1e-2)
    replay_buffer = ReplayBuffer(buffer_size)

    # load the prior preference model
    prior_model_save_path = "results/prior_model/{}/".format(args.env_name)
    priorModel = tf.saved_model.load(prior_model_save_path)
    # initial our model using parameters in the config file
    pplModel = PPLModel(priorModel, args=args)

    losses = []
    all_rewards = []
    episode_reward = 0
    crash = False
    
    env = gym.make(args.env_name)
    observation = env.reset()

    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        pplModel.epsilon.assign(epsilon)
        
        obv = tf.convert_to_tensor(observation, dtype=tf.float32)
        obv = tf.expand_dims(obv, axis=0)
        
        action = pplModel.act(obv)
        action = action.numpy().squeeze()

        next_obv, reward, done, _ = env.step(action)
        replay_buffer.push(observation, action, reward, next_obv, done)
        
        observation = next_obv
        episode_reward += reward

        if done:
            observation = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            batch_obv, batch_action, batch_reward, batch_next_obv, batch_done = replay_buffer.sample(batch_size)
            batch_action = tf.one_hot(batch_action, depth=args.a_width)
            grads_efe, grads_model, grads_l2, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done)
            
            if tf.math.is_nan(loss_efe):
                print("loss_efe nan at frame #", frame_idx)
                break
            
            # clip gradients
            crash = False
            grads_model_clipped = []
            for grad in grads_model:
                if grad is not None:
                    grad = tf.clip_by_norm(grad, clip_norm=grad_norm_clip)
                    if tf.math.reduce_any(tf.math.is_nan(grad)):
                        print("grad_model nan at frame # ", frame_idx)
                        crash = True
                grads_model_clipped.append(grad)
            
            grads_efe_clipped = []
            for grad in grads_efe:
                if grad is not None:
                    grad = tf.clip_by_norm(grad, clip_norm=grad_norm_clip)
                    if tf.math.reduce_any(tf.math.is_nan(grad)):
                        print("grad_efe nan at frame # ", frame_idx)
                        #TODO print all weights,bias of EFE net
                        crash = True
                grads_efe_clipped.append(grad)

            grads_l2_clipped = []
            for grad in grads_l2:
                if grad is not None:
                    grad = tf.clip_by_norm(grad, clip_norm=grad_norm_clip)
                    if tf.math.reduce_any(tf.math.is_nan(grad)):
                        print("grad_l2 nan at frame # ", frame_idx)
                        crash = True
                grads_l2_clipped.append(grad)

            if crash:
                break

            opt.apply_gradients(
                (grad, var) 
                for (grad, var) in zip(grads_model_clipped, pplModel.trainable_variables) 
                if grad is not None
                )
            opt.apply_gradients(
                (grad, var) 
                for (grad, var) in zip(grads_efe_clipped, pplModel.trainable_variables) 
                if grad is not None
                )
            opt.apply_gradients(
                (grad, var) 
                for (grad, var) in zip(grads_l2_clipped, pplModel.trainable_variables) 
                if grad is not None
                )

            # losses.append(loss_model.numpy())
            weights_maxes = [tf.math.reduce_max(var) for var in pplModel.trainable_variables]
            weights_mins = [tf.math.reduce_min(var) for var in pplModel.trainable_variables]
            weights_max = tf.math.reduce_max(weights_maxes)
            weights_min = tf.math.reduce_min(weights_mins)

            # grads_model_maxes = [tf.math.reduce_max(grad) for grad in grads_model if grad is not None]
            # grads_model_mins = [tf.math.reduce_min(grad) for grad in grads_model if grad is not None]
            # grads_max = tf.math.reduce_max(grads_model_maxes)
            # grads_min = tf.math.reduce_min(grads_model_mins)

            grads_efe_maxes = [tf.math.reduce_max(grad) for grad in grads_efe if grad is not None]
            grads_efe_mins = [tf.math.reduce_min(grad) for grad in grads_efe if grad is not None]
            grads_max = tf.math.reduce_max(grads_efe_maxes)
            grads_min = tf.math.reduce_min(grads_efe_mins)
            
            tf.summary.scalar('loss_model', loss_model.numpy(), step=frame_idx)
            tf.summary.scalar('loss_efe', loss_efe.numpy(), step=frame_idx)
            tf.summary.scalar('loss_l2', loss_l2.numpy(), step=frame_idx)
            tf.summary.scalar('weights_max', weights_max.numpy(), step=frame_idx)
            tf.summary.scalar('weights_min', weights_min.numpy(), step=frame_idx)
            tf.summary.scalar('grads_max', grads_max.numpy(), step=frame_idx)
            tf.summary.scalar('grads_min', grads_min.numpy(), step=frame_idx)
            tf.summary.scalar('R_ti_max', tf.math.reduce_max(R_ti), step=frame_idx)
            tf.summary.scalar('R_ti_min', tf.math.reduce_min(R_ti), step=frame_idx)
            tf.summary.scalar('R_te_max', tf.math.reduce_max(R_te), step=frame_idx)
            tf.summary.scalar('R_te_min', tf.math.reduce_min(R_te), step=frame_idx)
            tf.summary.scalar('EFE_old_max', tf.math.reduce_max(efe_t), step=frame_idx)
            tf.summary.scalar('EFE_old_min', tf.math.reduce_min(efe_t), step=frame_idx)
            tf.summary.scalar('EFE_new_max', tf.math.reduce_max(efe_target), step=frame_idx)
            tf.summary.scalar('EFE_new_min', tf.math.reduce_min(efe_target), step=frame_idx)

        if frame_idx % target_update_freq == 0:
            pplModel.update_target()

        if frame_idx % 200 == 0 and len(all_rewards) > 0:
            # plot(frame_idx, all_rewards, losses)
            print("frame {}, loss_model {}, loss_efe {}, episode_reward {}".format(frame_idx, loss_model.numpy(), loss_efe.numpy(), all_rewards[-1]))
            tf.summary.scalar('last_episode_rewards', all_rewards[-1], step=frame_idx)

    env.close()