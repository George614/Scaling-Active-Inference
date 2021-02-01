import logging
import numpy as np
import math
import os
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


@tf.function
def train_step(model, optimizer, s_prev, a_prev, o_cur):
    ''' training one step for the VAE-like components of active inference agent '''
    # mask out zero samples from shorter sequences
    mask = tf.not_equal(o_cur[:, 0], 0)
    
    with tf.GradientTape() as tape:
        total_loss, gnll, kld, state_post = model(s_prev, a_prev, o_cur, mask)
    
    # since all layer/model classes are subclassed from tf.Modules, it's
    # eaiser to gather all trainable variables from all layers then calculate
    # gradients for all of them. Note that tf.Module is used instead of 
    # keras.Layer or keras.Model because tf.Module is cleaner / bare-bone
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # use back-prop algorithms for optimization for now
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, gnll, kld, state_post


def config4performance(dataset, batch_size=32, buffer_size=1024):
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


if __name__ == "__main__":
    print("training data path: ", args.data_path)
    all_data = np.load(args.data_path + "/all_data.npy", allow_pickle=True)
    train_dataset = tf.data.Dataset.from_tensor_slices(all_data)
    train_dataset = train_dataset.shuffle(buffer_size=args.buffer_size).batch(args.vae_batch_size)
    model_save_path = "results/{}/{}/vae_ai".format(args.exp_name, args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # use tensorboard for monitoring training if needed
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    opt = tf.keras.optimizers.get(args.vae_optimizer)
    opt.__setattr__('learning_rate', args.vae_learning_rate)

    # initial our model using parameters in the config file
    stateModel = StateModel(args=args)
    # fake data
    # x_fake = tf.random.normal((1024, 100, 8))  # batch x seq_length x data_dim
    total_train_steps = 0

    for epoch in range(args.vae_num_epoch):
        print("=================== Epoch {} ==================".format(epoch+1))
        loss_accum = 0
        for training_step, x_batch in enumerate(train_dataset):
            # split the batch data into observation and action (disgard reward for now)
            o_cur_batch, a_prev_batch = x_batch[:, :, 0:1], x_batch[:, :, 2:3] # x_batch[:, :, 3:4]
            # if stateModel.posterior.mu is None:
            #     initial_states = tf.random.normal((args.vae_batch_size, args.z_size))
            # else:
            #     initial_states = stateModel.posterior.mu + stateModel.posterior.std * tf.random.normal((args.vae_batch_size, args.z_size))
            initial_states = tf.zeros((args.vae_batch_size, args.z_size))
            initial_actions = np.zeros((args.vae_batch_size, args.a_width))
            # idx_rand_action = np.random.uniform(size=args.vae_batch_size) > 0.5
            # initial_actions[idx_rand_action, :] = 2
            initial_actions = tf.convert_to_tensor(initial_actions, dtype=tf.float32)
            
            for time_step in range(tf.shape(o_cur_batch)[1]):
                if time_step == 0:
                    loss_episode = 0
                    gnll_episode = 0
                    kld_episode = 0
                    total_loss, gnll, kld, state_post = train_step(stateModel, opt, initial_states, initial_actions, o_cur_batch[:, 0, :])
                else:
                    total_loss, gnll, kld, state_post = train_step(stateModel, opt, state_post, a_prev_batch[:, time_step, :], o_cur_batch[:, time_step, :])
                loss_episode += total_loss.numpy()
                gnll_episode += gnll.numpy()
                kld_episode += kld.numpy()
            
            total_train_steps += 1
            tf.summary.scalar('total', loss_episode, step=total_train_steps)
            tf.summary.scalar('gnll', gnll_episode, step=total_train_steps)
            tf.summary.scalar('kld', kld_episode, step=total_train_steps)
            print("training_step {}, total_loss: {:.3f}, gnll: {:.3f}, kld: {:.3f}".format(training_step+1, loss_episode, gnll_episode, kld_episode))
            # loss_accum += loss_episode
            # print("training_step {}, loss: {:.4f}".format(training_step+1, loss_accum / (training_step+1)))
    
    #%% fine-tune with human data
    fine_tune_epoch = 100
    path = "D:/Projects/TF2_ML/openai.gym.human3"
    d_human = np.load(path+'/all_data.npy', allow_pickle=True)
    d_human = d_human[1:]
    # make the temporal length of human data to be the length of the
    # longest episode
    pad_length = None
    for i in range(np.shape(d_human)[1]):
        if np.alltrue(d_human[:, i, :] == 0):
            pad_length = i
            break
    d_human = d_human[:, :pad_length, :]
    batch_tune = len(d_human)
    opt.__setattr__('learning_rate', args.vae_learning_rate*0.3)
    human_data = tf.data.Dataset.from_tensor_slices(d_human)
    human_data = human_data.shuffle(len(d_human)).batch(batch_tune, drop_remainder=True)
    print("Starting fine-tuning...")
    
    # @tf.function
    def train_step_ft(model, optimizer, s_prev, a_prev, o_cur):
        ''' fine-tune the VAE-AI model '''
        # mask out zero samples from shorter sequences
        mask = tf.not_equal(o_cur[:, 0], 0)
        # tf.print(mask)
        with tf.GradientTape() as tape:
            total_loss, gnll, kld, state_post = model(s_prev, a_prev, o_cur, mask)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        # use back-prop algorithms for optimization for now
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return total_loss, gnll, kld, state_post

    for epoch in range(fine_tune_epoch):
        print("=================== Epoch {} ==================".format(epoch+1))
        loss_accum = 0
        for training_step, x_batch in enumerate(human_data):
            # split the batch data into observation and action (disgard reward for now)
            o_cur_batch, a_prev_batch = x_batch[:, :, 0:1], x_batch[:, :, 2:3] # x_batch[:, :, 3:4]
            initial_states = tf.zeros((batch_tune, args.z_size))
            initial_actions = tf.zeros((len(x_batch), args.a_width))
            
            for time_step in range(tf.shape(o_cur_batch)[1]):
                if time_step == 0:
                    loss_episode = 0
                    gnll_episode = 0
                    kld_episode = 0
                    total_loss, gnll, kld, state_post = train_step(stateModel, opt, initial_states, initial_actions, o_cur_batch[:, 0, :])
                else:
                    total_loss, gnll, kld, state_post = train_step(stateModel, opt, state_post, a_prev_batch[:, time_step, :], o_cur_batch[:, time_step, :])
                loss_episode += total_loss.numpy()
                gnll_episode += gnll.numpy()
                kld_episode += kld.numpy()
            
            print("training_step {}, total_loss: {:.3f}, gnll: {:.3f}, kld: {:.3f}".format(training_step+1, loss_episode, gnll_episode, kld_episode))
    
    # save the trained model
    tf.saved_model.save(stateModel, model_save_path)
    print("> Finished training. Model saved in: ", model_save_path)
    # evaluate the model using the reconstruction error (on observation)