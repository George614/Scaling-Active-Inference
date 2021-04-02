import logging
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from datetime import datetime
from utils import PARSER
from AI_models import StateModel

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


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    ''' Cyclical Annealing Schedule: A Simple Approach to Mitigating {KL}
    Vanishing. Fu etal NAACL 2019 '''
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


if __name__ == "__main__":
    print("training data path: ", args.data_path)
    all_data = np.load(args.data_path + "/all_random_data.npy", allow_pickle=True)
    
    # for MountainCar data, map the action values from {0, 2} to {0, 1}
    # then one-hot encoded the action vector
    all_data = all_data[:, 1:, :]  # exclude first sample with empty action
    a_idx_right = all_data[:, :, 2] == 2
    all_data[a_idx_right, 2] = 1
    all_actions = all_data[:, :, 2].reshape(-1)
    all_actions_onehot = tf.keras.utils.to_categorical(all_actions)
    all_actions_onehot = np.reshape(all_actions_onehot, (all_data.shape[0], all_data.shape[1], args.a_width))
    all_data = np.concatenate((all_data[:, :, 0:1], all_actions_onehot), axis=-1)
    
    train_dataset = tf.data.Dataset.from_tensor_slices(all_data)
    train_dataset = train_dataset.shuffle(buffer_size=args.buffer_size).batch(args.vae_batch_size)
    model_save_path = "results/{}/{}/vae_ai".format(args.exp_name, args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # use tensorboard for monitoring training if needed
    now = datetime.now()
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard', now.strftime("%b-%d-%Y %H-%M-%S"))
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    opt = tf.keras.optimizers.get(args.vae_optimizer)
    opt.__setattr__('learning_rate', args.vae_learning_rate)

    # initial our model using parameters in the config file
    stateModel = StateModel(args=args)
    # fake data
    # x_fake = tf.random.normal((1024, 100, 8))  # batch x seq_length x data_dim
    total_train_steps = 0
    n_iters = (len(all_data) // args.vae_batch_size) * args.vae_num_epoch
    beta_array = frange_cycle_linear(n_iters, start=0.0, stop=args.vae_kl_weight, n_cycle=4, ratio=0.5)

    for epoch in range(args.vae_num_epoch):
        print("=================== Epoch {} ==================".format(epoch+1))
        for training_step, x_batch in enumerate(train_dataset):
            # split the batch data into observation and action (disgard reward for now)
            o_cur_batch, a_prev_batch = x_batch[:, :, 0:1], x_batch[:, :, 1:]
            # if stateModel.posterior.mu is None:
            #     initial_states = tf.random.normal((args.vae_batch_size, args.z_size))
            # else:
            #     initial_states = stateModel.posterior.mu + stateModel.posterior.std * tf.random.normal((args.vae_batch_size, args.z_size))
            initial_states = tf.zeros((args.vae_batch_size, args.z_size))
            # initial_actions = tf.zeros((args.vae_batch_size, args.a_width))

            stateModel.kl_weight.assign(beta_array[total_train_steps])
            
            # for time_step in range(tf.shape(o_cur_batch)[1]):
            #     if time_step == 0:
            #         loss_episode = 0
            #         gnll_episode = 0
            #         kld_episode = 0
            #         total_loss, gnll, kld, state_post = train_step(stateModel, opt, initial_states, a_prev_batch[:, 0, :], o_cur_batch[:, 0, :])
            #     else:
            #         total_loss, gnll, kld, state_post = train_step(stateModel, opt, state_post, a_prev_batch[:, time_step, :], o_cur_batch[:, time_step, :])

            #     loss_episode += total_loss.numpy()
            #     gnll_episode += gnll.numpy()
            #     kld_episode += kld.numpy()
            
            x_batches = [tf.split(x_batch, num_or_size_splits=2, axis=1)]
            loss_episode = 0
            gnll_episode = 0
            kld_episode = 0
            for x_batch in x_batches:
                o_cur_batch, a_prev_batch = x_batch[:, :, 0:1], x_batch[:, :, 1:]
                loss_accum = 0
                with tf.GradientTape() as tape:
                    for time_step in range(tf.shape(o_cur_batch)[1]):
                        mask = tf.not_equal(o_cur_batch[:, time_step, 0], 0)
                        if time_step == 0:
                            loss_chunk = 0
                            gnll_chunk = 0
                            kld_chunk = 0
                            total_loss, gnll, kld, state_post = stateModel(initial_states, a_prev_batch[:, time_step, :], o_cur_batch[:, time_step, :], mask)
                        else:
                            total_loss, gnll, kld, state_post = stateModel(state_post, a_prev_batch[:, time_step, :], o_cur_batch[:, time_step, :], mask)
                        loss_chunk += total_loss
                        gnll_chunk += gnll.numpy()
                        kld_chunk += kld.numpy()
                gradients = tape.gradient(loss_chunk, stateModel.trainable_variables)
                opt.apply_gradients(zip(gradients, stateModel.trainable_variables))

                loss_episode += loss_chunk.numpy()
                gnll_episode += gnll_chunk
                kld_episode += kld_chunk
            
            total_train_steps += 1
            tf.summary.scalar('total', loss_episode, step=total_train_steps)
            tf.summary.scalar('gnll', gnll_episode, step=total_train_steps)
            tf.summary.scalar('kld', kld_episode, step=total_train_steps)
            tf.summary.scalar('kld_weight', beta_array[total_train_steps-1], step=total_train_steps)
            print("training_step {}, total_loss: {:.3f}, gnll: {:.3f}, kld: {:.3f}".format(training_step+1, loss_episode, gnll_episode, kld_episode))
    
    # save the trained model
    tf.saved_model.save(stateModel, model_save_path)
    print("> Finished training. Model saved in: ", model_save_path)

    #%% fine-tune with human data
    fine_tune_epoch = 1000
    path = args.human_data_path
    d_human = np.load(path+'/all_human_data.npy', allow_pickle=True)
    d_human = d_human[1:]
    
    # make the temporal length of human data to be the length of the
    # longest episode
    pad_length = None
    for i in range(np.shape(d_human)[1]):
        if np.alltrue(d_human[:, i, :] == 0):
            pad_length = i
            break
    d_human = d_human[:, 1:pad_length, :] # exclude first sample with empty action

    # for MountainCar data, map the action values from {0, 2} to {0, 1}
    # then one-hot encoded the action vector
    a_idx_right = d_human[:, :, 2] == 2
    d_human[a_idx_right, 2] = 1
    all_actions = d_human[:, :, 2].reshape(-1)
    all_actions_onehot = tf.keras.utils.to_categorical(all_actions)
    all_actions_onehot = np.reshape(all_actions_onehot, (d_human.shape[0], d_human.shape[1], args.a_width))
    d_human = np.concatenate((d_human[:, :, 0:1], all_actions_onehot), axis=-1)

    batch_tune = len(d_human)
    opt.__setattr__('learning_rate', args.vae_learning_rate)
    human_data = tf.data.Dataset.from_tensor_slices(d_human)
    human_data = human_data.shuffle(len(d_human)).batch(batch_tune, drop_remainder=True)
    tune_iters = (len(d_human) // batch_tune) * fine_tune_epoch
    beta_array_tune = frange_cycle_linear(tune_iters, start=0.0, stop=args.vae_kl_weight, n_cycle=4, ratio=0.5)
    total_tuning_steps = 0
    print("Starting fine-tuning...")

    for epoch in range(fine_tune_epoch):
        print("=================== Epoch {} ==================".format(epoch+1))
        loss_accum = 0
        for training_step, x_batch in enumerate(human_data):
            # split the batch data into observation and action (disgard reward for now)
            o_cur_batch, a_prev_batch = x_batch[:, :, 0:1], x_batch[:, :, 1:]
            initial_states = tf.zeros((batch_tune, args.z_size))
            
            stateModel.kl_weight.assign(beta_array_tune[total_tuning_steps])
            
            for time_step in range(tf.shape(o_cur_batch)[1]):
                if time_step == 0:
                    loss_episode = 0
                    gnll_episode = 0
                    kld_episode = 0
                    total_loss, gnll, kld, state_post = train_step(stateModel, opt, initial_states, a_prev_batch[:, 0, :], o_cur_batch[:, 0, :])
                else:
                    total_loss, gnll, kld, state_post = train_step(stateModel, opt, state_post, a_prev_batch[:, time_step, :], o_cur_batch[:, time_step, :])
                loss_episode += total_loss.numpy()
                gnll_episode += gnll.numpy()
                kld_episode += kld.numpy()
            
            total_tuning_steps += 1
            print("training_step {}, total_loss: {:.3f}, gnll: {:.3f}, kld: {:.3f}".format(training_step+1, loss_episode, gnll_episode, kld_episode))
    
    # save the fine-tuned model
    model_save_path = model_save_path + '/fine-tune'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tf.saved_model.save(stateModel, model_save_path)
    print("> Finished fine-tuning. Model saved in: ", model_save_path)
    