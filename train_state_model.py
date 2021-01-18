import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import PARSER
from AI_model import StateModel, Planner

AUTOTUNE = tf.data.experimental.AUTOTUNE
args = PARSER.parse_args()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


@tf.function
def train_step(model, optimizer, s_prev, a_prev, o_cur, training=True):
    ''' training one step for the VAE-like components of active inference agent '''
    with tf.GradientTape() as tape:
        loss, state_post = model(s_prev, a_prev, o_cur, training=training)
    # since all layer/model classes are subclassed from tf.Modules, it's
    # eaiser to gather all trainable variables from all layers then calculate
    # gradients for all of them. Note that tf.Module is used instead of 
    # keras.Layer or keras.Model because tf.Module is cleaner / bare-bone
    gradients = tape.gradient(loss, model.trainable_variables)
    # use back-prop algorithms for optimization for now
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, state_post


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
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    opt = tf.keras.optimizers.get(args.vae_optimizer)
    opt.__setattr__('learning_rate', args.vae_learning_rate)
    stateModel = StateModel(args=args)
    # fake data
    # x_fake = tf.random.normal((1024, 100, 8))  # batch x seq_length x data_dim
    
    for epoch in range(args.vae_num_epoch):
        print("=================== Epoch {} ==================".format(epoch+1))
        loss_accum = 0
        for training_step, x_batch in enumerate(train_dataset):
            # split the batch data into observation and action (disgard reward for now)
            o_cur_batch, a_prev_batch = x_batch[:, :, 0:1], x_batch[:, :, 2:3] # x_batch[:, :, 3:4]
            if stateModel.posterior.mu is None:
                initial_states = tf.random.normal((args.vae_batch_size, args.z_size))
                # initial_states = tf.convert_to_tensor(np.random.normal(size=(args.vae_batch_size, args.z_size)), dtype=tf.float32)
            else:
                initial_states = stateModel.posterior.mu + stateModel.posterior.std * tf.random.normal((args.vae_batch_size, args.z_size))
            
            initial_actions = np.zeros((args.vae_batch_size, args.a_width))
            idx_rand_action = np.random.uniform(size=args.vae_batch_size) > 0.5
            initial_actions[idx_rand_action, :] = 2
            initial_actions = tf.convert_to_tensor(initial_actions, dtype=tf.float32)
            
            for time_step in range(tf.shape(o_cur_batch)[1]):
                if time_step == 0:
                    loss_episode = 0
                    loss, state_post = train_step(stateModel, opt, initial_states, initial_actions, o_cur_batch[:, 0, :])
                else:
                    loss, state_post = train_step(stateModel, opt, state_post, a_prev_batch[:, time_step, :], o_cur_batch[:, time_step, :])
                loss_episode += loss.numpy().sum()
            
            loss_accum += loss_episode
            print("training_step {}, loss: {:.4f}".format(training_step+1, loss_accum / (training_step+1)))
        
    # save the trained model
    tf.saved_model.save(stateModel, model_save_path)

    # evaluate the model using the reconstruction error (on observation)
    