import logging
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from datetime import  datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from utils import PARSER
from AI_model import StateModel

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


if __name__ == "__main__":
    path = "D:/Projects/TF2_ML/openai.gym.human3"
    d_human = np.load(path+'/all_data.npy', allow_pickle=True)
    d_human = d_human[40] # take only 1 episode
    pad_length = None
    for i in range(np.shape(d_human)[0]):
        if np.alltrue(d_human[i, :] == 0):
            pad_length = i
            break
    # discard the first sample which doesn't contain a meaningful action
    # and clip the sequence to exclude empty samples
    d_human = d_human[1:pad_length, :]
    # for MountainCar data, map the action values from {0, 2} to {0, 1}
    # then one-hot encoded the action vector
    a_idx_right = d_human[:, 2] == 2
    d_human[a_idx_right, 2] = 1
    all_actions = d_human[:, 2]
    all_actions_onehot = tf.keras.utils.to_categorical(all_actions)
    d_human = np.concatenate((d_human[:, 0:1], all_actions_onehot), axis=-1)

    human_data = tf.convert_to_tensor(d_human)
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

    n_epochs = 1000
    for epoch in range(n_epochs):
        print("\n=================== Epoch {} ==================".format(epoch+1))
        loss_accum = 0
        # split the batch data into observation and action (disgard reward for now)
        o_cur_seq, a_seq = human_data[:, 0:1], human_data[:, 1:]
        initial_states = tf.zeros((1, args.z_size))
        
        for time_step in range(tf.shape(o_cur_seq)[0]):
            if time_step == 0:
                loss_episode = 0
                gnll_episode = 0
                kld_episode = 0
                total_loss, gnll, kld, state_post = train_step(stateModel, opt, initial_states, a_seq[0:1, :], o_cur_seq[0:1, :])
            else:
                total_loss, gnll, kld, state_post = train_step(stateModel, opt, state_post, a_seq[time_step:time_step+1, :], o_cur_seq[time_step:time_step+1, :])
            loss_episode += total_loss.numpy()
            gnll_episode += gnll.numpy()
            kld_episode += kld.numpy()
        
        tf.summary.scalar('total', loss_episode, step=epoch)
        tf.summary.scalar('gnll', gnll_episode, step=epoch)
        tf.summary.scalar('kld', kld_episode, step=epoch)
        print("\nEpoch {}, total_loss: {:.3f}, gnll: {:.3f}, kld: {:.3f}".format(epoch+1, loss_episode, gnll_episode, kld_episode))

    #%% evaluate the VAE model using the reconstruction error (on observation)
    o_rec_post_seq = tf.zeros(tf.shape(o_cur_seq)).numpy()
    o_rec_tran_seq = tf.zeros(tf.shape(o_cur_seq)).numpy()
    state_post_seq = np.zeros((len(o_cur_seq), args.z_size), dtype=np.float32)
    state_tran_seq = np.zeros((len(o_cur_seq), args.z_size), dtype=np.float32)

    for time_step in range(tf.shape(o_cur_seq)[0]):
        if time_step == 0:
            state_post, mu_s_post, std_s_post = stateModel.posterior(tf.concat([initial_states, a_seq[0:1, :], o_cur_seq[0:1, :]], axis=-1))
            state_tran, mu_s_tran, std_tran = stateModel.transition(tf.concat([mu_s_post, a_seq[0:1, :]], axis=-1))
        else:
            state_post, mu_s_post, std_s_post = stateModel.posterior(tf.concat([state_post, a_seq[time_step:time_step+1, :], o_rec_post], axis=-1))
            state_tran, mu_s_tran, std_tran = stateModel.transition(tf.concat([state_tran, a_seq[time_step:time_step+1, :]], axis=-1))

        o_rec_post, mu_o_post, std_o_post = stateModel.likelihood(state_post)
        o_rec_tran, mu_o_tran, std_o_tran = stateModel.likelihood(state_tran)
        o_rec_post_seq[time_step, :] = mu_o_post.numpy().squeeze()
        o_rec_tran_seq[time_step, :] = mu_o_tran.numpy().squeeze()
        state_post_seq[time_step, :] = mu_s_post.numpy().squeeze()
        state_tran_seq[time_step, :] = mu_s_tran.numpy().squeeze()

    o_cur_seq_np = o_cur_seq.numpy().squeeze()
    steps = len(o_cur_seq_np)
    fig = plt.figure(constrained_layout=True)
    ymax = max(o_cur_seq_np.max(), o_rec_post_seq.max(), o_rec_tran_seq.max())
    ymin = min(o_cur_seq_np.min(), o_rec_post_seq.min(), o_rec_tran_seq.min())
    ax = plt.axes(xlim=(0, steps), ylim=(ymin, ymax)) 
    line1, = ax.plot(np.arange(steps), o_cur_seq_np, lw=1, linestyle="-", label="ground truth")
    line2, = ax.plot(np.arange(steps), o_rec_tran_seq, lw=1, linestyle="-.", label="transition model")
    line3, = ax.plot(np.arange(steps), o_rec_post_seq, lw=1, linestyle="--", label="posterior model")
    ax.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
               ncol=3, mode="expand", fontsize='small', borderaxespad=0.)
    plt.xlabel('Time steps')
    plt.ylabel('Observation / postion')
    plt.title("Predicted observations plotted on top of the ground truth observations")
    plt.show()
    fig.savefig(os.path.join(model_save_path, 'reconstruction_compare.pdf'), dpi=600, bbox_inches="tight")

    fig = plt.figure(constrained_layout=True)
    ymax = state_post_seq.max()
    ymin = state_post_seq.min()
    ax = plt.axes(xlim=(0, steps), ylim=(ymin, ymax)) 
    line1, = ax.plot(np.arange(steps), state_post_seq[:, 0], lw=1, linestyle="-", label="1st dim")
    line2, = ax.plot(np.arange(steps), state_post_seq[:, 1], lw=1, linestyle="-.", label="2nd dim")
    line3, = ax.plot(np.arange(steps), state_post_seq[:, 2], lw=1, linestyle="--", label="3rd dim")
    line3, = ax.plot(np.arange(steps), state_post_seq[:, 3], lw=1, linestyle="--", label="4th dim")
    ax.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
               ncol=4, mode="expand", fontsize='small', borderaxespad=0.)
    plt.xlabel('Time steps')
    plt.ylabel('Hidden state values')
    plt.title("Hidden states of the approximate posterior model")
    plt.show()
    fig.savefig(os.path.join(model_save_path, 'posterior_states.pdf'), dpi=600, bbox_inches="tight")

    fig = plt.figure(constrained_layout=True)
    ymax = state_tran_seq.max()
    ymin = state_tran_seq.min()
    ax = plt.axes(xlim=(0, steps), ylim=(ymin, ymax)) 
    line1, = ax.plot(np.arange(steps), state_tran_seq[:, 0], lw=1, linestyle="-", label="1st dim")
    line2, = ax.plot(np.arange(steps), state_tran_seq[:, 1], lw=1, linestyle="-.", label="2nd dim")
    line3, = ax.plot(np.arange(steps), state_tran_seq[:, 2], lw=1, linestyle="--", label="3rd dim")
    line3, = ax.plot(np.arange(steps), state_tran_seq[:, 3], lw=1, linestyle="--", label="4th dim")
    ax.legend(bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
               ncol=4, mode="expand", fontsize='small', borderaxespad=0.)
    plt.xlabel('Time steps')
    plt.ylabel('Hidden state values')
    plt.title("Hidden states of the transition model")
    plt.show()
    fig.savefig(os.path.join(model_save_path, 'transition_states.pdf'), dpi=600, bbox_inches="tight")