import logging
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from datetime import datetime
from utils import PARSER
from AI_models import PPLModel
from buffers import ReplayBuffer, NaivePrioritizedBuffer
from scheduler import Linear_schedule, Exponential_schedule

AUTOTUNE = tf.data.experimental.AUTOTUNE
args = PARSER.parse_args()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    num_frames = 200000  # total number of training steps
    num_episodes = 2000  # total number of training episodes
    test_episodes = 1  # number of episodes for testing
    target_update_freq = 500  # in terms of steps
    target_update_ep = 2  # in terms of episodes
    buffer_size = 100000
    prob_alpha = 0.6  # power value used in the PER
    batch_size = 256
    grad_norm_clip = 10.0
    log_interval = 4
    keep_expert_batch = True
    use_per_buffer = True
    vae_reg = False
    epistemic_anneal = True
    plot_eps_schedule = False
    plot_rewards = False
    is_stateful = args.is_stateful
    seed = args.seed
    # epsilon exponential decay schedule
    epsilon_start = 0.9
    epsilon_final = 0.02
    epsilon_decay = num_frames / 20
    epsilon_by_frame = Exponential_schedule(epsilon_start, epsilon_final, epsilon_decay)
    # gamma linear schedule for VAE regularization
    gamma_start = 0.01
    gamma_final = 0.99
    gamma_ep_duration = 200
    gamma_by_episode = Linear_schedule(gamma_start, gamma_final, gamma_ep_duration)
    # rho linear schedule for annealing epistemic term
    anneal_start_reward = -180
    rho_start = 1.0
    rho_final = 0.1
    rho_ep_duration = 200
    rho_by_episode = Linear_schedule(rho_start, rho_final, rho_ep_duration)
    # beta linear schedule for prioritized experience replay
    beta_start = 0.4
    beta_final = 1.0
    beta_ep_duration = 400
    beta_by_episode = Linear_schedule(beta_start, beta_final, beta_ep_duration)
    if plot_eps_schedule:
        # plot the epsilon schedule
        plt.plot([epsilon_by_frame(i) for i in range(num_frames)])
        plt.xlabel('steps')
        plt.ylabel('epsilon')
        plt.title("Exponential decay schedule for epsilon")
        plt.show()

    ### use tensorboard for monitoring training if needed ###
    now = datetime.now()
    model_save_path = "results/ppl_model/{}/".format(args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard', now.strftime("%b-%d-%Y %H-%M-%S"))
    model_save_path = os.path.join(model_save_path, now.strftime("%b-%d-%Y_%H-%M-%S"))
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()

    ### initialize optimizer and buffers ###
    if args.vae_optimizer == "AdamW":
        import tensorflow_addons as tfa
        opt = tfa.optimizers.AdamW(learning_rate=args.vae_learning_rate, weight_decay=args.vae_weight_decay, epsilon=1e-5)
    else:
        opt = tf.keras.optimizers.get(args.vae_optimizer)
        opt.__setattr__('learning_rate', args.vae_learning_rate)
        opt.__setattr__('epsilon', 1e-5)
    if use_per_buffer:
        per_buffer = NaivePrioritizedBuffer(buffer_size * 2, prob_alpha=prob_alpha)
    else:
        expert_buffer = ReplayBuffer(buffer_size, seed=seed)
        replay_buffer = ReplayBuffer(buffer_size, seed=seed)
    
    # set seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ### load and pre-process human expert-batch data ###
    # print("Human expert batch data path: ", args.prior_data_path)
    # all_data = np.load(args.prior_data_path + "/all_human_data.npy", allow_pickle=True)
    # all_data = all_data[1:-1, :, :]  # exclude samples with imcomplete sequence

    # for i in range(len(all_data)):
    #     for j in range(1, np.shape(all_data)[1]):
    #         o_t = all_data[i, j-1, :2]
    #         o_tp1 = all_data[i, j, :2]
    #         reward = all_data[i, j, 3]
    #         action = all_data[i, j, 2]
    #         done = 0
    #         if j == np.shape(all_data)[1]-1:
    #             done = 1
    #             expert_buffer.push(o_t, action, reward, o_tp1, done)
    #             break
    #         if np.equal(all_data[i, j+1, 0], 0):
    #             done = 1
    #             expert_buffer.push(o_t, action, reward, o_tp1, done)
    #             break
    #         expert_buffer.push(o_t, action, reward, o_tp1, done)

    ### load and pre-process rl-zoo expert-batch data ###
    print("RL-zoo expert data path: ", args.zoo_data_path)
    all_data = np.load(args.zoo_data_path + "/zoo-agent-mcar.npy", allow_pickle=True)
    idx_done = np.where(all_data[:, 6] == 1)[0]
    idx_done = idx_done - 1  # fix error on next_obv when done in original data
    mask = np.not_equal(all_data[:, 6], 1)
    for idx in idx_done:
        all_data[idx, 6] = 1
    all_data = all_data[mask]

    for i in range(min(len(all_data), buffer_size)):
        o_t, action, reward, o_tp1, done = all_data[i, :2], all_data[i, 2], all_data[i, 3], all_data[i, 4:6], all_data[i, 6]
        if use_per_buffer:
            per_buffer.push(o_t, action, reward, o_tp1, done)
        else:
            expert_buffer.push(o_t, action, reward, o_tp1, done)

    if not keep_expert_batch and not use_per_buffer:
        replay_buffer = expert_buffer

    ### load the prior preference model ###
    prior_model_save_path = "results/prior_model/{}/".format(args.env_name)
    prior_model_save_path = os.path.join(prior_model_save_path, "zoo_data_old_gnll")
    priorModel = tf.saved_model.load(prior_model_save_path)
    # initial our model using parameters in the config file
    pplModel = PPLModel(priorModel, args=args)

    mean_ep_reward = []
    std_ep_reward = []
    frame_idx = 0
    crash = False
    rho_anneal_start = False
    
    env = gym.make(args.env_name)
    observation = env.reset()

    # for frame_idx in range(1, num_frames + 1): # deprecated
    for ep_idx in range(num_episodes):  # training using episode as cycle
        ### training the PPL model ###
        pplModel.training.assign(True)
        done = False
        ## linear schedule for VAE model regularization
        if vae_reg:
            gamma = gamma_by_episode(ep_idx)
            pplModel.gamma.assign(gamma)
        if use_per_buffer:
            beta = beta_by_episode(ep_idx)
        while not done:
            frame_idx += 1
            epsilon = epsilon_by_frame(frame_idx)
            pplModel.epsilon.assign(epsilon)
            
            obv = tf.convert_to_tensor(observation, dtype=tf.float32)
            obv = tf.expand_dims(obv, axis=0)
            
            action = pplModel.act(obv)
            action = action.numpy().squeeze()

            next_obv, reward, done, _ = env.step(action)
            
            if use_per_buffer:
                per_buffer.push(observation, action, reward, next_obv, done)
                observation = next_obv
                batch_data = per_buffer.sample(batch_size, beta=beta)
                [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done, batch_indices, batch_weights] = batch_data
                batch_action = tf.one_hot(batch_action, depth=args.a_width)
                grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target, priorities = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done, batch_weights)
                per_buffer.update_priorities(batch_indices, priorities.numpy())
            else:
                replay_buffer.push(observation, action, reward, next_obv, done)
                observation = next_obv

                if keep_expert_batch:
                    total_samples = len(expert_buffer) + len(replay_buffer)
                    if total_samples / len(replay_buffer) < batch_size:
                        # sample from both expert buffer and replay buffer
                        n_replay_samples = np.floor(batch_size * len(replay_buffer) / total_samples)
                        n_expert_samples = batch_size - n_replay_samples
                    # if len(replay_buffer) < batch_size // 2:
                    #     continue
                    # else:
                        # sample equal number of samples from expert buffer and replay buffer
                        # n_expert_samples = batch_size // 2
                        # n_replay_samples = batch_size // 2
                        expert_batch = expert_buffer.sample(int(n_expert_samples))
                        replay_batch = replay_buffer.sample(int(n_replay_samples))
                        batch_data = [np.concatenate((expert_sample, replay_sample)) for expert_sample, replay_sample in zip(expert_batch, replay_batch)]
                    else:
                        # sample from expert buffer only
                        batch_data = expert_buffer.sample(batch_size)
                else:
                    batch_data = replay_buffer.sample(batch_size)

                [batch_obv, batch_action, batch_reward, batch_next_obv, batch_done] = batch_data
                
                batch_action = tf.one_hot(batch_action, depth=args.a_width)
                grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target = pplModel.train_step(batch_obv, batch_next_obv, batch_action, batch_done)
            
            if tf.math.is_nan(loss_efe):
                print("loss_efe nan at frame #", frame_idx)
                break
            
            ### clip gradients  ###
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
                        crash = True
                grads_efe_clipped.append(grad)

            if crash:
                break

            ### Gradient descend by Adam optimizer excluding variables with no gradients ###
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

            weights_maxes = [tf.math.reduce_max(var) for var in pplModel.trainable_variables]
            weights_mins = [tf.math.reduce_min(var) for var in pplModel.trainable_variables]
            weights_max = tf.math.reduce_max(weights_maxes)
            weights_min = tf.math.reduce_min(weights_mins)

            grads_model_maxes = [tf.math.reduce_max(grad) for grad in grads_model if grad is not None]
            grads_model_mins = [tf.math.reduce_min(grad) for grad in grads_model if grad is not None]
            grads_model_max = tf.math.reduce_max(grads_model_maxes)
            grads_model_min = tf.math.reduce_min(grads_model_mins)

            grads_efe_maxes = [tf.math.reduce_max(grad) for grad in grads_efe if grad is not None]
            grads_efe_mins = [tf.math.reduce_min(grad) for grad in grads_efe if grad is not None]
            grads_efe_max = tf.math.reduce_max(grads_efe_maxes)
            grads_efe_min = tf.math.reduce_min(grads_efe_mins)
            

            ### tensorboard logging for training statistics ###
            if frame_idx % log_interval == 0:
                tf.summary.scalar('loss_model', loss_model, step=frame_idx)
                tf.summary.scalar('loss_efe', loss_efe, step=frame_idx)
                tf.summary.scalar('loss_l2', loss_l2, step=frame_idx)
                tf.summary.scalar('weights_max', weights_max, step=frame_idx)
                tf.summary.scalar('weights_min', weights_min, step=frame_idx)
                tf.summary.scalar('grads_model_max', grads_model_max, step=frame_idx)
                tf.summary.scalar('grads_model_min', grads_model_min, step=frame_idx)
                tf.summary.scalar('grads_efe_max', grads_efe_max, step=frame_idx)
                tf.summary.scalar('grads_efe_min', grads_efe_min, step=frame_idx)
                tf.summary.scalar('R_ti_max', tf.math.reduce_max(R_ti), step=frame_idx)
                tf.summary.scalar('R_ti_min', tf.math.reduce_min(R_ti), step=frame_idx)
                tf.summary.scalar('R_te_max', tf.math.reduce_max(R_te), step=frame_idx)
                tf.summary.scalar('R_te_min', tf.math.reduce_min(R_te), step=frame_idx)
                tf.summary.scalar('EFE_old_max', tf.math.reduce_max(efe_t), step=frame_idx)
                tf.summary.scalar('EFE_old_min', tf.math.reduce_min(efe_t), step=frame_idx)
                tf.summary.scalar('EFE_new_max', tf.math.reduce_max(efe_target), step=frame_idx)
                tf.summary.scalar('EFE_new_min', tf.math.reduce_min(efe_target), step=frame_idx)

            # if frame_idx % target_update_freq == 0:
            #     pplModel.update_target()

            if frame_idx % 200 == 0:
                print("frame {}, loss_model {:.3f}, loss_efe {:.3f}".format(frame_idx, loss_model.numpy(), loss_efe.numpy()))

        if ep_idx % target_update_ep == 0:
            pplModel.update_target()

        ### after each training episode is done ###
        observation = env.reset()
        if is_stateful:
            pplModel.clear_state()

        ### evaluate the PPL model using a number of episodes ###
        pplModel.training.assign(False)
        pplModel.epsilon.assign(0.0) # use greedy policy when testing
        reward_list = []
        for _ in range(test_episodes):
            episode_reward = 0
            done_test = False
            while not done_test:
                obv = tf.convert_to_tensor(observation, dtype=tf.float32)
                obv = tf.expand_dims(obv, axis=0)
                action = pplModel.act(obv)
                action = action.numpy().squeeze()
                observation, reward, done_test, _ = env.step(action)
                episode_reward += reward
            observation = env.reset()
            if is_stateful:
                pplModel.clear_state()
            reward_list.append(episode_reward)

        mean_reward = np.mean(reward_list)
        std_reward = np.std(reward_list)
        mean_ep_reward.append(mean_reward)
        std_ep_reward.append(std_reward)
        print("episode {}, mean reward {:.3f}, std reward {:.3f}".format(ep_idx+1, mean_reward, std_reward))
        tf.summary.scalar('mean_ep_rewards', mean_ep_reward[-1], step=ep_idx+1)

        # annealing of the epistemic term based on the average test rewards
        if epistemic_anneal:
            if not rho_anneal_start and mean_reward > anneal_start_reward:
                start_ep = ep_idx
                rho_anneal_start = True
            if rho_anneal_start:
                rho = rho_by_episode(ep_idx - start_ep)
                pplModel.rho.assign(rho)

    env.close()
    if crash:
        print("Training crashed!!")
    else:
        # save the PPL model
        tf.saved_model.save(pplModel, model_save_path)
        print("> Trained the PPL model. Saved in:", model_save_path)
        if plot_rewards:
            # plot the mean and standard deviation of episode rewards
            mean_ep_reward = np.asarray(mean_ep_reward)
            std_ep_reward = np.asarray(std_ep_reward)
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(mean_ep_reward)), mean_ep_reward, alpha=0.7, color='red', label='mean', linewidth = 0.5)
            ax.fill_between(np.arange(len(mean_ep_reward)), np.clip(mean_ep_reward - std_ep_reward, -200, None), mean_ep_reward + std_ep_reward, color='#888888', alpha=0.4)
            ax.legend(loc='upper left')
            ax.set_ylabel("Rewards")
            ax.set_xlabel("N_episode")
            ax.set_title("Episode rewards")
            plt.show()