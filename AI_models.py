import tensorflow as tf
import numpy as np
import math
import common_nn as nn
import metrics as mcs


class PPLModel(tf.Module):
    '''
    Prior Preference Learning model with Inverse Q-Learning. Learn EFE and
    forward dynamics of an environment given learned prior preference from
    human expert
    '''
    def __init__(self, priorModel, args):
        super().__init__(name='PPLModel')
        self.dim_z = args.z_size  # latent/state space size
        self.dim_obv = args.o_size  # observation size
        self.a_size = args.a_width  # action size
        self.n_samples = args.vae_n_samples
        self.dropout_rate = args.vae_dropout_rate
        self.layer_norm = args.layer_norm
        self.kl_weight = tf.Variable(args.vae_kl_weight, trainable=False)
        self.encoder = nn.FlexibleEncoder((self.dim_obv, 128, 128, self.dim_z), name='Encoder', dropout_rate=self.dropout_rate, layer_norm=False)
        self.decoder = nn.FlexibleEncoder((self.dim_z, 128, 128, self.dim_obv), name='Decoder', dropout_rate=self.dropout_rate, layer_norm=False)
        self.transition = nn.FlexibleEncoder((self.dim_z + self.a_size, 128, 128, self.dim_z), name='Transition', dropout_rate=self.dropout_rate, layer_norm=False)
        # self.encoder = nn.Encoder(self.dim_obv, self.dim_z, name='Encoder')
        # self.decoder = nn.Encoder(self.dim_z, self.dim_obv, name='Decoder')
        # self.transition = nn.Encoder(self.dim_z + self.a_size, self.dim_z, name='Transition')
        self.EFEnet = nn.FlexibleMLP((self.dim_z, 128, 128, self.a_size), name='EFEnet', layer_norm=self.layer_norm)
        self.EFEnet_target = nn.FlexibleMLP((self.dim_z, 128, 128, self.a_size), name='EFEnet_target', trainable=False, layer_norm=self.layer_norm)
        self.update_target()
        self.priorModel = priorModel
        self.obv_t = None
        self.a_t = None
        self.epsilon = tf.Variable(1.0, trainable=False)  # epsilon greedy parameter
        self.training = tf.Variable(True, trainable=False) # training mode
        self.is_stateful = args.is_stateful  # whether the model maintain a hidden state
        self.double_q = args.double_q  # use double-Q learning or not
        self.z_state = None
        self.l2_reg = args.l2_reg
        self.gamma = tf.Variable(1.0, trainable=False)  # gamma weighting factor for balance KL-D on transition vs unit Gaussian
        self.rho = tf.Variable(0.0, trainable=False)  # weight term on the epistemic value

    @tf.function
    def act(self, obv_t):
        if self.is_stateful and self.z_state is None:
            state, _, _ = self.encoder(obv_t)
            self.z_state = tf.identity(state)
        
        if tf.random.uniform(shape=()) > self.epsilon:
            # action selection using o_t and EFE network
            if not self.is_stateful:
                states, _, _ = self.encoder(obv_t)
            else:
                states = self.z_state
            efe_t = self.EFEnet(states)
            action = tf.argmax(efe_t, axis=-1, output_type=tf.int32)
        else:
            action = tf.random.uniform(shape=(1,), maxval=self.a_size, dtype=tf.int32)

        if self.is_stateful:
            a_t = tf.one_hot(action, depth=self.a_size)
            state_next, _, _ = self.transition(tf.concat([self.z_state, a_t], axis=-1))
            self.z_state = tf.identity(state_next)

        return action

    def clear_state(self):
        self.z_state = None

    def update_target(self):
        for target_var, var in zip(self.EFEnet_target.variables, self.EFEnet.variables):
            target_var.assign(var)

    @tf.function
    def train_step(self, obv_t, obv_next, action, done, weights=None):
        with tf.GradientTape(persistent=True) as tape:
            ### run s_t and a_t through the PPL model ###
            states, state_mu, state_std = self.encoder(obv_t)
            # states, state_mu, state_std, state_log_sigma = self.encoder(obv_t)

            efe_t = self.EFEnet(states)  # in batch, output shape (batch x a_space_size)

            states_next_tran, s_next_tran_mu, s_next_tran_std = self.transition(tf.concat([states, action], axis=-1))
            # states_next_tran, s_next_tran_mu, s_next_tran_std, s_next_tran_log_sigma = self.transition(tf.concat([states, action], axis=-1))

            o_next_hat, o_next_mu, o_next_std = self.decoder(states_next_tran)
            # o_next_hat, o_next_mu, o_next_std, o_next_log_sigma = self.decoder(states_next_tran)

            states_next_enc, s_next_enc_mu, s_next_enc_std = self.encoder(obv_next)
            # states_next_enc, s_next_enc_mu, s_next_enc_std, s_next_enc_log_sigma = self.encoder(obv_next)

            with tape.stop_recording():
                o_next_prior, o_prior_mu, o_prior_std = self.priorModel(obv_t)
                
                # Alternative: difference between preferred future and predicted future
                # R_ti = -1.0 * mcs.g_nll(o_next_hat,
                #                     o_prior_mu,
                #                     o_prior_std * o_prior_std,
                #                     keep_batch=True)
                
                # difference between preferred future and actual future, i.e. instrumental term
                # R_ti = -1.0 * mcs.g_nll_old(o_prior_mu, o_prior_std, obv_next, keep_batch=True)
                R_ti = -1.0 * mcs.g_nll(obv_next,
                                    o_prior_mu,
                                    o_prior_std * o_prior_std,
                                    # o_prior_log_sigma,
                                    keep_batch=True)

                ### negative almost KL-D between state distribution from transition model and from
                # approximate posterior model (encoder), i.e. epistemic value. Assumption is made. ###
                # R_te = -1.0 * mcs.kl_d_old(s_next_tran_mu, s_next_tran_std, s_next_enc_mu, s_next_enc_std, keep_batch=True)
                R_te = -1.0 * mcs.kl_d(s_next_tran_mu,
                                    s_next_tran_std * s_next_tran_std,
                                    tf.math.log(s_next_tran_std),
                                    s_next_enc_mu,
                                    s_next_enc_std * s_next_enc_std,
                                    tf.math.log(s_next_enc_std),
                                    keep_batch=True)
                ### alternative epistemic term using sampled next state as X ###
                # R_te = mcs.g_nll(states_next_tran,
                #             s_next_tran_mu,
                #             s_next_tran_std * s_next_tran_std,
                #             # s_next_tran_log_sigma,
                #             keep_batch=True) - g_nll(states_next_tran,
                #             s_next_enc_mu,
                #             s_next_enc_std * s_next_enc_std,
                #             # s_next_enc_log_sigma,
                #             keep_batch=True)
                # clip the epistemic value
                R_te = tf.clip_by_value(R_te, -50.0, 50.0)

                # the nagative EFE value, i.e. the reward. Note the sign here
                R_t = R_ti + self.rho * R_te

            ## model reconstruction loss ##
            loss_reconst = mcs.g_nll(obv_next, o_next_mu, o_next_std * o_next_std) #, o_next_log_sigma)
            # regularization for weights
            loss_l2 = tf.add_n([tf.nn.l2_loss(var) for var in self.trainable_variables if 'W' in var.name])
            loss_l2 *= self.l2_reg
            # latent penalty term
            loss_latent = mcs.kl_d(s_next_enc_mu,
                                s_next_enc_std * s_next_enc_std,
                                tf.math.log(s_next_enc_std),
                                s_next_tran_mu,
                                s_next_tran_std * s_next_tran_std,
                                tf.math.log(s_next_tran_std))
            # latent regularization term
            unit_Gaussian_mu = tf.zeros(tf.shape(s_next_enc_mu))
            unit_Gaussian_std = tf.ones(tf.shape(s_next_enc_std))
            loss_latent_reg = mcs.kl_d(s_next_enc_mu,
                                    s_next_enc_std * s_next_enc_std,
                                    tf.math.log(s_next_enc_std),
                                    unit_Gaussian_mu,
                                    unit_Gaussian_std * unit_Gaussian_std,
                                    tf.math.log(unit_Gaussian_std))

            loss_model = loss_reconst + self.gamma * loss_latent + (1 - self.gamma) * loss_latent_reg + loss_l2

            # take the old EFE values given action indices
            efe_old = tf.math.reduce_sum(efe_t * action, axis=-1)

            with tape.stop_recording():
                # EFE values for next state, s_t+1 is from transition model instead of encoder
                efe_target = self.EFEnet_target(states_next_tran)
                if self.double_q:
                    idx_a_next = tf.math.argmax(efe_t, axis=-1, output_type=tf.dtypes.int32)
                else:
                    idx_a_next = tf.math.argmax(efe_target, axis=-1, output_type=tf.dtypes.int32)
                onehot_a_next = tf.one_hot(idx_a_next, depth=self.a_size)
                # take the new EFE values
                efe_new = tf.math.reduce_sum(efe_target * onehot_a_next, axis=-1)

            ## TD loss ##
            done = tf.cast(done, dtype=tf.float32)
            # Alternative: use Huber loss instead of MSE loss
            if weights is not None:
                loss_efe_batch = mcs.huber(efe_old, (R_t + efe_new * (1 - done)), keep_batch=True)
                priorities = tf.math.abs(loss_efe_batch + 1e-5)
                loss_efe_batch *= weights
                loss_efe = tf.math.reduce_mean(loss_efe_batch)
            else:
                loss_efe = mcs.huber(efe_old, (R_t + efe_new * (1 - done)))
            # use MSE loss for the TD error
            # loss_efe = mcs.mse(efe_old, (R_t + efe_new * (1 - done)))

        # calculate gradient w.r.t model reconstruction and TD respectively
        grads_model = tape.gradient(loss_model, self.trainable_variables)
        grads_efe = tape.gradient(loss_efe, self.trainable_variables)
        
        if weights is not None:
            return grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target, priorities
        
        return grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target


class StateModel(tf.Module):
    '''
    State model parameterized by a VAE-like architecture for active inference
    framework, consists of transition model, posterior model and likelihood
    model.
    '''
    def __init__(self, args):
        super().__init__(name='StateModel')
        self.dim_z = args.z_size  # latent space size
        self.dim_obv = args.o_size  # observation size
        self.a_size = args.a_width  # action size
        self.n_samples = args.vae_n_samples
        self.kl_weight = tf.Variable(args.vae_kl_weight, trainable=False)
        self.kl_regularize_weight = args.vae_kl_regularize_weight
        self.transition = nn.Encoder(self.dim_z + self.a_size, self.dim_z, n_samples=self.n_samples, name='transition')
        self.posterior = nn.Encoder(self.dim_z + self.a_size + self.dim_obv, self.dim_z, n_samples=self.n_samples, name='posterior')
        self.likelihood = nn.Encoder(self.dim_z, self.dim_obv, name='likelihood')


    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.bool)])
    def __call__(self, s_prev, a_prev, o_cur, mask):
        ''' forward function of the AIF state model used during training '''
        # add some noise to the observation
        o_cur_batch = o_cur + tf.random.normal(tf.shape(o_cur), stddev=0.05)
        _, mu_tran, std_tran = self.transition(tf.concat([s_prev, a_prev], axis=-1))
        state_post, mu_post, std_post = self.posterior(tf.concat([s_prev, a_prev, o_cur_batch], axis=-1))
        # o_reconst = self.likelihood(state_post)
        o_reconst, mu_o, std_o = self.likelihood(state_post)

        # mask out empty samples in the batch when calculating the loss
        # MSE loss
        # mse_loss = mse_with_sum(o_reconst[mask], o_cur[mask])

        # Guassian log-likelihood loss for reconstruction of observation
        gnll = mcs.g_nll(mu_o[mask], std_o[mask], o_cur[mask])

        # KL Divergence loss
        kld = mcs.kl_d(mu_post[mask], std_post[mask], mu_tran[mask], std_tran[mask])
        standard_mean = mu_post[mask] * 0.0
        standard_std = std_post[mask] * 0.0 + 1.0
        kld_regularize = mcs.kl_d(mu_post[mask], std_post[mask], standard_mean, standard_std)

        total_loss = self.kl_weight * kld + self.kl_regularize_weight * kld_regularize + gnll

        return total_loss, gnll, kld, state_post


    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32)])
    def serve(self, s_prev, a_prev, o_cur):
        ''' forward function of the AIF state model used during inference '''
        o_cur += tf.random.normal(tf.shape(o_cur), stddev=0.05)  # add noise
        state_tran, mu_tran, std_tran = self.transition(tf.concat([s_prev, a_prev], axis=-1))
        o_reconst_tran, mu_o_tran, std_o_tran = self.likelihood(state_tran)
        state_post, mu_post, std_post = self.posterior(tf.concat([s_prev, a_prev, o_cur], axis=-1))
        o_reconst_post, mu_o_post, std_o_post = self.likelihood(state_post)

        return state_tran, state_post, mu_o_tran, mu_o_post
