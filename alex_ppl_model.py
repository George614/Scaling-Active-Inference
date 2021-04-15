import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../utils/')
from utils import softmax, sample_gaussian, sample_gaussian_with_logvar, \
                  g_nll_from_logvar, kl_div_loss, g_nll, kl_d, mse, load_object, \
                  entropy_gaussian_from_logvar, huber
from prob_mlp import ProbMLP
from config import Config

class PPLModel:
    def __init__(self, prior, args):
        self.prior = prior
        self.args = args

        self.dim_z = int(args.getArg("dim_z"))  # latent/state space size
        self.dim_o = int(args.getArg("dim_o"))  # observation size
        self.dim_a = int(args.getArg("dim_a"))  # action size
        self.layer_norm = (args.getArg("layer_norm").strip().lower() == 'true')
        self.all_layer_norm = False
        if args.hasArg("all_layer_norm") is True:
            self.all_layer_norm = (args.getArg("all_layer_norm").strip().lower() == 'true')
        self.l2_reg = float(args.getArg("l2_reg"))
        self.act_fx = args.getArg("act_fx")
        self.efe_act_fx = args.getArg("efe_act_fx")
        self.epist_type = args.getArg("epistemic_term")
        self.instru_type = args.getArg("instrumental_term")
        self.efe_loss = args.getArg("efe_loss")
        self.gamma_d = float(args.getArg("gamma_d"))
        self.use_sum_q = (args.getArg("use_sum_q").strip().lower() == 'true')
        self.is_stateful = (args.getArg("is_stateful").strip().lower() == 'true')

        hid_dims = [128, 128]

        # set up dimensions for each sub-model
        trans_dims = [(self.dim_z + self.dim_a)]
        trans_dims = trans_dims + hid_dims
        trans_dims.append(self.dim_z)

        enc_dims = [self.dim_o]
        enc_dims = enc_dims + hid_dims
        enc_dims.append(self.dim_z)

        dec_dims = [self.dim_z]
        dec_dims = dec_dims + hid_dims
        dec_dims.append(self.dim_o)

        efe_dims = [self.dim_z]
        efe_dims = efe_dims + hid_dims
        efe_dims.append(self.dim_a)

        act_fun = self.act_fx #"relu"
        efe_act_fun = self.efe_act_fx #"relu6"
        wght_sd = 0.025
        init_type = "alex_uniform"
        sigma_fun = "softplus"
        self.z_state = None

        # transition model
        self.transition = ProbMLP(name="Trans",z_dims=trans_dims, act_fun=act_fun, wght_sd=wght_sd,
                                  init_type=init_type, model_variance=True, sigma_fun=sigma_fun,
                                  use_layer_norm=self.all_layer_norm)

        # encoder
        self.encoder = ProbMLP(name="Enc",z_dims=enc_dims, act_fun=act_fun, wght_sd=wght_sd,
                               init_type=init_type, model_variance=True, sigma_fun=sigma_fun,
                               use_layer_norm=self.all_layer_norm)
        # decoder
        self.decoder = ProbMLP(name="Dec",z_dims=dec_dims, act_fun=act_fun, wght_sd=wght_sd,
                               init_type=init_type, model_variance=True, sigma_fun=sigma_fun,
                               use_layer_norm=self.all_layer_norm)
        # EFE model
        self.efe = ProbMLP(name="EFE",z_dims=efe_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                           init_type=init_type,use_layer_norm=self.layer_norm)
        self.efe_target = ProbMLP(name="EFE_targ",z_dims=efe_dims, act_fun=efe_act_fun, wght_sd=wght_sd,
                                  init_type=init_type,use_layer_norm=self.layer_norm)


        # clump all parameter variables of sub-models/modules into one shared pointer list
        self.param_var = []
        self.param_var = self.param_var + self.transition.extract_params()
        self.param_var = self.param_var + self.encoder.extract_params()
        self.param_var = self.param_var + self.decoder.extract_params()
        self.param_var = self.param_var + self.efe.extract_params()

        self.epsilon = tf.Variable(1.0, trainable=False)  # epsilon greedy parameter
        self.gamma = tf.Variable(1.0, trainable=False)  # gamma weighting factor for balance KL-D on transition vs unit Gaussian
        self.rho = tf.Variable(1.0, trainable=False)  # weight term on the epistemic value
        self.tau = -1 # if set to 0, then no Polyak averaging is used for target network
        self.update_target()

    def update_target(self):
        self.efe_target.set_weights(self.efe, tau=self.tau)

    def act(self, o_t):
        if self.is_stateful:
            if self.z_state is None:
                z_mu_enc, z_sigma_enc, z_logvar_enc = self.encoder.predict(o_t)
                z_t = sample_gaussian(n_s=o_t.shape[0], mu=z_mu_enc, sig=z_sigma_enc)
                self.z_state = z_t

        if tf.random.uniform(shape=()) > self.epsilon:
            # action selection using o_t and EFE network
            if self.is_stateful is False:
                z_mu_enc, z_sigma_enc, z_logvar_enc = self.encoder.predict(o_t)
                z_t = sample_gaussian(n_s=o_t.shape[0], mu=z_mu_enc, sig=z_sigma_enc)
            else:
                z_t = self.z_state
            # run EFE model given state at time t
            efe_t, _, _ = self.efe.predict(z_t)
            action = tf.argmax(efe_t, axis=-1, output_type=tf.int32)
        else:
            action = tf.random.uniform(shape=(), maxval=self.dim_a, dtype=tf.int32)

        if self.is_stateful:
            #print(action)
            a_t = tf.expand_dims(tf.one_hot(action.numpy().squeeze(), depth=self.dim_a),0)
            s_next_tran_mu, s_next_tran_std, _ = self.transition.predict(tf.concat([self.z_state, a_t], axis=-1))
            z_tp1 = sample_gaussian(n_s=o_t.shape[0], mu=s_next_tran_mu, sig=s_next_tran_std)
            self.z_state = z_tp1

        return action

    def clear_state(self):
        self.z_state = None

    def train_step(self, obv_t, obv_next, action, done, weights=None, reward=None):
        with tf.GradientTape(persistent=True) as tape:
            ### run s_t and a_t through the PPL model ###
            state_mu, state_std, _ = self.encoder.predict(obv_t)
            states = sample_gaussian(n_s=obv_t.shape[0], mu=state_mu, sig=state_std)

            efe_t, _, _ = self.efe.predict(states) # in batch, output shape (batch x a_space_size)

            s_next_tran_mu, s_next_tran_std, _ = self.transition.predict(tf.concat([states, action], axis=-1))
            states_next_tran = sample_gaussian(n_s=obv_next.shape[0], mu=s_next_tran_mu, sig=s_next_tran_std)

            o_next_mu, o_next_std, _ = self.decoder.predict(states_next_tran)
            o_next_hat = sample_gaussian(n_s=obv_next.shape[0], mu=o_next_mu, sig=o_next_std)

            s_next_enc_mu, s_next_enc_std, _ = self.encoder.predict(obv_next)
            states_next_enc = sample_gaussian(n_s=obv_next.shape[0], mu=s_next_enc_mu, sig=s_next_enc_std)

            with tape.stop_recording():
                ### instrumental term ###
                R_ti = None
                if self.instru_type == "prior_actual":
                    o_prior_mu, o_prior_std, _ = self.prior.predict(obv_t)
                    #o_next_prior = sample_gaussian(n_s=obv_next.shape[0], mu=o_prior_mu, sig=o_prior_std)
                    # difference between preferred future and actual future, i.e. instrumental term
                    R_ti = -1.0 * g_nll(obv_next, o_prior_mu, o_prior_std * o_prior_std, keep_batch=True)
                elif self.instru_type == "prior_pred":
                    o_prior_mu, o_prior_std, _ = self.prior.predict(obv_t)
                    #o_next_prior = sample_gaussian(n_s=obv_next.shape[0], mu=o_prior_mu, sig=o_prior_std)
                    R_ti = -1.0 * g_nll(o_next_hat, o_prior_mu, o_prior_std * o_prior_std, keep_batch=True)
                else:
                    R_ti = -reward
                    if len(R_ti.shape) < 2:
                        R_ti = tf.expand_dims(R_ti,axis=1)
                ### epistemic term ###
                R_te = None
                if self.epist_type == "kl":
                    R_te = -1.0 * kl_d(s_next_tran_mu, s_next_tran_std * s_next_tran_std, tf.math.log(s_next_tran_std),
                                       s_next_enc_mu, s_next_enc_std * s_next_enc_std, tf.math.log(s_next_enc_std), keep_batch=True)
                    #R_te = kl_d(s_next_tran_mu, s_next_tran_std * s_next_tran_std, tf.math.log(s_next_tran_std),
                    #                   s_next_enc_mu, s_next_enc_std * s_next_enc_std, tf.math.log(s_next_enc_std), keep_batch=True)
                elif self.epist_type == "log_diff":
                    z_tp1 = states_next_tran
                    R_te = g_nll(z_tp1, s_next_tran_mu, s_next_tran_std * s_next_tran_std, keep_batch=True) - \
                           g_nll(z_tp1, s_next_enc_mu, s_next_enc_std * s_next_enc_std, keep_batch=True)
                else: # entropy_diff
                    entr = entropy_gaussian_from_logvar(tf.math.log(s_next_tran_std * s_next_tran_std)) + \
                           entropy_gaussian_from_logvar(tf.math.log(s_next_enc_std * s_next_enc_std))
                    R_te = -1 * tf.reduce_sum(entr, axis=1) # collapse to get entropy per sample in mini-batch

                R_te = tf.clip_by_value(R_te, -50.0, 50.0)

                # the nagative EFE value, i.e. the reward. Note the sign here
                R_t = R_ti + self.rho * R_te
                #R_t = R_ti + R_te

            ## model reconstruction loss ##
            loss_reconst = g_nll(obv_next, o_next_mu, o_next_std * o_next_std)
            # regularization for weights
            loss_l2 = 0.0
            if self.l2_reg > 0.0:
                loss_l2 = (tf.add_n([tf.nn.l2_loss(var) for var in self.param_var if 'W' in var.name])) * self.l2_reg
            # latent penalty term
            loss_latent = kl_d(s_next_enc_mu, s_next_enc_std * s_next_enc_std, tf.math.log(s_next_enc_std),
                               s_next_tran_mu, s_next_tran_std * s_next_tran_std, tf.math.log(s_next_tran_std))
            # latent regularization term
            unit_Gaussian_mu = tf.zeros(tf.shape(s_next_enc_mu))
            unit_Gaussian_std = tf.ones(tf.shape(s_next_enc_std))
            loss_latent_reg = kl_d(s_next_enc_mu, s_next_enc_std * s_next_enc_std, tf.math.log(s_next_enc_std),
                                   unit_Gaussian_mu, unit_Gaussian_std * unit_Gaussian_std, tf.math.log(unit_Gaussian_std))

            loss_model = loss_reconst + self.gamma * loss_latent + (1 - self.gamma) * loss_latent_reg + loss_l2

            # take the old EFE values given action indices
            done = tf.expand_dims(tf.cast(done, dtype=tf.float32),axis=1)
            if self.use_sum_q is True:
                efe_old = tf.math.reduce_sum(efe_t * action, axis=-1)
                with tape.stop_recording():
                    # EFE values for next state, s_t+1 is from transition model instead of encoder
                    efe_target, _, _ = self.efe_target.predict(states_next_tran)
                    idx_a_next = tf.math.argmax(efe_target, axis=-1, output_type=tf.dtypes.int32)
                    onehot_a_next = tf.one_hot(idx_a_next, depth=self.dim_a)
                    # take the new EFE values
                    efe_new = tf.math.reduce_sum(efe_target * onehot_a_next, axis=-1)
                    y_j = R_t + (efe_new * self.gamma_d) * (1 - done)
            else:
                efe_old = efe_t
                with tape.stop_recording():
                    # EFE values for next state, s_t+1 is from transition model instead of encoder
                    efe_target, _, _ = self.efe_target.predict(states_next_tran)
                    efe_new = tf.expand_dims(tf.reduce_max(efe_target, axis=1),axis=1) # max EFE values at t+1
                    y_j = R_t + (efe_new * self.gamma_d) * (1.0 - done)
                    y_j = (action * y_j) + ( efe_old * (1.0 - action) )

            ## Temporal Difference loss ##
            if weights is not None:
                loss_efe_batch = None
                if self.efe_loss == "huber":
                    loss_efe_batch = huber(x_true=y_j, x_pred=efe_old, keep_batch=True)
                else:
                    loss_efe_batch = mse(x_true=y_j, x_pred=efe_old, keep_batch=True)
                loss_efe_batch = tf.squeeze(loss_efe_batch)
                loss_efe_batch *= weights
                priorities = loss_efe_batch + 1e-5
                loss_efe = tf.math.reduce_mean(loss_efe_batch)
            else:
                if self.efe_loss == "huber":
                    loss_efe = huber(x_true=y_j, x_pred=efe_old)
                else:
                    loss_efe = mse(x_true=y_j, x_pred=efe_old)

        # calculate gradient w.r.t model reconstruction and TD respectively
        grads_model = tape.gradient(loss_model, self.param_var)
        grads_efe = tape.gradient(loss_efe, self.param_var)

        if weights is not None:
            return grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target, priorities

        return grads_efe, grads_model, loss_efe, loss_model, loss_l2, R_ti, R_te, efe_t, efe_target
