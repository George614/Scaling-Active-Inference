import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from VAE_models import Dense

EPSILON = 1e-5

@tf.function
def kl_d(mu_p, std_p, mu_q, std_q, keep_batch=False):
    ''' KL-Divergence function for 2 diagonal Gaussian distributions '''
    # reference: https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    k = tf.cast(tf.shape(mu_p)[-1], tf.float32)
    log_var = tf.math.log(tf.reduce_prod(tf.math.square(std_q), axis=-1)/tf.reduce_prod(tf.math.square(std_p), axis=-1) + EPSILON)
    mu_var_multip = tf.math.reduce_sum((mu_p-mu_q) / (tf.math.square(std_q) + EPSILON) * (mu_p-mu_q), axis=-1)
    trace = tf.math.reduce_sum(tf.math.square(std_p) / (tf.math.square(std_q) + EPSILON), axis=-1)
    kld = 0.5 * (log_var - k + mu_var_multip + trace)
    if not keep_batch:
        kld = tf.math.reduce_mean(kld)
    return kld


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                               tf.TensorSpec(shape=None, dtype=tf.float32)])
def mse_with_sum(x_reconst, x_true):
    ''' Mean Squared Error with sum over last dimension '''
    sse = tf.math.reduce_sum(tf.math.square(x_reconst - x_true), axis=-1)
    return tf.math.reduce_mean(sse)


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                               tf.TensorSpec(shape=None, dtype=tf.float32)])
def mse(x_reconst, x_true):
    ''' Mean Squared Error '''
    se = tf.math.square(x_reconst - x_true)
    return tf.math.reduce_mean(se)


@tf.function
def g_nll(mu, std, x_true, keep_batch=False):
    ''' Gaussian Negative Log Likelihood loss function '''
    nll = 0.5 * tf.math.log(2 * math.pi * tf.math.square(std)) + tf.math.square(x_true - mu) / (2 * tf.math.square(std) + EPSILON)
    nll = tf.reduce_sum(nll, axis=-1)
    if not keep_batch:
        nll = tf.reduce_mean(nll)
    return nll


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def swish(x):
    ''' swish activation function '''
    return x * tf.math.sigmoid(x)


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32, name='mean'),
                              tf.TensorSpec(shape=None, dtype=tf.float32, name='std')])
def sample(mu, std):
    # reparameterization trick
    z_sample = mu + std * tf.random.normal(tf.shape(std))
    return z_sample


@tf.function
def sample_k_times(k, mu, std):
    multiples = tf.constant((1, k, 1), dtype=tf.int32)
    # shape of mu_, std_ and z_sample: (batch x n_samples x vec_dim)
    mu_ = tf.tile(tf.expand_dims(mu, axis=1), multiples)
    std_ = tf.tile(tf.expand_dims(std, axis=1), multiples)
    # reparameterization trick
    z_sample = mu_ + std_ * tf.random.normal(tf.shape(std_))
    return z_sample


class Encoder(tf.Module):
    ''' transition / posterior model of the active inference framework '''
    def __init__(self, dim_input, dim_z, n_samples=1, name='Encoder', activation='relu6'):
        super().__init__(name=name)
        self.dim_z = dim_z  # latent dimension
        self.N = n_samples
        self.dense_input = Dense(dim_input, 64, name='dense_input')
        self.dense_e1 = Dense(64, 64, name='dense_1')
        self.dense_mu = Dense(64, dim_z, name='dense_mu')
        self.dense_raw_std = Dense(64, dim_z, name='dense_raw_std')
        self.mu = None
        self.std = None
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu6':
            self.activation = tf.nn.relu6
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')
    
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, X):
        z = self.dense_input(X)
        z = self.activation(z)
        z = self.dense_e1(z)
        z = self.activation(z)
        mu = self.dense_mu(z)
        raw_std = self.dense_raw_std(z)
        # softplus is supposed to avoid numerical overflow
        std = tf.clip_by_value(tf.math.softplus(raw_std), 0.01, 10.0)
        if self.mu is None:
            batch_size = tf.shape(X)[0]
            self.mu = tf.Variable(tf.zeros((batch_size, self.dim_z)), trainable=False)
            self.std = tf.Variable(tf.zeros((batch_size, self.dim_z)), trainable=False)
        else:
            self.mu.assign(mu)
            self.std.assign(std)
        z_sample = self.sample(mu, std)

        return z_sample, mu, std

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32, name='mean'),
                                  tf.TensorSpec(shape=None, dtype=tf.float32, name='std')])
    def sample(self, mu, std):
        batch_size = tf.shape(mu)[0]
        # reparameterization trick
        z_sample = mu + std * tf.random.normal((batch_size, self.dim_z))
        return z_sample

    @tf.function
    def sample_k_times(self):
        multiples = tf.constant((1, self.N, 1), dtype=tf.int32)
        # shape of mu_, std_ and z_sample: (batch x n_samples x vec_dim)
        mu_ = tf.tile(tf.expand_dims(self.mu, axis=1), multiples)
        std_ = tf.tile(tf.expand_dims(self.std, axis=1), multiples)
        # reparameterization trick
        z_sample = mu_ + std_ * tf.random.normal(tf.shape(std_))
        return z_sample


class FlexibleEncoder(tf.Module):
    ''' Generic Gaussian encoder model based on Dense layer. Output mean and std as well
    as samples from the learned distribution '''
    def __init__(self, layer_dims, n_samples=1, name='Encoder', activation='relu6'):
        super().__init__(name=name)
        self.dim_z = layer_dims[-1]
        self.N = n_samples
        self.layers =[]
        for i in range(len(layer_dims)-1):
            self.layers.append(Dense(layer_dims[i], layer_dims[i+1]))
        # add another group of neurons for mu/std in the last layer
        self.layers.append(Dense(layer_dims[-2], self.dim_z))
        self.mu = None
        self.std = None
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu6':
            self.activation = tf.nn.relu6
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')
    
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, x):
        for layer in self.layers[:-2]:
            x = layer(x)
            x = self.activation(x)
        mu = self.layers[-2](x)
        raw_std = self.layers[-1](x)
        # softplus is supposed to avoid numerical overflow
        std = tf.clip_by_value(tf.math.softplus(raw_std), 0.01, 10.0)
        if self.mu is None:
            batch_size = tf.shape(x)[0]
            self.mu = tf.Variable(tf.zeros((batch_size, self.dim_z)), trainable=False)
            self.std = tf.Variable(tf.zeros((batch_size, self.dim_z)), trainable=False)
        else:
            self.mu.assign(mu)
            self.std.assign(std)
        z_sample = sample(mu, std)

        return z_sample, mu, std


class FlexibleMLP(tf.Module):
    ''' Simple multi-layer perceptron model taking layer parameters as input '''
    def __init__(self, layer_dims, name='MLP', activation='relu6', trainable=True): # relu6, or tanh
        super().__init__(name=name)
        self.layers =[]
        for i in range(len(layer_dims)-1):
            self.layers.append(Dense(layer_dims[i], layer_dims[i+1], trainable=trainable))
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu6':
            self.activation = tf.nn.relu6
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')
    
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)  # linear activation for the last layer
        return x


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
        self.kl_weight = tf.Variable(args.vae_kl_weight, trainable=False)
        self.encoder = FlexibleEncoder((self.dim_obv, 64, 64, self.dim_z), name='Encoder')
        self.decoder = FlexibleEncoder((self.dim_z, 64, 64, self.dim_obv), name='Decoder')
        self.transition = FlexibleEncoder((self.dim_z + self.a_size, 64, 64, self.dim_z), name='Transition')
        # self.encoder = Encoder(self.dim_obv, self.dim_z, name='Encoder')
        # self.decoder = Encoder(self.dim_z, self.dim_obv, name='Decoder')
        # self.transition = Encoder(self.dim_z + self.a_size, self.dim_z, name='Transition')
        self.EFEnet = FlexibleMLP((self.dim_z, 128, 64, self.a_size), name='EFEnet')
        self.EFEnet_target = FlexibleMLP((self.dim_z, 128, 64, self.a_size), name='EFEnet_target', trainable=False)
        self.update_target()
        self.priorModel = priorModel
        self.obv_t = None
        self.a_t = None
        self.epsilon = tf.Variable(1.0, trainable=False)  # epsilon greedy parameter


    def act(self, obv_t):
        # action selection using o_t and EFE network
        states, state_mu, state_std = self.encoder(obv_t)
        efe_t = self.EFEnet(states)
        if tf.random.uniform(shape=()) > self.epsilon:
            action = tf.argmax(efe_t, axis=-1)
        else:
            action = tf.random.uniform(shape=(), maxval=self.a_size, dtype=tf.int32)

        return action


    def update_target(self):
        for target_var, var in zip(self.EFEnet_target.variables, self.EFEnet.variables):
            target_var.assign(var)


    def train_step(self, obv_t, obv_next, action, done):
        batch_size = len(action)
        a_t = np.zeros((batch_size, self.a_size), dtype=np.float32)
        a_t[np.arange(batch_size), action] = 1  # shape of (batch x a_space_size) one-hot
        with tf.GradientTape(persistent=True) as tape:
            # run s_t and a_t through the PPL model
            states, state_mu, state_std = self.encoder(obv_t)
            efe_t = self.EFEnet(states)  # in batch, output shape (batch x a_space_size)
            states_next_tran, s_next_tran_mu, s_next_tran_std = self.transition(tf.concat([states, a_t], axis=-1))
            o_next_hat, o_next_mu, o_next_std = self.decoder(states_next_tran)
            o_next_prior, o_prior_mu, o_prior_std = self.priorModel(obv_t)
            states_next_enc, s_next_enc_mu, s_next_enc_std = self.encoder(obv_next)

            # difference between preferred future and predicted future
            # R_ti = kl_d(o_prior_mu, o_prior_std, o_next_mu, o_next_std)

            # difference between preferred future and actual future, i.e. instrumental term
            R_ti = -1.0 * g_nll(o_prior_mu, o_prior_std, obv_next, keep_batch=True)
            
            # negative almost KL-D between state distribution from transition model and from
            # approximate posterior model (encoder), i.e. epistemic value. Assumption is made.
            R_te = -1.0 * kl_d(s_next_tran_mu, s_next_tran_std, s_next_enc_mu, s_next_enc_std, keep_batch=True)
            
            # the nagative EFE value, i.e. the reward. Note the sign here
            R_t = R_ti + R_te

            # model reconstruction loss
            loss_model = g_nll(o_next_mu, o_next_std, obv_next)
            
            # take the old EFE values given action indices
            idx_0 = tf.range(batch_size)
            a_idx = tf.stack([idx_0, action], axis=-1)
            efe_old = tf.gather_nd(efe_t, a_idx)
            
            with tape.stop_recording():
                # EFE values for next state, s_t+1 is from transition model instead of encoder
                efe_target = self.EFEnet_target(states_next_tran)
                idx_a_next = tf.math.argmax(efe_target, axis=-1, output_type=tf.dtypes.int32)
                onehot_a_next = np.zeros((batch_size, self.a_size), dtype=np.float32)
                onehot_a_next[np.arange(batch_size), idx_a_next.numpy()] = 1
                # take the new EFE values
                a_next_idx = tf.stack([idx_0, idx_a_next], axis=-1)
                efe_new = tf.gather_nd(efe_target, a_next_idx)

            # TD loss
            done = tf.convert_to_tensor(done, dtype=tf.float32)
            loss_efe = mse(efe_old, (R_t + efe_new * (1 - done))) #TODO:chnage to huber loss

            # calculate gradient w.r.t model reconstruction and TD respectively
            grads_model = tape.gradient(loss_model, self.trainable_variables)
            grads_efe = tape.gradient(loss_efe, self.trainable_variables)

        return grads_efe, grads_model, loss_efe, loss_efe, R_ti, R_te, efe_t, efe_target


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
        self.transition = Encoder(self.dim_z + self.a_size, self.dim_z, n_samples=self.n_samples, name='transition')
        self.posterior = Encoder(self.dim_z + self.a_size + self.dim_obv, self.dim_z, n_samples=self.n_samples, name='posterior')
        self.likelihood = Encoder(self.dim_z, self.dim_obv, name='likelihood')


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
        
        # tf.print("---------posterior mu--------")
        # tf.print(mu_post[mask])
        # tf.print("---------transition mu--------")
        # tf.print(mu_tran[mask])
        # tf.print("--------posterior std--------")
        # tf.print(std_post[mask])
        # tf.print("--------transition std--------")
        # tf.print(std_tran[mask])
        
        # Guassian log-likelihood loss for reconstruction of observation
        gnll = g_nll(mu_o[mask], std_o[mask], o_cur[mask])

        # KL Divergence loss
        kld = kl_d(mu_post[mask], std_post[mask], mu_tran[mask], std_tran[mask])
        standard_mean = mu_post[mask] * 0.0
        standard_std = std_post[mask] * 0.0 + 1.0
        kld_regularize = kl_d(mu_post[mask], std_post[mask], standard_mean, standard_std)

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


class Planner(tf.Module):
    ''' Planner class which encapsulates state rollout and policy selection '''
    def __init__(self, stateModel, args, s_mu_prefer, s_std_prefer=None):
        super().__init__(name='Planner')
        self.stateModel = stateModel
        self.dim_z = args.z_size
        self.dim_obv = args.o_size
        self.K = args.planner_lookahead  # lookahead
        self.D = args.planner_plan_depth  # recursive planning depth
        self.N = args.planner_n_samples  # number of samples per policy
        self.rho = args.planner_rho  # weight term on expected ambiguity
        self.gamma = args.planner_gamma  # temperature factor applied to the EFE before calculate belief about policies
        self.n_actions = args.n_actions  # for discrete action space, specify the number of possible actions
        self.n_pi = self.n_actions ** self.D
        self.action_space = tf.constant([0, 1], dtype=tf.float32) #TODO change it to a general setting
        self.action = 0
        # whether or not use the effects of switching to another policy
        self.include_extened_efe = args.planner_full_efe
        # self.true_state is the single true state of the agent/model
        self.true_state = None
        # self.stage_states contains all N samples of states used by the transition model
        # at the start / end of each stage of planning
        self.stage_states = None
        # empirical preferred state (prior preference)
        self.s_mu_prefer = s_mu_prefer
        if s_std_prefer is not None:
            self.s_std_prefer = s_std_prefer
        else:
            self.s_std_prefer = tf.ones((self.dim_z,))


    # @tf.function
    def evaluate_stage_efe_recursive(self, all_kld, all_H):
        '''
        This is the recursive version of calculating EFE for policies. The number policies
        equals to the number of possible actions or policies at the first/root stage.
        In other words, only the EFE values for the first stage are returned.
        All policies in future stages (expansion in a tree search) are accumulated
        recursively into the EFE at the root stage. This implementation uses an iterative
        approach instead of a recursive approach.
        '''
        reduced_efe = None
        for d in reversed(range(self.D)):
            stage_kld = all_kld.pop()
            stage_H = all_H.pop()
            efe_stage = stage_kld + 1/self.rho * stage_H
            if reduced_efe is not None:
                efe_stage += reduced_efe
            if d > 0:
                efe_group = [efe_stage[i:i+self.n_actions] for i in range(0, self.n_actions ** (d+1), self.n_actions)]
                prob_pi_group = [tf.nn.softmax(-self.gamma * efe_branch) for efe_branch in efe_group]
                efe_stage_weighted = [prob_pi * efe_stage for prob_pi, efe_stage in zip(prob_pi_group, efe_group)]
                efe_stage_weighted = tf.concat(efe_stage_weighted, axis=0)
                segments = tf.repeat(tf.range(self.n_actions ** d), self.n_actions)
                reduced_efe = tf.math.segment_sum(efe_stage_weighted, segments)
        
        return efe_stage


    def evaluate_stage_efe(self, all_kld, all_H):
        '''
        This is the non-recursive version of calculating EFE for policies which cover
        a complete horizon H = K x D. The number of policies evaluated equals to the
        number of possible actions to the power of D.
        '''
        horizon_kld = tf.zeros((len(all_kld[-1]),))
        horizon_H = tf.zeros((len(all_kld[-1]),))

        for d in range(1, self.D+1):
            stage_kld = all_kld.pop(0)
            stage_H = all_H.pop(0)
            repeated_kld = tf.repeat(stage_kld, repeats=self.n_actions**(self.D-d))
            repeated_H = tf.repeat(stage_H, repeats=self.n_actions**(self.D-d))
            horizon_kld = horizon_kld + repeated_kld
            horizon_H = horizon_H + repeated_H
        
        horizon_efe = horizon_kld + 1/self.rho * horizon_H
        
        return horizon_efe, horizon_kld, horizon_H


    def evaluate_2_term_efe(self, all_kld, all_H):
        '''
        This funciton calculate the EFE values for self.n_action ** self.D policies
        using 2 terms. The first term is the EFE value for a planning horizon H=KxD.
        The second term acounts for the effects of switching to another policy after
        the first H steps.
        '''
        # EFE values for a horzion of H = KxD, self.n_pi policies in total
        horizon_efe, horizon_kld, horizon_H = self.evaluate_stage_efe(all_kld[:self.D], all_H[:self.D])
        # EFE values for policies after H steps, i.e. the effect of switching to a new policy
        extended_efe, extended_kld, extended_H = self.evaluate_stage_efe(all_kld[self.D:], all_H[self.D:])
        
        efe_group = [extended_efe[i:i+self.n_pi] for i in range(0, self.n_actions ** (2*self.D), self.n_pi)]
        prob_pi_group = [tf.nn.softmax(-self.gamma * efe_branch) for efe_branch in efe_group]
        efe_extended_weighted = [prob_pi * efe_extended for prob_pi, efe_extended in zip(prob_pi_group, efe_group)]
        efe_extended_weighted = tf.concat(efe_extended_weighted, axis=0)
        segments = tf.repeat(tf.range(self.n_pi), self.n_pi)
        switch_efe = tf.math.segment_sum(efe_extended_weighted, segments)
        total_efe = horizon_efe + switch_efe

        return total_efe


    # @tf.function
    def __call__(self, cur_obv):
        if self.true_state is None:
            # self.true_state =  self.stateModel.posterior.mu + self.stateModel.posterior.std * tf.random.normal(tf.shape(self.stateModel.posterior.mu))
            # self.true_state = tf.math.reduce_mean(self.true_state, axis=0, keepdims=True)
            self.true_state = tf.zeros((1, self.dim_z))
        
        # take current observation, previous action and previous true state to calculate
        # current true state using the posterior model
        a_prev = np.zeros((1, self.n_actions), dtype=np.float32)
        a_prev[:, self.action] = 1
        a_prev = tf.convert_to_tensor(a_prev)
        cur_obv += tf.random.normal(tf.shape(cur_obv), stddev=0.05) # add noise
        cur_obv = tf.expand_dims(cur_obv, axis=0)
        multiples = tf.constant((self.N, 1), dtype=tf.int32)
        a_prev = tf.tile(a_prev, multiples)
        cur_obv = tf.tile(cur_obv, multiples)
        self.true_state = tf.tile(self.true_state, multiples)
        self.true_state, _, _ = self.stateModel.posterior(tf.concat([self.true_state, a_prev, cur_obv], axis=-1))
        self.true_state = tf.math.reduce_mean(self.true_state, axis=0, keepdims=True)

        # KL-D and entropy values for each policy at each planning stage (D stages in total)
        all_kld = []
        all_H = []

        planning_depth = self.D * 2 if self.include_extened_efe else self.D
        # plan for 2D / D times of K consecutive steps with repeated actions lookahead
        for d in range(planning_depth):
            # each branch is split into N_actions branchs at each stage
            if d == 0:
                # make N_samples copies of the true state
                multiples = tf.constant((self.n_actions, self.N, 1), dtype=tf.int32)
                self.stage_states = tf.Variable(tf.tile(tf.expand_dims(self.true_state, axis=0), multiples))
            else:
                multiples = tf.constant([self.n_actions, 1, 1], dtype=tf.int32)
                self.stage_states = tf.Variable(tf.tile(self.stage_states, multiples))
            
            # KL-D and entropy values for each policy at each time step within K steps
            step_kld = np.zeros((self.n_actions ** (d+1), self.K), dtype=np.float32)
            step_H = np.zeros((self.n_actions ** (d+1), self.K), dtype=np.float32)
            
            # rollout for each branch, only for finite discretized actions
            for idx_b in range(self.n_actions ** (d+1)):
                action = idx_b % self.n_actions
                # convert action to one-hot encoded vector
                action_onehot = np.zeros((1, self.n_actions))
                action_onehot[:, action] = 1
                multiples = tf.constant((self.N, 1), dtype=tf.int32)
                action_t = tf.tile(action_onehot, multiples)

                rolling_states = self.stage_states[idx_b]

                # plan for K steps in a roll (repeat an action K times given a policy)
                for t in range(self.K):
                    # use only the transition model to rollout the states
                    rolling_states, _, _ = self.stateModel.transition(tf.concat([rolling_states, action_t], axis=-1))
                    # use the likelihood model to map states to observations
                    obvSamples, o_mu_batch, o_std_batch = self.stateModel.likelihood(rolling_states)
                    # gather batch statistics
                    states_mean = tf.math.reduce_mean(rolling_states, axis=0)
                    states_std = tf.math.reduce_std(rolling_states, axis=0)
                    obv_std = tf.math.reduce_std(obvSamples, axis=0)
                    # KL Divergence between expected states and preferred states
                    step_kld_point = kl_d(states_mean, states_std, self.s_mu_prefer, self.s_std_prefer)
                    # print("step_kld_point shape: ", step_kld_point.shape)
                    step_kld[idx_b, t] = step_kld_point
                    # entropy on the expected observations
                    n = tf.shape(obv_std)[-1]
                    H = float(n/2) + float(n/2) * tf.math.log(2*math.pi * tf.math.pow(tf.math.reduce_prod(tf.math.square(obv_std), axis=-1), float(1/n)))
                    step_H[idx_b, t] = tf.reduce_mean(H)
                
                # update self.stage_states, which must be a tf.Variable since tensors are immutable
                self.stage_states[idx_b, :, :].assign(rolling_states[:, :])
                
            # gather KL-D value and entropy for each branch
            all_kld.append(np.sum(step_kld, axis=1))
            all_H.append(np.sum(step_H, axis=1))
        
        # calculate EFE values for the root policies
        # efe_root = self.evaluate_stage_efe_recursive(all_kld, all_H)
        if self.include_extened_efe:
            efe_root = self.evaluate_2_term_efe(all_kld, all_H)
        else:
            efe_root, _, _ = self.evaluate_stage_efe(all_kld, all_H)
        
        # sample a policy given probabilities of each policy
        # prob_pi = tf.math.softmax(-self.gamma * efe_root)
        # self.action =  np.random.choice([0, 1], p=prob_pi.numpy())
        
        # use the policy with the lowest efe value
        self.action = tf.argmin(efe_root)
        self.action = tf.cast(tf.reshape(self.action, [-1]), dtype=tf.float32)
        
        if self.action < 4:
            self.action = 0
        else:
            self.action = 1
        
        print("self.action: ", self.action)
        return self.action