import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from VAE_models import Dense


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32, name='p_mean'),
                              tf.TensorSpec(shape=None, dtype=tf.float32, name='p_std'),
                              tf.TensorSpec(shape=None, dtype=tf.float32, name='q_mean'),
                              tf.TensorSpec(shape=None, dtype=tf.float32, name='q_std')])
def kl_d(mu_p, std_p, mu_q, std_q):
    ''' KL-Divergence function for 2 diagonal Gaussian distributions '''
    k = tf.cast(tf.shape(mu_p)[-1], tf.float32)
    log_var = tf.math.log(tf.reduce_prod(tf.math.square(std_q), axis=-1)/tf.reduce_prod(tf.math.square(std_p), axis=-1))
    mu_var_multip = tf.math.reduce_sum((mu_p-mu_q) / tf.math.square(std_q) * (mu_p-mu_q), axis=-1)
    trace = tf.math.reduce_sum(tf.math.square(std_p) / tf.math.square(std_q), axis=-1)
    kld = 0.5 * (log_var - k + mu_var_multip + trace)
    kld = tf.math.reduce_mean(kld)
    return kld

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def swish(x):
    ''' swish activation function '''
    return x * tf.math.sigmoid(x)


class Encoder(tf.Module):
    ''' transition / posterior model of the active inference framework '''
    def __init__(self, dim_input, dim_z, name='Encoder', activation='leaky_relu'):
        super().__init__(name=name)
        self.dim_z = dim_z  # latent dimension
        self.dense_input = Dense(dim_input, 20, name='dense_input')
        self.dense_e1 = Dense(20, 20, name='dense_1')
        self.dense_mu = Dense(20, dim_z, name='dense_mu')
        self.dense_raw_std = Dense(20, dim_z, name='dense_raw_std')
        self.mu = None
        self.std = None
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'leaky_relu':
            self.activation = tf.nn.leaky_relu
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')    
    
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, X):
        print("X in encoder", X)
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


class Likelihood(tf.Module):
    ''' likelihood model of the active inference framework '''
    def __init__(self, dim_input, dim_x, name='likelihood', activation='leaky_relu'):
        super().__init__(name=name)
        self.dim_x = dim_x
        self.dense_z_input = Dense(dim_input, 20, name='dense_z_input')
        self.dense_d1 = Dense(20, 20, name='dense_d1')
        self.dense_output = Dense(20, self.dim_x, name='dense_x_output')
        if activation == 'tanh': 
            self.activation = tf.nn.tanh
        elif activation == 'leaky_relu':
            self.activation = tf.nn.leaky_relu
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')
       
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, z):
        x_output = self.dense_z_input(z)
        x_output = self.activation(x_output)
        x_output = self.dense_d1(x_output)
        x_output = self.activation(x_output)
        x_output = self.dense_output(x_output)
        return x_output


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
        self.a_size = args.a_width
        self.kl_weight = args.vae_kl_weight
        self.transition = Encoder(self.dim_z + self.a_size, self.dim_z, name='transition')
        self.posterior = Encoder(self.dim_z + self.a_size + self.dim_obv, self.dim_z, name='posterior')
        self.likelihood = Likelihood(self.dim_z, self.dim_obv)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                                   tf.TensorSpec(shape=None, dtype=tf.float32)])
    def mse(self, x_reconst, x_true):
        sse = tf.math.reduce_sum(tf.math.square(x_reconst-x_true), axis=-1)
        return tf.math.reduce_mean(sse)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, s_prev, a_prev, o_cur):
        # add some noise to the observation
        o_cur_batch = o_cur + tf.random.normal(tf.shape(o_cur)) * 0.1
        print("s_prev in stateModel: ", s_prev)
        print("a_prev in stateModel: ", a_prev)
        _, mu_tran, std_tran = self.transition(tf.concat([s_prev, a_prev], axis=-1))
        state_post, mu_post, std_post = self.posterior(tf.concat([s_prev, a_prev, o_cur_batch], axis=-1))
        o_reconst = self.likelihood(state_post)
        # reconstruction loss (std assume to be 1, only reconst of mean)
        mse_loss = self.mse(o_reconst, o_cur)
        # KL Divergence loss
        kld = kl_d(mu_post, std_post, mu_tran, std_tran)
        total_loss = self.kl_weight * kld + mse_loss
        return total_loss, state_post
    
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32)])
    def serve(self, s_prev, a_prev, o_cur):
        state_tran, mu_tran, std_tran = self.transition(tf.concat([s_prev, a_prev], axis=-1))
        o_reconst = self.likelihood(state_tran)
        state_post, mu_post, std_post = self.posterior(tf.concat([s_prev, a_prev, o_cur], axis=-1))
        return state_tran, o_reconst, state_post


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
        self.n_pi = math.pow(self.n_actions, self.D)
        self.action_space = tf.constant([0, 2], dtype=tf.float32) #TODO change it to a general setting
        self.action = tf.constant([0], dtype=tf.float32)
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
        horizon_kld = tf.zeros((self.n_actions**self.D,))
        horizon_H = tf.zeros((self.n_actions**self.D,))

        for d in range(1, self.D+1):
            stage_kld = all_kld.pop(0)
            stage_H = all_H.pop(0)
            repeated_kld = tf.repeat(stage_kld, repeats=self.n_actions**(self.D-d))
            repeated_H = tf.repeat(stage_H, repeats=self.n_actions**(self.D-d))
            horizon_kld = horizon_kld + repeated_kld
            horizon_H = horizon_H + repeated_H
        
        horizon_efe = horizon_kld + 1/self.rho * horizon_H
        return horizon_efe, horizon_kld, horizon_H

    # @tf.function
    def __call__(self, cur_obv):
        if self.true_state is None:
            self.true_state =  self.stateModel.posterior.mu + self.stateModel.posterior.std * tf.random.normal(tf.shape(self.stateModel.posterior.mu))
            self.true_state = tf.math.reduce_mean(self.true_state, axis=0, keepdims=True)
        
        # take current observation, previous action and previous true state to calculate
        # current true state using the posterior model
        a_prev = tf.expand_dims(self.action, axis=0)
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
        # plan for D times of K consecutive steps with repeated actions lookahead
        for d in range(self.D):
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
                action = self.action_space[idx_b % self.n_actions]
                if action == 1:
                    action = 2
                rolling_states = self.stage_states[idx_b]
                # plan for K steps in a roll (repeat an action K times given a policy)
                for t in range(self.K):
                    action_t = action * tf.ones((self.N, 1))
                    # use only the transition model to rollout the states
                    rolling_states, _, _ = self.stateModel.transition(tf.concat([rolling_states, action_t], axis=-1))
                    # use the likelihood model to map states to observations
                    obvSamples = self.stateModel.likelihood(rolling_states)
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
                    step_H[idx_b, t] = H
                
                # update self.stage_states, which must be a tf.Variable since tensors are immutable
                self.stage_states[idx_b, :, :].assign(rolling_states[:, :])
                
            # gather KL-D value and entropy for each branch
            all_kld.append(np.sum(step_kld, axis=1))
            all_H.append(np.sum(step_H, axis=1))
        
        # calculate EFE values for the root policies
        # efe_root = self.evaluate_stage_efe_recursive(all_kld, all_H)
        efe_root, _, _ = self.evaluate_stage_efe(all_kld, all_H)

        # sample a policy given probabilities of each policy
        # prob_pi = tf.math.softmax(-self.gamma * efe_root)
        # self.action =  np.random.choice([0, 2], p=prob_pi.numpy())
        
        # use the policy with the lowest efe value
        self.action = tf.argmin(efe_root)
        self.action = tf.cast(tf.reshape(self.action, [-1]), dtype=tf.float32)
        
        if self.action < 4:
            self.action = tf.constant([0], dtype=tf.float32)
        else:
            self.action = tf.constant([2], dtype=tf.float32)
        
        ### the block below is previous code for 2 policies
        # # account for the omission of actoion 1 for MountainCar
        # if self.action == 1:
        #     self.action = tf.constant([2], dtype=tf.float32)
        
        print("self.action: ", self.action.numpy())
        return self.action