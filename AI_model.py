import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from VAE_models import FlexibleDense, Sampler


# @tf.function
def kl_d(mu_p, std_p, mu_q, std_q):
    ''' KL-Divergence function for 2 diagonal Gaussian distributions '''
    k = tf.cast(tf.shape(mu_p)[-1], tf.float32)
    log_var = tf.math.log(tf.reduce_prod(tf.math.square(std_q), axis=-1)/tf.reduce_prod(tf.math.square(std_p), axis=-1))
    mu_var_multip = tf.math.reduce_sum((mu_p-mu_q) / tf.math.square(std_q) * (mu_p-mu_q), axis=-1)
    trace = tf.math.reduce_sum(tf.math.square(std_p) / tf.math.square(std_q), axis=-1)
    kl_d = 0.5 * (log_var - k + mu_var_multip + trace)
    kl_d = tf.math.reduce_mean(kl_d)
    return kl_d

# @tf.function
def swish(x):
	''' swish activation function '''
	return x * tf.math.sigmoid(x)


class Encoder(tf.Module):
    ''' transition / posterior model of the active inference framework '''
    def __init__(self, dim_z, name='Encoder', activation='leaky_relu'):
        super().__init__(name=name)
        self.dim_z = dim_z  # latent dimension
        self.dense_input = FlexibleDense(20, name='dense_input')
        self.dense_e1 = FlexibleDense(20, name='dense_1')
        self.dense_mu = FlexibleDense(dim_z, name='dense_mu')
        self.dense_raw_std = FlexibleDense(dim_z, name='dense_raw_std')
        self.sampler = Sampler()
        self.mu = None
        self.raw_std = None
        self.std = None
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'leaky_relu':
            self.activation = tf.nn.leaky_relu
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')    
    
    @tf.Module.with_name_scope
    def __call__(self, X):
        z = self.dense_input(X)
        z = self.activation(z)
        z = self.dense_e1(z)
        z = self.activation(z)
        mu = self.dense_mu(z)
        raw_std = self.dense_raw_std(z)
        self.mu = mu
        self.raw_std = raw_std
        z_sample, std = self.sampler((mu, raw_std))
        self.std = std
        return z_sample, mu, std

    @tf.Module.with_name_scope
    def sample(self):
        z_sample, _ = self.sampler((self.mu, self.raw_std))
        return z_sample


class Likelihood(tf.Module):
    ''' likelihood model of the active inference framework '''
    def __init__(self, dim_x, name='likelihood', activation='leaky_relu'):
        super().__init__(name=name)
        self.dim_x = dim_x
        self.dense_z_input = FlexibleDense(20, name='dense_z_input')
        self.dense_d1 = FlexibleDense(20, name='dense_d1')
        self.dense_output = FlexibleDense(self.dim_x, name='dense_x_output')
        if activation == 'tanh': 
            self.activation = tf.nn.tanh
        elif activation == 'leaky_relu':
            self.activation = tf.nn.leaky_relu
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')
       
    @tf.Module.with_name_scope
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
        self.kl_weight = args.vae_kl_weight
        self.transition = Encoder(self.dim_z, name='transition')
        self.posterior = Encoder(self.dim_z, name='posterior')
        self.likelihood = Likelihood(self.dim_obv)

    @tf.Module.with_name_scope
    def mse(self, x_reconst, x_true):
        sse = tf.math.reduce_sum(tf.math.square(x_reconst-x_true), axis=-1)
        return tf.math.reduce_mean(sse)

    @tf.Module.with_name_scope
    def __call__(self, s_prev, a_prev, o_cur=None, training=False):
        if training:
            # add some noise to the observation
            o_cur_batch = o_cur + tf.random.normal(tf.shape(o_cur)) * 0.1
            _, mu_tran, std_tran = self.transition(tf.concat([s_prev, a_prev], axis=-1))
            state_post, mu_post, std_post = self.posterior(tf.concat([s_prev, a_prev, o_cur_batch], axis=-1))
            o_reconst = self.likelihood(state_post)
            # reconstruction loss (std assume to be 1, only reconst of mean)
            mse_loss = self.mse(o_reconst, o_cur)
            # KL Divergence loss
            kld = kl_d(mu_post, std_post, mu_tran, std_tran)
            total_loss = self.kl_weight * kld + mse_loss
            return total_loss, state_post
        else:
            state_tran, mu_tran, std_tran = self.transition(tf.concat([s_prev, a_prev], axis=-1))
            o_reconst = self.likelihood(state_tran)
            if o_cur is not None:
                state_post, mu_post, std_post = self.posterior(tf.concat([s_prev, a_prev, o_cur], axis=-1))
                return state_tran, o_reconst, state_post
            return state_tran, o_reconst


class Planner(tf.Module):
    ''' Planner class which encapsulates state rollout and policy selection '''
    def __init__(self, stateModel, args):
        super().__init__(name='Planner')
        self.stateModel = stateModel
        self.dim_z = self.stateModel.dim_z
        self.dim_obv = self.stateModel.dim_obv
        self.K = args.planner_lookahead  # lookahead
        self.D = args.planner_plan_depth  # recursive planning depth
        self.N = args.planner_n_samples  # number of samples per policy
        self.rho = args.planner_rho  # weight term on expected ambiguity
        self.gamma = args.planner_gamma  # temperature factor applied to the EFE before calculate belief about policies
        self.n_actions = args.n_actions  # for discrete action space, specify the number of possible actions
        self.n_pi = math.pow(self.n_actions, self.D)
        self.action_space = tf.constant([0, 2], dtype=tf.int32) #TODO change it to a general setting
        self.action = 0
        #TODO need to find preferred state (prior preference) empirically
        self.s_mu_prefer = tf.ones((self.dim_z,))
        self.s_std_prefer = tf.ones((self.dim_z,))

    @tf.Module.with_name_scope
    def evaluate_stage_efe(self, all_kld, all_H):
        ## calculate expected free energy for each policy using an iterative approach
        # instead of the original recursive approach ##
        reduced_efe = None
        for d in reversed(range(self.D)):
            stage_kld = all_kld.pop()
            stage_H = all_H.pop()
            efe_stage = stage_kld + stage_H
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

    @tf.Module.with_name_scope
    def __call__(self, curStates=None):
        if curStates is None:
            curStates = self.stateModel.posterior.sample()
            curStates = tf.expand_dims(tf.expand_dims(curStates, axis=0), axis=0)
        all_kld = []
        all_H = []
        # plan for D times of K-step lookahead
        for d in range(self.D):
            # each branch is split into N_actions branchs
            if d == 0:
                multiples = tf.constant((self.n_actions, self.N, 1), dtype=tf.int32)
                state_results = tf.tile(curStates, multiples)
            else:
                multiples = tf.constant([self.n_actions, 1, 1], dtype=tf.int32)
                state_results = tf.tile(state_results, multiples)
            step_kld = tf.zeros((self.n_actions ** (d+1), self.K))
            step_H = tf.zeros((self.n_actions ** (d+1), self.K))
            # rollout for each branch
            for idx_b in range(self.n_actions ** (d+1)):
                action = self.action_space[idx_b % self.n_actions]
                curStates = state_results[idx_b]
                # plan for K steps in a roll (continuously follows a policy)
                for t in range(self.K):
                    action_t = action * tf.ones((self.N, 1))
                    # use only the transition model to rollout the states
                    curStates, _, _ = self.stateModel.transition(tf.concat([curStates, action_t], axis=-1))
                    # use the likelihood model to map states to observations
                    obvSamples = self.likelihood(curStates)
                    # gather batch statistics
                    states_mean = tf.math.reduce_mean(curStates, axis=0)
                    states_std = tf.math.reduce_std(curStates, axis=0)
                    obv_std = tf.math.reduce_std(obvSamples, axis=0)
                    # KL Divergence between expected states and preferred states
                    step_kld[idx_b, t] = kl_d(states_mean, states_std, self.s_mu_prefer, self.s_std_prefer)
                    # entropy on the expected observations
                    n = tf.shape(obv_std)[-1]
                    H = n/2 + n/2 * tf.math.log(2*math.pi * tf.math.pow(tf.math.reduce_prod(tf.math.square(obv_std), axis=-1), 1/n))
                    step_H[idx_b, t] = H / self.rho
            # gather KL-D value and entropy for each branch
            all_kld.append(tf.reduce_sum(step_kld, axis=1))
            all_H.append(tf.reduce_sum(step_H, axis=1))
        # calculate EFE values for the root policies
        efe_root = self.evaluate_stage_efe(all_kld, all_H)
        self.action = tf.argmin(efe_root)