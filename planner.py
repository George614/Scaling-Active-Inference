import tensorflow as tf
import numpy as np
import math
import metrics as mcs


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
                    step_kld_point = mcs.kl_d(states_mean, states_std, self.s_mu_prefer, self.s_std_prefer)
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