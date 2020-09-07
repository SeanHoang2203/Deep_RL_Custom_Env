import tensorflow as tf
import copy
import numpy as np
from tensorflow_probability import distributions as tfd
tf.compat.v1.disable_eager_execution()


class PPOTrain_single:
    def __init__(self, tf_session, Policy, Old_Policy, gamma=0.99, clip_value=0.2, c_1=1, c_2=0.01,lr=5e-5):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """
        act_space = 2     #should be modified accordingly
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma
        self.session = tf_session
        self.clip_value = clip_value
        self.c_1 = c_1
        self.c_2 = c_2
        self.lr = lr
        
        self.pi_trainable = self.Policy.get_trainable_variables()
        self.old_pi_trainable = self.Old_Policy.get_trainable_variables()

##        self.pi_params = tf.Variable(1., validate_shape=False) #tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='pi_params')
##        self.old_pi_params = tf.Variable(1., validate_shape=False) #tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='old_pi_params')

        # assign_operations for policy parameter values to old policy parameters
        with tf.compat.v1.variable_scope('assign_op'):            
            self.assign_ops = [tf.compat.v1.assign(v_old, v) for v_old, v in zip(self.old_pi_trainable, self.pi_trainable)]

        # inputs for train_op
        with tf.compat.v1.variable_scope('train_inp'):
            self.actions = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, act_space], name='actions')
            self.rewards = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.adv = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='adv')

        # act_sample = self.Policy.act_stochastic

        self.act_probs = self.Policy.act_probs.log_prob(self.actions)
        self.act_probs_old = self.Old_Policy.act_probs.log_prob(self.actions)

        # self.actions = tf.clip_by_value(self.actions, -1, 1)
        # probabilities of actions which agent took with policy
        # act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        # act_probs = tf.reduce_sum(act_probs, axis=1)
        # probabilities of actions which agent took with old policy
        # act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        # act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.compat.v1.variable_scope('loss'):
            # construct computation graph for loss_clip
            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(self.act_probs - self.act_probs_old)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value, clip_value_max=1 + self.clip_value)
            loss_clip = tf.minimum(tf.multiply(self.adv, ratios), tf.multiply(self.adv, clipped_ratios))
            loss_clip = tf.reduce_mean(input_tensor=loss_clip)
            tf.compat.v1.summary.scalar('loss_clip', loss_clip)
            # construct computation graph for loss of entropy bonus
            entropy = tf.reduce_mean(input_tensor=self.Policy.act_probs.entropy())
            #entropy = -tf.reduce_sum(act_probs *
            #                         tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0)))
            #entropy = tf.reduce_mean(act_probs_whole.entropy(),axis=0)  # mean of entropy of pi(obs)
            tf.compat.v1.summary.scalar('entropy', entropy)
            # construct computation graph for loss of value function
            v_preds = self.Policy.v_preds
            v_preds_clip = self.Old_Policy.v_preds + \
                           tf.clip_by_value(self.Policy.v_preds - self.Old_Policy.v_preds,\
                                            -c_2, c_2)
            
            loss_vf_unclip = tf.square(self.rewards + self.gamma * self.v_preds_next - v_preds)
            loss_vf_clip = tf.square(self.rewards + self.gamma * self.v_preds_next - v_preds_clip)
            loss_vf = 0.5 * tf.reduce_mean(tf.maximum(loss_vf_unclip,loss_vf_clip))
##            loss_vf = tf.math.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(input_tensor=loss_vf)
            tf.compat.v1.summary.scalar('value_difference', loss_vf)

            #l2_loss = self.Policy.get_l2_loss()
            # construct computation graph for loss
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy

            # minimize -loss == maximize loss
            loss = -loss
            tf.compat.v1.summary.scalar('total', loss)

        self.merged = tf.compat.v1.summary.merge_all()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        self.gradients = self.optimizer.compute_gradients(loss, var_list=self.pi_trainable)
        self.train_op = self.optimizer.minimize(loss, var_list=self.pi_trainable)
        self.kl_div = tfd.kl_divergence(self.Policy.act_probs,self.Old_Policy.act_probs)

    def train(self, obs, actions, adv, rewards, v_preds_next):
        #logstd_vector = np.ones([obs.shape[0],1])
        self.session.run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.adv: adv})
        indices = np.random.permutation(obs.shape[0])[:200]
        return self.session.run(self.kl_div, feed_dict={self.Policy.obs: obs[indices,:],\
                                                        self.Old_Policy.obs: obs[indices,:]})

    def get_summary(self, obs, actions, adv, rewards, v_preds_next):
        #logstd_vector = np.ones([obs.shape[0],1])
        return self.session.run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.actions: actions,
                                                                    self.rewards: rewards,
                                                                    self.v_preds_next: v_preds_next,
                                                                    self.adv: adv})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        self.session.run(self.assign_ops)



    def get_grad(self, obs, actions, adv, rewards, v_preds_next):
        #logstd_vector = np.ones([obs.shape[0],1])
        return self.session.run(self.gradients, feed_dict={self.Policy.obs: obs,
                                                                       self.Old_Policy.obs: obs,
                                                                       self.actions: actions,
                                                                       self.rewards: rewards,
                                                                       self.v_preds_next: v_preds_next,
                                                                       self.adv: adv})
    def see_prob(self, obs, actions):
        return self.session.run([self.act_probs, self.act_probs_old], feed_dict={self.Policy.obs: obs,
                                                                                             self.Old_Policy.obs: obs,
                                                                                             self.actions: actions})

    def set_params(self, clip_value=None, lr=None):
        if clip_value is not None:
            self.clip_value = clip_value
        if lr is not None:
            self.lr = lr
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
