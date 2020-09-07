import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd
tf.compat.v1.disable_eager_execution()

class Policy_net_single:
    def __init__(self, name: str, tf_session, units=64, temp=0.1, n_layers=2):
        """
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        :param n_layers: number of dense layers (default=2)
        """

        ob_space = 22 #should be modified accordingly
        act_space = 2 #should be modified accordingly
        self.session = tf_session
        self.weights_initializer = tf.compat.v1.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)
        self.bias_initializer = tf.compat.v1.constant_initializer(0.1)
        with tf.compat.v1.variable_scope(name):
            self.obs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,ob_space], name='obs')
            #self.logstd_normal = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='logstd_normal')
            with tf.compat.v1.variable_scope('policy_net'):
                layer_1 = tf.compat.v1.layers.dense(inputs=self.obs, units=units, activation=tf.nn.tanh, kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer)
                for _ in range(n_layers-1):
                    layer_1 = tf.compat.v1.layers.dense(inputs=layer_1, units=units, activation=tf.nn.tanh, kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer)
                layer_out = tf.compat.v1.layers.dense(inputs=layer_1, units=act_space, activation=tf.nn.tanh, kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer)
                #logstd = tf.get_variable(name="logstd", shape=[1, act_space], initializer=tf.zeros_initializer())
                logstd = tf.zeros([1, act_space])
                #logstd_spend = tf.matmul(self.logstd_normal, logstd)
                #self.mean1, self.mean2, self.mean3 = tf.split(layer_3, num_or_size_splits = act_space, axis=1)
                #self.logstd1, self.logstd2, self.logstd3 = tf.split(logstd_spend, num_or_size_splits = act_space, axis=1)
                #self.categorical_prob = tf.matmul(self.logstd_normal, self.categorical_prob)
                #self.act_probs = tfd.Normal(loc = layer_3, scale = tf.exp(logstd_spend))

            #self.act_probs = tfd.MultivariateNormalDiag(loc = layer_3, scale_diag = tf.exp(logstd_spend))
                #self.act_probs = tfd.Normal(loc = layer_out, scale = tf.exp(logstd))
                self.act_probs = tfd.MultivariateNormalDiag(loc = layer_out, scale_diag = tf.exp(logstd))
                self.act_direct_step = layer_out

            with tf.compat.v1.variable_scope('value_net'):
                layer_1_value = tf.compat.v1.layers.dense(inputs=self.obs, units=units, activation=tf.nn.tanh, kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer)
                for _ in range(n_layers-1):
                    layer_1_value = tf.compat.v1.layers.dense(inputs=layer_1_value, units=units, activation=tf.nn.tanh, kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer)
                self.v_preds = tf.compat.v1.layers.dense(inputs=layer_1_value, units=1, activation=None, kernel_initializer=self.weights_initializer, bias_initializer=self.bias_initializer)

            self.act_stochastic = self.act_probs.sample()
            #self.act_deterministic = tf.argmax(self.act_probs, axis=1)
            self.scope = tf.compat.v1.get_variable_scope().name

    def act(self, obs, stochastic=True):
        #logstd_vector = np.ones([obs.shape[0],1])
        return self.session.run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
                                                                                            #self.logstd_normal:logstd_vector})
    def get_action_prob(self, obs):
        #logstd_vector = np.ones([obs.shape[0],1])
        return self.session.run(self.act_probs, feed_dict={self.obs: obs})
                                                                       #self.logstd_normal:logstd_vector})
    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def action_to_np(self, action):
        return self.session.run(action)

    def act_direct(self, obs):
        return self.session.run(self.act_direct_step, feed_dict={self.obs: obs})

    def get_l2_loss(self):
        return tf.compat.v1.losses.get_regularization_loss()
