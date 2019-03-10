from gym import wrappers
import gym
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import random
from collections import deque



class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out, self.network_params = self.create_critic_network()

        # Target Network
        self.target_inputs, self.target_action, self.target_out, self.target_network_params = self.create_critic_network()

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        # self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))

        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        ### Get start var count ###
        istart = len(tf.trainable_variables())
        ### Input tensors ###
        with tf.name_scope("Critic" + '_in'):
            inputs = tf.placeholder(tf.float32, [None, self.s_dim], name="s")
            actions = tf.placeholder(tf.float32, [None, self.a_dim], name="a")

        ### ANN Model ###\n",
        with tf.name_scope("Critic" + '_' + 'ANN'):
            hl1_s = Dense(400, activation='relu', name='hl1_s',
                          kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.003))
            bn1_s = BatchNormalization()
            hl3 = Dense(300, activation='relu', name='hl3',
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.003))
            bn3 = BatchNormalization()
            hl4 = Dense(1, activation='linear', name='ol0',
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.003))

            a1s = hl1_s(inputs)
            a2 = Add(name="Merged")([bn1_s(a1s), actions])
            a3 = hl3(a2)

            out = hl4(a3)

        weights = tf.trainable_variables()[istart:]

        return inputs, actions, out, weights

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)