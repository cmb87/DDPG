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




class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out, self.network_params = self.create_actor_network()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out, self.target_network_params = self.create_actor_network()

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        ### Get start var count ###
        istart = len(tf.trainable_variables())
        ### Input tensors ###
        with tf.name_scope("Actor" + '_in'):
            inputs = tf.placeholder(tf.float32, [None, self.s_dim], name="s")

        ### ANN Model ###
        with tf.name_scope("Actor" + '_ANN'):
            ### Define layers ###

            hl1 = Dense(400, activation='relu', name="hl1",
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.003))
            bn1 = BatchNormalization()
            hl2 = Dense(300, activation='relu', name="hl2",
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.003))
            bn2 = BatchNormalization()
            #            hl3 = Dense(300, activation='relu',name="hl3",
            #                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.003))
            ol0 = Dense(self.a_dim, activation='tanh', name="actions",
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.003))

            ### Define activations ###
            a1 = hl1(inputs)
            a2 = hl2(bn1(a1))
            out = 2.0 * ol0(bn2(a2))

        ### Scaled output ###
        with tf.name_scope("Actor" + '_act_scaled'):
            scaled_out = tf.multiply(out, self.action_bound)

        ### Trainable weights ###
        weights = tf.trainable_variables()[istart:]

        return inputs, out, scaled_out, weights

    ### Train ###
    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars