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

from src.models.actor import ActorNetwork
from src.models.critic import CriticNetwork
from src.models.ounoise import OrnsteinUhlenbeckActionNoise
from src.models.replaybuffer import ReplayBuffer

class ActorCritic(object):
    """ Class which controls the actor and critic model"""

    def __init__(self, sess, state_dim, action_dim, action_bound):

        self.sess = sess
        self.actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(0.0001), float(0.001),
                             int(64))

        self.critic = CriticNetwork(sess, state_dim, action_dim,
                               float(0.001), float(0.001),
                               float(0.99),
                               self.actor.get_num_trainable_vars())

        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        self.minibatch_size = 64


    # ===========================
    #   Tensorflow Summary Ops
    # ===========================
    @staticmethod
    def build_summaries():
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar("Qmax_Value", episode_ave_max_q)

        summary_vars = [episode_reward, episode_ave_max_q]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars

    # ===========================
    #   Agent Training
    # ===========================

    def train(self, env):

        # Set up summary Ops
        summary_ops, summary_vars = ActorCritic.build_summaries()

        self.sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./results/tf_ddpg', self.sess.graph)

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()


        # Initialize replay memory
        replay_buffer = ReplayBuffer(1000000, 1234)

        # Needed to enable BatchNorm.
        # This hurts the performance on Pendulum but could be useful
        # in other environments.
        # tflearn.is_training(True)

        for episode in range(50000): # episoders

            s = env.reset()

            ep_reward = 0
            ep_ave_max_q = 0

            for epoche in range(1000): # Max epoche lengths

                if True:
                    env.render()

                # Predict action and add exploration noise
                a = self.actor.predict(np.reshape(s, (1, self.actor.s_dim))) + self.ou_noise()

                # Now step in environment
                s2, r, done, info = env.step(a[0])

                # Store results in replay_buffer
                replay_buffer.add(np.reshape(s, (self.actor.s_dim,)), np.reshape(a, (self.actor.a_dim,)), r, done, np.reshape(s2, (self.actor.s_dim,)))

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if replay_buffer.size() > self.minibatch_size:

                    # Sample a batch from the replay buffer
                    s_batch, a_batch, r_batch, done_batch, s2_batch = replay_buffer.sample_batch(self.minibatch_size)

                    # Calculate q_target from target critic network
                    target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

                    # Now create minibatch
                    y_i = []
                    for k in range(self.minibatch_size):
                        if done_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

                    # Get the predicted q value
                    predicted_q_value, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i,(self.minibatch_size, 1)))

                    # Take the maximum as epoche maximum to track
                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(s_batch)
                    grads = self.critic.action_gradients(s_batch, a_outs)
                    self.actor.train(s_batch, grads[0])

                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                s = s2
                ep_reward += r

                if done:

                    summary_str = self.sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / epoche
                    })

                    writer.add_summary(summary_str, episode)
                    writer.flush()

                    print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), episode, (ep_ave_max_q / float(epoche))))
                    break