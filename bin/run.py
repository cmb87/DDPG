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

from src.models.actorcritic import ActorCritic





if __name__ == '__main__':
    K.clear_session()

    with tf.Session() as sess:
        K.set_session(sess)

        env = gym.make("Pendulum-v0")
        # env = gym.make("MountainCarContinuous-v0")

        np.random.seed(int(1234))
        tf.set_random_seed(int(1234))
        env.seed(int(1234))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)



        if False:
            if not True:
                env = wrappers.Monitor(
                    env, './results/gym_ddpg', video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, './results/gym_ddpg', force=True)

        A2C = ActorCritic(sess, state_dim, action_dim, action_bound)
        A2C.train(env)

        if False:
            env.monitor.close()