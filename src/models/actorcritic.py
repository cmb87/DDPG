import numpy as np
import tensorflow as tf
import logging
from src.models.actor import ActorNetwork
from src.models.critic import CriticNetwork
from src.models.ounoise import OrnsteinUhlenbeckActionNoise
from src.models.replaybuffer import ReplayBuffer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)-8s] [%(name)-8s] [%(levelname)-1s] [%(message)s]')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class ActorCritic(object):
    """ Class which controls the actor and critic model"""

    def __init__(self, sess, state_dim, action_dim, action_bound):
        """
        Control structure for the actor, critic and training of both classes
        :param sess:
        :param state_dim:
        :param action_dim:
        :param action_bound:
        """

        self.sess = sess
        self.actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(0.0001), float(0.001),
                             int(64))

        self.critic = CriticNetwork(sess, state_dim, action_dim,
                               float(0.001), float(0.001),
                               float(0.99))

        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        self.minibatch_size = 64
        self.nepoches = 5000
        self.nepisodes = 5000

        # Initialize replay memory
        self.replay_buffer = ReplayBuffer(1000000, random_seed=1234)

        logger.info("Actor-Critic created and ready")

    # ===========================
    #   Agent Training
    # ===========================

    def train(self, env):
        """
        Train the actor critic model
        :param env: Simulation environment
        :return:
        """

        # Set up summary Ops
        summary_ops, summary_vars = ActorCritic.build_summaries()

        self.sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./results/tf_ddpg', self.sess.graph)

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

        # Needed to enable BatchNorm.
        # This hurts the performance on Pendulum but could be useful
        # in other environments.
        # tflearn.is_training(True)

        logger.info("Starting training")

        for episode in range(self.nepisodes): # episoders

            # Reset state and reward storage
            s = env.reset()
            ep_reward = 0
            ep_ave_max_q = 0

            for epoche in range(self.nepoches): # Max epoche lengths

                if True:
                    env.render()

                # Predict action and add exploration noise
                a = self.actor.predict(np.reshape(s, (1, self.actor.s_dim))) + self.ou_noise()

                # Now step in environment
                s2, r, done, info = env.step(a[0])

                # Store results in replay_buffer
                self.replay_buffer.add(np.reshape(s, (self.actor.s_dim,)), np.reshape(a, (self.actor.a_dim,)), r, done, np.reshape(s2, (self.actor.s_dim,)))

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if self.replay_buffer.size() > self.minibatch_size:

                    # Sample a batch from the replay buffer
                    s_batch, a_batch, r_batch, done_batch, s2_batch = self.replay_buffer.sample_batch(self.minibatch_size)

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

                # Proceed to next state and store reward
                s = s2
                ep_reward += r

                # Epoche is done
                if done:
                    summary_str = self.sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / epoche
                    })

                    writer.add_summary(summary_str, episode)
                    writer.flush()

                    logger.info('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), episode, (ep_ave_max_q / float(epoche))))
                    break


    # ===========================
    #   Tensorflow Summary Ops
    # ===========================
    @staticmethod
    def build_summaries():
        """
        Creates the variables for tensorboard monitoring
        :return:
        """
        # For Tensorboard
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar("Qmax_Value", episode_ave_max_q)

        summary_vars = [episode_reward, episode_ave_max_q]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars