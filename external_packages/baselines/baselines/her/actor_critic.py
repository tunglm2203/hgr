import tensorflow as tf
from baselines.her.util import nn


class ActorCritic:
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.inputs_tf = inputs_tf
        self.dimo = dimo
        self.dimg = dimg
        self.dimu = dimu
        self.max_u = max_u
        self.o_stats = o_stats
        self.g_stats = g_stats
        self.hidden = hidden
        self.layers = layers

        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_q, [self.hidden] * self.layers + [1])
            # for critic training
            input_q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_q  # exposed for tests
            self.Q_tf = nn(input_q, [self.hidden] * self.layers + [1], reuse=True)
