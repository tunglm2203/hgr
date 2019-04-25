import tensorflow as tf
from util import store_args, nn


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        Output:
            pi_tf: Output of policy
            q1_tf: Q-value 1 (for Double-Q)
            q2_tf: Q-value 2 (for Double-Q)
            q1_pi_tf: Q-value 1 take action from policy
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        # Actor Network
        with tf.variable_scope('pi'):
            input_pi = tf.concat(axis=1, values=[o, g])  # for actor
            self.pi_tf = self.max_u * tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu]))

        # Critic Networks
        with tf.variable_scope('q1'):
            # For critic training
            input_q1 = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self.q1_tf = nn(input_q1, [self.hidden] * self.layers + [1])

        with tf.variable_scope('q2'):
            # For critic training
            input_q2 = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])  # actions from the policy
            self.q2_tf = nn(input_q2, [self.hidden] * self.layers + [1])

        with tf.variable_scope('q1', reuse=True):
            # for policy training
            input_q1_pi = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])  # actions from the buffer
            self._input_q1 = input_q1_pi  # exposed for tests
            self.q1_pi_tf = nn(input_q1_pi, [self.hidden] * self.layers + [1], reuse=True)

