import numpy as np
import tensorflow as tf


class ZeroBaseline:
    def reset(self, session):
        pass

    def evaluate(self, episode, goal):
        return np.zeros(episode.length - 1, dtype=np.float32)

    def train_step(self, episodes):
        return 0.


class BatchVariables:
    def __init__(self, d_observations, d_goals):
        self.d_observations = np.reshape(d_observations, -1).tolist()
        self.d_goals = np.reshape(d_goals, -1).tolist()

        self.lengths = tf.placeholder(tf.int32, [None], name='lengths')

        self.observations = tf.placeholder(tf.float32, [None, None] +
                                           self.d_observations,
                                           name='observations')
        self.rewards = tf.placeholder(tf.float32, [None, None], name='rewards')

        self.goals = tf.placeholder(tf.float32, [None] + self.d_goals, 'goals')

        self.times = tf.placeholder(tf.float32, [None, None], name='times')

        self.batch_size = tf.shape(self.observations)[0]
        self.max_steps = tf.shape(self.observations)[1]

        self.goals_enc = tf.tile(self.goals, [1, self.max_steps])
        self.goals_enc = tf.reshape(self.goals_enc, [-1, self.max_steps] +
                                    self.d_goals)

        self.times_enc = tf.expand_dims(self.times, axis=2)


class ValueBaseline:
    def __init__(self, env, hidden_layers, learning_rate, use_vscaling_init):
        self.env = env
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.use_vscaling_init = use_vscaling_init
        self.create_network()
        self.setup()

    def create_network(self):
        self.bvars = BatchVariables(self.env.d_observations, self.env.d_goals)

        if len(self.bvars.d_observations) == 1:
            self.create_network_1d()
        else:
            with tf.variable_scope("baseline"):
                self.create_network_conv()

    def create_network_1d(self):
        d_input = self.bvars.d_observations[0] + self.bvars.d_goals[0] + 1
        inputs = tf.concat(
            [self.bvars.observations, self.bvars.goals_enc, self.bvars.times_enc], axis=2)

        output = tf.reshape(inputs, [-1, d_input])
        self.variables = []
        layers = [d_input] + self.hidden_layers + [1]
        for i in range(1, len(layers)):
            if self.use_vscaling_init:
                W_init = tf.variance_scaling_initializer(
                    mode="fan_avg", distribution="uniform")

                W = tf.get_variable('vW_{0}'.format(
                    i), [layers[i - 1], layers[i]], initializer=W_init)

                b = tf.get_variable('vb_{0}'.format(i), layers[i],
                                    initializer=tf.constant_initializer(0.0))

            else:
                W = tf.Variable(tf.truncated_normal([layers[i - 1], layers[i]],
                                                    stddev=0.001),
                                name='vW_{0}'.format(i))
                b = tf.Variable(tf.zeros(layers[i]), name='vb_{0}'.format(i))

            self.variables += [W, b]

            output = tf.matmul(output, W) + b
            if i < len(layers) - 1:
                output = tf.tanh(output)

        self.pred = tf.reshape(output, [-1, self.bvars.max_steps])

    def create_network_conv(self):
        n_filters = [32, 64, 64]
        filter_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        hidden_layer_sizes = self.hidden_layers

        assert isinstance(hidden_layer_sizes, list)

        reshaped_observations = tf.reshape(self.bvars.observations,
                                           [-1] + self.bvars.d_observations)

        reshaped_observations = reshaped_observations / 255
        with tf.device('/cpu:0'):
            prev_y = tf.image.resize_images(
                reshaped_observations, (84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        for i, (fs, ks, strides) in enumerate(
                zip(n_filters, filter_sizes, strides)):
            init = tf.variance_scaling_initializer(
                mode="fan_avg", distribution="uniform")
            prev_y = tf.layers.conv2d(prev_y, fs, ks,
                                      strides=strides,
                                      activation=tf.tanh,
                                      name="baseline_rep_conv_" + str(i),
                                      kernel_initializer=init)

        obs_enc_flat = tf.contrib.layers.flatten(prev_y)

        goals_reshaped = tf.reshape(self.bvars.goals_enc,
                                    [-1] + self.bvars.d_goals)

        times_reshaped = tf.reshape(self.bvars.times_enc,
                                    [-1, 1])

        inputs = tf.concat(
            [obs_enc_flat, goals_reshaped, times_reshaped], axis=1)

        prev_y = inputs
        for i, layer_size in enumerate(hidden_layer_sizes):
            init = tf.variance_scaling_initializer(
                mode="fan_avg", distribution="uniform")
            prev_y = tf.layers.dense(prev_y, layer_size,
                                     activation=tf.tanh,
                                     kernel_initializer=init,
                                     name="baseline_rep_dense_" + str(i))

        output = tf.layers.dense(prev_y, 1, name="op", activation=None)
        self.pred = tf.reshape(output, [-1, self.bvars.max_steps])

    def setup(self):
        # Ignoring prediction for last time step
        mask = tf.sequence_mask(self.bvars.lengths - 2,
                                self.bvars.max_steps - 1, dtype=tf.float32)
        pred = tf.stop_gradient(self.pred[:, 1:]) * mask

        mask = tf.sequence_mask(self.bvars.lengths - 1, dtype=tf.float32)
        rewards = self.bvars.rewards[:, 1:] * mask

        target = rewards + pred

        self.loss = ((self.pred[:, :-1] - target)**2) * mask
        self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(mask)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def reset(self, session):
        self.session = session

    def evaluate(self, episode, goal):
        feed = {self.bvars.lengths: [episode.length - 1],
                self.bvars.observations: [episode.observations[:-1]],
                self.bvars.goals: [goal],
                self.bvars.times: [np.arange(1, episode.length)]}

        return self.session.run(self.pred, feed)[0]

    def train_step(self, episodes):
        lengths = np.array([e.length for e in episodes], dtype=np.int32)

        batch_size = len(episodes)
        max_steps = max(lengths)

        observations = np.zeros([batch_size, max_steps] +
                                self.bvars.d_observations, dtype=np.float32)
        rewards = np.zeros((batch_size, max_steps), dtype=np.float32)
        goals = np.array([e.goal for e in episodes], dtype=np.float32)

        times = np.zeros((batch_size, max_steps), dtype=np.float32)

        for i in range(batch_size):
            for j in range(lengths[i]):
                observations[i, j] = episodes[i].observations[j]

            rewards[i, :lengths[i]] = episodes[i].rewards

            times[i, :lengths[i]] = np.arange(1, lengths[i] + 1)

        feed = {self.bvars.lengths: lengths,
                self.bvars.observations: observations,
                self.bvars.rewards: rewards, self.bvars.goals: goals,
                self.bvars.times: times}

        loss, _ = self.session.run([self.loss, self.train_op], feed)

        return loss
