import numpy as np
import tensorflow as tf


class ZeroBaseline:
    def reset(self, session):
        pass

    def evaluate(self, episode, goal):
        return np.zeros(episode.length - 1, dtype=np.float32)

    def train_step(self, episodes):
        return 0.


class AdaptiveOptimumValueBaseline:
    def __init__(self, env, learning_rate=0.2):
        self.env = env
        self.learning_rate = learning_rate

    def reset(self, session):
        self.correction = np.zeros(self.env.max_steps - 1, dtype=np.float32)

    def evaluate(self, episode, goal):
        v = self.env.evaluate_values(episode, goal)
        return v + self.correction[: episode.length - 1]

    def train_step(self, episodes):
        max_steps = max([e.length for e in episodes])

        correction = np.zeros(max_steps - 1, dtype=np.float32)
        correction_n = np.zeros(max_steps - 1, dtype=np.float32)

        for i, e in enumerate(episodes):
            v = self.env.evaluate_values(e, e.goal)

            crewards = np.cumsum(e.rewards[1:][::-1])[::-1]

            correction[:e.length - 1] += crewards - v
            correction_n[:e.length - 1] += 1.0

        correction = correction/correction_n
        d = correction - self.correction[:max_steps - 1]
        self.correction[:max_steps - 1] += self.learning_rate*d

        return np.mean(d**2)


class BatchVariables:
    def __init__(self, d_observations, d_goals):
        self.d_observations = d_observations
        self.d_goals = d_goals

        self.d_input = d_observations + d_goals + 1

        self.lengths = tf.placeholder(tf.int32, [None], name='lengths')

        self.observations = tf.placeholder(tf.float32, [None, None,
                                                        d_observations],
                                           name='observations')
        self.rewards = tf.placeholder(tf.float32, [None, None], name='rewards')

        self.goals = tf.placeholder(tf.float32, [None, d_goals], 'goals')

        self.times = tf.placeholder(tf.float32, [None, None], name='times')

        self.batch_size = tf.shape(self.observations)[0]
        self.max_steps = tf.shape(self.observations)[1]

        self.goals_enc = tf.tile(self.goals, [1, self.max_steps])
        self.goals_enc = tf.reshape(self.goals_enc, [-1, self.max_steps,
                                                     self.d_goals])

        times = tf.expand_dims(self.times, axis=2)
        self.inputs = tf.concat([self.observations, self.goals_enc, times],
                                axis=2)


class ValueBaseline:
    def __init__(self, env, hidden_layers, learning_rate):
        self.env = env
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        self.create_network()
        self.setup()

    def create_network(self):
        self.bvars = BatchVariables(self.env.d_observations, self.env.d_goals)

        output = tf.reshape(self.bvars.inputs, [-1, self.bvars.d_input])

        self.variables = []
        layers = [self.bvars.d_input] + self.hidden_layers + [1]
        for i in range(1, len(layers)):
            W = tf.Variable(tf.truncated_normal([layers[i - 1], layers[i]],
                                                stddev=0.01),
                            name='vW_{0}'.format(i))
            b = tf.Variable(tf.zeros(layers[i]), name='vb_{0}'.format(i))

            self.variables += [W, b]

            output = tf.matmul(output, W) + b
            if i < len(layers) - 1:
                output = tf.tanh(output)

        self.pred = tf.reshape(output, [-1, self.bvars.max_steps])

    def setup(self):
        # Ignoring prediction for last time step
        mask = tf.sequence_mask(self.bvars.lengths - 2,
                                self.bvars.max_steps - 1, dtype=tf.float32)
        pred = tf.stop_gradient(self.pred[:, 1:])*mask

        mask = tf.sequence_mask(self.bvars.lengths - 1, dtype=tf.float32)
        rewards = self.bvars.rewards[:, 1:]*mask

        target = rewards + pred

        self.loss = ((self.pred[:, :-1] - target)**2)*mask
        self.loss = tf.reduce_sum(self.loss)/tf.reduce_sum(mask)

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

        observations = np.zeros((batch_size, max_steps,
                                 self.bvars.d_observations), dtype=np.float32)
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
