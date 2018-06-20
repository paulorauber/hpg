import tensorflow as tf
import numpy as np

from hpg.agent import Agent


class BatchVariables:
    def __init__(self, d_observations, d_goals, n_actions):
        self.d_observations = d_observations
        self.d_goals = d_goals
        self.n_actions = n_actions

        self.d_input = d_observations + d_goals

        self.lengths = tf.placeholder(tf.int32, [None], name='lengths')

        self.actions = tf.placeholder(tf.int32, [None, None], name='actions')
        self.actions_enc = tf.one_hot(self.actions, self.n_actions)

        self.observations = tf.placeholder(tf.float32, [None, None,
                                                        d_observations],
                                           name='observations')

        self.rewards = tf.placeholder(tf.float32, [None, None], name='rewards')
        self.baselines = tf.placeholder(tf.float32, [None, None],
                                        name='baselines')

        self.goals = tf.placeholder(tf.float32, [None, self.d_goals],
                                    name='goals')

        self.batch_size = tf.shape(self.observations)[0]
        self.max_steps = tf.shape(self.observations)[1]

        self.goals_enc = tf.tile(self.goals, [1, self.max_steps])
        self.goals_enc = tf.reshape(self.goals_enc, [-1, self.max_steps,
                                                     self.d_goals])

        self.inputs = tf.concat([self.observations, self.goals_enc], axis=2)


class PolicyNetworkAgent(Agent):
    def __init__(self, env, hidden_layers, learning_rate, baseline,
                 init_session):
        Agent.__init__(self, env)

        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.baseline = baseline

        self.create_network()
        self.setup()

        self.saver = tf.train.Saver()

        if init_session:
            self.init()

    def create_network(self):
        self.bvars = BatchVariables(self.env.d_observations, self.env.d_goals,
                                    self.env.n_actions)

        output = tf.reshape(self.bvars.inputs, [-1, self.bvars.d_input])

        self.variables = []
        layers = [self.bvars.d_input] + self.hidden_layers +\
            [self.bvars.n_actions]
        for i in range(1, len(layers)):
            W = tf.Variable(tf.truncated_normal([layers[i - 1], layers[i]],
                                                stddev=0.01),
                            name='pW_{0}'.format(i))
            b = tf.Variable(tf.zeros(layers[i]), name='pb_{0}'.format(i))

            self.variables += [W, b]

            output = tf.matmul(output, W) + b
            if i < len(layers) - 1:
                output = tf.tanh(output)
            else:
                # Note: Probabilities lower bounded by 1e-12
                output = tf.nn.softmax(output) + 1e-12

        self.policy = tf.reshape(output, [-1, self.bvars.max_steps,
                                          self.bvars.n_actions])

    def feed(self, episodes):
        feed = {}

        lengths = np.array([e.length for e in episodes], dtype=np.int32)

        batch_size = len(episodes)
        max_steps = max(lengths)

        actions = np.zeros((batch_size, max_steps - 1), dtype=np.int32)
        observations = np.zeros((batch_size, max_steps,
                                 self.bvars.d_observations), dtype=np.float32)
        rewards = np.zeros((batch_size, max_steps), dtype=np.float32)
        baselines = np.zeros((batch_size, max_steps - 1), dtype=np.float32)

        for i in range(batch_size):
            actions[i, :lengths[i] - 1] = episodes[i].actions

            for j in range(lengths[i]):
                observations[i, j] = episodes[i].observations[j]

            rewards[i, :lengths[i]] = episodes[i].rewards

            v = self.baseline.evaluate(episodes[i], episodes[i].goal)
            baselines[i, :lengths[i] - 1] = v

        feed[self.bvars.lengths] = lengths
        feed[self.bvars.actions] = actions
        feed[self.bvars.observations] = observations
        feed[self.bvars.rewards] = rewards
        feed[self.bvars.baselines] = baselines
        feed[self.bvars.goals] = np.array([e.goal for e in episodes],
                                          dtype=np.float32)

        return feed

    def init(self):
        config = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        self.baseline.reset(self.session)

    def close(self):
        self.session.close()

    def act(self, observation, goal, greedy):
        goal = np.array([goal], dtype=np.float32)
        observation = np.array([[observation]], dtype=np.float32)

        feed = {self.bvars.lengths: [1], self.bvars.goals: goal,
                self.bvars.observations: observation}

        policy = self.session.run(self.policy, feed)[0][0]

        if greedy:
            return np.argmax(policy)

        return np.random.choice(self.bvars.n_actions, p=policy)

    def train(self, n_train_batches, batch_size, eval_freq, eval_size):
        ereturns = []
        treturns = []
        plosses = []
        blosses = []

        print('Training for {0} batches.'.format(n_train_batches))

        episodes = self.interact(eval_size, greedy=True, render=False)
        ereturns.append(np.mean([np.sum(e.rewards) for e in episodes]))
        print('Batch 0. Return (e): {0:.3f}.'.format(ereturns[-1]))

        for t in range(1, n_train_batches + 1):
            episodes = self.interact(batch_size, greedy=False, render=False)
            treturns.append(np.mean([np.sum(e.rewards) for e in episodes]))

            plosses.append(self.train_step(episodes))
            blosses.append(self.baseline.train_step(episodes))

            if t % eval_freq == 0:
                episodes = self.interact(eval_size, greedy=True, render=False)
                ereturns.append(np.mean([np.sum(e.rewards) for e in episodes]))

                msg = 'Batch {0}. Return (e): {1:.3f}. Return (t): {2:.3f}. '
                msg += 'Policy loss (t): {3:.3f}. Baseline loss (t): {4:.3f}.'
                aret = np.mean(treturns[-eval_freq:])
                aploss = np.mean(plosses[-eval_freq:])
                abloss = np.mean(blosses[-eval_freq:])
                print(msg.format(t, ereturns[-1], aret, aploss, abloss))

        ereturns, treturns = np.array(ereturns), np.array(treturns)
        plosses, blosses = np.array(plosses), np.array(blosses)

        return ereturns, treturns, plosses, blosses

    def setup(self):
        raise NotImplementedError()

    def train_step(self, episodes):
        raise NotImplementedError()

    def save(self, filepath):
        self.saver.save(self.session, filepath)

    def load(self, filepath):
        self.saver.restore(self.session, filepath)


class PolicyGradientAgent(PolicyNetworkAgent):
    def __init__(self, env, hidden_layers, learning_rate, baseline,
                 init_session):
        PolicyNetworkAgent.__init__(self, env, hidden_layers, learning_rate,
                                    baseline, init_session)

    def setup(self):
        mask = tf.sequence_mask(self.bvars.lengths - 1, dtype=tf.float32)

        policy = self.policy[:, :-1]
        proba = tf.reduce_sum(policy*self.bvars.actions_enc, axis=2)
        self.lproba = tf.log(proba)*mask

        creturn = tf.cumsum(self.bvars.rewards, axis=1, exclusive=True,
                            reverse=True)
        creturn = creturn[:, :-1] - self.bvars.baselines

        self.rlproba = tf.reduce_sum(self.lproba*creturn, axis=1)
        self.rlproba = tf.reduce_mean(self.rlproba)

        self.loss = -self.rlproba

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def train_step(self, episodes):
        feed = self.feed(episodes)
        loss, _ = self.session.run([self.loss, self.train_op], feed)

        return loss


class HindsightGradientAgent(PolicyNetworkAgent):
    def __init__(self, env, hidden_layers, learning_rate,
                 per_decision, weighted, subgoals_per_episode, baseline,
                 init_session):
        self.per_decision = per_decision
        self.weighted = weighted
        self.subgoals_per_episode = subgoals_per_episode

        PolicyNetworkAgent.__init__(self, env, hidden_layers, learning_rate,
                                    baseline, init_session)

    def setup(self):
        mask = tf.sequence_mask(self.bvars.lengths - 1, dtype=tf.float32)

        policy = self.policy[:, :-1]
        proba = tf.reduce_sum(policy*self.bvars.actions_enc, axis=2)
        self.lproba = tf.log(proba)*mask

        self.olproba = tf.placeholder(tf.float32, [None, None], name='olproba')

        lratios = tf.exp(tf.cumsum(self.lproba - self.olproba, axis=1,
                                   exclusive=False, reverse=False))*mask

        if self.weighted:
            lratios += 1e-12
            lratios = lratios / tf.reduce_sum(lratios, axis=0)

        lratios = tf.stop_gradient(lratios)

        baselines = lratios*self.bvars.baselines

        if not self.per_decision:
            lratios = tf.exp(tf.reduce_sum(self.lproba - self.olproba, axis=1))
            lratios = tf.reshape(lratios, [-1, 1])

            if self.weighted:
                lratios += 1e-12
                lratios = lratios/tf.reduce_sum(lratios)

            lratios = tf.stop_gradient(lratios)

        rewards = lratios*self.bvars.rewards[:, 1:]
        creturn = tf.cumsum(rewards, axis=1, exclusive=False, reverse=True)
        creturn = creturn - baselines

        self.rlproba = tf.reduce_sum(self.lproba*creturn, axis=1)
        self.rlproba = tf.reduce_mean(self.rlproba)

        self.gradrlproba = tf.gradients(self.rlproba, self.variables)

        self.grads = []
        for v in self.variables:
            self.grads.append(tf.placeholder(tf.float32, v.shape))

        grads_and_vars = list(zip(self.grads, self.variables))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(grads_and_vars)

    def train_step(self, episodes):
        # Note: Goal distribution is uniform by assumption
        feed = self.feed(episodes)

        batch_size = len(episodes)
        max_steps = max([e.length for e in episodes])

        olproba = self.session.run(self.lproba, feed)
        feed[self.olproba] = olproba

        # Note: Goals for which there would be some non-zero reward
        goals = self.env.subgoals(episodes, self.subgoals_per_episode)

        loss = 0.
        grads = [0.]*len(self.variables)
        for goal in goals:
            feed[self.bvars.goals] = np.array([goal]*batch_size)

            rewards = np.zeros((batch_size, max_steps), dtype=np.float32)
            baselines = np.zeros((batch_size, max_steps - 1), dtype=np.float32)

            for i, e in enumerate(episodes):
                length = self.env.evaluate_length(e, goal)

                r = self.env.evaluate_rewards(e, goal)[:length]
                rewards[i, :length] = r

                v = self.baseline.evaluate(e, goal)[:length - 1]
                baselines[i, :length - 1] = v

            feed[self.bvars.rewards] = rewards
            feed[self.bvars.baselines] = baselines

            rlproba, agrad = self.session.run([self.rlproba, self.gradrlproba],
                                              feed)

            loss -= rlproba
            grads = [g - ag for (g, ag) in zip(grads, agrad)]

        self.session.run(self.train_op, dict(zip(self.grads, grads)))

        return loss
