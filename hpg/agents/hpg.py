import numpy as np
import tensorflow as tf

from hpg.agents.agent import PolicyNetworkAgent

from hpg.agents.baselines import ZeroBaseline
from hpg.agents.baselines import ValueBaseline


class HindsightPolicyGradient(PolicyNetworkAgent):
    def __init__(self, env, hidden_layers, learning_rate, baseline_parameters,
                 subgoals_per_episode, use_vscaling_init, use_gpu, seed):
        # Inelegant
        np.random.seed(seed)
        tf.set_random_seed(seed)
        # Workaround
        env.seed(seed)
        
        if baseline_parameters:
            baseline = ValueBaseline(env, **baseline_parameters)
        else:
            baseline = ZeroBaseline()
            
        self.subgoals_per_episode = subgoals_per_episode

        PolicyNetworkAgent.__init__(self, env=env, hidden_layers=hidden_layers,
                                    learning_rate=learning_rate,
                                    baseline=baseline,
                                    use_vscaling_init=use_vscaling_init,
                                    init_session=False, use_gpu=use_gpu)

    def setup(self):
        mask = tf.sequence_mask(self.bvars.lengths - 1, dtype=tf.float32)

        policy = self.policy[:, :-1]
        proba = tf.reduce_sum(policy*self.bvars.actions_enc, axis=2)
        self.lproba = tf.log(proba)*mask

        self.olproba = tf.placeholder(tf.float32, [None, None], name='olproba')

        lratios = tf.exp(tf.cumsum(self.lproba - self.olproba, axis=1,
                                   exclusive=False, reverse=False))*mask

        lratios += 1e-12
        lratios = lratios / tf.reduce_sum(lratios, axis=0)
        lratios = tf.stop_gradient(lratios)

        baselines = lratios*self.bvars.baselines
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
