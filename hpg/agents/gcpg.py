import numpy as np
import tensorflow as tf

from hpg.agents.agent import PolicyNetworkAgent

from hpg.agents.baselines import ZeroBaseline
from hpg.agents.baselines import ValueBaseline


class GoalConditionalPolicyGradient(PolicyNetworkAgent):
    def __init__(self, env, hidden_layers, learning_rate,
                     baseline_parameters, use_vscaling_init, use_gpu, seed):
        # Inelegant
        np.random.seed(seed)
        tf.set_random_seed(seed)
        # Workaround
        env.seed(seed)
        
        if baseline_parameters:
            baseline = ValueBaseline(env, **baseline_parameters)
        else:
            baseline = ZeroBaseline()
        
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