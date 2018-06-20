import os

import numpy as np
import tensorflow as tf

import pandas as pd

import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import seaborn as sns

import sacred
sacred.SETTINGS.CAPTURE_MODE = 'sys'
from sacred import Experiment
from sacred.observers import FileStorageObserver

from hpg.environments.envs import make_env

from hpg.baselines import ZeroBaseline
from hpg.baselines import ValueBaseline

from hpg.pg import PolicyGradientAgent
from hpg.pg import HindsightGradientAgent

plt.switch_backend('Agg')

filepath = 'results/'

ex = Experiment()
ex.observers.append(FileStorageObserver.create(filepath))


def plot_returns(experiment_folder, xlabel, ylabel, values):
    plt.figure()
    sns.set(style='darkgrid')

    sns.tsplot(data=values, value=ylabel,
               time=pd.Series(np.arange(1, values.shape[1] + 1), name=xlabel),
               estimator=np.mean)

    fname = '{0}_{1}.png'.format(ylabel.replace(' ', '_'), xlabel)
    plt.savefig(os.path.join(experiment_folder, fname), dpi=300)


@ex.config
def config():
    seed = 1

    env_name = 'flipbit_6'
    max_steps = 7

    use_hindsight = True
    per_decision = True
    weighted = True
    subgoals_per_episode = 0

    policy_hidden_layers = [16]
    policy_learning_rate = 5e-3

    use_baseline = True
    baseline_hidden_layers = [16]
    baseline_learning_rate = 5e-3

    n_train_batches = 3000
    batch_size = 2
    eval_freq = 100
    eval_size = 128

    n_restarts = 2


@ex.automain
def main(seed, env_name, max_steps, use_hindsight, per_decision, weighted,
         subgoals_per_episode, policy_hidden_layers, policy_learning_rate,
         use_baseline, baseline_hidden_layers, baseline_learning_rate,
         n_train_batches, batch_size, eval_freq, eval_size, n_restarts):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    experiment_folder = os.path.join(filepath, str(ex.current_run._id))

    env = make_env(env_name, max_steps=max_steps)
    env.reset()
    env.render()

    if use_baseline:
        baseline = ValueBaseline(env, baseline_hidden_layers,
                                 baseline_learning_rate)
    else:
        baseline = ZeroBaseline()

    if use_hindsight:
        agent = HindsightGradientAgent(env, policy_hidden_layers,
                                       policy_learning_rate,
                                       per_decision, weighted,
                                       subgoals_per_episode, baseline,
                                       init_session=False)
    else:
        agent = PolicyGradientAgent(env, policy_hidden_layers,
                                    policy_learning_rate, baseline,
                                    init_session=False)

    ereturns, treturns, plosses, blosses = [], [], [], []

    for i in range(n_restarts):
        agent.init()

        stats = agent.train(n_train_batches, batch_size, eval_freq, eval_size)

        ereturns.append(stats[0])
        treturns.append(stats[1])
        plosses.append(stats[2])
        blosses.append(stats[3])

        agent.save(os.path.join(experiment_folder, 'agent_{0}.ckpt'.format(i)))

        agent.close()

    ereturns = np.array(ereturns)
    treturns = np.array(treturns)
    plosses = np.array(plosses)
    blosses = np.array(blosses)

    np.save(os.path.join(experiment_folder, 'ereturns.npy'), ereturns)
    np.save(os.path.join(experiment_folder, 'treturns.npy'), treturns)
    np.save(os.path.join(experiment_folder, 'plosses.npy'), plosses)
    np.save(os.path.join(experiment_folder, 'blosses.npy'), blosses)

    plot_returns(experiment_folder, 'evaluation', 'average return', ereturns)
