import os
import argparse
from itertools import product
from collections import namedtuple


ExpConfig =\
     namedtuple('ExpConfig',
                ['env_name', 'max_steps', 'n_train_batches', 'batch_size',
                 'eval_freq', 'eval_size', 'n_restarts'])

ParamConfig =\
     namedtuple('ParamConfig',
                ['seed', 'policy_hidden_layers', 'policy_learning_rate',
                 'use_baseline', 'baseline_hidden_layers',
                 'baseline_learning_rate', 'use_hindsight', 'per_decision',
                 'weighted', 'subgoals_per_episode'])


def create_combinations(g):
    combos = map(lambda c: ParamConfig(*c), product(*g))

    def filter_layers(c):
        return c.policy_hidden_layers == c.baseline_hidden_layers

    # Note: assumes baseline_hidden_layers is tied to policy_hidden_layers
    def filter_baseline(c):
        if c.use_baseline:
            return True

        return c.baseline_learning_rate == g.baseline_learning_rate[0]

    def filter_hindsight(c):
        if c.use_hindsight:
            return True

        valid_pd = c.per_decision == g.per_decision[0]
        valid_w = c.weighted == g.weighted[0]
        valid_pe = c.subgoals_per_episode == g.subgoals_per_episode[0]

        return valid_pd and valid_w and valid_pe

    combos = filter(lambda c: filter_layers(c), combos)
    combos = filter(lambda c: filter_baseline(c), combos)
    combos = filter(lambda c: filter_hindsight(c), combos)

    return combos


def main():
    e_config = ExpConfig('flipbit_8', 9, 3000, 2, 100, 128, 5)

    g_config =\
        ParamConfig(seed=[1],
                    policy_hidden_layers=[[16, 16], [64, 64], [256, 256]],
                    policy_learning_rate=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4,
                                          1e-4, 5e-5],
                    use_baseline=[True, False],
                    baseline_hidden_layers=[[16, 16], [64, 64], [256, 256]],
                    baseline_learning_rate=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4,
                                            1e-4, 5e-5],
                    use_hindsight=[True, False],
                    per_decision=[True],
                    weighted=[True],
                    subgoals_per_episode=[0])

    parser = argparse.ArgumentParser(usage='Hyperparameter search definition.')
    parser.add_argument('-d', '--dirpath', default='results')

    args = parser.parse_args()

    if not os.path.exists(args.dirpath):
        os.makedirs(args.dirpath)

    items = e_config._asdict().items()
    fname = ['{0}_{1}_'.format(k, v) for (k, v) in items]
    fname = ''.join(fname)[:-1]

    f = open(os.path.join(args.dirpath, fname + '.txt'), 'w')

    prefix = 'python3 hpg/experiments/pg.py with '
    prefix += ''.join(['\'{0}={1}\' '.format(k, v) for (k, v) in items])

    combos = create_combinations(g_config)
    for i, combo in enumerate(combos):
        items = combo._asdict().items()
        config = ''.join(['\'{0}={1}\' '.format(k, v) for (k, v) in items])

        f.write(prefix + config + '\n')

    f.close()

    print('Success ({0} combinations).'.format(i + 1))


if __name__ == "__main__":
    main()
