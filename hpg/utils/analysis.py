import argparse
import os
import json

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def log(msg, logfile):
    print(msg)
    print(msg, file=logfile)


def smooth_returns(returns, window_size):
    if window_size == 1:
        return np.array(returns)

    smoothed = []
    for j in range(returns.shape[0]):
        row = pd.rolling_mean(returns[j], window=window_size)
        smoothed.append(row[window_size - 1:])

    return np.array(smoothed)


def plot_returns(env, env_to_dir, dir_to_config, diff, directory, colors,
                 window_size, logfile, omit_plots):
    if not omit_plots:
        plt.figure()

    stats = {}
    for j, d in enumerate(env_to_dir[env]):
        returns = np.load(os.path.join(d, 'ereturns.npy'))

        condition = ''
        for param in diff:
            condition += '{0}_{1}_'.format(param, dir_to_config[d][param])
        condition = condition[:-1]

        mean_per_restart = np.mean(returns, axis=1)
        mean = np.mean(mean_per_restart)
        std = np.std(mean_per_restart)

        stats[(condition, d)] = (mean, std)

        if not omit_plots:
            returns = smooth_returns(returns, window_size)
            sns.tsplot(data=returns, value='average return',
                       time=pd.Series(np.arange(0, returns.shape[1]) +
                                      window_size, name='evaluation step'),
                       condition=condition, estimator=np.mean, ci=95,
                       color=colors[j])

    results = sorted(stats.keys(), key=lambda c: stats[c][0] - stats[c][1])
    for (condition, d) in reversed(results):
        mean, std = stats[(condition, d)]
        d = os.path.split(d)[1]
        log('\t({0}) {1}: Mean return: {2:.5f} ± {3:.5f}.'.format(d, condition,
            mean, std), logfile)

    if not omit_plots:
        legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.))

        plt.savefig(os.path.join(directory, 'ereturns_' + env + '.png'),
                    bbox_inches='tight', additional_artists=[legend], dpi=300)


def main():
    parser = argparse.ArgumentParser(usage='Result analysis.')

    parser.add_argument('directory')
    parser.add_argument('-o', '--omit_plots', action='store_true')
    parser.add_argument('-w', '--window-size', type=int, default=1)

    args = parser.parse_args()

    logfile = open(os.path.join(args.directory, 'analysis.txt'), 'w')

    dirs = [os.path.join(args.directory, d)
            for d in os.listdir(args.directory) if not d.startswith('_')]

    dirs = [d for d in dirs if os.path.isdir(d)]

    env_to_dir = {}
    dir_to_config = {}

    for d in dirs:
        if not os.path.exists(os.path.join(d, 'ereturns.npy')):
            log('Warning: Skipping {0}.'.format(d), logfile)
        else:
            f = open(os.path.join(d, 'config.json'))
            config = json.load(f)
            f.close()

            if config['env_name'] in env_to_dir:
                env_to_dir[config['env_name']].append(d)
            else:
                env_to_dir[config['env_name']] = [d]

            dir_to_config[d] = config

    sns.set_style('darkgrid')
    sns.set(rc={'lines.linewidth': 1.0, 'lines.markersize': 1.0})

    for env in env_to_dir.keys():
        log('\nEnvironment: {0}.'.format(env), logfile)

        configs = [dir_to_config[d] for d in env_to_dir[env]]

        colors = sns.color_palette()
        if len(configs) > 6:
            colors = sns.color_palette('hls', len(configs))

        for config in configs[1:]:
            if config.keys() != configs[0].keys():
                raise Exception('The configuration files do not match.')

        merged_configs = {k: set() for k in configs[0].keys()}
        for config in configs:
            for k, v in config.items():
                merged_configs[k].update([str(v)])

        diff = sorted([k for k in merged_configs.keys()
                       if len(merged_configs[k]) > 1])

        plot_returns(env, env_to_dir, dir_to_config, diff, args.directory,
                     colors, args.window_size, logfile, args.omit_plots)

    logfile.close()


if __name__ == "__main__":
    main()
