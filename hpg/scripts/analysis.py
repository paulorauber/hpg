import os
import argparse
import json
from collections import namedtuple

import seaborn as sns
import pandas as pd


def log(msg, logfile):
    print(msg)
    print(msg, file=logfile)


def plot(i, exp, df, directory):
    sns.set_style('darkgrid')
            
    ax = sns.lineplot(x='evaluation step', y='average return',
                      hue='method', data=df)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.set_title(json.dumps(exp, sort_keys=True))
    ax.set_xlim(xmin=0)
    
    ax.figure.savefig(os.path.join(directory, 'experiment_{0}.pdf'.format(i)),
                      bbox_inches='tight', dpi=300)
    
    ax.figure.clf()


def average_performance(df):
    dff = df[['method','seed','average return']]
    means_per_seed = dff.groupby(['method', 'seed'], as_index=False).mean()
    
    dff = means_per_seed[['method','average return']]
    means = dff.groupby('method').mean()
    stds = dff.groupby('method').std(ddof=0)
    
    df_ap = means.rename(columns={'average return': 'average performance'})
    df_ap['standard deviation'] = stds['average return']
    
    df_ap = df_ap.sort_values(by='average performance', ascending=False)

    return df_ap


Experiment = namedtuple('Experiment', ['environment', 'environment_parameters',
                                       'training_parameters'])


Method = namedtuple('Method', ['agent', 'agent_parameters'])


def main():
    parser = argparse.ArgumentParser(usage='Result analysis.')

    parser.add_argument('directory')
    parser.add_argument('-o', '--omit_plots', action='store_true')

    args = parser.parse_args()

    logfile = open(os.path.join(args.directory, 'average_performance.txt'), 'w')

    dirs = [os.path.join(args.directory, d)
            for d in os.listdir(args.directory) if not d.startswith('_')]

    dirs = [d for d in dirs if os.path.isdir(d)]

    dir_to_config = {}
    exps = []
    exp_to_dirs = []
    
    for d in dirs:
        if not os.path.exists(os.path.join(d, 'eval.csv')):
            log('Warning: Skipping {0}.'.format(d), logfile)
        else:
            f = open(os.path.join(d, 'config.json'))
            config = json.load(f)
            f.close()
            
            dir_to_config[d] = config
            
            exp = Experiment(config['environment'],
                             config['environment_parameters'],
                             config['training_parameters'])
            
            try:
                index = exps.index(exp)
            except ValueError:
                index = -1
                exps.append(exp)
                exp_to_dirs.append([])
                
            exp_to_dirs[index].append(d)
            
    for i, exp in enumerate(exps):
        log('# '+ json.dumps(exp, sort_keys=True), logfile)
        
        dirs = exp_to_dirs[i]
        configs = [dir_to_config[d] for d in dirs]
        
        df = pd.DataFrame()
        
        for config, d in zip(configs, dirs):
            seed = config['agent_parameters'].pop('seed')
            method = Method(config['agent'], config['agent_parameters'])
            df_eval = pd.read_csv(os.path.join(d, 'eval.csv'))
            
            df_cfg = pd.DataFrame({'method': json.dumps(method, sort_keys=True),
                                   'seed': seed,
                                   'evaluation step': range(len(df_eval)),
                                   'average return': df_eval['Average Return (e)']})
            
            df = df.append(df_cfg, ignore_index=True)

        df.to_csv(os.path.join(args.directory, 'experiment_{0}.csv'.format(i)),
                  index=False)

        df_ap = average_performance(df)
        with pd.option_context('max_colwidth', -1):
            log(str(df_ap), logfile)

        if not args.omit_plots:
            plot(i, exp, df, args.directory)

    logfile.close()


if __name__ == "__main__":
    main()
