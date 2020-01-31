import argparse
import os
import json

from hpg.agents.agents import agents
from hpg.environments.environments import environments


def run(config):
    """ Example configuration file: 
    {
        "environment" : "FlipBit",
        "environment_parameters": {"n_bits" : 8, "max_steps": 9},
        
        "agent" : "HindsightPolicyGradient",
        "agent_parameters": {"hidden_layers" : [16],
                             "learning_rate" : 5e-3,
                             "baseline_parameters": {"hidden_layers" : [16],
                                                     "learning_rate" : 5e-3,
                                                     "use_vscaling_init": false},
                             "subgoals_per_episode": 0,
                             "use_vscaling_init": false,
                             "use_gpu": false,
                             "seed" : 1
                             },
        
        "training_parameters": {"n_batches" : 3000,
                                "batch_size" : 2,
                                "eval_freq" : 100,
                                "eval_size" : 128
                                }
    }
    """
    env = environments[config['environment']](**config['environment_parameters'])
    agent = agents[config['agent']](env, **config['agent_parameters'])
    
    agent.init()
    data = agent.train(**config['training_parameters'])
    agent.close()
    
    return data


def write(data, directory):
    df_train, df_eval = data

    df_train.to_csv(os.path.join(directory, 'train.csv'), index=False)
    df_eval.to_csv(os.path.join(directory, 'eval.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Experiment directory.')
    args = parser.parse_args()
    
    f = open(os.path.join(args.directory, 'config.json'), 'r')
    config = json.load(f)
    f.close()
    
    data = run(config)
    write(data, args.directory)


if __name__ == '__main__':
    main()