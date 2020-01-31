import os
import argparse
import json
import shutil


def main():
    """Create `n` replicas of each experiment in an experiments directory"""
    parser = argparse.ArgumentParser('Replicate experiments.')
    parser.add_argument('directory', type=str, help='Experiments directory.')
    parser.add_argument('n', type=int, help='Number of replicas.')
    
    args = parser.parse_args()
    dirs = [os.path.join(args.directory, d)
            for d in os.listdir(args.directory) if not d.startswith('_')]
    dirs = [d for d in dirs if os.path.isdir(d)]
    
    for d in dirs:
        f = open(os.path.join(d, 'config.json'))
        config = json.load(f)
        f.close()
        
        shutil.rmtree(d)
        
        for i in range(1, args.n + 1):
            path = d + '_{0}'.format(i)
            os.makedirs(path)
            config['agent_parameters']['seed'] = i
            
            f = open(os.path.join(path, 'config.json'), 'w')
            json.dump(config, f)
            f.close()
            
    print('Success.')


if __name__ == "__main__":
    main()