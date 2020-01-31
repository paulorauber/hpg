import os
import time
import datetime
import argparse
import json

import ray

from hpg.scripts.run import run
from hpg.scripts.run import write


@ray.remote
def remote_run(config):
    return run(config)


def main():
    """ Run multiple experiments in parallel.
    
    In order to optionally distribute experiments across machines:
        * Ensure modules are compatible and available across machines.
        * Start the head node: ``ray start --head --redis-port=6379``
        * Connect additional nodes: ``ray start --redis-address <head address>``
        * Run this script on the head node with the corresponding address.
    """
    parser = argparse.ArgumentParser('Run multiple experiments.')
    parser.add_argument('directory', help='Experiments directory.')
    parser.add_argument('--address', default=None, help='Ray head address.')
    args = parser.parse_args()
    
    dirs = [os.path.join(args.directory, d)
            for d in os.listdir(args.directory) if not d.startswith('_')]
    dirs = [d for d in dirs if os.path.isdir(d)]
    
    start_time = time.time()
    ray.init(redis_address=args.address)
    
    id_to_dir = {}
    for d in dirs:
        f = open(os.path.join(d, 'config.json'), 'r')
        config = json.load(f)
        f.close()
        
        id_to_dir[remote_run.remote(config)] = d
        
    unready = list(id_to_dir.keys())
    while unready:
        ready, unready = ray.wait(unready)
        
        for data_id in ready:
            data = ray.get(data_id)
            write(data, id_to_dir[data_id])

    elapsed = datetime.timedelta(seconds=time.time() - start_time)
    print('Success. Time elapsed: {0}.'.format(elapsed))
    

if __name__ == "__main__":
    main()