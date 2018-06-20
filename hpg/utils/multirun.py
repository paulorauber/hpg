import os
import time
import datetime
import argparse


def main():
    if 'TMUX' not in os.environ:
        raise Exception('This script should be called from a tmux session.')

    parser = argparse.ArgumentParser(usage='Hyperparameter search execution.')

    parser.add_argument('filepath')
    parser.add_argument('jobs', type=int)
    parser.add_argument('-v', '--venv', default='../env_hpg/bin/activate')

    args = parser.parse_args()

    f = open(args.filepath)
    cmds = [line.strip() for line in f.readlines()]
    f.close()

    dname, fname = os.path.split(args.filepath)
    fname = os.path.splitext(fname)[0]

    run = os.system

    tokens = []
    for j in range(args.jobs):
        tokens.append(os.path.join(dname, '.token_{0}_{1}'.format(j, fname)))
        run('touch {0}'.format(tokens[-1]))

    src = 'source {0}'.format(args.venv)

    start_time = time.time()
    for j, cmd in enumerate(cmds, 1):
        free_tokens = []
        while len(free_tokens) < 1:
            free_tokens = list(filter(os.path.exists, tokens))
            time.sleep(1)

        os.remove(free_tokens[0])

        run('tmux new-window -d -t {0}'.format(j))

        run('tmux send-keys -t {0} \"{1}\" C-m'.format(j, src))
        run('tmux send-keys -t {0} \"{1}\" C-m'.format(j, cmd))

        touch = 'touch {0}'.format(free_tokens[0])
        run('tmux send-keys -t {0} \"{1}\" C-m'.format(j, touch))

        run('tmux send-keys -t {0} \"exit\" C-m'.format(j))

        print('Running command {0}/{1}.'.format(j, len(cmds)))

    free_tokens = []
    while len(free_tokens) < args.jobs:
        free_tokens = list(filter(os.path.exists, tokens))
        time.sleep(1)

    for token in tokens:
        os.remove(token)

    elapsed = datetime.timedelta(seconds=time.time() - start_time)
    print('Success. Time elapsed: {0}.'.format(elapsed))


if __name__ == "__main__":
    main()
