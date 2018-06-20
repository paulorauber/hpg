import re
import numpy as np
from scipy.sparse.csgraph import shortest_path


class FlipBit:
    def __init__(self, n_bits, max_steps):
        self.n_actions = n_bits
        self.d_observations = n_bits
        self.d_goals = n_bits

        self.max_steps = max_steps

    def reset(self):
        self.n_steps = 1

        self.state = np.zeros(self.d_observations)
        self.goal = np.random.randint(0, 2, size=self.d_goals)

        state, goal = np.array(self.state), np.array(self.goal)

        return state, goal, 0.0

    def step(self, a):
        if a >= self.n_actions:
            raise Exception('Invalid action')

        self.n_steps += 1

        self.state[a] = 1 - self.state[a]

        if np.allclose(self.state, self.goal):
            reward = self.max_steps - self.n_steps + 1
        else:
            reward = 0.0

        done = (self.max_steps <= self.n_steps) or (reward > 0.0)

        return np.array(self.state), reward, done

    def evaluate_length(self, episode, goal):
        for t in range(1, episode.length):
            if np.allclose(episode.observations[t], goal):
                return t + 1

        return episode.length

    def evaluate_rewards(self, episode, goal):
        rewards = np.zeros(episode.length)

        for t in range(1, episode.length):
            if np.allclose(episode.observations[t], goal):
                rewards[t] = self.max_steps - t

        return rewards

    def evaluate_values(self, episode, goal):
        values = np.zeros(episode.length - 1)

        start = 1
        if np.allclose(episode.observations[0], goal):
            values[0] = self.max_steps - 2
        else:
            start = 0

        for t in range(start, len(values)):
            moves = np.sum((episode.observations[t] - goal)**2)
            values[t] = max(self.max_steps - moves - t, 0)

        return values

    def subgoals(self, episodes, subgoals_per_episode):
        goals = [e.observations for e in episodes]
        if subgoals_per_episode > 0:
            goals = []
            for e in episodes:
                observations = np.unique(e.observations, axis=0)

                size = min(subgoals_per_episode, observations.shape[0])
                indices = np.random.choice(observations.shape[0], size, False)
                goals.append(observations[indices])

        return np.unique(np.concatenate(goals, axis=0), axis=0)

    def render(self):
        print(self.__repr__())

    def __repr__(self):
        return 'State: {0}. Goal: {1}.'.format(self.state, self.goal)


class Maze:
    def __init__(self, layout, max_steps, entries, exits=None, epsilon=0.0):
        self.layout = np.array(layout, dtype=np.int)
        validr, validc = np.nonzero(self.layout)
        self.valid_positions = set(zip(validr, validc))

        self.max_steps = max_steps

        self.entries = set(entries)

        self.exits = self.valid_positions - self.entries
        if exits is not None:
            self.exits = set(exits)

        self.epsilon = epsilon

        self.check_consistency()
        self.compute_distance_matrix()

        self.n_actions = 4
        self.d_observations = 2
        self.d_goals = 2

    def check_consistency(self):
        given = self.entries.union(self.exits)

        if not given.issubset(self.valid_positions):
            raise Exception('Invalid entry or exit.')

        if len(self.entries.intersection(self.exits)) > 0:
            raise Exception('Entries and exits must be disjoint.')

    def compute_distance_matrix(self):
        shape = self.layout.shape
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        adj_matrix = np.zeros((self.layout.size, self.layout.size))

        for (r, c) in self.valid_positions:
            index = np.ravel_multi_index((r, c), shape)

            for move in moves:
                nr, nc = r + move[0], c + move[1]

                if (nr, nc) in self.valid_positions:
                    nindex = np.ravel_multi_index((nr, nc), shape)
                    adj_matrix[index, nindex] = 1

        self.dist_matrix = shortest_path(adj_matrix)

    def distance(self, orig, dest):
        shape = self.layout.shape

        o_index = np.ravel_multi_index((int(orig[0]), int(orig[1])), shape)
        d_index = np.ravel_multi_index((int(dest[0]), int(dest[1])), shape)

        distance = self.dist_matrix[o_index, d_index]
        if not np.isfinite(distance):
            raise Exception('There is no path between origin and destination.')

        return distance

    def reset(self):
        self.n_steps = 1

        i = np.random.choice(len(self.entries))
        self.position = sorted(self.entries)[i]

        i = np.random.choice(len(self.exits))
        self.goal = sorted(self.exits)[i]

        return np.array(self.position), np.array(self.goal), 0.0

    def step(self, a):
        """a: up, down, left, right"""
        if a >= self.n_actions:
            raise Exception('Invalid action')

        if np.random.random() < self.epsilon:
            a = np.random.choice(self.n_actions)

        self.n_steps += 1

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        newr = self.position[0] + moves[a][0]
        newc = self.position[1] + moves[a][1]

        if (newr, newc) in self.valid_positions:
            self.position = (newr, newc)

        if self.position == self.goal:
            reward = self.max_steps - self.n_steps + 1
        else:
            reward = 0.0

        done = (self.max_steps <= self.n_steps) or (reward > 0.0)

        return np.array(self.position), reward, done

    def evaluate_length(self, episode, goal):
        for t in range(1, episode.length):
            if np.allclose(episode.observations[t], goal):
                return t + 1

        return episode.length

    def evaluate_rewards(self, episode, goal):
        rewards = np.zeros(episode.length)

        for t in range(1, episode.length):
            if np.allclose(episode.observations[t], goal):
                rewards[t] = self.max_steps - t

        return rewards

    def evaluate_values(self, episode, goal):
        values = np.zeros(episode.length - 1)

        for t in range(len(values)):
            moves = self.distance(episode.observations[t], goal)
            values[t] = max(self.max_steps - moves - t, 0)

        return values

    def subgoals(self, episodes, subgoals_per_episode):
        goals = [e.observations for e in episodes]
        if subgoals_per_episode > 0:
            goals = []
            for e in episodes:
                observations = np.unique(e.observations, axis=0)

                size = min(subgoals_per_episode, observations.shape[0])
                indices = np.random.choice(observations.shape[0], size, False)
                goals.append(observations[indices])

        return np.unique(np.concatenate(goals, axis=0), axis=0)

    def render(self):
        print(self.__repr__())

    def __repr__(self):
        s = []

        for i in range(len(self.layout)):
            for j in range(len(self.layout[0])):
                if (i, j) == self.position:
                    s.append('@')
                elif (i, j) == self.goal:
                    s.append('$')
                else:
                    s.append('.' if self.layout[i, j] else '#')
            s.append('\n')

        return ''.join(s)


def make_empty_maze(h, w, max_steps):
    return Maze(np.ones((h, w), dtype=np.int), max_steps, entries=[(0, 0)])


def make_random_layout(h, w):
    """Adapted from https://rosettacode.org/wiki/Maze_generation."""
    maze_string = ''

    vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
    ver = [["| "] * w + ['|'] for _ in range(h)] + [[]]
    hor = [["+-"] * w + ['+'] for _ in range(h + 1)]

    def walk(x, y):
        vis[y][x] = 1

        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        np.random.shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]:
                continue
            if xx == x:
                hor[max(y, yy)][x] = "+ "
            if yy == y:
                ver[y][max(x, xx)] = "  "
            walk(xx, yy)

    walk(np.random.randint(w), np.random.randint(h))
    for (a, b) in zip(hor, ver):
        maze_string += ''.join(a + ['\n'] + b) + '\n'

    A = [[]]
    for c in maze_string[: -2]:
        if c == '\n':
            A.append([])
        elif c == ' ':
            A[-1].append(1)
        else:
            A[-1].append(0)

    return np.array(A, dtype=np.int)


def make_random_maze(h, w, max_steps):
    return Maze(make_random_layout(h, w), max_steps, [(1, 1)])


def make_tmaze(length, max_steps):
    layout = np.zeros(shape=(3, length+1), dtype=np.int)

    layout[:, 0] = 1
    layout[1, :] = 1
    layout[:, -1] = 1

    return Maze(layout, max_steps, [(0, 0)])


def make_cheese_maze(length, max_steps):
    """Adapted from Bakker, Pieter Bram. The state of mind. 2004, pg. 155"""
    layout = np.ones(shape=(length, 5), dtype=np.int)

    layout[1:, 1] = 0
    layout[1:, 3] = 0

    return Maze(layout, max_steps, [(length - 1, 2)])


def make_wine_maze(max_steps):
    """Adapted from Bakker, Pieter Bram. The state of mind. 2004, pg. 155"""
    layout = np.array([[0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0, 1, 1],
                       [0, 1, 0, 1, 0, 1, 0],
                       [1, 1, 0, 1, 0, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0]], dtype=np.int)

    return Maze(layout, max_steps, [(1, 0)])


def make_four_rooms_maze(max_steps):
    """Adapted from Sutton et al. Between MDPs and semi-MDPs: ... (1999)"""
    layout = np.ones(shape=(11, 11), dtype=np.int)

    # Walls
    layout[:, 5] = 0
    layout[5, :5] = 0
    layout[6, 6:] = 0

    # Doors
    layout[5, 1] = 1
    layout[2, 5] = 1
    layout[6, 8] = 1
    layout[9, 5] = 1

    # Average distance to exit is close to 10
    entries = [(0, 0), (0, 10), (10, 0), (10, 10)]

    return Maze(layout, max_steps, entries, epsilon=0.2)


def make_env(env_string, max_steps):
    env = None

    match = re.match('flipbit_(\d+)', env_string)
    if match:
        env = FlipBit(n_bits=int(match.group(1)), max_steps=max_steps)

    match = re.match('empty_maze_(\d+)_(\d+)', env_string)
    if match:
        env = make_empty_maze(h=int(match.group(1)), w=int(match.group(2)),
                              max_steps=max_steps)

    match = re.match('random_maze_(\d+)_(\d+)', env_string)
    if match:
        env = make_random_maze(h=int(match.group(1)), w=int(match.group(2)),
                               max_steps=max_steps)

    match = re.match('tmaze_(\d+)', env_string)
    if match:
        env = make_tmaze(length=int(match.group(1)), max_steps=max_steps)

    match = re.match('cheese_maze_(\d+)', env_string)
    if match:
        env = make_cheese_maze(length=int(match.group(1)), max_steps=max_steps)

    match = re.match('wine_maze', env_string)
    if match:
        env = make_wine_maze(max_steps=max_steps)

    match = re.match('four_rooms_maze', env_string)
    if match:
        env = make_four_rooms_maze(max_steps=max_steps)

    if env is None:
        raise Exception('Invalid environment string.')

    return env


def play(maze, show_observations=True, show_rewards=True):
    udlr = ['8', '2', '4', '6']

    obs, goal, r = maze.reset()

    if show_observations:
        print('Goal: {0}.\n'.format(goal))

    done = False

    while not done:
        print('State:')
        maze.render()

        if show_observations:
            print('Observation: {0}.'.format(obs))
        if show_rewards:
            print('Reward: {0}.'.format(r))

        c = input('\nMove:')
        if c not in udlr:
            raise Exception('Invalid action')

        print('')

        obs, r, done = maze.step(udlr.index(c))

    print('State:')
    maze.render()
    if show_observations:
        print('Observation:\n{0}.'.format(obs))
    if show_rewards:
        print('Reward: {0}.'.format(r))


def main():
    np.random.seed(0)
    maze = make_env('four_rooms_maze', 32)
    play(maze)


if __name__ == "__main__":
    main()
