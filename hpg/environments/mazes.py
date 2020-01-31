import numpy as np
from scipy.sparse.csgraph import shortest_path


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

        return np.array(self.position), np.array(self.position), np.array(self.goal), 0.0

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

        return np.array(self.position), np.array(self.position), reward, done

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
    
    def seed(self, seed):
        pass
        
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


class EmptyRoom(Maze):
    def __init__(self, h, w, max_steps):
        Maze.__init__(self, np.ones((h, w), dtype=np.int), max_steps, entries=[(0, 0)])


class FourRooms(Maze):
    def __init__(self, max_steps):
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
        
        Maze.__init__(self, layout, max_steps, entries, epsilon=0.2)


def play(maze, show_observations=True, show_rewards=True):
    udlr = ['8', '2', '4', '6']

    obs, _, goal, r = maze.reset()

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

        obs, _, r, done = maze.step(udlr.index(c))

    print('State:')
    maze.render()
    if show_observations:
        print('Observation:\n{0}.'.format(obs))
    if show_rewards:
        print('Reward: {0}.'.format(r))


def main():
    seed = 0
    np.random.seed(seed)
    maze = FourRooms(32)
    play(maze)


if __name__ == "__main__":
    main()
