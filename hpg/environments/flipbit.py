import numpy as np


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

        return state, state, goal, 0.0

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
        
        state = np.array(self.state)
        return state, state, reward, done

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
        return 'State: {0}. Goal: {1}.'.format(self.state, self.goal)
