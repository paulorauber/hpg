import itertools
import numpy as np
import gym


def generate_itoa_dict(
        bucket_values=[-0.33, 0, 0.33], valid_movement_direction=[1, 1, 1, 1]):
    """
    Set cartesian product to generate action combination
        spaces for the fetch environments
    valid_movement_direction: To set
    """
    action_space_extended = [bucket_values if m == 1 else [0]
                             for m in valid_movement_direction]
    return list(itertools.product(*action_space_extended))


class FetchReach:
    def __init__(self, max_steps=50,
                 action_mode="cart", action_buckets=[-1, 0, 1],
                 action_stepsize=1.0,
                 reward_mode="sparse"):
        """
        Parameters:
            action_mode {"cart","cartmixed","cartprod","impulse","impulsemixed"}
            action_stepsize: Step size of the action to perform.
                            Int for cart and impulse
                            List for cartmixed and impulsemixed
            action_buckets: List of buckets used when mode is cartprod
            reward_mode = {"sparse","dense"}

        Reward Mode:
            `sparse` rewards are like the standard HPG rewards.
            `dense` rewards (from the paper/gym) give -(distance to goal) at every timestep.

        Modes:
            `cart` is for manhattan style movement where an action moves the arm in one direction
                for every action.

            `impulse` treats the action dimensions as velocity and adds/decreases
                the velocity by action_stepsize depending on the direction picked.
                Adds current direction
                velocity to state


            `impulsemixed` and `cartmixed` does the above with multiple magnitudes of action_stepsize.
                Needs the action_stepsize as a list.

            `cartprod` takes combinations of actions as input
        """

        try:
            self.env = gym.make("FetchReach-v1")
        except Exception as e:
            print(
                "You do not have the latest version of gym (gym-0.10.5). Falling back to v0 with movable table")
            self.env = gym.make("FetchReach-v0")

        self.action_directions = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.valid_action_directions = np.float32(
            np.any(self.action_directions, axis=0))

        self.action_mode = action_mode

        self.n_actions = self.generate_action_map(
            action_buckets, action_stepsize)
        self.d_observations = 10 + 4 * \
            (action_mode == "impulse" or action_mode == "impulsemixed")
        self.d_goals = 3

        self.max_steps = max_steps
        self.reward_mode = reward_mode

    def generate_action_map(self, action_buckets, action_stepsize=1.):

        action_directions = self.action_directions
        if self.action_mode == "cart" or self.action_mode == "impulse":
            assert isinstance(action_stepsize, float)
            self.action_space = np.vstack(
                (action_directions * action_stepsize, -action_directions * action_stepsize))

        elif self.action_mode == "cartmixed" or self.action_mode == "impulsemixed":
            assert isinstance(action_stepsize, list)
            action_space_list = []
            for ai in action_stepsize:
                action_space_list += [action_directions * ai,
                                      -action_directions * ai]
            self.action_space = np.vstack(action_space_list)

        elif self.action_mode == "cartprod":
            self.action_space = generate_itoa_dict(
                action_buckets, self.valid_action_directions)

        return len(self.action_space)

    def seed(self, seed):
        self.env.seed()

    def action_map(self, action):
        # If the modes are direct, just map the action as an index
        # else, accumulate them

        if self.action_mode in ["cartprod", "cart", "cartmixed"]:
            return self.action_space[action]
        else:
            self.action_vel += self.action_space[action]
            self.action_vel = np.clip(self.action_vel, -1, 1)
            return self.action_vel

    def reset(self):

        self.action_vel = np.zeros(4)  # Initialize/reset

        self.n_steps = 1
        obs = self.env.reset()

        self.state = obs["observation"]
        self.state_enc = obs["achieved_goal"]
        self.goal = obs["desired_goal"]

        if self.action_mode == "impulse" or self.action_mode == "impulsemixed":
            self.state = np.hstack((self.state, self.action_vel))

        state, state_enc, goal = np.array(
            self.state), np.array(
            self.state_enc), np.array(
            self.goal)

        return state, state_enc, goal, 0.0

    def goal_distance(self, goal_a, goal_b):
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def step(self, a):
        if a >= self.n_actions:
            raise Exception('Invalid action')
        self.n_steps += 1

        action_vec = self.action_map(a)
        obs, reward, done, info = self.env.step(action_vec)

        self.state = obs["observation"]
        self.state_enc = obs["achieved_goal"]

        if self.action_mode == "impulse" or self.action_mode == "impulsemixed":
            self.state = np.hstack(
                (self.state, np.clip(self.action_vel, -1, 1)))

        reached_goal = False

        sparsereward = (self.reward_mode == "sparse")
        if self.env.env._is_success(self.state_enc, self.goal):
            reward = (
                self.max_steps -
                self.n_steps +
                1) if sparsereward else 0.0
            reached_goal = True
        else:
            reward = 0.0 if sparsereward else (
                -self.goal_distance(self.state_enc, self.goal))

        done = (self.max_steps <= self.n_steps) or reached_goal

        return np.array(self.state), np.array(self.state_enc), reward, done

    # def evaluate(self, episode, goal):
    #     rewards = np.zeros(episode.length)

    #     for t in range(1, episode.length):
    #         if self.reward_mode=="sparse":
    #             if self.env.env._is_success(episode.observations_enc[t],goal):
    #                 rewards[t] = self.max_steps - t
    #                 return rewards
    #         else:
    #             if self.env.env._is_success(episode.observations_enc[t],goal):
    #                 rewards[t] = 0.0
    #                 return rewards
    #             else:
    #                 rewards[t] = -self.goal_distance(episode.observations_enc[t],goal)
    #     return rewards

    def evaluate_length(self, episode, goal):
        for t in range(1, episode.length):
            if self.env.env._is_success(episode.observations_enc[t], goal):
                return t + 1

        return episode.length

    def evaluate_rewards(self, episode, goal):
        rewards = np.zeros(episode.length)

        for t in range(1, episode.length):
            if self.env.env._is_success(episode.observations_enc[t], goal):
                rewards[t] = self.max_steps - t

        return rewards

    def subgoals(self, episodes, subgoals_per_episode):
        goals = [e.observations_enc for e in episodes]
        if subgoals_per_episode > 0:
            goals = []
            for e in episodes:
                observations_enc = np.unique(e.observations_enc, axis=0)

                size = min(subgoals_per_episode, observations_enc.shape[0])
                indices = np.random.choice(
                    observations_enc.shape[0], size, False)
                goals.append(observations_enc[indices])

        return np.unique(np.concatenate(goals, axis=0), axis=0)

    def render(self, **args):
        return
        # The args parameter was to pass close=True to gym that's supposed to close the window.
        # Unfortunately, that doesn't work.
        # self.env.render(*args)
        # else:
        # print(self.__repr__())

    def __repr__(self):
        return 'State: {0}. Goal: {1}.'.format(self.state, self.goal)

    def __del__(self):
        self.env.close()


class FetchPush(FetchReach):
    def __init__(self, max_steps=50,
                 action_mode="impulsemixed", action_buckets=[-1, 0, 1],
                 action_stepsize=[0.1, 1.0],
                 reward_mode="sparse"):

        try:
            self.env = gym.make("FetchPush-v1")
        except Exception as e:
            print(
                "You do not have the latest version of gym (gym-0.10.5). Falling back to v0 with movable table")
            self.env = gym.make("FetchPush-v0")

        self.action_directions = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.valid_action_directions = np.float32(
            np.any(self.action_directions, axis=0))

        self.action_mode = action_mode

        self.n_actions = self.generate_action_map(
            action_buckets, action_stepsize)
        self.d_observations = 25 + 4 * \
            (action_mode == "impulse" or action_mode == "impulsemixed")
        self.d_goals = 3

        self.max_steps = max_steps
        self.reward_mode = reward_mode


class FetchSlide(FetchReach):
    def __init__(self, max_steps=50,
                 action_mode="cart", action_buckets=[-1, 0, 1],
                 action_stepsize=1.0,
                 reward_mode="sparse"):

        try:
            self.env = gym.make("FetchSlide-v1")
        except Exception as e:
            print(
                "You do not have the latest version of gym (gym-0.10.5). Falling back to v0 with movable table")
            self.env = gym.make("FetchSlide-v0")

        self.action_directions = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.valid_action_directions = np.float32(
            np.any(self.action_directions, axis=0))

        self.action_mode = action_mode

        self.n_actions = self.generate_action_map(
            action_buckets, action_stepsize)
        self.d_observations = 25 + 4 * \
            (action_mode == "impulse" or action_mode == "impulsemixed")
        self.d_goals = 3

        self.max_steps = max_steps
        self.reward_mode = reward_mode
