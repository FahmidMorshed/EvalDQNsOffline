import gym
from gym import spaces
import numpy as np
import pandas as pd
from random import randint, random

STEP_COST = 0


class GridWorld(gym.Env):
    """A gym environment for pesudo experiment using a gridworld
    action: LEFT(0), RIGHT(1), UP(2), DOWN(3)
    space: (X, Y)
    reward and transitions are 'deterministic'
    TODO: NON DETERMINISTIC TRANSITIONS AND REWARDS
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_model='delayed', deterministic=True):
        self.reward_model = reward_model

        # 4 possible actions - Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # The input variables that an agent can observe before making a choice -
        #   the agent only knows where they are
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]),
                                            high=np.array([9.0, 9.0]), dtype=np.int32)

        self.current_state = (0, 0)
        self.current_action = -1
        self.attempted_action = -1

        self.world = np.zeros((10, 10))
        self.total_reward = 0
        self.step_count = 0

        self.deterministic = deterministic
        self.stochastic_probability = 0.3
        self.random_action_chosen = False

    def create_world(self):
        self.world = np.zeros((10, 10))

        self.world[0, 7] = 2
        self.world[2, 2] = 5
        self.world[2, 8] = -4
        self.world[3, 6] = 7
        self.world[4, 0] = -3
        self.world[4, 9] = -9
        self.world[5, 3] = -4
        self.world[6, 7] = 9
        self.world[7, 0] = 4
        self.world[7, 7] = -5
        self.world[8, 3] = 16
        self.world[8, 9] = 6
        self.world[9, 9] = -3

    def reset(self):
        """
        Reset the state of the environment to an initial state
        If initial state has a reward, it counts that
        """
        self.create_world()
        self.current_state = (0, 0)
        self.current_action = -1
        self.random_action_chosen = False

        self.total_reward = 0
        self.step_count = 0

        return self.current_state

    def step(self, action):
        """
        Execute one time step within the environment

        Action Key:
            0 - Up
            1 - Right
            2 - Down
            3 - Left
        """
        self.step_count += 1

        # If we want non-determinism, the agent will randomly take a different action with some probability
        self.random_action_chosen = False
        self.attempted_action = action

        if not self.deterministic:
            if random() < self.stochastic_probability:
                action = randint(0, self.action_space.n - 1)
                self.random_action_chosen = True

        self.current_action = action

        if action == 0:
            self.current_state = (self.current_state[0], self.current_state[1] + 1)
        elif action == 1:
            self.current_state = (self.current_state[0] + 1, self.current_state[1])
        elif action == 2:
            self.current_state = (self.current_state[0], self.current_state[1] - 1)
        elif action == 3:
            self.current_state = (self.current_state[0] - 1, self.current_state[1])

        done = False
        if self.current_state[0] > 9 or self.current_state[0] < 0 or self.current_state[1] > 9 or self.current_state[
            1] < 0:
            done = True
            if 'delayed' in self.reward_model:
                reward = self.total_reward - 1
            else:
                reward = -1
            self.current_state = (-1, -1)

            return self.current_state, reward, done, {"immediate_reward": -1, "total rewards": self.total_reward,
                                                      "total steps": self.step_count,
                                                      "current state": self.current_state}

        reward = self.world[self.current_state[0], self.current_state[1]]
        self.world[self.current_state[0], self.current_state[1]] = 0

        self.total_reward += reward

        immediate_reward = reward
        if 'delayed' in self.reward_model:
            reward = 0

        return self.current_state, reward, done, {"immediate_reward": immediate_reward,
                                                  "total rewards": self.total_reward, "total steps": self.step_count,
                                                  "current state": self.current_state}

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        """
        # Render the environment to the screen

        print(f'Step: {self.step_count}')
        print(
            f'Action: {self.current_action}{", Random [Attempted " + str(self.attempted_action) + "]" if self.random_action_chosen else ""}')
        print(f'Current State: {self.current_state}')
        print(f'Total Reward: {self.total_reward}')
        print()

        return self.step_count, self.current_state, self.total_reward

    def seed(self, random_seed=1):
        np.random.seed(random_seed)