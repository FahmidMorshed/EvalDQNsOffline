import gym
from gym import spaces
import numpy as np
import pandas as pd

STEP_COST = 0

class GridWorld(gym.Env):
    """A gym environment for pesudo experiment using a gridworld
    action: UP(0), RIGHT(1)
    space: (X, Y)
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, deterministic=True):
        
        self.deterministic=deterministic
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), 
                                            high=np.array([9.0, 9.0]), dtype=np.int32)

        

    def create_world(self):
        self.world = np.zeros((10,10))

        self.world[0,7] = 2
        self.world[2,2] = 5
        self.world[2,8] = -4
        self.world[3,6] = 7
        self.world[4,0] = -3
        self.world[4,9] = -9
        self.world[5,3] = -4
        self.world[6,7] = 9
        self.world[7,0] = 4
        self.world[7,7] = -5
        self.world[8,3] = 27
        self.world[8,9] = 6
        self.world[9,9] = -3

        
    def reset(self):
        """
        Reset the state of the environment to an initial state, randomly to a grid
        If initial state has a reward, it counts that
        """ 
        self.create_world()
        self.current_state = (0, 0)
        
        self.step_count = 0
        self.reward = 0
        self.total_reward = 0
        return self.current_state


    
    def step(self, action):
        self.step_count += 1
        if self.deterministic is False:
            if np.random.uniform() < 0.3:
                action = np.random.choice([0,1])

        if action == 0:
            self.current_state = (self.current_state[0], self.current_state[1] + 1)
        elif action == 1:
            self.current_state = (self.current_state[0]+1, self.current_state[1])
        
        self.reward = 0
        done = False
        if self.current_state[0] > 9 or self.current_state[0] < 0 or self.current_state[1] > 9 or self.current_state[1] < 0:
            done = True
            
            self.reward = -1
            self.current_state = (-1, -1)
            self.total_reward -= self.reward
            return self.current_state, self.reward, done, {"reward": self.reward, "total steps": self.step_count, "current state": self.current_state}

        
        self.reward = self.world[self.current_state[0], self.current_state[1]] - STEP_COST
        self.world[self.current_state[0], self.current_state[1]] = 0
        self.total_reward += self.reward

        return self.current_state, self.reward, done, {"reward": self.reward, "total steps": self.step_count, "current state": self.current_state}
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("Last Action:", self.action, "| Total Step:", self.step_count, "| Current State:", self.current_state, \
            "| Total Reward:", self.total_reward, "| Reward:", self.reward)
        

        return self.step_count, self.current_state, self.action, self.total_reward, self.reward
   
    def seed(self, random_seed=1):
        np.random.seed(random_seed)