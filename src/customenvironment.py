import gym
import random
import csv
import pandas as pd
import numpy as np
import gym_gridworld

env = gym.make('gridworld-v0')
observation_space = env.observation_space.shape[0]
action_space = env.action_space
print(observation_space, action_space)


array_of_values = []
episode_id = 0
fieldname = ['episode_id', 'transition_id', 'current state', 'action', 'reward', 'delayed_reward', 'done', 'next_state', 'info']
filename = "GridWorld-v0_20k.csv"
filename_pkl = "GridWorld-v0_20k.pkl"

for _ in range(20000):
    state = env.reset()
    episode_id += 1
    print(episode_id)
    transition_id = -1
    delayed_reward = 0
    terminal = False
    while terminal is False:
        transition_id += 1
        action = random.randint(0,3)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        delayed_reward += reward
        if terminal:
            state_info = {
                "episode_id": episode_id,
                "transition_id": transition_id,
                "state": tuple(state),
                "action": action,
                "immediate_reward": reward,
                "delayed_reward":delayed_reward,
                "done": terminal,
                "next_state": tuple(state_next),
                "info": info
            }
            array_of_values.append(state_info)
            break
        else:
            state_info = {
                "episode_id": episode_id,
                "transition_id": transition_id,
                "state": tuple(state),
                "action": action,
                "immediate_reward": reward,
                "delayed_reward": 0.0,
                "done": terminal,
                "next_state": tuple(state_next),
                "info": info
            }
            array_of_values.append(state_info)
            state = state_next


pd.DataFrame(array_of_values).to_csv(filename, index=False)
pd.DataFrame(array_of_values).to_pickle(filename_pkl)
