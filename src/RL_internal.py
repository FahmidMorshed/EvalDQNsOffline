import gym
import numpy as np
import pandas as pd
import gym_gridworld
import random

random_state = 0
np.random.seed(random_state)
random.seed(random_state)

env = gym.make('gridworld-v0', deterministic=False)## False for non deterministic
action_size = 2
episodes = 10000 #10000
df = pd.DataFrame(columns=['episode_id', 'transition_id', 'state', 'action', 'immediate_reward',
                           'delayed_reward', 'done', 'next_state'])

for ep in range(episodes):
    state = env.reset()
    done = False
    delayed_reward = 0
    transition_id = 0
    while not done:
        #         env.render()
        action = np.random.choice(range(action_size))
        next_state, reward, done, info = env.step(action)
        delayed_reward += reward

        if done:
            if ep % 100 == 0:
                print("Episode:", ep, "| Total Reward:", round(delayed_reward, 2))
            df = df.append(
                {'episode_id': ep, 'transition_id': transition_id, 'state': np.array(state), 'action': action,
                 'immediate_reward': reward, 'delayed_reward': delayed_reward, 'done': done,
                 'next_state': np.array(next_state)},
                ignore_index=True)
            break

        df = df.append({'episode_id': ep, 'transition_id': transition_id, 'state': np.array(state), 'action': action,
                        'immediate_reward': reward, 'delayed_reward': 0, 'done': done,
                        'next_state': np.array(next_state)},
                       ignore_index=True)
        transition_id += 1
        state = next_state

df.to_pickle('../data/gridworld_ndm_10k.pkl') ## ndm instead of dm
df
