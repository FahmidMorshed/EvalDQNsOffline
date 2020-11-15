import gym
import random
import csv
import pandas as pd
import numpy as np

env = gym.make("CartPole-v1")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
array_of_values = []
episode_id = 0
fieldname = ['episode_id', 'transition_id', 'current state', 'action', 'reward', 'delayed_reward', 'done', 'next_state', 'info']
filename = "CartPole-v1_50k.csv"
filename_pkl = "CartPole-v1_50k.pkl"

"""""
with open(filename, "a") as csvfile:
#csvfile = open(filename, 'w')
    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldname)
    csvwriter.writeheader()
    for _ in range(25000):
            state = env.reset()
            episode_id += 1
            transition_id = 0
            delayed_reward = 0
            terminal = False
            while terminal is False:
                transition_id += 1
                action = random.randint(0,1)
                state_next, reward, terminal, info = env.step(action)
                reward = reward if not terminal else -reward
                delayed_reward += reward
                state_info = {
                    "episode_id": episode_id,
                    "transition_id": transition_id,
                    "current state": state,
                    "action": action,
                    "reward": reward,
                    "delayed_reward":0.0,
                    "done": terminal,
                    "next_state": state_next,
                    "info": info
                }
                #print(state_info)
                csvwriter.writerow(state_info)
                state = state_next
                if terminal:
                    state_info = {
                        "episode_id": episode_id,
                        "transition_id": transition_id,
                        "current state": state,
                        "action": action,
                        "reward": reward,
                        "delayed_reward":delayed_reward,
                        "done": terminal,
                        "next_state": state_next,
                        "info": info
                    }
                    #print(state_info)
                    csvwriter.writerow(state_info)
                    break
csvfile.close()

"""""
for _ in range(50000):
    state = env.reset()
    episode_id += 1
    print(episode_id)
    transition_id = -1
    delayed_reward = 0
    terminal = False
    while terminal is False:
        transition_id += 1
        action = random.randint(0,1)
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


#df2 = pd.read_csv('CartPole-v1_dummy.csv')
#print(type(df2.loc[0,'state']))
#print(type(df2['state'].values))