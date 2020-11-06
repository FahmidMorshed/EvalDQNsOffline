#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import sys
import os
import pdb
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime

from keras.models import Sequential, Model
import keras.layers as layers
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.merge import _Merge, Multiply
import ast
import gym

random_state=0
np.random.seed(random_state)
random.seed(random_state)


# # Agent

# In[7]:


class QLayer(_Merge):
    """
    Q Layer that merges an advantage and value layer'''
    Needed for dueling dqn only
    """
    def _merge_function(self, inputs):
        '''Assume that the inputs come in as [value, advantage]'''
        output = inputs[0] + (inputs[1] - K.mean(inputs[1], axis=1, keepdims=True))
        return output

class DQNAgent:
    def __init__(self, df_batch, state_size, action_size, 
                 minibatch_size=64, gamma=.9, lr=0.0001, units=128, hidden_layers=1,
                 dueling=False, double_param=0, priority_alpha=0,
                 copy_online_to_target_ep=100, eval_after=100):
        
        # NOT FOR NOW. FUTURE WORK
        #adding priority as noise in batch
        df_batch.at[:, 'weight'] = 0.0
        for i, row in df_batch.iterrows():
            df_batch.at[i, 'priority'] = (0 + np.random.uniform(0, 0.001))**priority_alpha

        
        # setting parameters
        self.state_size = state_size
        self.action_size = action_size
        self.batch = df_batch
        
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.learning_rate = lr
        self.units = units
        self.hidden_layers = hidden_layers
        
        self.dueling = dueling
        self.double_param = double_param
        self.priority_alpha = priority_alpha
        
        self.copy_online_to_target_ep = copy_online_to_target_ep
        self.eval_after = eval_after
        
        
        # setting up the models
        if self.dueling:
            # TODO
            self.model_1 = self._build_model_dueling()
            self.model_2 = self._build_model_dueling()
        else:
            self.model_1 = self._build_model()
            self.model_2 = self._build_model()
        
        # evaluation variables
        self.R = []
        self.ecrs = []
    
    def _build_model_dueling(self):
        inputs = layers.Input(shape=(self.state_size,))
        z = layers.Dense(self.units, kernel_initializer='glorot_normal', activation='relu')(inputs)
        for layer in range(self.hidden_layers-1):
            z = layers.Dense(self.units, kernel_initializer='glorot_normal', activation='relu')(z)
            
        value = layers.Dense(1, kernel_initializer='glorot_normal', activation='linear')(z)
        
        adv = layers.Dense(self.action_size, kernel_initializer='glorot_normal', activation='linear')(z)

        q = QLayer()([value, adv])
        
        model = Model(inputs=inputs, outputs=q)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
        
    def _build_model(self):
        """
        Standard DQN model
        """
        model = Sequential()
        
        model.add(layers.Dense(self.units, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_normal'))
        for layer in range(self.hidden_layers-1):
            model.add(layers.Dense(self.units, activation='relu', kernel_initializer='glorot_normal'))
        
        model.add(layers.Dense(self.action_size, activation='linear', kernel_initializer='glorot_normal'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])
        return model
    
    def act(self, state):
        state_array = np.array(state.reshape(1, self.state_size))
        act_values = self.model_2.predict(state_array)
        return np.argmax(act_values[0]), np.max(act_values[0])
    
    def learn(self, epoch, env=None):
        for i in range(epoch):
            self._learn_minibatch()
            
            if (i+1)%self.copy_online_to_target_ep==0:
                self.model_2.set_weights(self.model_1.get_weights())
            
            if (i+1)%self.eval_after==0:
                r = self.run_env(env)
                self.R.append(r)
                ecr = 0
                ecr = self.ecr_reward()
                print("--epoch: {}/{} | ECR: {:.5f} | R: {:.2f} --".format(i+1, epoch, ecr, r))
        
        print("--final run--")
        self.model_2 = self.model_1
        r = self.run_env(env)
        self.R.append(r)
        self.predict()
        ecr = self.ecr_reward()
        print("--epoch: {}/{} | ECR: {:.5f} | R: {:.2f} --".format(i+1, epoch, ecr, r))
    
    
    def _learn_row(self, row):
        i = row.name
        state, action, reward, next_state, done = row['state'], row['action'], row['reward'], row['next_state'], row['done']

        target_q = reward

        # For Double DQN
        rand = random.random()
        
        if rand >= self.double_param:
            if not done: 
                ns_act_values = self.model_1.predict(next_state.reshape(1,self.state_size))[0]
                a_prime = np.argmax(ns_act_values)

                target_ns_act_values = self.model_2.predict(next_state.reshape(1,self.state_size))[0]
                target_ns_q = target_ns_act_values[a_prime]

                target_q = reward + self.gamma*target_ns_q

                self.batch.at[i, 'pred_action'] = a_prime
                self.batch.at[i, 'pred_reward'] = target_q
                
            target_f = self.model_1.predict(state.reshape(1,self.state_size))
            # Prioritized Experience Reply with noise
            self.batch.loc[i, 'priority'] = (abs(target_q - target_f[0][action]) + np.random.uniform(0, 0.001))**self.priority_alpha

            target_f[0][action] = target_q
            self.model_1.fit(state.reshape(1,self.state_size), target_f, epochs=1, verbose=0)
        else:
            if not done: 
                ns_act_values = self.model_2.predict(next_state.reshape(1,self.state_size))[0]
                a_prime = np.argmax(ns_act_values)

                target_ns_act_values = self.model_1.predict(next_state.reshape(1,self.state_size))[0]
                target_ns_q = target_ns_act_values[a_prime]

                target_q = reward + self.gamma*target_ns_q

                self.batch.at[i, 'pred_action'] = a_prime
                self.batch.at[i, 'pred_reward'] = target_q

            target_f = self.model_2.predict(state.reshape(1,self.state_size))
            # Prioritized Experience Reply with noise
            self.batch.loc[i, 'priority'] = (abs(target_q - target_f[0][action]) + np.random.uniform(0, 0.001))**self.priority_alpha

            target_f[0][action] = target_q
            self.model_2.fit(state.reshape(1,self.state_size), target_f, epochs=1, verbose=0)
            
                
    
    def _learn_minibatch(self):
        # For PER
        priority_sum = self.batch['priority'].sum()
        self.batch['weight'] = self.batch['priority']/priority_sum
        
        
        minibatch = self.batch.sample(self.minibatch_size, weights=self.batch['weight'])
        minibatch.apply(self._learn_row, axis=1)
        
            
    def ecr_reward(self):
        reward = 0.0
        count = 0
        for i, row in self.batch.loc[self.batch['transition_id']==1].iterrows():
            state = row['state']
            next_state = row['next_state']
                
            reward += self.act(state)[1]
            count += 1
            
        ecr = reward/count
        self.ecrs.append(ecr)
        return ecr
    
    def _predict_row(self, row):
        i = row.name
        state = row['state']
        next_state = row['next_state']

        act, q = self.act(state)
        self.batch.loc[i, 'pred_action'] = act
        self.batch.loc[i, 'pred_reward'] = q
        
    def predict(self):
        self.batch.apply(self._predict_row, axis=1)
        
        return self.batch
    
    def run_env(self, env):
        if env is None:
            return 0
        state = env.reset()
        total_reward = 0
        while True:
            action = self.act(state)[0]
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                state = env.reset()
                return total_reward
            
    def get_all_eval_df(self):
        eval_df = pd.DataFrame(columns=['ECR', 'R'])
        
        eval_df['ECR'] = self.ecrs
        eval_df['R'] = self.R
        
        return eval_df


# # RUN

# In[3]:

total_time = 0

# Selecting a GPU for tensorflow
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# SOME FORMATTING ISSUES WITH CSV
df = pd.read_csv('../data/CartPole-v1_10k.csv')
for i, row in df.iterrows():
    state = ast.literal_eval(row['state'])
    df.at[i, 'state'] = np.array(state)
    
    next_state = ast.literal_eval(row['next_state'])
    df.at[i, 'next_state'] = np.array(next_state)

dlt_lst = []
for i, row in df.iterrows():
    if row['done']==True and row['delayed_reward']==0:
        dlt_lst.append(i)
len(dlt_lst)    

df.drop(dlt_lst, inplace=True)
df = df.sort_values(by=['episode_id', 'transition_id'])
df.reset_index(inplace=True, drop=True)
df

print("Cartpole", "| Total transitions:", len(df), " | Total episodes:", len(df['episode_id'].unique()))
org_df = df.copy()


# In[8]:


result_dir = '../results/'
env = gym.make("CartPole-v1")
epoch = 10000
action_size = 2
random_state = 0
env_name = 'cartpole'
prefix = ''
for ep_size in [1000, 5000, 10000]:
    df_run = org_df.copy()
    if ep_size < len(df_run['episode_id'].unique()):
        eps = np.random.choice(df_run['episode_id'].unique(), ep_size)
        df_run = df_run.loc[df_run['episode_id'].isin(eps)]
        df_run.reset_index(drop=True, inplace=True)
    
    for reward_type in ['immediate_reward', 'delayed_reward']:
        df_run['reward'] = df_run[reward_type]
        
        for dueling, double_param, priority_alpha in [(False, 0, 0), (False, 0, 0.05), (False, 0.5, 0.05), (True, 0.5, 0.05)]:
            start_time = datetime.datetime.now()

            df = df_run.copy()

            prefix = env_name + '_' + 'ep_size_' + str(ep_size) + '_' + reward_type + '_' +                     'dueling_' + str(dueling) + '_double_' + str(double_param) + '_priority_' +                     str(priority_alpha) + '_' + 'rs_' + str(random_state) + '_'


            np.random.seed(random_state)
            random.seed(random_state)

            print("==" + prefix + "==")
            agent = DQNAgent(df_batch=df, state_size=len(df.iloc[0]['state']), action_size=action_size,
                             dueling=dueling, double_param=double_param, priority_alpha=priority_alpha,
                             copy_online_to_target_ep=100, eval_after=100)

            agent.learn(epoch, env)


            result = agent.batch
            eval_df = agent.get_all_eval_df()
            eval_df.to_pickle(result_dir + prefix +'eval.pkl')
            result.to_pickle(result_dir + prefix +'result.pkl')
            eval_df.to_csv(result_dir + prefix +'eval.csv')
            result.to_csv(result_dir + prefix +'result.csv')

            print('==run ends==')

            run_time = datetime.datetime.now() - start_time
            total_time += run_time.seconds
            print(f'Run took {run_time.seconds} seconds')
            print(f'Total time so far: {total_time} seconds')
            print()


# In[ ]:




