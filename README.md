# PROJECT: Evaluating robustness of DQN variants for offline/batch learning

This repository contains Deep Reinforcement Learning project for CSC 722 Advance Machine Learning from Fall 2020 course at NC State University, taught by D. Bahler. 

### FILES and Directories:
The directories and their uses are as follows:
1. /data
  - contains generated 10k episodes for each environments
  - files are in pickle format (binary)
  - these files can be generated using the script /src/generate_data.ipynb
  
2. /results
  - contains all the results for each of the runs
  - for each environments, there are 24 different runs, all with different parameters. the filenames are the parameter choices
  - this can be generated using the script /src/RL.ipynb

3. /Graphs
  - contains two xlsx files for generating graphs
  - images for our resutls are also here
  - R Script.R is the script that was used to generate resulted graphs

4. /Images
  - for github readme file

5. /src
  - the /src/gym_gridworld folder is our custom gym environment. one can install the environment by doing *pip install -e .* inside the folder
  - the cartpole.py script was used to generate cartpole data
  - the generate_data.ipynb was used to generate custom gridworld data (both deterministic and non-deterministic)
  - the RL.ipynb was used for the DQN implementations and generating results
  
*Note: the .ipynb are jupyter files and can be run using jupyter notebook or MS visual code or something similar*


### Abstract
Deep Q-Networks (DQN) are the modern standard for deep reinforcement learning. Recently, many popular variations in DQN algorithms have emerged, 
such as Dueling DQN, DQN with experience replay, and priority-based DQN. While these models have demonstrated excellent performance on model-based learning 
environments, where the agent acts in a completely known environment that can be simulated and learned from, the applicability of DQN models to batch learning 
has not been thoroughly investigated. In batch learning, where the agent’s model of its environment is incomplete and uncertain, many of these algorithms fail 
to achieve satisfactory performance. Also, in the case of a delayed reward, DQN algorithms often suffer with limited data. In this paper, we analyze and 
evaluate the robustness of different existing DQN algorithms in a batch learning context. Our results show that DQNs are somewhat effective in a batch learning 
context when there are immediate rewards available in the environments but are substantially worse in the case of delayed rewards. We also found that increasing 
data size only helps when the environment is large and complex enough to learn. Our results suggest that Vanilla DQN performs better than other more complex 
DQN variants in terms of convergence time and reward collection in presence of immediate reward.

### Environments
To do our comparative study, we have used 3 environments. Two of the environments are custom made, while one is off-the-self from gym environment. The environments are as follows: 
- Cartpole-v1
- Gridworld (deterministic)
- Gridworld (non-deterministic)

<img src="/images/gridworld.png" width="600">


### Data
For batch learning, we collected 10k, 5k and 1k randomly played episodes from each environments. We trained DQNs based on the collected data.
A typical dataset looks like as follows:

<img src="/images/dataset.png" width="600">

### Result
<p float="center">
  <img src="/Graphs/Hypothesis 1 cartpole F.png" width="320" />
  <img src="/Graphs/Hypothesis 1 grd dm F.png" width="320" /> 
  <img src="/Graphs/Hypothesis 1 grd ndm F.png" width="320" />
</p>
- DQNs are somewhat useful when immediate rewards are used, but almost useless when the rewards are delayed
- Among different variants of DQNs, vanilla DQN has higher performace than others when used in offline learning
- Increasing episodes (data size) doesn't effect the learning much
