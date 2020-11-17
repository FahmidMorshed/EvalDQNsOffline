# PROJECT NAME: Evaluating robustness of DQN variants for offline/batch learning

This repository contains project work of CSC 722 Advance Machine Learning from Fall 2020 course at NC State University, taught by D. Bahler. 

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




