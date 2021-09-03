# Emergency-evacuation-Deep-reinforcement-learning
Deep reinforcement learning model with a particle dynamics environment applied to emergency evacuation of a room with obstacles.

## About
We developed a deep reinforcement learning algorithm in association with a particle dynamics model to train agents to find the fastest path to evacuate. This report accompanies the 
"[Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles](https://doi.org/10.1016/j.physa.2021.125845) " paper which appears on the 2021 Physica A: Statistical Mechanics and its Applications.


## Abstract

Efficient emergency evacuation is crucial for survival. However, it is not clear if the application of the self-driven force of the social-force model results in optimal evacuation, especially in complex environments with obstacles. In this report, we develop a deep reinforcement learning algorithm in association with the social force model to train agents to find the fastest evacuation path. During training, we penalize every step of an agent in the room and give zero reward at the exit. We adopt the Dyna-Q learning approach, which incorporates both the model-free Q-learning algorithm and the model-based reinforcement learning method, to update a deep neural network used to approximate the action value functions. We show that our model, based on the Dyna-Q reinforcement learning approach, can efficiently handle modeling of emergency evacuation in complex environments with multiple room exits and obstacles where it is difficult to obtain an intuitive rule for fast evacuation.

## How to use

We used a particle dynamics model to train agents to find the fastest path to evacuate.
There are many available arguments.

| Argument                 | Type     | Default    | Description                                                  |
| ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
| door_size                | float    | 1.0        | size of door                                                 |
| agent_size               | float    | 0.5        | size of agent (particle)                                     |
| reward                   | float    | -0.1       | reward                                                       |
| end_reward               | float    | 0          | end_reward                                                   |
| dis_lim                  | float    | 0.75       | distance from the exit which the agent is regarded as left   |
| action_force             | float    | 1.0        | unit action force                                            |
| desire_velocity          | float    | 2.0        | desire velocity                                              |
| relaxation_time          | float    | 0.5        | relaxation_time                                              |
| delta_t                  | float    | 0.1        | time difference of simulation                                |
| xmax                     | float    | 50.0       | x-direction size of the cell space                           |
| ymax                     | float    | 50.0       | y-direction size of the cell space                           |
| cfg_save_step            | int      | 5          | time steps interval for saving Cfg file                      |



### Setup

Please check the requiremnts document to setup the environment. The versions of main package are as following.

`
NVIDIA driver Verison    470.63.01
`

`
cudatoolkit              10.0.130
`

`
cuDNN                    7.6.5
`

`
Tensorflow-gpu           1.14.0
`

`
tensorflow-probability   0.7
`

`
trfl                     1.1
`

`
numpy                    1.16.4
`

Install the requirements with your package manager, i.e.  ` pip install tensorflow-probability==0.7`


### Train

You can use the framework to train Evacuation_Continuum_4Exits_DQN_Fully model and Evacuation_Continuum_3Exits_Ob_DQN_Fully model:  The full list of parameters can be found in the table below.
`
python Evacuation_Continuum_4Exits_DQN_Fully
`
or
`
python Evacuation_Continuum_3Exits_Ob_DQN_Fully
`

There are many available arguments.

| Argument                 | Type     | Default    | Description                                                  |
| ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
| num_episodes             | int      | 10000      | max number of episodes to learn from                         |
| max_steps                | int      | 10000      | max steps in an episode                                      |
| gamma                    | float    | 0.999      | future reward discount                                       |
| memory_size              | int      | 1000       | memory capacity                                              |
| batch_size               | int      | 50         | batch size                                                   |
| explore_start            | float    | 1.0        | exploration probability at start                             |
| explore_stop             | float    | 0.1        | minimum exploration probability                              |
| num_agent                | int      | 1          | how many workers for the training                            |
| update_target_every      | int      | 1          | target update frequency                                      |
| tau                      | float    | 0.1        | target update factor                                         |
| save_step                | int      | 1000       | steps to save the model                                      |
| train_step               | int      | 1          | steps to train the model                                     |
| learning_rate            | float    | 1e-04      | Learning rate to use                                         |
| Cfg_save_freq            | int      | 100        | Cfg save frequency (episode)                                 |



### Test

This is used to assess the generalization capabilities of a model. The test run on the optimal policy learned from a single agent. We have put the a trained policy in the model folder. If you want to train a new policy, you need to run the train file firstly.

You can use the test document to test the result of the trained model. You can check the optimal distribution figure by using `matplotlib`. You can also check the test Cfg in the test folder by using `Ovito 2.9.0`.

`python Evacuation_Continuum_DQN_Fully_test`

Available arguments:

| Argument                 | Type     | Default    | Description                                                  |
| ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
| test_episodes            | int      | 1          | max number of episodes to test                               |
| Number_Agent             | int      | 80         | How many agents to evacuate from the cell space during test  |
| max_steps                | int      | 10000      | max steps in an episode                                      |
| Cfg_save_freq            | int      | 1          | Cfg save frequency (episode)                                 |
| cfg_save_step            | int      | 2          | Cfg save frequency (step)                                    |
| arrow_len                | float    | 0.07       | the arrow length in optimal distribution figure              |

## Cite

If you want to cite this work please use this:
```
@article{ZHANG2021125845,
title = {Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles},
journal = {Physica A: Statistical Mechanics and its Applications},
volume = {571},
pages = {125845},
year = {2021},
issn = {0378-4371},
doi = {https://doi.org/10.1016/j.physa.2021.125845},
url = {https://www.sciencedirect.com/science/article/pii/S0378437121001175},
author = {Yihao Zhang and Zhaojie Chai and George Lykotrafitis},
keywords = {Dyna-Q learning, Particle dynamics simulation, Social-force model, Pedestrian dynamics},
}
```

