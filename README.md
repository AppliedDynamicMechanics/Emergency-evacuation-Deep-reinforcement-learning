# Emergency-evacuation-Deep-reinforcement-learning
Deep reinforcement learning model with a particle dynamics environment applied to emergency evacuation of a room with convex and concave obstacles.

## About
We developed a deep reinforcement learning algorithm in association with a particle dynamics model to train agents to find the fastest path to evacuate a room with obstacles. This report accompanies the 
"[Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles](https://doi.org/10.1016/j.physa.2021.125845) " paper paper which appeared on the 2021 Physica A: Statistical Mechanics and its Applications.


## Abstract

Efficient emergency evacuation is crucial for survival. However, it is not clear if the application of the self-driven force of the social-force model results in optimal evacuation, especially in complex environments with obstacles. In this work, we developed a deep reinforcement learning algorithm in association with the social force model to train agents to find the fastest evacuation path. During training, we penalized every step of an agent in the room and gave zero reward at the exit. We adopted the Dyna-Q learning approach. We showed that our model can efficiently handle modeling of emergency evacuation in complex environments with multiple room exits and convex and concave obstacles where it is difficult to obtain an intuitive rule for fast evacuation using just the social force model.

## How to use

We use a particle dynamics model to train agents to find the fastest path to evacuate.
There are many available arguments.

| Argument                 | Type     | Default    | Description                                                               |
| ------------------------ | -------- | ---------- | ------------------------------------------------------------------------- |
| door_size                | float    | 1.0        | Size of door                                                              |
| agent_size               | float    | 0.5        | Size of agent (particle)                                                  |
| reward                   | float    | -0.1       | Reward                                                                    |
| end_reward               | float    | 0          | End_reward                                                                |
| dis_lim                  | float    | 0.75       | Direct distance from the center of the agent to the center of the exit.   |
| action_force             | float    | 1.0        | Unit action force                                                         |
| desire_velocity          | float    | 2.0        | Desire velocity                                                           |
| relaxation_time          | float    | 0.5        | Relaxation_time                                                           |
| delta_t                  | float    | 0.01       | Time step                                            |
| xmax                     | float    | 50.0       | X-direction size of the cell space                                        |
| ymax                     | float    | 50.0       | Y-direction size of the cell space                                        |
| cfg_save_step            | int      | 5          | Time interval for saving Cfg file                                   |



### Setup

Please check the requirements document to setup the environment. The versions of main package are as following.

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
| num_episodes             | int      | 10000      | Max number of episodes to learn from                         |
| max_steps                | int      | 10000      | Max steps in an episode                                      |
| gamma                    | float    | 0.999      | Future reward discount                                       |
| memory_size              | int      | 1000       | Memory capacity                                              |
| batch_size               | int      | 50         | Batch size                                                   |
| explore_start            | float    | 1.0        | Exploration probability at start                             |
| explore_stop             | float    | 0.1        | Minimum exploration probability                              |
| num_agent                | int      | 1          | How many agents for the training                            |
| update_target_every      | int      | 1          | Target update frequency                                      |
| tau                      | float    | 0.1        | Target update factor                                         |
| save_step                | int      | 1000       | Steps to save the model                                      |
| train_step               | int      | 1          | Steps to train the model                                     |
| learning_rate            | float    | 1e-04      | Learning rate to use                                         |
| Cfg_save_freq            | int      | 100        | Cfg save frequency (episode)                                 |



### Test

This is used to assess the generalization capabilities of the model. The test run on the optimal policy learned from a single agent. We have put a learned policy in the model folder. If you want to train a new policy, you need to first run the train file.

You can use the test document to test the result of the trained model. You can check the optimal distribution figure by using `matplotlib`. You can also check the test Cfg in the test folder by using `Ovito 2.9.0`.

`python Evacuation_Continuum_DQN_Fully_test`

Available arguments:

| Argument                 | Type     | Default    | Description                                                  |
| ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
| test_episodes            | int      | 1          | Max number of episodes to test                               |
| Number_Agent             | int      | 80         | How many agents to evacuate from the cell space during test  |
| max_steps                | int      | 10000      | Max steps in an episode                                      |
| Cfg_save_freq            | int      | 1          | Cfg save frequency (episode)                                 |
| cfg_save_step            | int      | 2          | Cfg save frequency (step)                                    |
| arrow_len                | float    | 0.07       | The arrow length in optimal distribution figure              |

## Cite

To cite the work please use:
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

