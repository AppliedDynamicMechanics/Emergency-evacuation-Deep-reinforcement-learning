# Emergency-evacuation-Deep-reinforcement-learning
This code accompanies "[Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles](https://doi.org/10.1016/j.physa.2021.125845)", which appeared on *Physica A: Statistical Mechanics and its Applications* in 2021.

## About
This project uses a deep reinforcement learning algorithm in association with a particle dynamics model to train agents to find the fastest path to evacuate a room with obstacles.

Efficient emergency evacuation is crucial for survival. However, it is not clear if the application of the self-driven force of the social-force model results in optimal evacuation, especially in complex environments with obstacles. In this work, we developed a deep reinforcement learning algorithm in association with the social force model to train agents to find the fastest evacuation path. During training, we penalized every step of an agent in the room and gave zero reward at the exit. We adopted the Dyna-Q learning approach. We showed that our model can efficiently handle modeling of emergency evacuation in complex environments with multiple room exits and convex and concave obstacles where it is difficult to obtain an intuitive rule for fast evacuation using just the social force model.

# *How to use this code*
## Setup
*Note:* This code was designed to be used with Python 3.6.13, and is not compatible with later versions.

To install this project's package dependencies, please run the following command:

    pip install -r requirements.txt

## Train
This project provides framework to train an agent for emergency evacuation in two scenarios:

1. An empty room with four exits. To train this model, run the following command:
```
    python Evacuation_Continuum_4Exits_DQN_Fully.py
```

2. A room with three exits and two obstacles. To train this model, run the following command:
```
    python Evacuation_Continuum_3Exits_Ob_DQN_Fully.py
```

## Test
A built-in testing script can be used to assess generalization capabilities, illustrating the optimal policy learned by an agent during training. A pre-trained policy has been included in the model folder, which can be tested for reference. To run the testing framework, you can use the following command:

    python Evacuation_Continuum_DQN_Fully_test.py

## Configure
This code was developed with many customizable parameters to facilitate its application to different evacuation environments. You can modify the following parameters in the source code to appropriately configure your training:

- In file `Continuum_Cellspace.py`:

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

- In file `Evacuation_Continuum_4Exits_DQN_Fully.py` or `Evacuation_Continuum_3Exits_Ob_DQN_Fully.py`:

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

- In file `Evacuation_Continuum_DQN_Fully_test.py`:

    | Argument                 | Type     | Default    | Description                                                  |
    | ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
    | test_episodes            | int      | 1          | Max number of episodes to test                               |
    | Number_Agent             | int      | 80         | How many agents to evacuate from the cell space during test  |
    | max_steps                | int      | 10000      | Max steps in an episode                                      |
    | Cfg_save_freq            | int      | 1          | Cfg save frequency (episode)                                 |
    | cfg_save_step            | int      | 2          | Cfg save frequency (step)                                    |
    | arrow_len                | float    | 0.07       | The arrow length in optimal distribution figure              |

## Cite

To reference this work, please use the following:
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
