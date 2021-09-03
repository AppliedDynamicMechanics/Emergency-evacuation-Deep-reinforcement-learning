# Emergency-evacuation-Deep-reinforcement-learning
Deep reinforcement learning model with a particle dynamics environment applied to emergency evacuation of a room with obstacles.

## About
We developed a deep reinforcement learning algorithm in association with a particle dynamics model to train agents to find the fastest path to evacuate. This report accompanies the 
"[Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles]" (https://doi.org/10.1016/j.physa.2021.125845) paper which appears on the 2021 Physica A: Statistical Mechanics and its Applications.


## Abstract

Efficient emergency evacuation is crucial for survival. However, it is not clear if the application of the self-driven force of the social-force model results in optimal evacuation, especially in complex environments with obstacles. In this report, we develop a deep reinforcement learning algorithm in association with the social force model to train agents to find the fastest evacuation path. During training, we penalize every step of an agent in the room and give zero reward at the exit. We adopt the Dyna-Q learning approach, which incorporates both the model-free Q-learning algorithm and the model-based reinforcement learning method, to update a deep neural network used to approximate the action value functions. We show that our model, based on the Dyna-Q reinforcement learning approach, can efficiently handle modeling of emergency evacuation in complex environments with multiple room exits and obstacles where it is difficult to obtain an intuitive rule for fast evacuation.

## How to use

### Setup

Please check the requiremnts document to setup the environment.
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
The following trains a U-Net with a specific weight decay coefficient:

`python train_network.py --experiment_name unet_wd_1e-5 --model_type --weight_decay_coefficient 1e-5 `

There are many available arguments.

| Argument                 | Type     | Default    | Description                                                  |
| ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
| model_type               | str      | NA         | Network architecture for training [ar_lstm convlstm, resnet, unet, predrnn] |
| num_epochs               | int      | 50         | The experiment's epoch budget                                |
| num_input_frames         | int      | 5          | How many frames to insert initially                          |
| num_output_frames        | int      | 20         | How many framres to predict in the future                    |
| dataset                  | str      | 'original' | select which dataset to use [original, fixed_tub]             |
| batch_size               | int      | 16         | Batch size                                                   |
| samples_per_sequence     | int      | 10         | How may training points to generate from each simulation sequence |
| experiment_name          | str      | 'dummy'    | Experiment name - used for building the experiment folder    |
| normalizer_type          | str      | 'normal'   | how to normalize the images [normal, m1to1 (-1 to 1), none]  |
| num_workers              | int      | 8          | how many workers for the dataloader                          |
| seed                     | int      | 12345      | Seed to use for random number generator for experiment       |
| seed_everything          | str2bool | True       | Use seed for everything random (numpy, pytorch, python)      |
| debug                    | str2bool | False      | For debugging purposes                                       |
| weight_decay_coefficient | float    | 1e-05      | Weight decay to use for Adam                                 |
| learning_rate            | float    | 1e-04      | Learning rate to use for Adam                                |
| scheduler_patience       | int      | 7          | Epoch patience before reducing learning_rate                 |
| scheduler_factor         | float    | 0.1        | Factor to reduce learning_rated                              |
| continue_experiment      | str2bool | False      | Whether the experiment should continue from the last epoch   |
| back_and_forth           | bool     | False      | If training will be with predicting both future and past     |
| reinsert_frequency       | int      | 10         | LSTM: how often to use the reinsert mechanism                |


### Test

This is used to assess the generalization capabilities of a model. The test are run on all the datasets that are provided above. If you want to change that you'll need to change the dataloaders in the `evaluate_experiment` function in `utils/experiment_evaluatory.py`,

`python test_network.py --experiment_name unet_wd_1e-5`

Available arguments:

| Argument                 | Type     | Default    | Description                                                  |
| ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
| test_starting_point      | int      | 15         | Which frame to start the test                                |
| num_total_output_frames  | int      | 80         | How many frames to predict to the future during evaluation   |
| get_sample_predictions   | str2bool | True       | Print sample predictions figures or not                      |
| num_output_keep_frames   | int      | 20         | How many frames to keep from each propagation in RNN models |
| refeed                   | str2bool | False      | Whether to use the refeed mechanism in RNNs                  |

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

