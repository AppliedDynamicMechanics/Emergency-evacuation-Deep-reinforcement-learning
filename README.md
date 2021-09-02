# Emergency-evacuation-Deep-reinforcement-learning
Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles


Abstract

Efficient emergency evacuation is crucial for survival. A very successful model for simulating emergency evacuation is the social-force model. At the heart of the model is the self-driven force that is applied to an agent and is directed towards the exit. However, it is not clear if the application of this force results in optimal evacuation, especially in complex environments with obstacles. We develop a deep reinforcement learning algorithm in association with a particle dynamics model to train agents to find the fastest path to evacuate. During training, we penalize every step of an agent in the room and give zero reward at the exit. We adopt the Dyna-Q learning approach, which incorporates both the model-free Q-learning algorithm and the model-based reinforcement learning method, to update a deep neural network used to approximate the action value functions. We first show that the resulting self-driven force points directly towards the exit as in the social force model in the case of a room without obstacles. To quantitatively validate our method, we compare the total time elapsed for agents to escape a room with one door and without obstacles with the results obtained from the social-force model. The distributions of the exit time intervals calculated using the two methods are shown to not differ significantly. We then study evacuation of a room with multiple exits following a similar training procedure. We show that agents are able to evacuate efficiently from the nearest exit through a shared network trained for a single agent. In addition, we test the robustness of the Dyna-Q learning approach in a complex environment with multiple exits and obstacles. Overall, we show that our model, based on the Dyna-Q reinforcement learning approach, can efficiently handle modeling of emergency evacuation in complex environments with multiple room exits and obstacles where it is difficult to obtain an intuitive rule for fast evacuation.
