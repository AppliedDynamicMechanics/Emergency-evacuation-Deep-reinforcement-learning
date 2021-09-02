####Test of trained model for evacuation

import numpy as np
import tensorflow as tf
import os
import shutil
from Continuum_Cellspace import *

Number_Agent = 80
delta_t = 0.05

#4Exits
Exit.append(np.array([0.5, 1.0, 0.5]))  ##Add up exit
Exit.append(np.array([0.5, 0.0, 0.5]))  ##Add down exit
Exit.append(np.array([0.0, 0.5, 0.5]))  ##Add left exit
Exit.append(np.array([1.0, 0.5, 0.5]))  ##Add right Exit

#3Exits with ob
#Exit.append(np.array([0.7, 1.0, 0.5]))  ##Add up exit
#Exit.append(np.array([0.5, 0, 0.5]))    ##Add down Exit
#Exit.append(np.array([0, 0.7, 0.5]))    ##Add left exit
#
#Ob1 = []                                ##Obstacle #1
#Ob1.append(np.array([0.8, 0.8, 0.5]))
#Ob.append(Ob1)
#Ob_size.append(2.0)
#
#Ob2 = []                                ##Obstacle #2
#Ob2.append(np.array([0.3, 0.5, 0.5]))
#Ob.append(Ob2)
#Ob_size.append(3.0)

output_dir = './Test'
model_saved_path_4exits = './model/Continuum_4Exits_DQN_Fully'
model_saved_path_3exits_ob = './model/Continuum_3Exits_Ob_DQN_Fully'
  
class DQN_4exit:
    def __init__(self, name, learning_rate = 0.0001, gamma = 0.99,
                 action_size = 8, batch_size = 20):
        
        self.name = name
        
        # state inputs to the Q-network
        with tf.variable_scope(name):
            
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name = 'inputs')  

            with tf.contrib.framework.arg_scope(
                    [tf.contrib.layers.fully_connected],
                    activation_fn=tf.nn.elu, 
                    weights_initializer=tf.initializers.he_normal()
                    ):
                self.f1 = tf.contrib.layers.fully_connected(self.inputs_, 64)
                self.f2 = tf.contrib.layers.fully_connected(self.f1, 128)
                self.f3 = tf.contrib.layers.fully_connected(self.f2, 128)
                self.f4 = tf.contrib.layers.fully_connected(self.f3, 64)

            self.output = tf.contrib.layers.fully_connected(self.f4, action_size, activation_fn = None)
    
    def get_qnetwork_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]  

class DQN_3exit_Ob:
    def __init__(self, name, learning_rate = 0.0001, gamma = 0.99,
                 action_size = 8, batch_size = 20):
        
        self.name = name
        
        # state inputs to the Q-network
        with tf.variable_scope(name):
            
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name = 'inputs')  

            with tf.contrib.framework.arg_scope(
                    [tf.contrib.layers.fully_connected],
                    activation_fn = tf.nn.relu,                    
                    ):
                self.f1 = tf.contrib.layers.fully_connected(self.inputs_, 64)
                self.f2 = tf.contrib.layers.fully_connected(self.f1, 64)
                self.f3 = tf.contrib.layers.fully_connected(self.f2, 64)
                self.f4 = tf.contrib.layers.fully_connected(self.f3, 64)
                self.f5 = tf.contrib.layers.fully_connected(self.f4, 64)
                self.f6 = tf.contrib.layers.fully_connected(self.f5, 64)

            self.output = tf.contrib.layers.fully_connected(self.f6, action_size, activation_fn = None)
            
    def get_qnetwork_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.name)] 

if __name__ == '__main__':
    
    test_episodes = 10        # max number of episodes to test
    max_steps = 10000                # max steps in an episode
    gamma = 0.999                   # future reward discount

    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.1            # minimum exploration probability 
#    decay_rate = 0.00002            # exponential decay rate for exploration prob
    decay_percentage = 0.5        
    decay_rate = 4 / decay_percentage
            
    # Network parameters
    learning_rate = 1e-4         # Q-network learning rate 
    
    # Memory parameters
    memory_size = 10000          
    batch_size = 50               
    pretrain_length = batch_size
    
    Cfg_save_freq = 1
    cfg_save_step = 2
    
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut = 1.5, dt = delta_t, Number = Number_Agent)
    state = env.reset()
        
    tf.reset_default_graph()

    mainQN_4Exits = DQN_4exit(name='main_qn_4exits', learning_rate=learning_rate, batch_size=batch_size, gamma = gamma)
    mainQN_3Exits_Ob = DQN_3exit_Ob(name='main_qn_3exits_ob', learning_rate=learning_rate, batch_size=batch_size, gamma = gamma)
 
    Four_list = mainQN_4Exits.get_qnetwork_variables()
    Ob_list = mainQN_3Exits_Ob.get_qnetwork_variables()
    
    init = tf.global_variables_initializer()
    saver_4exits = tf.train.Saver(Four_list)
    saver_3exits_ob = tf.train.Saver(Ob_list)

    ######GPU usage fraction
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    with tf.Session(config = config) as sess:   
        
        sess.run(init)
        ####check saved model      
        if not os.path.isdir(model_saved_path_4exits):
            os.mkdir(model_saved_path_4exits)
        
        if not os.path.isdir(model_saved_path_3exits_ob):
            os.mkdir(model_saved_path_3exits_ob)
            
        checkpoint_4exits = tf.train.get_checkpoint_state(model_saved_path_4exits)
        if checkpoint_4exits and checkpoint_4exits.model_checkpoint_path:
            saver_4exits.restore(sess, checkpoint_4exits.model_checkpoint_path)
#            saver_4exits.restore(sess, checkpoint_4exits.all_model_checkpoint_paths[3])
            print("Successfully loaded:", checkpoint_4exits.model_checkpoint_path)            
            
        checkpoint_3exits_ob = tf.train.get_checkpoint_state(model_saved_path_3exits_ob)
        if checkpoint_3exits_ob and checkpoint_3exits_ob.model_checkpoint_path:
#            saver_3exits_ob.restore(sess, checkpoint_3exits_ob.model_checkpoint_path)
            saver_3exits_ob.restore(sess, checkpoint_3exits_ob.all_model_checkpoint_paths[2])  
            print("Successfully loaded:", checkpoint_3exits_ob.model_checkpoint_path) 

        ############Illustration of force direction
        x, y = np.meshgrid(np.linspace(0,1,100) - offset[0], np.linspace(0,1,100) - offset[1])
        x_arrow, y_arrow = np.meshgrid(np.linspace(0.05,0.95,15) - offset[0], np.linspace(0.05,0.95,15) - offset[1])
        xy = np.vstack([x.ravel(), y.ravel()]).T
        xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T
        
        ###random velocity
        vxy = np.random.randn(*xy.shape) * 0.
        vxy_arrow = np.random.randn(*xy_arrow.shape) * 0.
        
        ####constant velocity
        vxy[:,1] = 0
        vxy_arrow[:,1] = 0
        
        xtest = np.hstack([xy, vxy])
        x_arrow_test = np.hstack([xy_arrow, vxy_arrow])

        ypred = sess.run(mainQN_4Exits.output, feed_dict = {mainQN_4Exits.inputs_ : xtest})
        ypred_arrow = sess.run(mainQN_4Exits.output, feed_dict = {mainQN_4Exits.inputs_ : x_arrow_test})
 
#        ypred = sess.run(mainQN_3Exits_Ob.output, feed_dict = {mainQN_3Exits_Ob.inputs_ : xtest})
#        ypred_arrow = sess.run(mainQN_3Exits_Ob.output, feed_dict = {mainQN_3Exits_Ob.inputs_ : x_arrow_test})
 
        action_pred = np.argmax(ypred, axis = 1)
        action_arrow_pred = np.argmax(ypred_arrow, axis = 1)
 
        action_grid = action_pred.reshape(x.shape)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize = (5,5), subplot_kw = {'xlim' : (-0.5, 0.5),
                               'ylim' : (-0.5, 0.5)})
        
        ####Contour plot
#        c_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
#        contour = ax.contourf(x,y,action_grid+0.1,colors = c_map, alpha = 0.8)
        
        contour = ax.contourf(x,y,action_grid + 0.1, cmap = plt.cm.get_cmap('rainbow'), alpha = 0.8)       
#        cbar = fig.colorbar(contour, ticks = range(8))
#        cbar.set_label('Force direction')
#        cbar.set_ticklabels(['Up', 'Up-Left', 'Left','Down-Left','Down',
#                             'Down-right', 'Right', 'Up-Right', 'Right'])
    
        ###text annotation
#        for idx, p in enumerate(xy_arrow):
#            ax.annotate(str(action_arrow_pred[idx]), xy = p)
    
        ###Arrow
        arrow_len = 0.07
        angle = np.sqrt(2) / 2
        arrow_map = {0 : [0, arrow_len], 1: [-angle * arrow_len, angle * arrow_len],
                     2 : [-arrow_len, 0], 3: [-angle * arrow_len, -angle * arrow_len],
                     4 : [0, -arrow_len], 5: [angle * arrow_len, -angle * arrow_len],
                     6 : [arrow_len, 0], 7: [angle * arrow_len, angle * arrow_len],}
        for idx, p in enumerate(xy_arrow):
            ax.annotate('', xy = p, xytext = np.array(arrow_map[action_arrow_pred[idx]]) + p,
                        arrowprops = dict(arrowstyle = '<|-',color = 'k',lw = 1.5))
        
#        ax.add_patch(plt.Circle((0,0),0.1, alpha = 0.5))
#        ax.add_patch(plt.Circle((0,0.2),0.1, alpha = 0.5))
#        ax.add_patch(plt.Circle((0.3,0.3),0.1, alpha = 0.5))
#        ax.add_patch(plt.Circle((-0.2,0),0.15, alpha = 0.5))
        
        ax.tick_params(labelsize = 'large')
        plt.show()
        
#        fig.savefig('fs1a.png',dpi=600)
        
        step = 0     
        
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)  
        
        for ep in range(0, test_episodes):
            total_reward = 0
            t = 0
            
            print("Testing episode: {}".format(ep))
            
            if ep % Cfg_save_freq == 0:
                
                pathdir = os.path.join(output_dir, 'case_' + str(ep) )             
                if not os.path.isdir(pathdir):
                    os.mkdir(pathdir)
                    
                else:            
                    for filename in os.listdir(output_dir):
                        filepath = os.path.join(output_dir, filename)
                        
                    try:
                        shutil.rmtree(filepath)
                    except OSError:
                        os.remove(filepath)  
                    
                env.save_output(pathdir + '/s.' + str(t))
            
            while t < max_steps:

                # Get action from Q-network                
                #########ALL Particles
                done = env.step_all(sess, mainQN_4Exits, Normalized=True)
                # done = env.step_all(sess, mainQN_3Exits_Ob, Normalized=True)
                
                # done = env.step_optimal() 
                
                step += 1
                t += 1
                
                if done:
                    if ep % Cfg_save_freq == 0:
                        env.save_output(pathdir + '/s.' + str(t))

                    state = env.reset()
                    break

                else:

                    if ep % Cfg_save_freq == 0:
                        if t % cfg_save_step == 0:
                            env.save_output(pathdir + '/s.' + str(t))
            
            print("Total steps in episode {} is : {}".format(ep, t))