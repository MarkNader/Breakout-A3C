'''
Author : Mark Nader Morcos 
Year of Publication : 2017 (Thesis ==> DeepMind-like : Breakout)
Based on code of "Arthur Juliani" : Simple Reinforcement Learning with Tensorflow Part 8 (A3C)
Please Cite The author's name and "Arthur Juliani" in any personal or project use
'''

import multiprocessing
import os
import threading
from random import choice

import gym
from gym import wrappers
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow.contrib.layers as layer

save_frequency = 20 # how many global episodes needed to save model
update_frequency = 30 # how many local steps needed to update global network
mode = "TRAIN_MODE" #or PLAY_MODE , whether we are currently training or playing
device = "/gpu:0"   #or /cpu:0 , whether we use CPU or GPU
num_workers = 1 if mode == "PLAY_MODE" else 8 # Num of Workers , 1 if playing , 8 if training
experiment_path = './algorithms/AdamGray_1/Experiment' # when Uploading to Gym , Path of Experiment

gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are rgb frames of 84*84
a_size = 3 # 3 actions
actions = [1, 4, 5]
load_model = True
restore_path = './algorithms/AdamGray/model'
model_path = './algorithms/AdamGray/model'
UPLOAD_AT = 300

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Return Here
# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def convert_to_gray(x):
    x = x[49:193, 8:152] #crop un-needed borders
    x = np.dot(x[..., :3],[0.299, 0.587, 0.114]) #Convert to Gray Scale
    x = scipy.misc.imresize(x, [84, 84]) #resize to 84X84 frames
    return np.reshape(np.array(x), [-1]) / 255.0 #flattening and normalizing


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.out = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            self.out = layer.convolution2d(
                activation_fn=tf.nn.relu, inputs=self.out, num_outputs=32,
                kernel_size=5, stride=1)
            self.out = layer.max_pool2d(self.out, 2)

            self.out = layer.convolution2d(
                activation_fn=tf.nn.relu, inputs=self.out, num_outputs=32, kernel_size=5)
            self.out = layer.max_pool2d(self.out, 2)

            self.out = layer.convolution2d(
                activation_fn=tf.nn.relu, inputs=self.out, num_outputs=64, kernel_size=4) 
            self.out = layer.max_pool2d(self.out, 2)

            self.out = layer.convolution2d(
                activation_fn=tf.nn.relu, inputs=self.out, num_outputs=64, kernel_size=3) 


            #Output layers for policy and value estimations
            self.out = layer.fully_connected(
                layer.flatten(self.out), 512, activation_fn=tf.nn.relu)
            self.policy = layer.fully_connected(
                self.out, a_size, activation_fn=tf.nn.softmax)
            self.value = layer.fully_connected(
                self.out, 1, activation_fn=None)
            self.policy = tf.reshape(self.policy, [-1, a_size])
            self.value = tf.reshape(self.value, [-1, 1])
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(
                    tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy+ 1e-6))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1) # to increment global episodes , a node in our graph is used
        self.episode_rewards = [] #saves rewards of episodes
        self.episode_lengths = [] #saves lengths of episodes
        self.episode_mean_values = [] #saves the mean of values obtained during an episode
        self.upload_episodes = 0 # will be needed just if we upload to gym , to specify after how many epsiodes the upload will be done
        self.summary_writer = tf.summary.FileWriter("./algorithms/AdamGray/tgp_"+str(self.number)) #Path to save our summary for Tensorboard visualization

        #Create the local copy of the network and copy global params to local ones
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name) #node to constantly copy global network to local one

        #The Below code is related to setting up the Breakout Environment
        self.env = gym.make('Breakout-v0')
        if mode == "PLAY_MODE":
            self.env = wrappers.Monitor(self.env, experiment_path) #Creating a Wrapper just in PlayMode

    def train(self, rollout, sess, gamma, bootstrap_value):
        #Each item in rollout has this format : [state, action, reward, next_state, done, value] 
        #Below we obtain each item of the tuple and stack them together
        rollout = np.array(rollout)
        observations = rollout[:, 0] #states
        actions = rollout[:, 1] #actions
        rewards = rollout[:, 2] #rewards
        next_observations = rollout[:, 3] #next_states
        values = rollout[:, 5] #values

        ''' Here we take the rewards and values from the rollout, and use them to
        generate the advantage and discounted returns.
        The advantage function uses "Generalized Advantage Estimation" '''

        '''bootstrap value is estimation of value of last current state
        if the current state is a terminal state , then bootstrap value = 0
        otherwise we use the last state to bootstrap it's current value from our network and pass that value to this method '''

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)
        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
                     self.local_AC.inputs:np.vstack(observations),
                     self.local_AC.actions:actions,
                     self.local_AC.advantages:advantages}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        ''' we return some params for plotting(statistical) issues as average of (value loss , policy loss , entropy loss)
         as well as the current gradient norms and trainable variables norms '''
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n


    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print "Starting worker " + str(self.number)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0

                done = False
                s = self.env.reset()
                s = convert_to_gray(s)

                while not done:
                    #Take an action using probabilities from policy network output.
                    a_dist, v = sess.run([self.local_AC.policy, self.local_AC.value],
                                         feed_dict={self.local_AC.inputs:[s]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist[0] == a)

                    s1, reward, done, _ = self.env.step(actions[a])
                    s1 = convert_to_gray(s1)
                    if done == True:
                        s1 = s #next_state to be same as previous one if the episode has done

                    #Creating our rollout as mentioned in "train" method" 
                    episode_buffer.append([s, a, reward, s1, done, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += reward
                    s = s1 #after making the rollout , preparing for next time step by making the next_state --> a current state
                    total_steps += 1
                    episode_step_count += 1


                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == update_frequency and done == False and mode == "TRAIN_MODE":
                        # we don't know the true final return is, so we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs:[s]})[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops) #Running our node to copy global params to local ones
                    if done == True:
                        break

                #saving episode lengths , rewards, mean values for plotting in tensorboard
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and mode == "TRAIN_MODE":
                    '''Bootstrapping with 0 in terminal state
                    Training is valid only in TRAIN_MODE '''
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically saving model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % save_frequency == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print self.episode_rewards[-save_frequency:]
                        print "Saved Model"

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    
                    if mode == "TRAIN_MODE":
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                        summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        self.summary_writer.add_summary(summary, episode_count)
                        self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                '''
                self.upload_episodes += 1
                print self.upload_episodes
                if self.upload_episodes == UPLOAD_AT :
                    self.env.close()
                    #gym.upload(experiment_path, api_key='ADD_YOUR_GYM_API_KEY')
                '''







tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)


with tf.device(device): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
if mode == "TRAIN_MODE" and device == "/gpu:0":
    config.gpu_options.allow_growth = True #This line allows threads to perform on gpu


with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(restore_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the threading happens.
    # Start the "work" process for each worker in a separate thread.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads) 
