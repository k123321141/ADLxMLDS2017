from agent_dir.agent import Agent
import gym
import numpy as np

from keras.models import *
from keras.layers import * 
from keras.optimizers import * 
import scipy,keras
import keras.backend as K
import tensorflow as tf
import sys,os,random
import scipy.signal
import signal, threading, time
from collections import deque
import multiprocessing
def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()
def real_act(action):
    if action == 0:
        return 0
    elif action == 1:
        return 2
    elif action == 2:
        return 3
    else:
        print('no such action', action)
        sys.exit(1)
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def build_model(action_size, state_size, scope):
    pixel_input = Input(shape=(state_size,))

    #actor
    x = Reshape((80, 80, 1), name='shared_reshape')(pixel_input)
    x = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), padding='same', name='shared_conv2d',
                            activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
    x = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
    x = Flatten(name='shared_flatten')(x)
    x = Dense(256,name='shared_dense64', activation='relu', kernel_initializer='he_uniform')(x)
    actor_output = Dense(action_size, activation='softmax')(x)
    actor = Model(inputs=pixel_input, outputs=actor_output)
    
    #critic
    '''
    x = Reshape((80, 80, 1))(pixel_input)
    x = Conv2D(32, kernel_size=(6, 6), strides=(3, 3), padding='same',
            activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    '''
    critic_output = Dense(1, activation='linear')(x)
    critic = Model(inputs=pixel_input, outputs=critic_output)

    #whole model
    model = Model(inputs=pixel_input, outputs=[actor_output, critic_output])
    return actor, critic, model 

def set_train_fn(local_info, global_info, action_size=3, state_size=6400):
    actor, critic, local_net = local_info 
    
    states = K.placeholder(shape=(None, state_size), dtype='float32')
    one_hot_actions = K.placeholder(shape=(None, action_size), dtype='float32')
    target = K.placeholder(shape=(None, 1), dtype='float32')
    advantage_fn = K.placeholder(shape=(None, 1), dtype='float32')

    action_probs = actor([states,])
    critic_value = critic([states,]) 

    actor_loss = -K.sum(K.log(K.sum(action_probs * one_hot_actions, axis=-1)) * advantage_fn)
    critic_loss = K.sum(K.square(target - critic_value))  
    
    # 0.9*log(0.9)+0.1*log(0.1) = -0.14 > 0.4*log(0.4)+0.6*log(0.6) = -0.29
    entropy = K.sum(action_probs * K.log(action_probs))
   
    loss = actor_loss + 0.5*critic_loss + 0.01 * entropy

    grads = tf.gradients(loss, local_net.trainable_weights)
    
    #global
    global_net, global_opt = global_info
    updates = global_opt.apply_gradients(zip(grads, global_net.trainable_weights))
    update_fn = K.function(
            inputs=[states, one_hot_actions, target, advantage_fn],
            outputs=[loss,],
                updates=[updates,])
    return update_fn 
def set_predict_fn(state_size, model):
    states = K.placeholder(shape=(None, state_size), dtype='float32')
    outputs = model([states,])
    predict_fn  = K.function(
        inputs=[states,],
        outputs=[outputs,],
            updates=[])
    return predict_fn
def worker_thread(worker):
    worker.train()


class Worker():
    def __init__(self, agent, name, env_name, build_net, global_info):
        self.name = name 
        self.env = gym.make(env_name)
        self.global_net, self.global_opt = global_info
        self.agent = agent
        #self.loss, self.opt = set_train_fn()

        self.actor, self.critic, self.local_net = build_net(agent.action_size, agent.state_size, self.name)
        #local_opt = tf.train.RMSPropOptimizer(learning_rate=self.agent.learning_rate)
        self.pull()
        global_info = [self.global_net, self.global_opt]
        local_info = [self.actor, self.critic, self.local_net]
        self.update_fn = set_train_fn(local_info, global_info)
        #polocy gradient loss 
        #counting the loss of every trajectory with discounted reward, then summerize them. 
        self.actor_predict_fn  = set_predict_fn(self.agent.state_size, self.actor)
        self.critic_predict_fn  = set_predict_fn(self.agent.state_size, self.critic)
    def train(self): 
        env = self.env
        self.states, self.next_states, self.actions, self.rewards = [],[],[],[]
        

        
        self.prev_x = None
        state = env.reset()
        steps = 1
        while not self.agent.stop:
            #if args.do_render:
            #    env.env.render()
            x = prepro(state)
            '''
            cur_x = prepro(state)
            x = cur_x - self.prev_x if self.prev_x is not None else cur_x
            self.prev_x = cur_x
            '''
            action = self.act(x)
            
            state, reward, terminal, info = env.step(real_act(action))
            #next_x = prepro(state) - cur_x
            next_x = prepro(state)
            
            done = reward != 0  #someone get the point
            self.remember(x, next_x, action, reward)
            if done:
                self.prev_x = None
                self.update(done)
                self.agent.update_count += 1
                self.states, self.next_states, self.actions, self.rewards = [],[],[],[]
                #print(self.name, self.agent.update_count)
            if terminal:    #game over
                state = env.reset()
                self.pull()
                #sess.run(self.pull_op)
                #every 21 point per update 
            steps += 1
        print('quit thread %s' % self.name)
    def remember(self, state, next_state, action, reward):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def update(self, done):

        states = np.vstack(self.states)
        next_states = np.vstack(self.next_states)
        actions = np.vstack(self.actions)
        rewards = np.vstack(self.rewards).reshape([-1,1])
        one_hot_actions = keras.utils.to_categorical(actions, self.agent.action_size)
        
        state_values = self.critic_predict_fn([states, ])[0]
        #next_state_values = self.critic_predict_fn([next_states, ])[0]

    
        # td error, but eliminate the bias of one step
        discounted_rewards = discount(rewards.copy(), self.agent.gamma)
        target = discounted_rewards
        #next_state_values[-1,0] = 0
        #advantage_fn = rewards - ( state_values - next_state_values )
        advantage_fn = discounted_rewards - state_values

        self.update_fn([states, one_hot_actions, target, advantage_fn])
    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        probs = self.actor_predict_fn([state,])[0].flatten()
        action = np.random.choice(self.agent.action_size, 1, p=probs)[0]
        return action 
    def get_discount_state_value(self, states):
        inputs = [ states[-1:, :] ]
        state_values = np.zeros([len(states), 1])
        state_values[-1, 0] = self.critic_predict_fn(inputs)[0][0,0]
        state_values = discount(state_values, self.agent.gamma)
        return state_values
     
    #get weights from global net 
    def pull(self):
        self.local_net.set_weights(self.global_net.get_weights())

class Agent_A3C(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_A3C,self).__init__(env)



        self.state_size = 80 * 80 
        self.env = env
        self.args = args
        
        self.action_size = 3
        #['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        #0, 2, 3
        if args.test_ac:
            #you can load your model here
            if os.path.isfile(args.a3c_model): 
                print('testing : load model from %s.' % args.a3c_model)
                self.learning_rate = 0.
                self.prev_x = None
                self.action_size = 3
                self.actor, self.critic, self.model = self.build_model('global')

                self.load(args.a3c_model)
            else:
                print('no model for testing!\nerror path %s' % args.a3c_model)
                sys.exit(1)



    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prev_x = None


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        env = self.env
        args = self.args
        self.learning_rate = 0.001
        self.gamma = args.a3c_discount_factor        
        self.update_count = 0
        self.global_actor, self.global_critic, self.global_net = build_model(self.action_size, self.state_size, 'global')
        opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        global_info = [self.global_net, opt]
        if os.path.isfile(args.a3c_model) and args.keep_train: 
            print('load model from %s.' % args.a3c_model)
            self.load(args.a3c_model)
        else:
            print('train a new model')
        
        self.predict_fn  = set_predict_fn(self.state_size, self.global_actor)

        #multi-threading
        thread_list = []
        self.stop = False
        num_workers = multiprocessing.cpu_count()
        #for i in range(args.a3c_worker_count):
        print('work on %d cpu' % num_workers)
        for i in range(num_workers):
            name = 'worker_%d' % i
            worker = Worker(self, name, 'Pong-v0',build_model, global_info)
            t = threading.Thread(target=worker_thread, args=[worker, ])
            thread_list.append(t)

        try:
            for t in thread_list:
                t.start()
            steps = 1
            self.prev_x = None
            state = env.reset()
            score = 0
            episode = 1
            start_time = time.time()
            while True:
                action = self.make_action(state)
                
                state, reward, terminal, info = env.step(action)
                score += reward 
                if terminal:    #game over
                    state = env.reset()
                    #every 21 point per update 
                    print('Episode: %d - Score: %2.1f - Update Counter: %5d - Spent Time %6.2f' % (episode, score, self.update_count, time.time()-start_time ))

                    episode += 1
                    score = 0
                    if episode > 0 and episode % args.a3c_save_interval == 0:
                        print('save model to %s' % args.a3c_model)
                        self.save(args.a3c_model)

                    time.sleep(1)
                steps += 1

        except (KeyboardInterrupt, Exception) as e:
            print('stop program',e)
            self.stop = True
            for t in thread_list:
                t.join()
            print("Quitting Program.")
        print('Done')

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        cur_x = prepro(observation)
        '''
        cur_x = prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else cur_x 
        self.prev_x = cur_x
        '''
        
        cur_x = cur_x.reshape([1,-1])
        #prob = self.global_actor.predict(cur_x, batch_size=1).flatten()
        prob = self.predict_fn([cur_x,])[0].flatten()
        
        #stochastic
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        #action = np.argmax(prob)
        return real_act(action)
    def load(self, name):
        self.global_net.load_weights(name)
    
    def save(self, name):
        self.global_net.save_weights(name)
    def pull_weights(self):
        self.global_weights = self.global_net.get_weights()

