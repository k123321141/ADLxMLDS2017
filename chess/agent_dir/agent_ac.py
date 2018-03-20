from agent_dir.agent import Agent
import gym
import numpy as np

from keras.models import *
from keras.layers import * 
from keras.optimizers import Adam
import scipy,keras
import keras.backend as K
import tensorflow as tf
import sys,os,random
import scipy.signal
from collections import deque
hidden_state_units = 64
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
def discount_TD_error(rewards, last_next_state_values, done, gamma):
    assert len(rewards.shape) == 2
    state_values = np.zeros_like(rewards)
    num = len(rewards)
    R = 0. if done else last_next_state_values
    for i in reversed(range(num)):
        R = rewards[i, 0] + R * gamma
        state_values[i, 0] = R 
    return state_values 

def build_model(action_size, state_size):
    pixel_input = Input(shape=(state_size,))
    hi_st = Input(shape=(hidden_state_units,))
    #actor
    x = Reshape((80, 80, 1))(pixel_input)
    for i in range(2):
        x = Conv2D(32 * 2**i, kernel_size=(5, 5), strides=(3, 3), padding='same',
                                activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
    cnn_out = Reshape([1,-1])(x)
    
    x, st = GRU(hidden_state_units, activation='relu',return_state=True)(cnn_out, hi_st)
    
    actor_output = Dense(action_size, activation='softmax')(x)
    actor = Model(inputs=[pixel_input, hi_st], outputs=actor_output)

    
    #critic
    critic_output = Dense(1, activation='linear')(x)
    critic = Model(inputs=[pixel_input, hi_st], outputs=critic_output)

    #whole model
    model = Model(inputs=[pixel_input, hi_st], outputs=[actor_output, critic_output, st])
    return actor, critic, model

class Agent_AC(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_AC,self).__init__(env)



        self.state_size = 6400 
        self.env = env
        self.args = args
        
        self.action_size = 3
        #['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        #0, 2, 3
        if args.test_ac:
            #you can load your model here
            if os.path.isfile(args.ac_model): 
                print('testing : load model from %s.' % args.ac_model)
                self.learning_rate = 0.
                self.prev_x = None
                self.action_size = 3
                self.actor, self.critic, self.model = self.build_model()

                self.load(args.ac_model)
            else:
                print('no model for testing!\nerror path %s' % args.ac_model)
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

        self.gamma = args.ac_discount_factor
        self.learning_rate = 1e-4
        self.actor, self.critic, self.model = build_model(env.action_space.n, 6400)
        self.actor_target, self.critic_target, self.model_target = build_model(env.action_space.n, 6400)
        self.set_a2c_train_fn()
        
        #summary
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            args.ac_summary  , self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        
        self.update_counter = 0
        self.update_target_counter = 0
        if os.path.isfile(args.ac_model) and args.keep_train: 
            print('load model from %s.' % args.ac_model)
            self.load(args.ac_model)
        else:
            print('train a new model')

        score, win, lose, step = 0,0,0,0
        episode = 0

        done = False
        state = env.reset()
        self.states,  self.actions, self.rewards, self.done, self.hi_sts = [], [], [], [], []
        hi_st = np.zeros([1, hidden_state_units], dtype=np.float32)
        score_que = deque([-21.,], maxlen=30)
        while True:
            if args.do_render:
                env.env.render()
            
            #random eploration action
            action, next_hi_st = self.act([state, hi_st])
            next_state, reward, terminal, info = env.step(action)
            
            score += reward
            done = reward != 0  #someone get the point
            self.remember(state, action, reward, done, hi_st)
            hi_st = next_hi_st
            state = next_state
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1
            step += 1
            #if done:
            if terminal or step % 20 ==0:
                done = terminal
                loss, actor_loss, critic_loss, entropy = self.update_actor_critic(done, next_state, next_hi_st)
                self.states, self.actions, self.rewards, self.done, self.hi_sts = [], [], [], [], []
                hi_st = np.zeros([1, hidden_state_units], dtype=np.float32)
                #summary 
                status = [loss, actor_loss, critic_loss, entropy, np.mean(score_que)]

                for i in range(len(status)):
                    K.get_session().run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(status[i])
                    })
                summary_str = K.get_session().run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.update_counter)
                
                self.update_counter += 1
            if terminal:    #game over
                state = env.reset()
                #every 21 point per update 


                #for log
                episode += 1
                print('Episode: %d - Score: %2.1f - Update Counter: %5d ' % (episode, score, self.update_counter ))
                sys.stdout.flush()
                if episode > 1 and episode % args.ac_save_interval == 0:
                    print('save model to %s.' % args.ac_model)
                    self.save(args.ac_model)

                
                score_que.append(score)
                score, win, lose, step = 0,0,0,0

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
        state = observation.reshape([1, -1])  
        prob = self.model.predict(state, batch_size=1).flatten()
        
        #stochastic
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action

    def act(self, inputs):
        state, hi_st = inputs
        state = state.reshape([1, -1])
        probs, _, hi_st = self.model.predict([state, hi_st]) 
        
        probs = probs.flatten()
        action = np.random.choice(self.action_size, 1, p=probs)[0]
        return action, hi_st 

    #train funcfion
    def remember(self, state, action, reward, done, hi_st):
        state = state.reshape([1, -1])
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.done.append(done)
        self.hi_sts.append(hi_st)
    
    def set_a2c_train_fn(self):
        #polocy gradient loss 
        #counting the loss of every trajectory with discounted reward, then summerize them. 
        states = K.placeholder(shape=(None, 6400), dtype='float32')
        hi_sts = K.placeholder(shape=(None, hidden_state_units), dtype='float32')
        target = K.placeholder(shape=(None, 1), dtype='float32')
        one_hot_actions = K.placeholder(shape=(None, self.action_size), dtype='float32')
        advantage_fn = K.placeholder(shape=(None, 1), dtype='float32')

        action_probs, critic_value, _ = self.model([states, hi_sts])

        action_probs = tf.clip_by_value(action_probs, 1e-20, 1.0)

        actor_loss = -K.sum(K.log(K.sum(action_probs * one_hot_actions, axis=-1)) * advantage_fn)
        critic_loss = K.sum(K.square(target - critic_value))  
        
        # 0.9*log(0.9)+0.1*log(0.1) = -0.14 > 0.4*log(0.4)+0.6*log(0.6) = -0.29
        entropy = K.sum(K.sum(action_probs * K.log(action_probs), axis=-1))
       
        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy
        
        opt = Adam(lr=self.learning_rate)
        #trainable_weights
        updates = opt.get_updates(
                                params=self.model.trainable_weights, 
                                loss=loss)
        self.a2c_train_fn = K.function(
                inputs=[states, hi_sts, one_hot_actions, target, advantage_fn],
                outputs=[loss, actor_loss, critic_loss, entropy],
                updates=updates)
        

    #train funcfion
    def update_actor_critic(self, done, next_state, next_hi_st):
        next_state = next_state.reshape([1, -1])
        
        states = np.vstack(self.states + [next_state])  #None,6400 or None,128
        actions = np.vstack(self.actions)   #None, 1
        rewards = np.vstack(self.rewards)   #None, 1
        hi_sts = np.vstack(self.hi_sts + [next_hi_st])  #None,rnn_units
        one_hot_actions = keras.utils.to_categorical(actions, self.action_size)
        
        _, state_values, _ = self.model.predict([states, hi_sts])
        last_next_state_values = state_values[-1, 0]
        target = discount_TD_error(rewards, last_next_state_values, done, self.gamma) 
        #adv = r + gamma*next_v - v 
        advantage_fn = rewards + self.gamma * state_values[1:,:] - state_values[:-1,:]
        
        train_ret = self.a2c_train_fn([states[:-1, :], hi_sts[:-1, :], one_hot_actions, target, advantage_fn])
        #critic
        if self.update_target_counter % self.args.ac_update_target_frequency == 0:
            weights = self.model.get_weights()
            target_weights = self.model_target.get_weights()
            for i in range(len(weights)):
                target_weights[i] = self.args.TAU * weights[i] + (1. - self.args.TAU) * target_weights[i]
            self.model_target.set_weights(target_weights)
            #print('update target')
        #print('update target')
        self.update_target_counter += 1
        return train_ret
    def load(self, name):
        self.model.load_weights(name)
        self.model_target.set_weights(self.model.get_weights())
    def save(self, name):
        self.model_target.save_weights(name)
    def setup_summary(self):
        total_loss = tf.Variable(0.)
        actor_loss = tf.Variable(0.)
        critic_loss = tf.Variable(0.)
        entropy = tf.Variable(0.)
        p1 = tf.Variable(0.) #probs of acitons
        p2 = tf.Variable(0.) #probs of acitons
        p3 = tf.Variable(0.) #probs of acitons
        episode_avg_score = tf.Variable(0.)

        tf.summary.scalar('Total Loss / Updates', total_loss)
        tf.summary.scalar('Actor Loss / Updates', actor_loss)
        tf.summary.scalar('Critic Loss / Updates', critic_loss)
        tf.summary.scalar('Entropy / Updates', entropy)
        '''
        tf.summary.scalar('Action1 / Updates', p1)
        tf.summary.scalar('Action2 / Updates', p2)
        tf.summary.scalar('Action3 / Updates', p3)
        '''
        tf.summary.scalar('Average Score/Episode', episode_avg_score)

        summary_vars = [total_loss, actor_loss,
        #                critic_loss, entropy,p1,p2,p3]#, episode_avg_score]
                        critic_loss, entropy, episode_avg_score]#, episode_avg_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


