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
    assert x[-1] != 0
    for i in x[:-1]:
        assert i == 0
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
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
        self.actor, self.critic, self.model = self.build_model(env.action_space.n, 6400)
        self.actor_target, self.critic_target, self.model_target = self.build_model(env.action_space.n, 6400)
        self.baseline = args.ac_baseline 
        self.set_a2c_train_fn()
        
        #summary
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            args.ac_summary  , self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.global_step = 0
        self.update_counter = 0
        self.update_target_counter = 0
        self.reply_buffer = deque(maxlen=args.reply_buffer)
        if os.path.isfile(args.ac_model) and args.keep_train: 
            print('load model from %s.' % args.ac_model)
            self.load(args.ac_model)
        else:
            print('train a new model')

        self.prev_x = None
        score, win, lose, step,que = 0,0,0,0,deque(maxlen=30)
        episode = 0

        terminal = False 
        done = False
        state = env.reset()
        self.states, self.next_states, self.actions, self.rewards, self.done, self.hi_sts = [], [], [], [], [], []
        hi_st = np.zeros([1, hidden_state_units], dtype=np.float32)
        while True:
            self.global_step += 1
            if args.do_render:
                env.env.render()
            if terminal:    #game over
                state = env.reset()
                #every 21 point per update 


                #for log
                episode += 1
                que.append(score)
                print('Episode: %d - Score: %2.1f - Update Counter: %5d ' % (episode, score, self.update_counter ))
                sys.stdout.flush()
                if episode > 1 and episode % args.ac_save_interval == 0:
                    print('save model to %s.' % args.ac_model)
                    self.save(args.ac_model)


                
                score, win, lose, step = 0,0,0,0
                    

            #random eploration action
            action, next_hi_st = self.act([state, hi_st])
            next_state, reward, terminal, info = env.step(action)
            
            score += reward
            done = reward != 0  #someone get the point
            self.remember(state, next_state, action, reward, done, hi_st)
            hi_st = next_hi_st
            state = next_state
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1
            step += 1
            if done:
                self.prev_x = None
                self.update_reply_buffer()
                loss, actor_loss, critic_loss, entropy = self.update_actor_critic()
                self.states, self.next_states, self.actions, self.rewards, self.done, self.hi_sts = [], [], [], [], [], []
                hi_st = np.zeros([1, hidden_state_units], dtype=np.float32)
                
                
                self.update_counter += 1

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
    def build_model(self, action_size, state_size):
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

    def act(self, inputs):
        state, hi_st = inputs
        state = state.reshape([1, -1])
        probs, _, hi_st = self.model.predict([state, hi_st]) 
        
        probs = probs.flatten()
        action = np.random.choice(self.action_size, 1, p=probs)[0]
        return action, hi_st 

    #train funcfion
    def remember(self, state, next_state, action, reward, done, hi_st):
        state = state.reshape([-1])
        next_state = next_state.reshape([-1])
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.done.append(done)
        self.hi_sts.append(hi_st)
    
    def update_reply_buffer(self ):
        actions = np.vstack(self.actions)
        rewards = np.array(self.rewards)
        hi_sts = np.vstack(self.hi_sts)
        discounted_rewards = discount(rewards, self.gamma) 
        num = len(self.rewards)
        for i in range(num):
            buf = [self.states[i:i+1], self.next_states[i:i+1], rewards[i:i+1], actions[i:i+1,:], \
                    self.done[i], discounted_rewards[i:i+1], hi_sts[i:i+1,:]]
            self.reply_buffer.append(buf)
    def set_a2c_train_fn(self):
        #polocy gradient loss 
        #counting the loss of every trajectory with discounted reward, then summerize them. 
        actor_states = K.placeholder(shape=(None, 6400), dtype='float32')
        critic_states = K.placeholder(shape=(None, 6400), dtype='float32')
        actor_hi_st = K.placeholder(shape=(None, hidden_state_units), dtype='float32')
        critic_hi_st = K.placeholder(shape=(None, hidden_state_units), dtype='float32')
        target = K.placeholder(shape=(None, 1), dtype='float32')
        one_hot_actions = K.placeholder(shape=(None, self.action_size), dtype='float32')
        advantage_fn = K.placeholder(shape=(None, 1), dtype='float32')

        action_probs = self.actor([actor_states, actor_hi_st])
        action_probs = tf.clip_by_value(action_probs, 1e-20, 1.0)
        critic_value = self.critic([critic_states, critic_hi_st]) 

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
                inputs=[actor_states, actor_hi_st, critic_states, critic_hi_st, one_hot_actions, target, advantage_fn],
                outputs=[loss, actor_loss, critic_loss, entropy],
                updates=updates)
        

    #train funcfion
    def update_actor_critic(self):
        #for actor
        a_states = np.vstack(self.states)
        next_states = np.vstack(self.next_states)
        a_hi_sts = np.vstack(self.hi_sts)
        actions = np.vstack(self.actions)
        rewards = np.array(self.rewards).reshape([-1,1])
        
        state_values = self.critic_target.predict([a_states, a_hi_sts]) 
        next_state_values = self.critic_target.predict([next_states, a_hi_sts])
        next_state_values[-1,0] = 0

        one_hot_actions = keras.utils.to_categorical(actions, self.action_size)
        advantage_fn = rewards + self.gamma*next_state_values - state_values 
        #advantage_fn = next_state_values - state_values
        #advantage_fn[-1, 0] = state_values[-1, 0]
        
        #for critic
        #state value
        '''
        batch = random.sample(self.reply_buffer, min(len(self.rewards), len(self.reply_buffer)))
        states, next_states, rewards, actions, done, discounted_rewards, c_hi_sts = [],[],[],[],[],[],[]
        for b in batch:
            states.append(b[0])
            #next_states.append(b[1])
            #rewards.append(b[2])
            #actions.append(b[3])
            #done.append(b[4])
            discounted_rewards.append(b[5])
            c_hi_sts.append(b[6])
        c_states = np.vstack(states)
        c_hi_sts = np.vstack(c_hi_sts)
        #next_states = np.vstack(next_states)
        #actions = np.vstack(actions)
        #rewards = np.vstack(rewards)
        discounted_rewards = np.vstack(discounted_rewards)
        
        target = discounted_rewards
        '''
        target = discount(rewards, self.gamma)
        c_states = a_states
        c_hi_sts = a_hi_sts
        #self.a2c_train_fn([states, one_hot_actions, discounted_rewards, advantage_fn])
        loss, actor_loss, critic_loss, entropy = self.a2c_train_fn([a_states, a_hi_sts, c_states, c_hi_sts, one_hot_actions, target, advantage_fn])
        #print('loss : %4.4f  actor_loss : %4.4f critic_loss : %4.4f entropy : %4.4f'  % (loss, actor_loss, critic_loss, entropy))
        self.update_target_networks()
        return loss, actor_loss, critic_loss, entropy
        
    def update_target_networks(self):
        '''
        #critic
        
        if self.update_target_counter % self.args.ac_update_target_frequency == 0:
            weights = self.model.get_weights()
            target_weights = self.model_target.get_weights()
            for i in range(len(weights)):
                target_weights[i] = self.args.TAU * weights[i] + (1. - self.args.TAU) * target_weights[i]
            self.model_target.set_weights(target_weights)
            #print('update target')
        #print('update target')
        self.actor_target.set_weights(self.actor.get_weights())
        self.update_target_counter += 1
        '''
        self.model_target.set_weights(self.model.get_weights())
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
        #episode_avg_score = tf.Variable(0.)

        tf.summary.scalar('Total Loss / Updates', total_loss)
        tf.summary.scalar('Actor Loss / Updates', actor_loss)
        tf.summary.scalar('Critic Loss / Updates', critic_loss)
        tf.summary.scalar('Entropy / Updates', entropy)
        tf.summary.scalar('Action1 / Updates', p1)
        tf.summary.scalar('Action2 / Updates', p2)
        tf.summary.scalar('Action3 / Updates', p3)
        #tf.summary.scalar('Average Score/Episode', episode_avg_score)

        summary_vars = [total_loss, actor_loss,
                        critic_loss, entropy,p1,p2,p3]#, episode_avg_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


