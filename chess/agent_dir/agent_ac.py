from agent_dir.agent import Agent
import gym
import numpy as np
from keras.models import *
from keras.layers import * 
from keras.optimizers import Adam
import scipy,keras
import keras.backend as K
import tensorflow as tf
import sys,os
from collections import deque

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

class Agent_AC(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_AC,self).__init__(env)



        self.state_size = 80 * 80 
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
                self.actor, self.critic = self.build_model()
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
        self.learning_rate = 0.0001
        self.actor, self.critic = self.build_model()
        self.baseline = args.ac_baseline 
        self.set_actor_train_fn()
        self.set_critic_train_fn()
        
        #summary
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            args.ac_summary  , self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.states = []
        self.actions = []
        self.rewards = []
        if os.path.isfile(args.ac_model) and args.keep_train:
            print('load model from %s.' % args.ac_model)
            self.load(args.ac_model)
        else:
            print('train a new model')

        self.prev_x = None
        score, win, lose, step,que = 0,0,0,0,deque()
        episode = 0

        terminal = False 
        done = False
        state = env.reset()
        while True:
            if args.do_render:
                env.env.render()
            if terminal:    #game over
                state = env.reset()
                #every 21 point per update 
                self.update_actor_critic()
                self.states, self.actions, self.rewards = [], [], []


                #for log
                episode += 1
                que.append(score)
                print('Episode: %d - Score: %f.' % (episode,score))
                sys.stdout.flush()
                if episode > 1 and episode % args.ac_save_interval == 0:
                    print('save model to %s.' % args.ac_model)
                    self.save(args.ac_model)
                #summary
                stats = [score, win, step, lose, np.mean(que) ]

                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, episode + 1)
                
                score, win, lose, step = 0,0,0,0
                if len(que) > 30:
                    que.popleft()
                    
            cur_x = prepro(state)
            x = cur_x - self.prev_x if self.prev_x is not None else cur_x
            self.prev_x = cur_x

            action = self.act(x)
            state, reward, terminal, info = env.step(self.real_act(action))
            score += reward
            self.remember(x, action, reward)
            done = reward != 0  #someone get the point
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1
            step += 1
            if done:
                self.prev_x = None
    
    def real_act(self, action):
        if action == 0:
            return 0
        elif action == 1:
            return 2
        elif action == 2:
            return 3
        else:
            print('no such action', action)
            sys.exit(1)

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
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.state_size)
        self.prev_x = cur_x
        
        cur_x = cur_x.reshape([1,cur_x.shape[0]])
        prob = self.model.predict(cur_x, batch_size=1).flatten()
        
        #stochastic
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        #action = np.argmax(prob)
        return self.real_act(action)
    def build_model(self):

        pixel_input = Input(shape=(self.state_size,))

        #shared cnn
        x = Reshape((80, 80, 1))(pixel_input)
        x = Conv2D(32, kernel_size=(6, 6), strides=(3, 3), padding='same',
                                activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
        cnn_output = Flatten()(x)


        #actor
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(cnn_output)
        x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        actor_output = Dense(self.action_size, activation='softmax')(x)
        actor = Model(inputs=pixel_input, outputs=actor_output)
        
        #critic
        action_input = Input(shape=(self.action_size,))
        x1 = Dense(64, activation='linear', kernel_initializer='he_uniform')(cnn_output)
        x2 = Dense(64, activation='linear', kernel_initializer='he_uniform')(action_input)
        x = Concatenate(axis=-1)([x1, x2])
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        critic_output = Dense(1, activation='linear')(x)
        critic = Model(inputs=[pixel_input, action_input], outputs=critic_output)
        return actor, critic

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        prob = self.actor.predict(state, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action 

    #train funcfion
    def remember(self, state, action, reward):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
    def set_actor_train_fn(self):
        #polocy gradient loss 
        #counting the loss of every trajectory with discounted reward, then summerize them. 
        action_probs = self.actor.output 
        action_one_hot = K.placeholder(shape=(None, self.action_size), dtype='float32')
        state_value = K.placeholder(shape=(None, ), dtype='float32')
        next_state_value = K.placeholder(shape=(None, ), dtype='float32')
        discounted_rewards = K.placeholder(shape=(None, ), dtype='float32')
        
        advantage_fn = discounted_rewards - (state_value - next_state_value)
        probs = K.sum(action_probs * action_one_hot, axis=1)
        log_probs = K.log(probs)
        loss = -log_probs * discounted_rewards
        loss = K.mean(loss)
        
        opt = Adam(lr=self.learning_rate)
        updates = opt.get_updates(
                                params=self.actor.trainable_weights, 
                                loss=loss)

        self.actor_train_fn = K.function(
                inputs=[self.actor.input, action_one_hot, discounted_rewards, state_value, next_state_value],
                outputs=[loss],
                updates=updates)
    def set_critic_train_fn(self):
        #counting the loss of every trajectory with discounted reward, then summerize them. 
        action_one_hot = K.placeholder(shape=(None, self.action_size), dtype='float32')
        states = K.placeholder(shape=(None, 6400), dtype='float32')
        discounted_rewards = K.placeholder(shape=(None, ), dtype='float32')

        critic_value = self.critic([states, action_one_hot])

        loss = -K.square(critic_value - discounted_rewards)
        loss = K.mean(loss)
        
        opt = Adam(lr=self.learning_rate)
        updates = opt.get_updates(
                                params=self.critic.trainable_weights, 
                                loss=loss)

        self.critic_train_fn = K.function(
                inputs=[states, action_one_hot, discounted_rewards],
                outputs=[loss],
                updates=updates)

    #train funcfion
    def discount_rewards(self, rewards):
        #summerize every trajectory discounted rewards
        #[ 0  0  0 -1  0  0  0  0  1]
        #[-0.97029901 -0.98009998 -0.99000001 -1.          0.96059602  0.970299010.98009998  0.99000001  1.        ]
        discounted_rewards = np.zeros_like(rewards, dtype='float32')
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    #train funcfion
    def update_actor_critic(self):
        actions = np.vstack(self.actions)
        actions = keras.utils.to_categorical(actions, self.action_size).astype(np.float32)
        rewards = np.array(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
         
        X = np.vstack([self.states])
        #state value
        state_values = self.critic.predict([X, actions]).flatten()
        next_state_values = np.zeros(state_values.shape, state_values.dtype)
        np.copyto(next_state_values[:-1], state_values[1:])
        self.actor_train_fn([X, actions, rewards, state_values, next_state_values])
        self.critic_train_fn([X, actions, rewards])

        
        
    def load(self, name):
        self.actor.load_weights(name.replace('.h5','_actor.h5'))
        self.critic.load_weights(name.replace('.h5','_critic.h5'))

    def save(self, name):
        self.actor.save_weights(name.replace('.h5','_actor.h5'))
        self.critic.save_weights(name.replace('.h5','_critic.h5'))
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_win = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_lose = tf.Variable(0.)
        episode_avg_score = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Win/Episode', episode_win)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Lose/Episode', episode_lose)
        tf.summary.scalar('Average Score/Episode', episode_avg_score)

        summary_vars = [episode_total_reward, episode_win,
                        episode_duration, episode_lose, episode_avg_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


