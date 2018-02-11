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
from collections import deque

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

class Agent_DDPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DDPG,self).__init__(env)



        self.state_size = 80 * 80 
        self.env = env
        self.args = args
        
        self.action_size = 3
        #['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        #0, 2, 3
        if args.test_ddpg:
            #you can load your model here
            actor_path = args.ddpg_model.replace('.h5','_actor.h5')
            critic_path = args.ddpg_model.replace('.h5','_critic.h5')
            if os.path.isfile(actor_path) and os.path.isfile(critic_path): 
                print('testing : load model from %s.' % args.ddpg_model)
                self.learning_rate = 0.
                self.prev_x = None
                self.action_size = 3
                self.actor, self.critic, _ = self.build_model()

                self.load(args.ddpg_model)
            else:
                print('no model for testing!\nerror path %s' % args.ddpg_model)
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


        self.gamma = args.ddpg_discount_factor
        self.learning_rate = 0.0001
        self.actor, self.critic = self.build_model()
        self.actor_target, self.critic_target = self.build_model()
        self.baseline = args.ddpg_baseline 
        self.set_ddpg_train_fn()
        #summary
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            args.ddpg_summary  , self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.reply_buffer = deque(maxlen=args.reply_buffer)
        self.update_target_counter = 0
        
        actor_path = args.ddpg_model.replace('.h5','_actor.h5')
        critic_path = args.ddpg_model.replace('.h5','_critic.h5')
        if os.path.isfile(actor_path) and os.path.isfile(critic_path) and args.keep_train: 
            print('load model from %s.' % args.ddpg_model)
            self.load(args.ddpg_model)
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
                self.update_reply_buffer()
                self.updates(len(self.actions))
                self.states, self.next_states, self.actions, self.rewards = [], [], [], []


                #for log
                episode += 1
                que.append(score)
                print('Episode: %d - Score: %f.' % (episode,score))
                sys.stdout.flush()
                if episode > 1 and episode % args.ddpg_save_interval == 0:
                    print('save model to %s.' % args.ddpg_model)
                    self.save(args.ddpg_model)
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
            next_state, reward, terminal, info = env.step(self.real_act(action))
            next_x = prepro(next_state)
            next_x = next_x - x
            
            score += reward
            self.remember(x, next_x, action, reward)
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
        x = Reshape((80, 80, 1), name='shared_reshape')(pixel_input)
        x = Conv2D(32, kernel_size=(6, 6), strides=(3, 3), padding='same', name='shared_conv2d',
                                activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
        x = Flatten(name='shared_flatten')(x)
        self.shared_net = Model(inputs=pixel_input, outputs=x, name='shared_network')
        cnn_output = self.shared_net(pixel_input)
        
        #actor
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(cnn_output)
        x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        actor_output = Dense(self.action_size, activation='softmax')(x)
        actor = Model(inputs=pixel_input, outputs=actor_output)
        
        #critic
        action_input = Input(shape=(self.action_size,))
        x = Concatenate(axis=-1)([cnn_output, action_input])
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
    def remember(self, state, next_state, action, reward):
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.actions.append(action)
    def update_reply_buffer(self):
        actions = np.vstack(self.actions)
        actions = keras.utils.to_categorical(actions, self.action_size).astype(np.float32)
        rewards = np.array(self.rewards)
        rewards = self.discount_rewards(rewards)
        
        num = len(self.rewards)
        for i in range(num):
            buf = [self.states[i:i+1], self.next_states[i:i+1], rewards[i:i+1], actions[i:i+1,:]]
            self.reply_buffer.append(buf)
    def set_ddpg_train_fn(self):
        #critic gradient loss 
        #counting the loss of every trajectory with discounted reward, then summerize them. 
        states = K.placeholder(shape=(None, 6400), dtype='float32')
        next_state_value = K.placeholder(shape=(None, 1), dtype='float32')
        discounted_rewards = K.placeholder(shape=(None, 1), dtype='float32')
        action_one_hot = K.placeholder(shape=(None, self.action_size), dtype='float32')
        
        action_probs = self.actor([states,])
        critic_value = self.critic([states, action_one_hot])

        actor_loss = - K.sum(self.critic([states, action_probs]) )
        critic_loss = K.sum(K.square( (discounted_rewards + next_state_value) - critic_value))
       
        loss = actor_loss + critic_loss 

        #trainable_weights
        trainable_weights = []
        for wi in self.actor.trainable_weights:
            if wi not in trainable_weights:
                trainable_weights.append(wi)
        for wi in self.critic.trainable_weights:
            if wi not in trainable_weights:
                trainable_weights.append(wi)

        opt = Adam(lr=self.learning_rate)
        updates = opt.get_updates(
                                params=trainable_weights, 
                                loss=loss)
        self.ddpg_train_fn = K.function(
                inputs=[states, next_state_value, discounted_rewards, action_one_hot],
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
    def updates(self, batch_size):
        batch = random.sample(self.reply_buffer, batch_size)
        states, next_states, rewards, actions = [],[],[],[]
        for b in batch:
            states.append(b[0])
            next_states.append(b[1])
            rewards.append(b[2])
            actions.append(b[3])
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        rewards = np.vstack(rewards)
        actions = np.vstack(actions)
        
        #need more details 
        next_state_values = self.critic_target.predict([next_states, self.actor_target.predict(next_states)])
        self.ddpg_train_fn([states, next_state_values, rewards, actions]) 
        if self.update_target_counter % self.args.update_target_frequency == 0:
            self.update_target_networks()


    def update_target_networks(self):
        #actor
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.args.TAU * actor_weights[i] + (1. - self.args.TAU) * actor_target_weights[i]
        self.actor_target.set_weights(actor_target_weights)
        #critic
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.args.TAU * critic_weights[i] + (1. - self.args.TAU) * critic_target_weights[i]
        self.critic_target.set_weights(critic_target_weights)
        
    def load(self, name):
        actor_path = name.replace('.h5','_actor.h5')
        critic_path = name.replace('.h5','_critic.h5')
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

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


