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

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
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
            if os.path.isfile(args.ddpg_model): 
                print('testing : load model from %s.' % args.ddpg_model)
                self.learning_rate = 0.
                self.prev_x = None
                self.action_size = 3
                self.actor, self.critic, self.model = self.build_model()

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
        self.actor, self.critic, self.model = self.build_model()
        self.actor_target, self.critic_target, self.model_target = self.build_model()
        self.baseline = args.ddpg_baseline 
        self.set_a2c_train_fn()
        
        self.train_start = args.ddpg_train_start
        self.epsilon = args.ddpg_epsilon
        self.epsilon_end = args.ddpg_epsilon_end
        self.exploration_steps = args.ddpg_exploration_steps
        self.epsilon_decay_step = (self.epsilon - self.epsilon_end) / self.exploration_steps
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
        self.done = []
        self.global_step = 0
        self.update_counter = 0
        self.update_target_counter = 0
        self.reply_buffer = deque(maxlen=args.reply_buffer)
        
        if os.path.isfile(args.ddpg_model) and args.keep_train: 
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
            self.global_step += 1
            if args.do_render:
                env.env.render()
            if self.epsilon > self.epsilon_end:
                self.epsilon -= self.epsilon_decay_step
            if self.global_step > self.train_start and len(self.reply_buffer) > self.args.ddpg_batch_size:
                if self.global_step % self.args.ddpg_train_frequency == 0:
                    self.update_actor_critic(self.args.ddpg_batch_size)
                    self.update_counter += 1
            if terminal:    #game over
                state = env.reset()
                #every 21 point per update 


                #for log
                episode += 1
                que.append(score)
                print('Episode: %d - Score: %2.1f - Epsilon: %.4f - Update Counter: %5d - Replay Buffer : %5d' % (episode,score,self.epsilon, self.update_counter, len(self.reply_buffer)))
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
                    
            cur_x = prepro(state)
            x = cur_x - self.prev_x if self.prev_x is not None else cur_x
            self.prev_x = cur_x

            #random eploration action
            if np.random.rand() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                action = self.act(x)
            next_state, reward, terminal, info = env.step(self.real_act(action))
            next_x = prepro(next_state)
            next_x = next_x - cur_x
            
            score += reward
            done = reward != 0  #someone get the point
            self.remember(x, next_x, action, reward, done)
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1
            step += 1
            if done:
                self.prev_x = None
                self.update_reply_buffer()
                self.states, self.next_states, self.actions, self.rewards, self.done = [], [], [], [], []
            state = next_state
    
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
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        shared_out = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
        #actor
        x = Dense(16, activation='relu', kernel_initializer='he_uniform')(shared_out)
        actor_output = Dense(self.action_size, activation='softmax')(x)
        actor = Model(inputs=pixel_input, outputs=actor_output)
        
        #critic
        x = Dense(16, activation='relu', kernel_initializer='he_uniform')(shared_out)
        critic_output = Dense(self.action_size, activation='linear')(x)
        critic = Model(inputs=pixel_input, outputs=critic_output)

        #whole model
        model = Model(inputs=pixel_input, outputs=[actor_output, critic_output])
        return actor, critic, model 

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        prob = self.actor_target.predict(state, batch_size=1).flatten()
        #print(prob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action 

    #train funcfion
    def remember(self, state, next_state, action, reward, done):
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.done.append(done)
    
    def update_reply_buffer(self):
        actions = np.vstack(self.actions)
        rewards = np.array(self.rewards)
        discounted_rewards = discount(rewards, self.gamma) 
        num = len(self.rewards)
        for i in range(num):
            buf = [self.states[i:i+1], self.next_states[i:i+1], rewards[i:i+1], actions[i:i+1,:], self.done[i], discounted_rewards[i:i+1]]
            self.reply_buffer.append(buf)
    def set_a2c_train_fn(self):
        #polocy gradient loss 
        #counting the loss of every trajectory with discounted reward, then summerize them. 
        states = K.placeholder(shape=(None, 6400), dtype='float32')
        target = K.placeholder(shape=(None, 1), dtype='float32')
        one_hot_actions = K.placeholder(shape=(None, self.action_size), dtype='float32')

        action_probs, critic_value = self.model([states,])

        actor_loss = - K.sum(action_probs * critic_value)
        critic_loss = K.sum(K.square(target - K.sum(critic_value * one_hot_actions, axis=-1)))  
        
        # 0.9*log(0.9)+0.1*log(0.1) > 0.4*log(0.4)+0.6*log(0.6)
        entropy = K.sum(action_probs * K.log(action_probs))
       
        
        opt = Adam(lr=self.learning_rate)
        #trainable_weights
        updates = opt.get_updates(
                                params=self.actor.trainable_weights, 
                                loss=actor_loss)
        self.actor_train_fn = K.function(
                inputs=[states,],
                outputs=[actor_loss,],
                updates=updates)
        
        opt = Adam(lr=self.learning_rate)
        updates = opt.get_updates(
                                params=self.critic.trainable_weights, 
                                loss=critic_loss)
        self.critic_train_fn = K.function(
                inputs=[states, one_hot_actions, target],
                outputs=[critic_loss,],
                updates=updates)
        
    '''
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
    '''
    #train funcfion
    def update_actor_critic(self, batch_size):
        
        batch = random.sample(self.reply_buffer, batch_size)
        states, next_states, rewards, actions, done, discounted_rewards = [],[],[],[],[],[]
        for b in batch:
            states.append(b[0])
            next_states.append(b[1])
            rewards.append(b[2])
            actions.append(b[3])
            done.append(b[4])
            discounted_rewards.append(b[5])
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        #discounted_rewards = np.vstack(discounted_rewards)
        one_hot_actions = keras.utils.to_categorical(actions, self.action_size)
        next_state_values = self.critic_target.predict(next_states)
        #state value
        action_probs = self.actor_target.predict(next_states)
        for i in range(len(done)):
            if done[i]:
                next_state_values[i,0] = 0
        target = rewards + np.sum(next_state_values * action_probs, axis=-1).reshape([-1,1]) 
        self.critic_train_fn([states, one_hot_actions, target])
        self.actor_train_fn([states,])
        self.update_target_networks()
        
    def update_target_networks(self):
        #critic
        if self.update_target_counter % self.args.ddpg_update_target_frequency == 0:
            weights = self.model.get_weights()
            target_weights = self.model_target.get_weights()
            for i in range(len(weights)):
                target_weights[i] = self.args.TAU * weights[i] + (1. - self.args.TAU) * target_weights[i]
            self.model_target.set_weights(target_weights)
            #print('update target')
        self.update_target_counter +=1
    def load(self, name):
        self.model.load_weights(name)
        self.model_target.load_weights(name)
    
    def save(self, name):
        #self.model.save_weights(name)
        self.model_target.save_weights(name)
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


