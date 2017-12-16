from agent_dir.agent import Agent
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D
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

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)



        self.state_size = 80 * 80 
        self.env = env
        self.args = args
        
        self.action_size = 3
        if args.test_pg:
            #you can load your model here
            if os.path.isfile(args.pg_model):
                print('load model from %s.' % args.pg_model)
                self.learning_rate = 0.
                self.prev_x = None
                self.action_size = 6 if args.pg_old_model else 3
                self.model = self.build_model()
                self.load(args.pg_model)
            else:
                print('no model for testing!\nerror path %s' % args.pg_model)
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
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        env = self.env
        args = self.args


        self.gamma = args.pg_discount_factor
        self.learning_rate = 0.0001
        self.model = self.build_model()
        self.baseline = args.pg_baseline 

        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        if os.path.isfile(args.pg_model) and args.keep_train:
            print('load model from %s.' % args.pg_model)
            self.load(args.pg_model)
        else:
            print('train a new model')
        #summary
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            args.pg_summary  , self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.prev_x = None
        score, win, lose, step,que = 0,0,0,0,deque()
        episode = 1643
        self.optimizer = self.gradient_optimizer()

        terminal = False 
        done = False
        state = env.reset()
        while True:
            if args.do_render:
                env.env.render()
            if terminal:    #game over
                self.update_src()
                state = env.reset()
                episode += 1
                que.append(score)
                print('Episode: %d - Score: %f.' % (episode,score))
                sys.stdout.flush()
                if episode > 1 and episode % args.pg_save_interval == 0:
                    print('save model to %s.' % args.pg_model)
                    self.save(args.pg_model)
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

            action, prob = self.act(x)
            real_action = self.real_act(action)
            state, reward, terminal, info = env.step(real_action)
            score += reward
            self.remember(x, action, prob, reward)
            done = reward != 0  #someone get the point
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1
            step += 1
            if done:
                self.prev_x = None
    


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
        #x = cur_x - self.prev_x if self.prev_x is not None else cur_x
        #src 
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.state_size)
        self.prev_x = cur_x
        
        cur_x = cur_x.reshape([1,cur_x.shape[0]])
        '''
        prob = self.model.predict(cur_x, batch_size=1).flatten()
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        #action = np.argmax(prob)
        return self.real_act(action)
        '''
        prob = self.model.predict(cur_x, batch_size=1).flatten()

        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action
    def build_model(self):
        model = Sequential()
        model.add(Reshape((80, 80, 1), input_shape=(self.state_size,)))
        model.add(Conv2D(32, kernel_size=(6, 6), strides=(3, 3), padding='same',
                                activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        # See note regarding crossentropy in cartpole_reinforce.py
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        prob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(prob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    #train funcfion
    def remember(self, state, action, prob, reward):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
    def gradient_optimizer(self):
        a = K.placeholder(shape=(None,self.action_size), dtype='float32')
        r = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=a)
        #loss = -a * K.log(py_x)
        loss = K.mean(r*loss)
        
        opt = Adam(lr=self.learning_rate)
        updates = opt.get_updates(self.model.trainable_weights, [], loss)

        train = K.function([self.model.input, a, r], [loss], updates=updates)
        return train 

    #train funcfion
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    #train funcfion
    def update_policy(self):
        rewards = np.vstack(self.rewards)
        actions = np.vstack(self.actions)
        x = np.squeeze(np.vstack([self.states]))#trajectory len,6400
        probs = np.vstack(self.probs)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        '''
        reward = np.sum(rewards) - self.baseline
        reward = reward * (self.gamma ** (len(self.rewards)-1))
        reward = np.sum(self.rewards)
        
        a = keras.utils.to_categorical(actions, self.action_size)
        loss = self.optimizer([x, a, rewards.ravel()])
        y = keras.utils.to_categorical(actions, self.action_size)
        #gradients = -reward * self.learning_rate * (y - probs)
        '''
        self.model.train_on_batch(x, y)
        self.states, self.probs, self.actions, self.rewards = [], [], [], []

        #print('after prob ',tt[17:20,:])
        #print('')
    def update_src(self):
        actions = np.vstack(self.actions)
        actions = keras.utils.to_categorical(actions, self.action_size).astype(np.float32)
        probs = np.vstack(self.probs)
        gradients = actions - probs
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.actions, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    def real_act(self, n):
        if self.args.pg_old_model:
            return n
        if n == 0:
            return 0 #nop
        elif n == 1:
            return 2 #up
        elif n == 2:
            return 3 #down
        else:
            print('error action number')
            sys.exit(1)
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


