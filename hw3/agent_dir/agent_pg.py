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

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)

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


        self.prev_x = None
        score = 0
        episode = 0
        self.optimizer = self.gradient_optimizer()

        terminal = False 
        done = False
        state = env.reset()
        while True:
            if args.do_render:
                env.env.render()
            if terminal:    #game over
                state = env.reset()
                episode += 1
                print('Episode: %d - Score: %f.' % (episode,score))
                sys.stdout.flush()
                score = 0
                if episode > 1 and episode % args.pg_save_interval == 0:
                    print('save model to %s.' % args.pg_model)
                    self.save(args.pg_model)
                    
            cur_x = prepro(state).ravel()
            x = cur_x - self.prev_x if self.prev_x is not None else cur_x
            self.prev_x = cur_x

            action, prob = self.act(x)
            real_action = self.real_act(action)
            state, reward, terminal, info = env.step(real_action)
            score += reward
            self.remember(x, action, prob, reward)

            done = reward != 0  #someone get the point
            if done:
                self.update_policy()
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
        cur_x = self.prepro(observation).ravel()
        x = cur_x - self.prev_x if self.prev_x is not None else cur_x 
        self.prev_x = cur_x

        action, prob = self.act(x)
        return self.real_act(action)
        #return self.env.get_random_action()
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
        '''
        #reward is the direction of gradient
        #y tell model how to improve, to get the expected distribution of action
        reward = np.sum(self.rewards)

        a = keras.utils.to_categorical(actions, self.action_size)
        #print(x.shape,a.shape,r.shape)
        loss = self.optimizer([x, a, rewards.ravel()])
        '''
        y = keras.utils.to_categorical(actions, self.action_size)
        #gradients = -reward * self.learning_rate * (y - probs)
        #y = probs + gradients
        print('reward ',reward)
        print('prob ',probs[17:20,:])
        print('y ',y[17:20,:])
        '''
        #self.model.train_on_batch(x, y)
        self.states, self.probs, self.actions, self.rewards = [], [], [], []
        tt = self.model.predict(x)

        #print('after prob ',tt[17:20,:])
        #print('')

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    def real_act(self, n):
        if n == 0:
            return 0 #nop
        elif n == 1:
            return 2 #up
        elif n == 2:
            return 3 #down
        else:
            print('error action number')
            sys.exit(1)


