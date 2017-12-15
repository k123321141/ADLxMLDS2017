from agent_dir.agent import Agent
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D
from keras.optimizers import Adam
import scipy
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

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################

        self.state_size = 80 * 80 
        self.env = env
        self.args = args
        self.action_size = env.env.action_space.n #
        self.gamma = args.pg_discount_factor
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.base_line = BASE_LINE
        self.prev_x = None
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        if os.path.isfile(args.pg_model):
            print('load model from %s.' % args.pg_model)
            self.load(args.pg_model)


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
        state = env.reset()
        self.prev_x = None
        self.score = 0
        self.episode = 0

        while True:
            if args.do_render:
                env.render()

            cur_x = self.preprocess(state)
            x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.state_size)
            self.prev_x = cur_x

            action, prob = self.act(x)
            state, reward, done, info = env.step(action)
            self.score += reward
            self.remember(x, action, prob, reward)

            if done:
                self.episode += 1
                self.update()
                print('Episode: %d - Score: %f.' % (self.episode, self.score))
                sys.stdout.flush()
                self.score = 0
                state = env.reset()
                self.prev_x = None
                if self.episode > 1 and self.episode % SAVE_INTERVAL == 0:
                    print('save model to %s.' % MODEL_PATH)
                    #self.save(MODEL_PATH)
    


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
        return action
        #return self.env.get_random_action()
    def _build_model(self):
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
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    #train funcfion
    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

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
    def update(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        #rewards = self.discount_rewards(rewards)
        #rewards = rewards / np.std(rewards - np.mean(rewards))
        print(BASE_LINE)
        rewards -= BASE_LINE
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


