import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D
from keras.optimizers import Adam
import sys,os

MODEL_PATH      =   '/tmp/rl/pong_pg_reward.h5'
SUMMARY_PATH    =   '/tmp/rl/summary/pong_pg_reward.h5'
SAVE_INTERVAL   =   10
STATE_WIDTH     =   210
STATE_HEIGHT    =   160
STATE_LENGTH    =   3
DO_RENDER       =   False
BASE_LINE       =   10
REWARD_GAMMA    =   0.2
class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()
        self.base_line = BASE_LINE
        if os.path.isfile(MODEL_PATH):
            print('load model from %s.' % MODEL_PATH)
            self.load(MODEL_PATH)

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

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        #print np.mean(rewards)[0]
        #print('rewards : %d, mean np.mean(rewards)')
        #rewards = rewards / np.std(rewards - np.mean(rewards))
        self.base_line = (1 - REWARD_GAMMA) * self.base_line + REWARD_GAMMA * np.mean(rewards)
        print('base line : %.2f' % float(self.base_line))
        rewards = rewards - (self.base_line / rewards.shape[0])
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def preprocess(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

if __name__ == "__main__":
    env = gym.make("Pong-v0")
    state = env.reset()
    prev_x = None
    score = 0
    episode = 0

    state_size = 80 * 80
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    while True:
        if DO_RENDER:
            env.render()

        cur_x = preprocess(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        prev_x = cur_x

        action, prob = agent.act(x)
        state, reward, done, info = env.step(action)
        score += reward
        agent.remember(x, action, prob, reward)

        if done:
            episode += 1
            agent.train()
            print('Episode: %d - Score: %f.' % (episode, score))
            sys.stdout.flush()
            score = 0
            state = env.reset()
            prev_x = None
            if episode > 1 and episode % SAVE_INTERVAL == 0:
                print('save model to %s.' % MODEL_PATH)
                agent.save(MODEL_PATH)
