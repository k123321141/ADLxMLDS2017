import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

EPISODES = 5000000

from main import parse
from environment import Environment
import sys,os
from os.path import expanduser,join
home = expanduser("~")

MODEL_PATH      =   join(home,'rl','/breakout_dqn.h5')
SUMMARY_PATH    =   join(home,'rl','summary','/breakout_dqn.h5')
SAVE_INTERVAL   =   100
STATE_WIDTH     =   84
STATE_HEIGHT    =   84
STATE_LENGTH    =   4
class DQNAgent:
    def __init__(self, action_size):
        self.load_model = True 
        # environment settings
        self.state_size = (STATE_WIDTH, STATE_HEIGHT, STATE_LENGTH)
        self.action_size = action_size
        # parameters about epsilon
        #self.epsilon = 1.
        self.epsilon = 0.2
        self.epsilon_end = 0.001
        self.exploration_steps = 500000.
        self.epsilon_decay_step = (self.epsilon - self.epsilon_end) \
                                  / self.exploration_steps
        # parameters about training
        self.batch_size = 32
        #self.train_start = 50000
        self.train_start = 10000
        #self.train_start = 3000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        #self.memory = deque(maxlen=400000)
        self.memory = deque(maxlen=100000)
        self.no_op_steps = 30
        # build model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
           SUMMARY_PATH , self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model and os.path.isfile(MODEL_PATH):
            print('load model from %s.' % MODEL_PATH)
            self.model.load_weights(MODEL_PATH)
        else:
            print('train a new model.')

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.square(y - q_value)
        """
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        """
        error = K.clip(error, 0.0, 3.0)
        loss = K.mean(error)
        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0])
            next_history[i] = np.float32(mini_batch[i][3] )
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def save_model(self, name):
        self.model.save_weights(name)

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


    def test(self,env):
        DO_RENDER = True
        if DO_RENDER:
            from gym.envs.classic_control import rendering
            viewer = rendering.SimpleImageViewer()
        done = True
        while True:
            if done:
                observe = env.reset()
            state = np.reshape(observe, (1, STATE_WIDTH, STATE_HEIGHT, STATE_LENGTH))
            if DO_RENDER:
                rgb = env.env.render('rgb_array')
                #rgb render
                upscaled=repeat_upsample(rgb,3, 3)
                viewer.imshow(upscaled)

            action = agent.get_action(state)
            # change action to real_action
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)
            """
            if reward != 0:
                print(reward)
            """

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)
if __name__ == "__main__":
    #env = gym.make('BreakoutDeterministic-v4')
    args = parse()
    env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper = True)
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    if args.test:
        DO_RENDER = True
        agent.test(env)
    else:

        for e in range(EPISODES):
            done = False
            dead = False
            # 1 episode = 5 lives
            step, score, start_life = 0, 0, 5
            observe = env.reset()

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, agent.no_op_steps)):
                observe, _, _, _ = env.step(1)

            # At start of episode, there is no preceding frame
            # So just copy initial states to make history
            state = observe
            state = np.reshape(state, (1, STATE_WIDTH, STATE_HEIGHT, STATE_LENGTH))
            history = state

            while not done:
                global_step += 1
                step += 1

                # get action for the current history and go one step in environment
                action = agent.get_action(history)
                # change action to real_action
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                observe, reward, done, info = env.step(real_action)
                # pre-process the observation --> history
                next_state = observe
                next_state = np.reshape([next_state], (1, STATE_WIDTH, STATE_HEIGHT, STATE_LENGTH))
                next_history = next_state 

                agent.avg_q_max += np.amax(
                    agent.model.predict(np.float32(history))[0])

                # if the agent missed ball, agent is dead --> episode is not over
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                reward = np.clip(reward, -1., 1.)

                # save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(history, action, reward, next_history, dead)
                # every some time interval, train model
                agent.train_replay()
                # update the target model with model
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

                score += reward

                # if agent is dead, then reset the history
                if dead:
                    dead = False
                else:
                    history = next_history

                # if done, plot the score over episodes
                if done:
                    mode = 'random'
                    if global_step > agent.train_start:
                        stats = [score, agent.avg_q_max / float(step), step,
                                 agent.avg_loss / float(step)]
                        for i in range(len(stats)):
                            agent.sess.run(agent.update_ops[i], feed_dict={
                                agent.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = agent.sess.run(agent.summary_op)
                        agent.summary_writer.add_summary(summary_str, e + 1)
                        mode = 'train'
                    if e % 100 == 0: 
                        print("episode:", e, "  score:", score, "  memory length:",
                          len(agent.memory), "  epsilon:", agent.epsilon,
                          "  global_step:", global_step, "  average_q:",
                          agent.avg_q_max / float(step), "  average loss:",
                          agent.avg_loss / float(step), "   mode:",mode
                          )
                        sys.stdout.flush()

                    agent.avg_q_max, agent.avg_loss = 0, 0

            if e % SAVE_INTERVAL == 0 and e > SAVE_INTERVAL:
                print('save model to %s.' % MODEL_PATH)
                agent.model.save_weights(MODEL_PATH)
