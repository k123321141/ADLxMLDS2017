from agent_dir.agent import Agent

from collections import deque
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input, Add, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D
from keras import backend as K
import os,random,sys
import tensorflow as tf
import numpy as np
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        print(args) 
        self.env = env
        self.args = args
        self.state_size = (84, 84, 4)
        self.action_size = env.action_space.n - 1   #ignore nop action

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            if args.dqn_dueling and os.path.isfile(args.dqn_model):
                print('load duel network model from %s.' % args.dqn_model)
                self.model = self.build_dueling_model()
                self.model.load_weights(args.dqn_model)
            elif os.path.isfile(args.dqn_model):
                print('load model from %s.' % args.dqn_model)
                self.model = self.build_model()
                self.model.load_weights(args.dqn_model)


            

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
        args = self.args
        #self.epsilon = 1.
        self.epsilon = args.dqn_epsilon
        self.epsilon_end = args.dqn_epsilon_end
        self.exploration_steps = args.dqn_exploration_steps
        self.epsilon_decay_step = (self.epsilon - self.epsilon_end) / self.exploration_steps
        # parameters about training
        self.batch_size = args.dqn_batch
        #self.train_start = 50000
        self.train_start = args.dqn_train_start
        #self.train_start = 3000
        self.update_target_rate = args.dqn_update_target 
        self.discount_factor = args.dqn_discount_factor
        #self.memory = deque(maxlen=400000)
        self.memory = deque(maxlen=args.dqn_memory)
        self.no_op_steps = args.dqn_no_ops
        # build model
        if args.dqn_dueling:
            self.model = self.build_dueling_model()
            self.model.name = 'dueling_network'        
            self.target_model = self.build_dueling_model()
            self.target_model.name = 'dueling_target_network'
        else:
            self.model = self.build_model()
            self.model.name = 'evaluate_network'
            self.target_model = self.build_model()
            self.target_model.name ='target_network'
        self.update_target_model()


        self.optimizer = self.optimizer()
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            args.dqn_summary  , self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        
        if self.args.keep_train:
            if args.dqn_dueling: 
                if os.path.isfile(args.dqn_model):
                    print('load duel network model from %s.' % args.dqn_model)
                    self.model.load_weights(args.dqn_model)
            elif os.path.isfile(self.args.dqn_model):
                print('load model from %s.' % args.dqn_model)
                self.model.load_weights(args.dqn_model)
            else:
                print('train a new model.')
        print('Training Mode : double DQN:[%s]  duel network:[%s]' % (args.dqn_double_dqn, args.dqn_dueling) )
        
        #training iteration
        scores, episodes, global_step = [], [], 0
        env = self.env
        STATE_WIDTH, STATE_HEIGHT, STATE_LENGTH = (84, 84, 4)
        e = 0
        step, score = 0, 0
        t = 0.
        while e <= args.dqn_max_spisode:
            done = False
            dead = False
            # 1 episode = 5 lives
            observe = env.reset()

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, self.no_op_steps)):
                observe, _, _, _ = env.step(env.get_random_action())

            # At start of episode, there is no preceding frame
            # So just copy initial states to make history
            state = np.reshape(observe, (1, STATE_WIDTH, STATE_HEIGHT, STATE_LENGTH))
            history = state

            while not dead:
                global_step += 1
                step += 1

                # get action for the current history and go one step in environment
                action = self.get_action(history)
                # change action to real_action
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                observe, reward, dead, info = env.step(real_action)
                # pre-process the observation --> history
                next_state = np.reshape(observe, (1, STATE_WIDTH, STATE_HEIGHT, STATE_LENGTH))
                next_history = next_state 

                self.avg_q_max += np.amax(
                    self.model.predict(np.float32(history))[0])

                # if the agent missed ball, agent is dead --> episode is not over
                if info['ale.lives'] == 0:
                    done = True

                #reward = np.clip(reward, -1., 1.)

                # save the sample <s, a, r, s'> to the replay memory
                self.replay_memory(history, action, reward, next_history, dead)
                # every some time interval, train model
                self.train_replay()
                # update the target model with model
                if global_step % self.update_target_rate == 0:
                    self.update_target_model()

                score += reward
                #print('dead done reward score',dead,done,reward,score)
                # if agent is dead, then reset the history
                if not dead:
                    history = next_history
                '''
                else:
                    print('dead',info['ale.lives'])
                if global_step %100 == 0:
                    from time import time
                    print('%.1f' % (time()-t))
                    t = time()
                '''
                # if done, plot the score over episodes
                if done:
                    mode = 'train' if global_step > self.train_start else 'random'
                    if global_step > self.train_start:
                        stats = [score, self.avg_q_max / float(step), step,
                                 self.avg_loss / float(step)]
                        for i in range(len(stats)):
                            self.sess.run(self.update_ops[i], feed_dict={
                                self.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = self.sess.run(self.summary_op)
                        self.summary_writer.add_summary(summary_str, e + 1)
                    if e % 10 == 0: 
                        print("episode:", e, "  score:", score, "  memory length:",
                          len(self.memory), "  epsilon:", self.epsilon,
                          "  global_step:", global_step, "  average_q:",
                          self.avg_q_max / float(step), "  average loss:",
                          self.avg_loss / float(step), "   mode:",mode
                          )
                        sys.stdout.flush()

                    self.avg_q_max, self.avg_loss = 0, 0
                    if e % args.dqn_save_interval == 0 and e >= args.dqn_save_interval:
                        if self.args.dqn_dueling:
                            print('save duel network model to %s.' % args.dqn_model)
                            self.model.save_weights(args.dqn_model)
                        else:
                            print('save model to %s    with double dqn : %s' % (args.dqn_model, self.args.dqn_double_dqn) )
                            self.model.save_weights(args.dqn_model)
                    e += 1
                    score = 0


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        state = np.reshape(observation, (1, 84, 84, 4))
        q_value = self.model.predict(state)
        action =  np.argmax(q_value[0])
        if action == 0:
            real_action = 1
        elif action == 1:
            real_action = 2
        else:
            real_action = 3
        return real_action
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
        #model.summary()
        return model
    def build_dueling_model(self):
        state = Input(shape=(self.state_size))

        buf = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(state)
        buf = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(buf)
        buf = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(buf)
        buf = Flatten()(buf)
        #action
        a = Dense(512, activation='relu')(buf)
        a = Dense(self.action_size)(a)
        #value
        v = Dense(512, activation='relu')(buf)
        v = Dense(1)(v)
        v = RepeatVector(self.action_size)(v)
        v = Reshape([self.action_size])(v)
        #sum
        q = Add()([v,a])
        model = Model(inputs=state, outputs=q)
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
        if self.args.dqn_double_dqn:
            eval_act = np.argmax(self.model.predict(next_history), axis = -1)
            #assert eval_act.shape == (self.batch_size,)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                if self.args.dqn_double_dqn:
                    target[i] = reward[i] + self.discount_factor * \
                                            target_value[i,eval_act[i]]
                else:
                    target[i] = reward[i] + self.discount_factor * \
                                            np.amax(target_value[i])
        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]


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

