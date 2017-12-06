from agent_dir.agent import Agent
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import *
import random

ENV_NAME = 'breakout'
NUM_EPISODES = 12000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH = 84,84,4
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 300000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 0.80  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 2000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 40000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 1000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.01  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 30000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = True 
TRAIN = True
SAVE_NETWORK_PATH = './saved_networks/' 
SAVE_SUMMARY_PATH = './summary/' + ENV_NAME
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time
DO_RENDER = False

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
        ##################
        # YOUR CODE HERE #
        ##################
        print(dir(env))
        print(env.get_action_space().n)

        self.env = env   
        self.network_init(env.get_action_space().n)

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
        DO_RENDER = True
        if DO_RENDER:
            from gym.envs.classic_control import rendering
            viewer = rendering.SimpleImageViewer()

        if TRAIN:  # Train mode
            for _ in range(NUM_EPISODES):
                terminal = False
                observation = env.reset()
                for _ in range(random.randint(1, NO_OP_STEPS)):
                    last_observation = observation
                    observation, _, _, _ = env.step(0)  # Do nothing
                while not terminal:
                    last_observation = observation
                    action = self.get_action(observation)
                    observation, reward, terminal, _ = env.step(action)
                    if DO_RENDER:
                        rgb = env.env.render('rgb_array')
                        #rgb render
                        upscaled=repeat_upsample(rgb,3, 3)
                        viewer.imshow(upscaled)
                    #
                    self.run(last_observation, action, reward, terminal, observation)
        else:  # Test mode
            # env.monitor.start(ENV_NAME + '-test')
            for _ in range(NUM_EPISODES_AT_TEST):
                terminal = False
                observation = env.reset()
                for _ in range(random.randint(1, NO_OP_STEPS)):
                    last_observation = observation
                    observation, _, _, _ = env.step(0)  # Do nothing
                while not terminal:
                    last_observation = observation
                    action = agent.get_action_at_test(observation)
                    observation, _, terminal, _ = env.step(action)
                    env.render()
                    self.run(last_observation, action, reward, terminal, observation)


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
        print('make action')
        return self.env.get_random_action()
    def network_init(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)
        
        '''
        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)
        '''
        self.sess.run(tf.initialize_all_variables())
        # Load network
        if LOAD_NETWORK:
            self.load_network()
        # Initialize target network
        self.sess.run(self.update_target_network)
    
    
    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (4,4), strides = (2,2), activation='relu', input_shape=(FRAME_WIDTH, FRAME_HEIGHT,STATE_LENGTH), data_format = 'channels_last'))
        #model.add(Conv2D(64, kernel_size = (4,4), strides = (2,2), activation='relu', data_format = 'channels_last'))
        model.add(Conv2D(64, kernel_size = (3,3), strides = (1,1), activation='relu', data_format = 'channels_last'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(4, activation='linear'))
        s = tf.placeholder(tf.float32, [None, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH])
        q_values = model(s)
        print(model.summary())
        return s, q_values, model
    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        '''
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
        '''
        loss = tf.reduce_mean(error)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update


    def get_action(self, observation):
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            #action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            probs = self.q_values.eval(feed_dict={self.s: [np.float32(observation)]})
            action = sgd_action(probs)
            #print softmax(probs[0]),action
        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step
        return action

    def run(self,last_observation, action, reward, terminal, observation):
        '''
        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)
        '''

        # Store transition in replay memory
        self.replay_memory.append((last_observation, action, reward, observation, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(last_observation)]}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                '''
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: tf.cast(stats[i],tf.float32)
                    })
                '''
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            #print('total loss = ' , (self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)) ))
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)) , mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0.
            self.duration = 0
            self.episode += 1

        self.t += 1

    def train_network(self):
        last_observation_batch = []
        action_batch = []
        reward_batch = []
        observation_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            last_observation_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            observation_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(observation_batch) )})
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(last_observation_batch)),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        
        #update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        
        summary_op = tf.summary.merge_all()
        #return summary_placeholders, update_ops, summary_op
        return summary_placeholders, None, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, observation):
        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            #action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            probs = self.q_values.eval(feed_dict={self.s: [np.float32(observation)]})
            action = sgd_action(probs)
            
        self.t += 1

        return action


def sgd_action(probs):
    probs = probs[0]
    probs = softmax(probs)
    temp = 0
    idx = 0
    r = random.random()
    assert len(probs.shape) == 1
    for i in range(len(probs)):
        temp += probs[i]
        if temp > r :
            idx = i
            break
    return idx

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)
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

