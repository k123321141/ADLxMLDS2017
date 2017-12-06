from agent_dir.agent import Agent
import tensorflow as tf

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
        print('init')
        ##################
        # YOUR CODE HERE #
        ##################
        print(env.observation_spec)
        NUM_EPISODES = 12000  # Number of episodes the agent plays
        STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
        FRAME_WIDTH = 210
        FRAME_HEIGHT = 160
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


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        print('init game 1')
        pass
        print('init game 2')


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        print('init train 1')
        pass
        print('init train 2')

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (8,8), activation='relu', input_shape=(210, 160, 3), data_format = 'channels_last'))
        model.add(MaxPooling2D((2,2) ))
        model.add(Conv2D(64, kernel_size = (4,4), activation='relu'))
        model.add(MaxPooling2D((2,2) ))
        model.add(Conv2D(64, kernel_size = (3,2), activation='relu'))
        model.add(MaxPooling2D((2,2) ))
        model.add(Conv2D(32, kernel_size = (8,8), activation='relu'))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(4, activation='linear'))

        s = tf.placeholder(tf.float32, [None, 210, 160, 3])
        q_values = model(s)

        return s, q_values, model

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

