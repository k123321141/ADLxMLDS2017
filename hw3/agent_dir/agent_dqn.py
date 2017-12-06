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

