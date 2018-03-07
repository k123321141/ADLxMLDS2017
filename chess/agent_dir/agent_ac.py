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
    '''
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()
    '''
    return I
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
class Agent_AC(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_AC,self).__init__(env)



        self.state_size = 128 
        self.env = env
        self.args = args
        
        self.action_size = 3
        #['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        #0, 2, 3
        if args.test_ac:
            #you can load your model here
            if os.path.isfile(args.ac_model): 
                print('testing : load model from %s.' % args.ac_model)
                self.learning_rate = 0.
                self.prev_x = None
                self.action_size = 3
                self.actor, self.critic, self.model = self.build_model()

                self.load(args.ac_model)
            else:
                print('no model for testing!\nerror path %s' % args.ac_model)
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

        env.seed(1337)
        np.random.seed(1337)

        self.gamma = args.ac_discount_factor
        self.learning_rate = 0.0001
        self.actor, self.critic, self.model = self.build_model()
        self.actor_target, self.critic_target, self.model_target = self.build_model()
        self.baseline = args.ac_baseline 
        self.set_a2c_train_fn()
        
        #summary
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            args.ac_summary  , self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.global_step = 0
        self.update_counter = 0
        self.update_target_counter = 0
        self.reply_buffer = deque(maxlen=args.reply_buffer)
        if os.path.isfile(args.ac_model) and args.keep_train: 
            print('load model from %s.' % args.ac_model)
            self.load(args.ac_model)
        else:
            print('train a new model')

        self.prev_x = None
        score, win, lose, step,que = 0,0,0,0,deque(maxlen=30)
        episode = 0

        terminal = False 
        done = False
        state = env.reset()
        self.states, self.next_states, self.actions, self.rewards, self.done, self.hi_sts = [], [], [], [], [], []
        hi_st = np.zeros([1, 16], dtype=np.float32)
        #self.actor.load_weights('./models/pong_pg.h5')
        #self.actor_target.load_weights('./models/pong_pg.h5')
        while True:
            self.global_step += 1
            if args.do_render:
                env.env.render()
            if terminal:    #game over
                state = env.reset()
                #every 21 point per update 


                #for log
                episode += 1
                que.append(score)
                print('Episode: %d - Score: %2.1f - Update Counter: %5d ' % (episode, score, self.update_counter ))
                sys.stdout.flush()
                if episode > 1 and episode % args.ac_save_interval == 0:
                    print('save model to %s.' % args.ac_model)
                    self.save(args.ac_model)


                
                score, win, lose, step = 0,0,0,0
                    
            x = prepro(state)
            #self.prev_x = x

            #random eploration action
            action, next_hi_st = self.act([x, hi_st])
            state, reward, terminal, info = env.step(self.real_act(action))
            next_x = prepro(state)
            
            score += reward
            done = reward != 0  #someone get the point
            self.remember(x, next_x, action, reward, done, hi_st)
            hi_st = next_hi_st
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1
            step += 1
            if done:
                self.prev_x = None
                self.update_reply_buffer()
                loss, actor_loss, critic_loss, entropy = self.update_actor_critic()
                self.states, self.next_states, self.actions, self.rewards, self.done, self.hi_sts = [], [], [], [], [], []
                
                
                self.update_counter += 1
                '''
                p1,p2,p3 = self.prev_probs.tolist()
                
                #summary
                stats = [loss, actor_loss, critic_loss, entropy, p1, p2, p3]

                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.update_counter)
                '''
            if terminal:
                hi_st = np.zeros([1, 16], dtype=np.float32)

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

        ram_input = Input(shape=(128,))
        hi_st = Input(shape=(16,))
        #actor
        
        x = Dense(64, activation='relu')(ram_input)
        x = Dense(32, activation='relu')(ram_input)
        x = Reshape([1,-1])(x)
        x, st = GRU(16, activation='relu',return_state=True)(x, hi_st)
        
        actor_output = Dense(self.action_size, activation='softmax')(x)
        actor = Model(inputs=[ram_input, hi_st], outputs=actor_output)

        
        #critic
        critic_output = Dense(1, activation='linear')(x)
        critic = Model(inputs=[ram_input, hi_st], outputs=critic_output)

        #whole model
        model = Model(inputs=[ram_input, hi_st], outputs=[actor_output, critic_output, hi_st])

        return actor, critic, model 

    def act(self, inputs):
        state, hi_st = inputs
        state = state.reshape([1, state.shape[0]])
        probs, _, hi_sts = self.model.predict([state, hi_st]) 
        
        probs = probs.flatten()
        action = np.random.choice(self.action_size, 1, p=probs)[0]
        return action, hi_sts 

    #train funcfion
    def remember(self, state, next_state, action, reward, done, hi_st):
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.done.append(done)
        self.hi_sts.append(hi_st)
    
    def update_reply_buffer(self ):
        actions = np.vstack(self.actions)
        rewards = np.array(self.rewards)
        hi_sts = np.vstack(self.hi_sts)
        discounted_rewards = discount(rewards*20., self.gamma) 
        num = len(self.rewards)
        for i in range(num):
            buf = [self.states[i:i+1], self.next_states[i:i+1], rewards[i:i+1], actions[i:i+1,:], \
                    self.done[i], discounted_rewards[i:i+1], hi_sts[i:i+1,:]]
            self.reply_buffer.append(buf)
    def set_a2c_train_fn(self):
        #polocy gradient loss 
        #counting the loss of every trajectory with discounted reward, then summerize them. 
        actor_states = K.placeholder(shape=(None, 128), dtype='float32')
        critic_states = K.placeholder(shape=(None, 128), dtype='float32')
        actor_hi_st = K.placeholder(shape=(None, 16), dtype='float32')
        critic_hi_st = K.placeholder(shape=(None, 16), dtype='float32')
        target = K.placeholder(shape=(None, 1), dtype='float32')
        one_hot_actions = K.placeholder(shape=(None, self.action_size), dtype='float32')
        advantage_fn = K.placeholder(shape=(None, 1), dtype='float32')

        action_probs = self.actor([actor_states, actor_hi_st])
        critic_value = self.critic([critic_states, critic_hi_st]) 

        actor_loss = -K.mean(K.log(K.sum(action_probs * one_hot_actions, axis=-1)) * advantage_fn)
        critic_loss = K.mean(K.square(target - critic_value))  
        
        # 0.9*log(0.9)+0.1*log(0.1) = -0.14 > 0.4*log(0.4)+0.6*log(0.6) = -0.29
        entropy = K.mean(action_probs * K.log(action_probs))
       
        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy
        
        opt = Adam(lr=self.learning_rate)
        #trainable_weights
        updates = opt.get_updates(
                                params=self.model.trainable_weights, 
                                loss=loss)
        self.a2c_train_fn = K.function(
                inputs=[actor_states, actor_hi_st, critic_states, critic_hi_st, one_hot_actions, target, advantage_fn],
                outputs=[loss, -actor_loss, critic_loss, entropy],
                updates=updates)
        

    #train funcfion
    def update_actor_critic(self):
        #for actor
        a_states = np.vstack(self.states)
        next_states = np.vstack(self.next_states)
        a_hi_sts = np.vstack(self.hi_sts)
        actions = np.vstack(self.actions)
        rewards = np.array(self.rewards).reshape([-1,1])
        
        state_values = self.critic_target.predict([a_states, a_hi_sts]) 
        next_state_values = self.critic_target.predict([a_states, a_hi_sts])
        next_state_values[-1,0] = 0

        one_hot_actions = keras.utils.to_categorical(actions, self.action_size)
        advantage_fn = rewards*20. + next_state_values - state_values 
        #advantage_fn = next_state_values - state_values
        #advantage_fn[-1, 0] = state_values[-1, 0]
        
        #for critic
        #state value
        batch = random.sample(self.reply_buffer, min(len(self.rewards)*3, len(self.reply_buffer)))
        states, next_states, rewards, actions, done, discounted_rewards, c_hi_sts = [],[],[],[],[],[],[]
        for b in batch:
            states.append(b[0])
            #next_states.append(b[1])
            #rewards.append(b[2])
            #actions.append(b[3])
            #done.append(b[4])
            discounted_rewards.append(b[5])
            c_hi_sts.append(b[6])
        c_states = np.vstack(states)
        c_hi_sts = np.vstack(c_hi_sts)
        #next_states = np.vstack(next_states)
        #actions = np.vstack(actions)
        #rewards = np.vstack(rewards)
        discounted_rewards = np.vstack(discounted_rewards)
        
        target = discounted_rewards
        #self.a2c_train_fn([states, one_hot_actions, discounted_rewards, advantage_fn])
        loss, actor_loss, critic_loss, entropy = self.a2c_train_fn([a_states, a_hi_sts, c_states, c_hi_sts, one_hot_actions, target, advantage_fn])
        #print('loss : %4.4f  actor_loss : %4.4f critic_loss : %4.4f entropy : %4.4f'  % (loss, actor_loss, critic_loss, entropy))
        self.update_target_networks()
        return loss, actor_loss, critic_loss, entropy
        
    def update_target_networks(self):
        #critic
        if self.update_target_counter % self.args.ac_update_target_frequency == 0:
            weights = self.model.get_weights()
            target_weights = self.model_target.get_weights()
            for i in range(len(weights)):
                target_weights[i] = self.args.TAU * weights[i] + (1. - self.args.TAU) * target_weights[i]
            self.model_target.set_weights(target_weights)
            #print('update target')
        #print('update target')
        self.actor_target.set_weights(self.actor.get_weights())
        self.update_target_counter += 1
    def load(self, name):
        self.model.load_weights(name)
        self.model_target.set_weights(self.model.get_weights())
    def save(self, name):
        self.model_target.save_weights(name)
    def setup_summary(self):
        total_loss = tf.Variable(0.)
        actor_loss = tf.Variable(0.)
        critic_loss = tf.Variable(0.)
        entropy = tf.Variable(0.)
        p1 = tf.Variable(0.) #probs of acitons
        p2 = tf.Variable(0.) #probs of acitons
        p3 = tf.Variable(0.) #probs of acitons
        #episode_avg_score = tf.Variable(0.)

        tf.summary.scalar('Total Loss / Updates', total_loss)
        tf.summary.scalar('Actor Loss / Updates', actor_loss)
        tf.summary.scalar('Critic Loss / Updates', critic_loss)
        tf.summary.scalar('Entropy / Updates', entropy)
        tf.summary.scalar('Action1 / Updates', p1)
        tf.summary.scalar('Action2 / Updates', p2)
        tf.summary.scalar('Action3 / Updates', p3)
        #tf.summary.scalar('Average Score/Episode', episode_avg_score)

        summary_vars = [total_loss, actor_loss,
                        critic_loss, entropy,p1,p2,p3]#, episode_avg_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


