from agent_dir.agent import Agent
import gym
import numpy as np

from keras.models import *
from keras.layers import * 
from keras.optimizers import * 
import scipy,keras
import keras.backend as K
import tensorflow as tf
import sys,os,random
import scipy.signal
import signal, threading, time
from collections import deque
import multiprocessing
from environment import getPongEnv

#hidden_state_units = 256
hidden_state_units = 64
graph = tf.get_default_graph()
np.random.seed(1337)
tf.set_random_seed(1337)
def discount(x, gamma):
    '''
    buf = x.flatten().tolist()
    assert buf[-1] != 0
    for a in buf[:-1]:
        if a != 0.:
            print(buf)
            ss = scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
            print(ss.flatten().tolist())
        assert a == 0.
    '''

    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def build_model(action_size, state_size, scope):
    with tf.variable_scope(scope):
        pixel_input = Input(shape=(state_size,))
        hi_st = Input(shape=(hidden_state_units,))
        #actor
        x = Reshape((80, 80, 1))(pixel_input)
        for i in range(2):
            x = Conv2D(32 * 2**i, kernel_size=(5, 5), strides=(3, 3), padding='same',
            #x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', \
                                    activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
        cnn_out = Reshape([1,-1])(x)
        
        x, st = GRU(hidden_state_units, activation='relu',return_state=True)(cnn_out, hi_st)
        
        actor_output = Dense(action_size, activation='softmax')(x)
        actor = Model(inputs=[pixel_input, hi_st], outputs=actor_output)

        
        #critic
        critic_output = Dense(1, activation='linear')(x)
        critic = Model(inputs=[pixel_input, hi_st], outputs=critic_output)

        #whole model
        model = Model(inputs=[pixel_input, hi_st], outputs=[actor_output, critic_output, hi_st])
        '''
        x = ram_input = Input(shape=(128,))
        hi_st = Input(shape=(hidden_state_units,))
        #actor
        x = Reshape([1,-1])(x)
        x, st = GRU(hidden_state_units, activation='relu',return_state=True, kernel_initializer='glorot_normal')(x, hi_st)
        
        actor_output = Dense(action_size, activation='softmax', kernel_initializer='glorot_normal')(x)
        actor = Model(inputs=[ram_input, hi_st], outputs=actor_output)

        
        #critic
        critic_output = Dense(1, activation='linear')(x)
        critic = Model(inputs=[ram_input, hi_st], outputs=critic_output)
        #whole model
        model = Model(inputs=[ram_input, hi_st], outputs=[actor_output, critic_output, hi_st])
        '''
    return actor, critic, model 

def set_train_fn(local_info, global_info, action_size, state_size, learning_rate):
    actor, critic, local_net = local_info 
    
    states = K.placeholder(shape=(None, state_size), dtype='float32')
    hi_st = K.placeholder(shape=(None, hidden_state_units), dtype='float32')
    one_hot_actions = K.placeholder(shape=(None, action_size), dtype='float32')
    target = K.placeholder(shape=(None, 1), dtype='float32')
    advantage_fn = K.placeholder(shape=(None, 1), dtype='float32')

    action_probs, critic_value, next_hi_st = local_net([states,hi_st])
    #action_probs = tf.clip_by_value(action_probs, 1e-20, 1.0)
    log_probs = K.log(action_probs)
    actor_loss = -K.sum(K.sum(log_probs * one_hot_actions, axis=-1) * advantage_fn)
    critic_loss = K.sum(K.square(target - critic_value))  
    
    # 0.9*log(0.9)+0.1*log(0.1) = -0.14 > 0.4*log(0.4)+0.6*log(0.6) = -0.29
    entropy = K.sum( K.sum(action_probs * log_probs, axis=-1) )
    loss = actor_loss + 0.5*critic_loss + 0.01 * entropy

    grads = tf.gradients(loss, local_net.trainable_weights)
    grads,_ = tf.clip_by_global_norm(grads, 40.)

    #each worker has own optimizer to update global network
    #opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    opt = tf.train.RMSPropOptimizer(learning_rate=1e-4 *5.)
    #opt = tf.train.AdamOptimizer(1e-4*5.)
    #global
    global_net = global_info
    update_op = opt.apply_gradients(zip(grads, global_net.trainable_weights))
    update_fn = K.function(
            inputs=[states, hi_st, one_hot_actions, target, advantage_fn],
            outputs=[loss, actor_loss, critic_loss, entropy],
                updates=[update_op,])
    predict_fn = K.function(
            inputs=[states, hi_st],
            outputs=[action_probs, critic_value, next_hi_st],
            updates=[])
    #return update_fn, sync_fn, predict_fn 
    return update_fn, predict_fn 


def set_predict_fn(state_size, net):
    states = K.placeholder(shape=(None, state_size), dtype='float32')
    hi_st = K.placeholder(shape=(None, hidden_state_units), dtype='float32')

    action_probs, critic_value, next_hi_st = net([states,hi_st])

    predict_fn = K.function(
            inputs=[states, hi_st],
            outputs=[action_probs, critic_value, next_hi_st],
            updates=[])
    return predict_fn
    
def setup_summary():
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



def worker_thread(worker):
    with graph.as_default():
        worker.train()


class Worker():
    def __init__(self, agent, name, env_name, build_net, global_info):
        self.name = name 
        self.env = getPongEnv(env_name)
        self.global_net = global_info
        self.agent = agent


        self.actor, self.critic, self.local_net = build_net(agent.action_size, agent.state_size, self.name)
        local_info = [self.actor, self.critic, self.local_net]
        self.update_fn, self.predict_fn = set_train_fn(local_info, global_info, agent.action_size, agent.state_size, agent.learning_rate)
    
    def train(self): 
        env = self.env
        self.states,self.actions, self.rewards, self.hi_sts = [],[],[],[],
        

        
        hi_st = np.zeros([1, hidden_state_units], dtype=np.float32)
        state = env.reset()
        steps = 1
        self.pull()
        if 'Worker_0' in self.name:
            summary_placeholders, update_ops, summary_op = setup_summary()
            summary_writer = tf.summary.FileWriter(
                self.agent.args.a3c_summary  , graph)
        while not self.agent.stop:
            #if args.do_render:
            #    env.env.render()
            action, next_hi_st = self.act([state, hi_st])
            next_state, reward, terminal, info = env.step(action)
            self.remember(state, action, reward, hi_st)
            hi_st = next_hi_st 
            state = next_state
            
            #done = reward != 0  #someone get the point

            if terminal or steps % 20 == 0:
                loss, actor_loss, critic_loss, entropy = self.update(terminal, next_state, next_hi_st)
                self.agent.update_count += 1
                self.states,self.actions, self.rewards, self.hi_sts = [],[],[],[],

                self.pull()
                #print('en', entropy)
                '''
                #Noop iteration
                for i in range(17):
                    env.step(0)
                hi_st = np.zeros([1, hidden_state_units], dtype=np.float32)
                #
                #print(self.name, self.agent.update_count)
                '''
                #summary
                if 'Worker_0' in self.name and steps > 2000:
                    s = state.reshape([1, -1])
                    h = hi_st
                    probs, _, _ = self.predict_fn([s, h]) 
                    p1,p2,p3 = probs.flatten().tolist()
                    
                    status = [loss, actor_loss, critic_loss, entropy, p1, p2, p3]

                    for i in range(len(status)):
                        K.get_session().run(update_ops[i], feed_dict={
                            summary_placeholders[i]: float(status[i])
                        })
                    summary_str = K.get_session().run(summary_op)
                    summary_writer.add_summary(summary_str, self.agent.update_count)
            if terminal:    #game over
                state = env.reset()
            steps += 1
        print('quit thread %s' % self.name)
    def remember(self, state, action, reward, hi_st):
        state = state.reshape([1,-1])
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.hi_sts.append(hi_st)
    def update(self, done, next_state, next_hi_st):
        next_state = next_state.reshape([1,-1])
        
        
        states = np.vstack(self.states + [next_state])  #None,80,80 or None,128
        actions = np.vstack(self.actions)   #None, 1
        rewards = np.vstack(self.rewards)   #None, 1
        hi_sts = np.vstack(self.hi_sts + [next_hi_st])  #None,rnn_units
        one_hot_actions = keras.utils.to_categorical(actions, self.agent.action_size)
        
        #print(states.shape, actions.shape, rewards.shape, hi_sts.shape) 
        _, state_values, _ = self.predict_fn([states, hi_sts])
        if done:
            state_values[-1,0] = 0
        target = np.vstack(self.rewards + [state_values[-1,0]])
        target = discount(target, self.agent.gamma)[:-1, :]
        
        #adv = r + gamma*next_v - v 
        #print(rewards.shape, state_values[1:,:].shape, state_values[:-1,:].shape)
        advantage_fn = rewards + self.agent.gamma * state_values[1:,:] - state_values[:-1,:]
        #advantage_fn = discount(advantage_fn, self.agent.gamma)

        loss, actor_loss, critic_loss, entropy = self.update_fn([states[:-1,:], hi_sts[:-1,:], one_hot_actions, target, advantage_fn])
        return loss, actor_loss, critic_loss, entropy 
    def act(self, inputs):
        state, hi_st = inputs
        state = state.reshape([1, -1])
        probs, _, hi_sts = self.predict_fn([state, hi_st]) 
        
        probs = probs.flatten()
        action = np.random.choice(self.agent.action_size, 1, p=probs)[0]
        return action, hi_sts 
     
    #get weights from global net 
    def pull(self):
        self.local_net.set_weights(self.global_net.get_weights())
        #self.sync_fn([])
class Agent_A3C(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_A3C,self).__init__(env)



        #self.state_size = 128
        self.state_size = 6400 
        self.env = env
        self.args = args
        
        self.action_size = self.env.action_space.n
        #['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        #0, 2, 3
        if args.test_a3c:
            K.set_learning_phase(0)
            #you can load your model here
            if os.path.isfile(args.a3c_model): 
                print('testing : load model from %s.' % args.a3c_model)
                self.learning_rate = 0.
                self.prev_x = None
                self.actor, self.critic, self.model = self.build_model('global')

                self.load(args.a3c_model)
            else:
                print('no model for testing!\nerror path %s' % args.a3c_model)
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
        self.learning_rate = 1e-4*5.
        self.gamma = args.a3c_discount_factor        
        self.update_count = 0
        self.global_actor, self.global_critic, self.global_net = build_model(self.action_size, self.state_size, 'global')
        global_info = self.global_net
        K.set_learning_phase(1)
        self.init_weights()

        if os.path.isfile(args.a3c_model) and args.keep_train: 
            print('load model from %s.' % args.a3c_model)
            self.load(args.a3c_model)
        else:
            print('train a new model')
        
        self.predict_fn  = set_predict_fn(self.state_size, self.global_net)

        #multi-threading
        thread_list = []
        self.stop = False
        num_workers = args.a3c_worker_count
        if num_workers == 0:
            num_workers = multiprocessing.cpu_count()

        #for i in range(args.a3c_worker_count):
        print('work on %d cpu with env %s' % (num_workers, env.env.spec.id))
        for i in range(num_workers):
            name = 'Worker_%d' % i
            worker = Worker(self, name, env.env.spec.id, build_model, global_info)
            t = threading.Thread(target=worker_thread, args=[worker, ])
            thread_list.append(t)
        try:
            for t in thread_list:
                t.start()
            state = env.reset()
            score = 0
            episode = 1
            start_time = time.time()
            self.hi_st = np.zeros([1, hidden_state_units], dtype=np.float32)
            mean_score = 0.
            threshold = -21.1
            que = deque([threshold, ],maxlen=3)
            while True:
                action = self.make_action(state)
                state, reward, terminal, info = env.step(action)
                score += reward 
                done = reward != 0
                if terminal:    #game over
                    state = env.reset()
                    self.hi_st = np.zeros([1, hidden_state_units], dtype=np.float32)
                    mean_score = np.mean(que) 
                    que.append(score)
                    #every 21 point per update 
                    print('Episode: %4d - Score: %2.0f - Update Counter: %7d - Spent Time %8.2f - Mean Score %2.4f'  % \
                            (episode, score, self.update_count, time.time()-start_time , mean_score))


                    episode += 1
                    score = 0
                    #if episode > 0 and episode % args.a3c_save_interval == 0:
                    if episode > 0 and mean_score > threshold:
                        threshold = mean_score
                        print('save model to %s' % args.a3c_model)
                        self.save(args.a3c_model)
                    time.sleep(1)

        except (KeyboardInterrupt, Exception) as e:
            print('stop program',e)
            self.stop = True
            for t in thread_list:
                t.join()
            print("Quitting Program.")
        print('Done')

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
        state = observation.reshape([1, -1])
        #print(state[0,:10], np.min(state), np.max(state))
        probs, _, self.hi_st = self.predict_fn([state, self.hi_st])
        probs = probs.flatten()
        
        #stochastic
        action = np.random.choice(self.action_size, 1, p=probs)[0]
        #action = np.argmax(prob)
        return action
    def load(self, name):
        self.global_net.load_weights(name)
    
    def save(self, name):
        self.global_net.save_weights(name)
    def init_weights(self):
        init = tf.global_variables_initializer()
        sess = K.get_session()
        sess.run(init)

