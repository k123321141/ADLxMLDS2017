from __future__ import division,print_function
import keras

from keras import backend as K
from keras.datasets import mnist
from keras.models import *
from keras.layers import *
from keras.initializers import *
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.datasets import imdb



class MyLayer(SimpleRNN):
    def lstm_cell(is_training = True):
        cell = tf.contrib.rnn.BasicLSTMCell(
                hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
    def build_lstm_graph(inputs):
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                #if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
        return output, state
    def __init__(self, **kwargs):
        units = 128
        cell = SimpleRNNCell(units)
        super(SimpleRNN, self).__init__(cell,return_sequences = True,**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.num_steps = input_shape[-2]
        self.kernel = self.add_weight(name='kernel',
                shape=(input_shape[1], self.output_dim),
                initializer='uniform',
                trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return build_lstm_graph(x)
    def compute_output_shape(self, input_shape):
        return (input_shape)
    
    

class GG(GRU):
    @interfaces.legacy_recurrent_support
    def __init__(self, units,attention_vec,
                 activation='tanh',
                 use_bias=True,
                 **kwargs):
        self.implementation = 1
        self.attention_vec = attention_vec
        super(GG,self).__init__(units,implementation = 1, **kwargs)
    def build(self, input_shape):
        super(GG,self).build(input_shape)

        self.timestep_len = input_shape[-2]
        #self.attention_vec = K.random_uniform(shape = (self.input_shape,self.units))

    def step(self, inputs, states):
        y,h =  super(GG,self).step(inputs,states) 
        return y,h


encoder_units = 300
match_units = 128
decoder_units = 100
match_dim  = 200
input_len = 80
def Encoder():
    model = Sequential()
    
    model.add(GRU(encoder_units,activation = 'tanh',return_sequences = True,input_shape = [input_len,4096]))
    return model
def Match():
    encoder_out = Input(shape = [input_len,encoder_units])
    z = Input(shape = [decoder_units])
    
    zz = RepeatVector(input_shape)(z)
    
    
    m = Concatenate(axis = -1)(encoder_out,zz)
    a = Dense(match_dim,activation = 'linear')(m)
    a = Dense(1,activation = 'softmax')(a)
    #(80,1)
    energy = Permute([1,0])(a)
    #(1,80)
    def weight_by_a(energy,x):
        #x (80,300)
        #a (80,1)
        return tf.matmul(energy,x)
    
    def weight_by_a_output_shape(shape_energy,shape_x):
        output_shape = tuple( list(shape_energy)[0] , list(shape_x)[1])
        return output_shape
    context = Lambda(weight_by_energy,output_shape = weight_by_energy_output_shape)(energy,encoder_out)

    model = Model(input=[encoder_out,z],output = context)
    return model

def Decoder():
    context = Input(shape = [1,encoder_units])

def end2end():
    encoder_model = Encoder()
    match_model = Match()
    decoder_model = Decoder()
    
    x = Input(shape = [80,4096])
    cheat_y = Input(shape = [])



    

from keras.models import *
from keras.layers import *
import config
import keras.backend as K
import tensorflow as tf
import numpy as np
import myinput
import config
import utils

assert K._BACKEND == 'tensorflow'

def model(input_len,input_dim,output_len,vocab_dim):
    print('Build model...')
    # Try replacing GRU, or SimpleRNN.
    #
    data = Input(shape=(input_len,input_dim))
    #in this application, input dim = vocabulary dim
    label = Input(shape=(output_len,vocab_dim))
    #masking
    x = data
    y = label
    #scaling data
    x = BatchNormalization()(x)
    #decoder
    for _ in range(config.DEPTH):
        #forward RNN
        ret1 = config.RNN(config.HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = False)(x)
        hi_st = ret1[1:] if config.RNN == LSTM else ret1[1]
        #backward RNN
        ret2  = config.RNN(config.HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = True)(x,initial_state = hi_st)
        #concatenate both side
        x = Concatenate(axis = -1)([ret1[0],ret2[0]])
        #prepare hidden state for encoder
        hi_st = ret2[1:] if config.RNN == LSTM else ret2[1]

    #word embedding
    y = Dense(config.EMBEDDING_DIM,activation = 'linear',use_bias = False)(y)
    y = Masking()(y)
    #encoder
    for _ in range(config.DEPTH):
        y  = config.RNN(config.HIDDEN_SIZE,activation = 'tanh',return_sequences = True)(y,initial_state = hi_st)
        
        #
    y = TimeDistributed(Dense(vocab_dim,activation='softmax'))(y)
    
    
    model = Model(inputs = [data,label],output=y)  
    model.compile(loss=utils.loss_with_mask,
                  optimizer='adam',
                  metrics=[utils.acc_with_mask])
    #model.summary()
    return model

def my_pred(model,x,output_len):
    #(80,4096)
    x = x.reshape([1,80,4096])
    y_pred = myinput.caption_one_hot('<bos>')
    y_pred[0,1:,:] = 0
    for i in range(1,output_len):
        '''
        print 'pred'
        print decode(y_pred[0,:,:]) 
        print np.argmax(y_pred[0,:,:],axis = -1)
        '''
        y = model.predict([x,y_pred])
        next_idx = np.argmax(y[:,i-1,:],axis = -1)[0]
        y_pred[0,i,next_idx] = 1
        #print 'next ',i,next_idx

    return y_pred[0,1:,:]
def batch_pred(model,x,output_len):
    num = x.shape[0]
    y_pred = np.repeat(myinput.caption_one_hot('<bos>'),num,axis = 0)
    y_pred = y_pred.astype(np.float32)
    y_pred[:,1:,:] = 0
    for i in range(1,output_len):
        '''
        print 'pred'
        print decode(y_pred[0,:,:]) 
        print np.argmax(y_pred[0,:,:],axis = -1)
        '''
        y = model.predict([x,y_pred])
        np.copyto(y_pred[:,i,:],y[:,i-1,:])
        #print 'next ',i,next_idx

    return y_pred[:,1:,:]
