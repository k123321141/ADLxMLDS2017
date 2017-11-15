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

