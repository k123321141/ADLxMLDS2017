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
    data = Input(shape=(input_len,input_dim),name='movie_feats')
    #in this application, input dim = vocabulary dim
    label = Input(shape=(output_len,vocab_dim),name='caption_labels')
    #masking
    x = data
    y = label
    #scaling data
    x = BatchNormalization()(x)
    #encoder, bidirectional RNN
    for _ in range(config.DEPTH):
        #forward RNN
        ret1 = config.RNN(config.HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = False,name = 'forward_RNN_encoder_depth_%d' % _)(x)
        hi_st = ret1[1:] if config.RNN == LSTM else ret1[1]
        #backward RNN
        ret2  = config.RNN(config.HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = True,name = 'backward_RNN_encoder_depth_%d'%_)(x,initial_state = hi_st)
        #concatenate both side
        x = Concatenate(axis = -1,name = 'bidirection_concat_depth_%d'%_)([ret1[0],ret2[0]])
        #prepare hidden state for encoder
        hi_st = ret2[1:] if config.RNN == LSTM else ret2[1]
        x = Dropout(config.DROPOUT)(x)

    #word embedding
    y = Dense(config.EMBEDDING_DIM,activation = 'linear',use_bias = False,name='embedding_look_up')(y)
    #y = Masking()(y)
    #concatenate x and label
    if config.RNN == LSTM:
        hi_concat = Concatenate(axis = -1)(hi_st)
    else:
        hi_concat = hi_st
    c = RepeatVector(output_len,name='peek_hole')(hi_concat)
    y = Concatenate(axis =-1,name='concate_with_labels')([c,y])

    #decoder
    for _ in range(config.DEPTH):
        y  = config.RNN(config.HIDDEN_SIZE,activation = 'tanh',return_sequences = True,name='decoder_depth_%d'%_)(y,initial_state = hi_st)
        y = Dropout(config.DROPOUT)(y) 
        #
    y = TimeDistributed(Dense(vocab_dim,activation='softmax',name='embedding_decode'))(y)
    
    
    model = Model(inputs = [data,label],output=y,name='seq2seq_model')  
    model.summary()
    return model

