from keras.models import *
from keras.layers import *
import config
import keras.backend as K
import tensorflow as tf
import numpy as np
import myinput
import config
import utils
import custom_recurrents
assert K._BACKEND == 'tensorflow'

def model(input_len,input_dim,output_len,vocab_dim):
    print('Build model...')
    # Try replacing GRU, or SimpleRNN.
    #
    data = Input(shape=(input_len,input_dim))
    label = Input(shape=(output_len,vocab_dim))
    #in this application, input dim = vocabulary dim
    #masking
    x = data
    #scaling data
    x = BatchNormalization()(x)

    #encoder, bidirectional RNN
    for _ in range(config.DEPTH):
        #forward RNN
        ret1 = config.RNN(config.HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = False,name='forwar_encoder_%d'%_)(x)
        hi_st = ret1[1:] if config.RNN == LSTM else ret1[1]
        #backward RNN
        ret2  = config.RNN(config.HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = True,name='backward_encoder_%d'%_)(x,initial_state = hi_st)
        #concatenate both side
        x = Concatenate(axis = -1)([ret1[0],ret2[0]])
        #prepare hidden state for encoder
        hi_st = ret2[1:] if config.RNN == LSTM else ret2[1]
        if _ != config.DEPTH -1 :
            x = Dropout(config.DROPOUT)(x)

    #word embedding
    y = TimeDistributed(Dense(config.EMBEDDING_DIM,activation = 'linear',use_bias = False,name='word_embedding'))(label)
    #y = label
    #decoder
    print('build attention')
    pred = custom_recurrents.AttentionDecoder(100,output_dim = config.EMBEDDING_DIM,train_by_label = True,name = 'decoder')([x,y])
    #pred = custom_recurrents.AttentionDecoder(100,output_dim = vocab_dim,train_by_label = True,name = 'decoder')([x,y])
    print('build attention done')
    pred = TimeDistributed(Dense(vocab_dim,activation ='softmax',use_bias = False))(pred) 
    model = Model(inputs = [data,label],output=pred)  
    model.summary()
    return model
def set_train_by_label(model,train_by_label):
    decoder_lay = 'None'
    for lay in model.layers:
        if lay.name == 'decoder':
            decoder_lay = lay
    assert decoder_lay != 'None'
    decoder_lay.train_by_label = train_by_label
