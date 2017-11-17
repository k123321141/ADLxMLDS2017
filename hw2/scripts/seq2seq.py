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
    #encoder, bidirectional RNN
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
    #y = Masking()(y)
    #concatenate x and label
    if config.RNN == LSTM:
        hi_concat = Concatenate(axis = -1)(hi_st)
    c = RepeatVector(output_len)(hi_concat)
    y = Concatenate(axis =-1)([c,y])

    #decoder
    for _ in range(config.DEPTH):
        y  = config.RNN(config.HIDDEN_SIZE,activation = 'tanh',return_sequences = True)(y,initial_state = hi_st)
        
        #
    y = TimeDistributed(Dense(vocab_dim,activation='softmax'))(y)
    
    
    model = Model(inputs = [data,label],output=y)  
    model.summary()
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
