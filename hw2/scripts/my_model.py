from keras.models import *
from keras.layers import *
import config
import keras.backend as K
import tensorflow as tf
import numpy as np
import myinput
import config

assert K._BACKEND == 'tensorflow'

def loss_with_mask(y_true, y_pred):
    #(batch,50,6528)
    #assert <pad> in t_true are all zeros.
    
    mask = tf.sign(tf.reduce_sum(y_true, axis = -1)) #(batch,50) -> 0,1 matrix

    cross_entropy = y_true * tf.log(y_pred)#(batch,50,6528)
    cross_entropy = -tf.reduce_sum(cross_entropy,axis=-1)
   
    cross_entropy = cross_entropy * mask
    return tf.reduce_mean(cross_entropy) 
    
    return (correct) / tf.reduce_sum(mask)
def loss_with_mask_src(y_true, y_pred):
    mask = tf.not_equal(tf.argmax(y_true,-1),mask_vector)
    mask = tf.cast(mask,tf.float32)

    cross_entropy = y_true * tf.log(y_pred)
    cross_entropy = -tf.reduce_sum(cross_entropy,axis=-1)
    
    cross_entropy = cross_entropy * mask
    return tf.reduce_mean(cross_entropy) 
def acc_with_mask(y_true, y_pred):
    #(batch,50,6528)
    #assert <pad> in t_true are all zeros.
    
    mask = tf.sign(tf.reduce_sum(y_true, axis = -1))  #(batch,50) -> 0,1 matrix


    correct = tf.equal(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)) #(batch,50) 

    
    mask = tf.cast(mask,tf.float32)
    correct = tf.cast(correct,tf.float32)
    
    correct = tf.reduce_sum(mask * correct)

    
    return (correct) / tf.reduce_sum(mask)

def mask_zeros(x,src):
    #not zero
    mask = tf.sign(tf.abs(src)) 
    x = tf.multiply(mask,x)
    return x
def data_length():
    mask = tf.sign(tf.abs(x)) 
    length = tf.reduce(mask)
    return length
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
    model.compile(loss=loss_with_mask,
                  optimizer='adam',
                  metrics=[acc_with_mask])
    #model.summary()
    return model

vocab_map = myinput.init_vocabulary_map()
decode_map = {vocab_map[k]:k for k in vocab_map.keys()}
def decode(y):
    output_len = y.shape[0]
    y = np.argmax(y,axis = -1)
    s = ''
    for j in range(output_len):
        vocab_idx = y[j]
        if vocab_idx != 0:      #<pad>
            s += decode_map[vocab_idx] + ' '
    s = s.strip() + '.' 
    return s.encode('utf-8')
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
