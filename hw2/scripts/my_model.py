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
    mask = tf.not_equal(tf.argmax(y_true,-1),mask_vector)
    mask = tf.cast(mask,tf.float32)

    cross_entropy = y_true * tf.log(y_pred)
    cross_entropy = -tf.reduce_sum(cross_entropy,axis=-1)
    
    cross_entropy = cross_entropy * mask
    return tf.reduce_mean(cross_entropy) 
def mask(x,src):
    mask = tf.sign(tf.abs(src)) 
    x = tf.multiply(mask,x)
    return x
def xor_mask(x):
    mask = tf.sign(tf.abs(src)) 
    return x

def data_length():
    mask = tf.sign(tf.abs(x)) 
    length = tf.reduce(mask)
    return length
def split_x(x):
    x = x[:,:MAXLEN,:] 
    return x
def split_x_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    shape[-2] = MAXLEN
    return tuple(shape)
def split_y(x):
    x = x[:,MAXLEN:,:] 
    return x
def split_y_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    shape[-2] = DIGITS+1
    return tuple(shape)
def model(input_len,input_dim,output_len,vocab_dim):
    print('Build model...')
    # Try replacing GRU, or SimpleRNN.
    RNN,HIDDEN_SIZE,DEPTH = config.model_config()
    #
    data = Input(shape=(input_len,input_dim))
    #in this application, input dim = vocabulary dim
    label = Input(shape=(output_len,vocab_dim))
    #masking
    x = data
    y = label
    #y = Masking()(y)

    #decoder
    x1,h,c  = RNN(HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = False)(x)
    x2,h,c  = RNN(HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = True)(x,[h,c])
    x = Concatenate(axis = -1)([x1,x2])
    x1,h,c  = RNN(HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = False,go_backwards = False)(x)
    x2,h2,c2  = RNN(HIDDEN_SIZE,activation = 'tanh',return_state = True,return_sequences = False,go_backwards = True)(x,[h,c])

    #encoder
    pred  = RNN(HIDDEN_SIZE,activation = 'tanh',return_state = False,return_sequences = True,go_backwards = False)(y,[h2,c2])
    
    #
    pred = TimeDistributed(Dense(vocab_dim,activation='softmax'))(pred)
    
    
    model = Model(inputs = [data,label],output=pred)  
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
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
def my_pred(model,x,input_len,output_len):
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

    return y_pred[:,1:,:]
def batch_pred(model,x,input_len,output_len):
    num = x.shape[0]
    y_pred = np.repeat(myinput.caption_one_hot('<bos>'),num,axis = 0)

    y_pred[:,1:,:] = 0
    for i in range(1,output_len):
        '''
        print 'pred'
        print decode(y_pred[0,:,:]) 
        print np.argmax(y_pred[0,:,:],axis = -1)
        '''
        y = model.predict([x,y_pred])
        y_pred[:,i,:] = y[:,i-1,:] 
        #print 'next ',i,next_idx

    return y_pred[:,1:,:]
