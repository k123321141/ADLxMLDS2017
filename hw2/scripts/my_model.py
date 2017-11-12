from keras.models import *
from keras.layers import *
import keras.backend as K
import tensorflow as tf
import numpy as np
import myinput
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
def model(input_len,vocab_dim,output_len):
    print('Build model...')
    # Try replacing GRU, or SimpleRNN.
    RNN = GRU
    HIDDEN_SIZE = 512
    LAYERS = 1
    DEPTH = 1
    #
    data = Input(shape=(input_len,vocab_dim))
    #in this application, input dim = vocabulary dim
    label = Input(shape=(output_len,vocab_dim))
    #masking
    x = data
    y = label

    _,hi_h = RNN(HIDDEN_SIZE,activation = 'relu',return_state = True)(x)
    
    pred = RNN(HIDDEN_SIZE,activation = 'relu',return_sequences = True)(y,hi_h) 

    #
    pred = TimeDistributed(Dense(len(chars),activation='softmax'))(pred)
    
    
    model = Model(inputs = [data,label],output=pred)  
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def my_pred(model,x,input_len,output_len):
    y_pred = myinput.caption_one_hot('<bos>')

    for i in range(1,output_len):
        y = model.predict([x,y_pred])
        y_pred[:,i,:] = y[:,i,:]

    return y_pred
