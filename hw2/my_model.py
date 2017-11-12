from keras.models import *
from keras.layers import *
import keras.backend as K
import tensorflow as tf
import numpy as np
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
def model(MAXLEN,chars,DIGITS):
    print('Build model...')
    # Try replacing GRU, or SimpleRNN.
    RNN = GRU
    HIDDEN_SIZE = 128
    LAYERS = 1
    DEPTH = 1
    #
    x = Input(shape=(MAXLEN,len(chars)))
    #in this application, input dim = vocabulary dim
    label = Input(shape=(DIGITS+1+2,len(chars)))
    #masking
    y = Masking()(label)

    _,hi_h = RNN(HIDDEN_SIZE,activation = 'relu',return_state = True)(x)
    
    pred = RNN(HIDDEN_SIZE,activation = 'relu',return_sequences = True)(y,hi_h) 

    #
    pred = TimeDistributed(Dense(len(chars),activation='softmax'))(pred)
    
    
    model = Model(inputs = [x,label],output=pred)  
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model
