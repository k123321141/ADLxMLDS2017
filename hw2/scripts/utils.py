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
