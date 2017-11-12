import tensorflow as tf
from configuration import *



mask_vector = tf.constant([num_classes]*max_len,tf.int64)


def acc_with_mask(y_true, y_pred):
    mask = tf.not_equal(tf.argmax(y_true,-1),mask_vector)
    correct = tf.equal(tf.argmax(y_true,-1),tf.argmax(y_pred,-1))
    #3696,777 
    
    mask = tf.cast(mask,tf.float32)
    correct = tf.cast(correct,tf.float32)
    
    correct = tf.reduce_sum(mask * correct)

    
    return (correct) / tf.reduce_sum(mask)

def softmax_cross_entropy(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits = y_pred)
def my_cross_entropy(y_true, y_pred):
    cross_entropy = y_true * tf.log(y_pred)

    cross_entropy = -tf.reduce_sum(cross_entropy,axis=-1)
    return tf.reduce_mean(cross_entropy)
def loss_with_mask(y_true, y_pred):
    mask = tf.not_equal(tf.argmax(y_true,-1),mask_vector)
    mask = tf.cast(mask,tf.float32)

    cross_entropy = y_true * tf.log(y_pred)
    cross_entropy = -tf.reduce_sum(cross_entropy,axis=-1)
    
    cross_entropy = cross_entropy * mask
    return tf.reduce_mean(cross_entropy) 
