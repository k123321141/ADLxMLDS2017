import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.models import Model

def build_network(num_actions, agent_history_length, resized_width, resized_height):
  with tf.device("/cpu:0"):
    state = tf.placeholder("float", [None, resized_width, resized_height, agent_history_length])
    inputs = Input(shape=(resized_width, resized_height,agent_history_length))
    x = Conv2D(16, kernel_size =(8,8), strides =(4,4), activation='relu', padding='same', data_format = 'channels_last')(inputs)
    x = Conv2D(32, kernel_size =(4,4), strides =(2,2), activation='relu', padding='same', data_format = 'channels_last', name='haha')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    q_values = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=inputs, outputs=q_values)
    model.summary()
  return state, model
