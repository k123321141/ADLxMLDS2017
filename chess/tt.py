import tensorflow as tf
import keras
from keras.models import *
from kears.layers import *

def build_net(action_size=3, state_size=6400):
    with tf.variable_scope('global'): 
        pixel_input = Input(shape=(state_size,))

        #actor
        x = Reshape((80, 80, 1), name='shared_reshape')(pixel_input)
        x = Conv2D(32, kernel_size=(6, 6), strides=(3, 3), padding='same', name='shared_conv2d',
                                activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
        x = Flatten(name='shared_flatten')(x)
        x = Dense(64,name='shared_dense64', activation='relu', kernel_initializer='he_uniform')(x)
        actor_output = Dense(action_size, activation='softmax')(x)
        actor = Model(inputs=pixel_input, outputs=actor_output)
        
        #critic
        x = Reshape((80, 80, 1))(pixel_input)
        x = Conv2D(32, kernel_size=(6, 6), strides=(3, 3), padding='same',
                activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        critic_output = Dense(1, activation='linear')(x)
        critic = Model(inputs=pixel_input, outputs=critic_output)

        #whole model
        model = Model(inputs=pixel_input, outputs=[actor_output, critic_output])
    return actor, critic, model 

def main():
    a,c,m = build_net()
    
    '''
    with tf.name_scope('sync'):
        with tf.name_scope('pull'):
            pull_a = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
            pull_c= [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
    '''
    
if __name__ = '__main__':
    main()
