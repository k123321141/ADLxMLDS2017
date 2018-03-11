import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
hidden_state_units = 128
def build_model(action_size, state_size, scope):
    with tf.variable_scope(scope):
        pixel_input = Input(shape=(state_size,))
        hi_st = Input(shape=(hidden_state_units,))
        #actor
        x = Reshape((80, 80, 1))(pixel_input)
        for i in range(4):
            x = Conv2D(16 * 2**i, kernel_size=(3, 3), strides=(2, 2), padding='same',
            #x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', \
                                    activation='relu', kernel_initializer='he_uniform', data_format = 'channels_last')(x)
            x = BatchNormalization()(x)
        cnn_out = Reshape([1,-1])(x)
        
        x, st = GRU(hidden_state_units, activation='relu',return_state=True)(cnn_out, hi_st)
        
        actor_output = Dense(action_size, activation='softmax')(x)
        actor = Model(inputs=[pixel_input, hi_st], outputs=actor_output)

        
        #critic
        critic_output = Dense(1, activation='linear')(x)
        critic = Model(inputs=[pixel_input, hi_st], outputs=critic_output)

        #whole model
        model = Model(inputs=[pixel_input, hi_st], outputs=[actor_output, critic_output, hi_st])
        '''
        x = ram_input = Input(shape=(128,))
        hi_st = Input(shape=(hidden_state_units,))
        #actor
        x = Reshape([1,-1])(x)
        x, st = GRU(hidden_state_units, activation='relu',return_state=True, kernel_initializer='glorot_normal')(x, hi_st)
        
        actor_output = Dense(action_size, activation='softmax', kernel_initializer='glorot_normal')(x)
        actor = Model(inputs=[ram_input, hi_st], outputs=actor_output)

        
        #critic
        critic_output = Dense(1, activation='linear')(x)
        critic = Model(inputs=[ram_input, hi_st], outputs=critic_output)
        #whole model
        model = Model(inputs=[ram_input, hi_st], outputs=[actor_output, critic_output, hi_st])
        '''
    return actor, critic, model 
def main():
    _,_,m = build_model(3, 6400,'test')
    m.load_weights('../models/pong_a3c.h5')
    print m.get_weights()
if __name__ == '__main__':
    main()
