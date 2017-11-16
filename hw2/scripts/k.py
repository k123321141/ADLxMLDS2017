# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.callbacks import *
import keras
import numpy as np
import seq2seq
import myinput
import config
import HW2_config
import os
import sys
import utils
from myinput import decode,batch_decode
from os.path import join
from keras.models import *



vocab_map = myinput.init_vocabulary_map()
vocab_dim = len(vocab_map)
class MyLayer(SimpleRNN):
    def lstm_cell(is_training = True):
        cell = tf.contrib.rnn.BasicLSTMCell(
                hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
    def build_lstm_graph(inputs):
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                #if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
        return output, state
    def __init__(self, **kwargs):
        units = 128
        cell = SimpleRNNCell(units)
        super(SimpleRNN, self).__init__(cell,return_sequences = True,**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.num_steps = input_shape[-2]
        self.kernel = self.add_weight(name='kernel',
                shape=(input_shape[1], self.output_dim),
                initializer='uniform',
                trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return build_lstm_graph(x)
    def compute_output_shape(self, input_shape):
        return (input_shape)
    
    

class GG(GRU):
    @interfaces.legacy_recurrent_support
    def __init__(self, units,attention_vec,
                 activation='tanh',
                 use_bias=True,
                 **kwargs):
        self.implementation = 1
        self.attention_vec = attention_vec
        super(GG,self).__init__(units,implementation = 1, **kwargs)
    def build(self, input_shape):
        super(GG,self).build(input_shape)

        self.timestep_len = input_shape[-2]
        #self.attention_vec = K.random_uniform(shape = (self.input_shape,self.units))

    def step(self, inputs, states):
        y,h =  super(GG,self).step(inputs,states) 
        return y,h


encoder_units = 300
match_units = 128
decoder_units = 100
match_dim  = 200
input_len = 80
def Encoder():
    model = Sequential()
    
    model.add(GRU(encoder_units,activation = 'tanh',return_sequences = True,input_shape = [input_len,4096]))
    return model
def Match():
    encoder_out = Input(shape = [input_len,encoder_units])
    z = Input(shape = [decoder_units])
    
    zz = RepeatVector(input_shape)(z)
    
    
    m = Concatenate(axis = -1)(encoder_out,zz)
    a = Dense(match_dim,activation = 'linear')(m)
    a = Dense(1,activation = 'softmax')(a)
    #(80,1)
    energy = Permute([1,0])(a)
    #(1,80)
    def weight_by_a(energy,x):
        #x (80,300)
        #a (80,1)
        return tf.matmul(energy,x)
    
    def weight_by_a_output_shape(shape_energy,shape_x):
        output_shape = tuple( list(shape_energy)[0] , list(shape_x)[1])
        return output_shape
    context = Lambda(weight_by_energy,output_shape = weight_by_energy_output_shape)(energy,encoder_out)

    model = Model(input=[encoder_out,z],output = context)
    return model

def Decoder():
    context = Input(shape = [1,encoder_units])

def end2end():
    encoder_model = Encoder()
    match_model = Match()
    decoder_model = Decoder()
    
    x = Input(shape = [80,4096])

    cheat_y = Input(shape = [50,vocab_dim])
    
    encoder_out = encoder_model(x)

    context = match_model(encoder_out,z0)


    

if __name__ == '__main__':
    x = myinput.read_x()
    y_generator = myinput.load_y_generator()

    #testing
    test_x = myinput.read_x('../data/testing_data/feat/')
    test_y_generator = myinput.load_y_generator('../data/testing_label.json')
    

    model = end2end()
   
    print 'start training' 
    for epoch_idx in range(2000000):
        #train by labels
        train_cheat = np.repeat(myinput.caption_one_hot('<bos>'),HW2_config.video_num,axis = 0)

        for caption_idx in range(HW2_config.caption_list_mean):
            y = y_generator.next()
            np.copyto(train_cheat[:,1:,:],y[:,:-1,:])
            model.fit(x=[x,train_cheat], y=y,
                      batch_size=10,verbose=1,
                      epochs=1)

        '''
        #after a epoch
        if epoch_idx % config.SAVE_ITERATION == 0:
            #model.save(join(config.CKS_PATH,'%d.cks'%epoch_idx))
            model.save(join(config.CKS_PATH,'mask.cks'))
            #test_y just for testing,no need for iter as a whole epoch 
            test_y = test_y_generator.next()
            # Select 2 samples from the test set at random so we can visualize errors.
            testing(model,x,y,test_x,test_y,2) 
        '''
    #
