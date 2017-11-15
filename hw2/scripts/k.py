from __future__ import division,print_function


from keras import backend as K
from keras.datasets import mnist
from keras.models import *
from keras.layers import *
import tensorflow as tf

from keras.preprocessing import sequence
from keras.datasets import imdb



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
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 **kwargs):
        super(GG,self).__init__(units,**kwargs)
        self.implementation = 1
    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(None, None, self.input_dim))

        self.states = [None]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory
        rec_dp_mask = states[2]

        if self.implementation == 1:
            x_z = K.dot(inputs , self.kernel_z)
            x_r = K.dot(inputs , self.kernel_r)
            x_h = K.dot(inputs , self.kernel_h)
            if self.use_bias:
                x_z = K.bias_add(x_z, self.bias_z)
                x_r = K.bias_add(x_r, self.bias_r)
                x_h = K.bias_add(x_h, self.bias_h)
        else:
            raise ValueError('Unknown `implementation` mode.')
        z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0],
                                                  self.recurrent_kernel_z))
        r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1],
                                                  self.recurrent_kernel_r))

        hh = self.activation(x_h + K.dot(r * h_tm1 * rec_dp_mask[2],
                                             self.recurrent_kernel_h))
        h = z * h_tm1 + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h]



max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(GG(128))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
