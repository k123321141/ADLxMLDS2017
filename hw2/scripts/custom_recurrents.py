import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent, _time_distributed_dense
from keras.engine import InputSpec

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

class AttentionDecoder(Recurrent):

    def __init__(self, units, output_dim,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias = True,
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 train_by_label=False,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states 
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space

        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 
            "Neural machine translation by jointly learning to align and translate." 
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences
        self.train_by_label = train_by_label

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape[0]
        self.batch_size, self.output_len, self.output_dim = input_shape[1]

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s
        """
            Matrices for GRU cells, copy source from keras.layers.recurrent.py with tag 2.0.7

        """
        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        """
            Setting matrics variables
        """
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]

        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,self.units : self.units * 2]

        self.kernel_h = self.kernel[:, self.units * 2:self.units * 3]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:self.units * 3]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:self.units * 3]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        """
            Output softmax matrics 
        """
        self.W_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Setting matrices for creating the context vector
        """
        self.W_a = self.recurrent_kernel[:, self.units * 3:]
        self.U_a = self.kernel[:, self.units * 3:]
        if self.use_bias:
            self.b_a = self.bias[self.units * 3:]
        else:
            self.b_a = None
        #for cosine similarity dense matrix
        self.m = self.add_weight(shape=(self.units*2, 1),
                                   name='match',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.m_a = self.add_weight(shape=(1,),
                                        name='match_bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        self.mix = self.add_weight(shape=(self.input_dim + self.output_dim, self.input_dim),
                                   name='mix',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.mix_a = self.add_weight(shape=(self.input_dim,),
                                        name='mix_bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)


        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim)),
            InputSpec(shape=(self.batch_size, self.output_len, self.output_dim))]
        self.built = True

    def call(self, inputs):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = inputs[0]
        self.y_seq = inputs[1]
        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:


        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a,# b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        return super(AttentionDecoder, self).call(self.y_seq)

    def get_initial_state(self, inputs):
        print('inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = inputs[:,0]
        return [y0, s0]

    def step(self, x, states):

        ytm, stm = states
        if self.train_by_label:
            ytm = x
        """
            Just cosine similarity.
        """
        _stm = K.repeat(stm, self.timesteps)
        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.

        #during a dense 
        combine = K.concatenate([_stm,self._uxpb],axis = -1)
        et = activations.sigmoid( K.dot(combine,self.m) + self.m_a) 
        #no softmax
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # veglobalctor of size (batchsize, timesteps, 1)
        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        #128

        #concatebate with label to mix the info
        combine = K.concatenate([context,ytm],axis=-1)
        context = K.dot(combine,self.mix) + self.mix_a

        # ~~~> calculate new hidden state
        """
            Original GRU cell operations.
        
        """

        x_z = K.dot(context, self.kernel_z)
        x_r = K.dot(context, self.kernel_r)
        x_h = K.dot(context, self.kernel_h)
        if self.use_bias:
            x_z = K.bias_add(x_z, self.bias_z)
            x_r = K.bias_add(x_r, self.bias_r)
            x_h = K.bias_add(x_h, self.bias_h)
        z = self.recurrent_activation(x_z + K.dot(stm, self.recurrent_kernel_z))
        r = self.recurrent_activation(x_r + K.dot(stm, self.recurrent_kernel_r))

        hh = self.activation(x_h + K.dot(r * stm, self.recurrent_kernel_h))
        h = z * stm + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        
        #yt = activations.softmax(
        
        yt = activations.tanh(
            K.dot(h, self.W_o)
            + self.b_o)
        st = h
        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.output_len, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
# check to see if it compiles
if __name__ == '__main__':
    from keras.layers import Input, LSTM
    from keras.models import Model
    from keras.layers.wrappers import Bidirectional
    i = Input(shape=(100,104), dtype='float32')
    enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(i)
    dec = AttentionDecoder(32, 4)(enc)
    model = Model(inputs=i, outputs=dec)
    model.summary()
