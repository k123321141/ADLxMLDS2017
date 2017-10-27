from keras.models import *
from combine_cnn_rnn import init_model

import output
import loss
import tensorflow as tf

#loss.mask_vector = tf.constant([48]*773,tf.int64)
#m = load_model('../checkpoints/comwithmask.09-0.30.cks',custom_objects = {'loss_with_mask' :loss.loss_with_mask,'acc_with_mask': loss.acc_with_mask})
m = init_model()
m.load_weights('../checkpoints/comwithmask.09-0.30.cks')
print('Done')
#output.compare_output(m)
