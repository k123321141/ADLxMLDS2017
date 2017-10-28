from keras.models import *
from keras.layers import *
#from keras.utils import plot_model
from keras.callbacks import *
from keras.optimizers import *
from keras.utils import to_categorical

import myinput
import dic_processing
import tensorflow as tf
import loss
from configuration import max_len,num_classes,max_out_len,features_count

loss.mask_vector = tf.constant([num_classes]*777,tf.int64)

def init_model():
    first_input = Input(shape=(max_len,features_count))
     
    #rnn layers
    seq_input = Masking()(first_input)    #whether the timestep masked depends on the prior cnn layer does use bias or not.
    rnn_lay = LSTM

    #depth 2 bidirection GRU
    x1,state_h,state_c  = rnn_lay(128,activation = 'tanh',return_state = True,return_sequences = True)(seq_input)
    x2,state_h,state_c  = rnn_lay(128,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = True)(seq_input,[state_h,state_c])
    xx = Concatenate(axis = -1)([x1,x2])
    xx = Dropout(0.5)(xx)
    x1,state_h,state_c  = rnn_lay(128,activation = 'tanh',return_state = True,return_sequences = True)(xx)
    x2 = rnn_lay(128,activation = 'tanh',return_state = False,return_sequences = True,go_backwards = True)(xx,[state_h,state_c])
    xx = Concatenate(axis = -1)([x1,x2])
    
    xx = Dropout(0.5)(xx)
    #softmax each timestep
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(xx)

    #define output model
    model = Model(input = first_input,output = result)
    
    #model visualization
    #plot_model(model, to_file='../model.png',show_shapes = True)
    
    return model
if __name__ == '__main__':
    #read input from pre-proccessing npz
    x,y = myinput.load_input('mfcc')
    
    #model setting
    model = init_model()

    #training attributes
    opt = RMSprop(lr = 0.001)

    model.compile(loss=loss.loss_with_mask, optimizer=opt,metrics=[loss.acc_with_mask],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    #make check points to trace the performance of model during training
    #cks = ModelCheckpoint('../checkpoints/rnn.{epoch:02d}-{val_loss:.2f}.cks',save_best_only=True,period = 2)
    
    #sample weight matrix in uesd or not
    model.fit(x,y,batch_size = 100,epochs = 200,callbacks=[early_stopping,cks],validation_split = 0.05)

