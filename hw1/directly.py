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

loss.mask_vector = tf.constant([num_classes]*775,tf.int64)

def init_model():
    first_input = Input(shape=(max_len,features_count))#(777,39)
    
    #cnn layers
    cnn_input = BatchNormalization(axis = -1)(first_input)
    cnn_input = Reshape( (max_len,features_count,1) ) (cnn_input)
    cnn_output = Conv2D(20,kernel_size = (3,5),use_bias = True,activation = 'relu',padding = 'valid')(cnn_input)
    #(777,39,1) -> (777,35,20)
    cnn_output = BatchNormalization(axis = -1)(cnn_output)
    
    
    #rnn layers
    seq_input = Reshape((775,35*20))(cnn_output)
    #seq_input = Masking()(seq_input)    #whether the timestep masked depends on the prior cnn layer does use bias or not.
    rnn_lay = GRU

    #depth 2 bidirection lstm
    x1,state_h    = rnn_lay(128,activation = 'tanh',return_state = True,return_sequences = True)(seq_input)
    x2,state_h    = rnn_lay(128,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = True)(seq_input,state_h)
    xx = Concatenate(axis = -1)([x1,x2])
    xx = Dropout(0.5)(xx)
    
    #trimming by cnn


    #softmax each timestep
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(xx)

    #define output model
    model = Model(input = first_input,output = result)
    
    #model visualization
    #plot_model(model, to_file='../model.png',show_shapes = True)
    #model.load_weights('../best.model')
    
    return model
def sample_weight(y):
    s_mat = np.ones(y.shape[0:2],dtype = np.float32)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j,-1] == 1:  #padding vlaue,the loss actually masked in my loss funtion
                #s_mat[i,j] =
                end = j+3 if y.shape[1] >= j+3 else  y.shape[1]
                s_mat[i,j:end] = 5
                #s_mat[i,j] = 5
               #s_mat[i,j] = 2
                break
            elif y[i,j,37] == 1: #sil
                s_mat[i,j] = 0.5
    return s_mat 
if __name__ == '__main__':
    #read input from pre-proccessing npz
    x,y = myinput.load_input('./mfcc.npz')
    
    #reduce y to fit output[773,49]
    y = y[:,1:-1,:]
    #model setting
    model = init_model()
    #directly train on custom sample weight
    #model.load_weights('../best.model')

    #training attributes
    opt = Adam(lr = 0.01)
    #model.compile(loss=loss.loss_with_mask, optimizer=opt,metrics=[loss.acc_with_mask],sample_weight_mode = 'temporal')
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=[loss.acc_with_mask,'accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    #make check points to trace the performance of model during training
    cks = ModelCheckpoint('../dirctly.{epoch:02d}-{val_loss:.2f}.cks',save_best_only=True,period = 1)
    
    #sample weight matrix in uesd or not
    model.fit(x,y,batch_size = 100,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = sample_weight(y))
    #model.fit(x,y,batch_size = 30,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05)
