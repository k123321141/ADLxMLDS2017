from keras.models import *
from keras.layers import *
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf
c = K.constant([49]*80,tf.int64)
def acc_with_mask(y_true, y_pred):
    mask = K.not_equal(K.argmax(y_true,-1),c)
    correct = K.equal(K.argmax(y_true,-1),K.argmax(y_pred,-1))
    #3696,777 
    mask = tf.cast(mask,tf.float32)
    correct = tf.cast(correct,tf.float32)
    correct = tf.reduce_sum(mask * correct)

    
    return (correct) / tf.reduce_sum(mask)

def cross_entropy(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits = y_pred)
def my_cross_entropy(y_true, y_pred):
    cross_entropy = y_true * tf.log(y_pred)

    cross_entropy = -tf.reduce_sum(cross_entropy,axis=-1)
    return tf.reduce_mean(cross_entropy)
def loss_with_mask(y_true, y_pred):
    mask = K.not_equal(K.argmax(y_true,-1),c)
    mask = tf.cast(mask,tf.float32)

    cross_entropy = y_true * tf.log(y_pred)
    cross_entropy = -tf.reduce_sum(cross_entropy,axis=-1)
    
    cross_entropy = cross_entropy * mask
    return tf.reduce_mean(cross_entropy) 
    #return cross_entropy

if __name__ == '__main__':
    import myinput
    import dic_processing
    from keras.utils import plot_model
    from keras.callbacks import *
    from keras.optimizers import *
#dic init setting,reshape
    max_len = 777
    num_classes = 48
    features_count = 39
    max_out_len = 80
    seq = True
    dic1 = myinput.load_input('mfcc')
    if seq == True:
        seq_dict = myinput.read_seq_Y('../data//mfcc/seq_y.lab')
        for sentenceID in sorted(seq_dict.keys()):
            frame_dic = seq_dict[sentenceID]
            seq_y = myinput.dic2ndarray(frame_dic)
            seq_y = seq_y.reshape(seq_y.shape[0],1)

            x,y = dic1[sentenceID]
            dic1[sentenceID] = x,seq_y
        dic_processing.pad_dic(dic1,max_len,max_out_len,49) #padding
    else:
       for sentenceID in sorted(dic1.keys()):
            x,y = dic1[sentenceID]
    dic_processing.catogorate_dic(dic1,num_classes+2)
    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]
    
    #model setting
    
    x = x.reshape(num,max_len,features_count,1)
    first_input = Input(shape=(max_len,features_count,1))
    cnn_input = BatchNormalization(axis = -1)(first_input)
    cnn_output = Conv2D(10,kernel_size = (1,5),use_bias = False,activation = 'relu',padding = 'valid')(cnn_input)
    #(777,39,1) -> (777,35,10)
    cnn_output = Dropout(0.5)(cnn_output)
    cnn_output = BatchNormalization(axis = -1)(cnn_output)
    cnn_output = Conv2D(30,kernel_size = (3,7),use_bias = False,activation = 'relu',padding = 'valid')(cnn_output)
    cnn_output = Dropout(0.5)(cnn_output)
    #(777,35,10) -> (775,29,30)
    cnn_output = BatchNormalization(axis = -1)(cnn_output)
    seq_input = Reshape((max_len-2,29*30))(cnn_output)
    seq_input = Masking()(seq_input)
    #
    rnn_lay = LSTM
    #auto decoder encoder
    x1,state_h, state_c  = rnn_lay(300,activation = 'tanh',recurrent_dropout = 0.5,return_state = True)(seq_input)
    x2,state_h, state_c  = rnn_lay(300,activation = 'tanh',recurrent_dropout = 0.5,return_state = True,go_backwards = True)(seq_input,[state_h, state_c])
    xx = Concatenate(axis = -1)([x1,x2])

    xx = RepeatVector(max_out_len)(xx)
    x1,state_h, state_c  = rnn_lay(300,activation = 'tanh',return_state = True,recurrent_dropout = 0.5,return_sequences = True)(xx,[state_h, state_c])
    x2 = rnn_lay(300,activation = 'tanh',return_state = False,return_sequences = True,recurrent_dropout = 0.5,go_backwards = True)(xx,[state_h, state_c])
    xx = Concatenate(axis = -1)([x1,x2])
    
    xx = Dropout(0.5)(xx)
    result = TimeDistributed(Dense(50,activation='softmax'))(xx)

    model = Model(input = first_input,output = result)
    plot_model(model, to_file='../model.png',show_shapes = True)
#    model.load_weights('../checkpoints/seq.09-1.46.cks')
    
    
    s_mat = np.ones(y.shape[0:2],dtype = np.float32)
    for i in range(y.shape[0]): #3696
            for j in range(y.shape[1]):#777
                if y[i,j,-1] == 1:
                    y[i,j,-1] = 0
                    y[i,j,-2] = 1   #end of sentence 
                    s_mat[i,j:] = 0
                    s_mat[i,j] = 5
                    break
                elif y[i,j,37] == 1: #sil
                    s_mat[i,j] = 0.5

    opt = Adam(lr = 0.001)
    model.compile(loss=loss_with_mask, optimizer=opt,metrics=[acc_with_mask],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    cks = ModelCheckpoint('../checkpoints/seq.{epoch:02d}-{val_loss:.2f}.cks',save_best_only=True,period = 2)

    model.fit(x,y,batch_size = 30,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    print 'Done'
