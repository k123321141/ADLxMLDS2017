from keras.models import *
from keras.layers import *
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf
c = K.constant([48]*773,tf.int64)
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
    dic1 = myinput.load_input('mfcc')
    dic_processing.pad_dic(dic1,max_len,max_len,num_classes)
    dic_processing.catogorate_dic(dic1,num_classes+1)
    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]
    
    print x.shape
    x = x.reshape(num,max_len,features_count,1)
    first_input = Input(shape=(max_len,features_count,1))
    cnn_input = BatchNormalization(axis = -1)(first_input)
    cnn_output = Conv2D(10,kernel_size = (3,5),use_bias = False,activation = 'relu',padding = 'valid')(cnn_input)
    #(777,39,1) -> (777,35,10)
    cnn_output = Conv2D(30,kernel_size = (3,5),use_bias = False,activation = 'relu',padding = 'valid')(cnn_output)
    #(777,35,10) -> (773,31,30)
    print cnn_output.shape
    seq_input = Reshape((max_len-4,31*30))(cnn_output)
    seq_input = Masking()(seq_input)
    #seq_input= Dropout(0.5)(seq_input)
    #
    rnn_lay = LSTM

    #xx,state_h, state_c = (rnn_lay(300,activation = 'tanh',return_state = True))(seq_input)
    #bidirection
    x1,state_h, state_c  = rnn_lay(300,activation = 'tanh',return_state = True,return_sequences = True)(seq_input)
    x2,state_h, state_c  = rnn_lay(300,activation = 'tanh',return_state = True,return_sequences = True,go_backwards = True)(seq_input,[state_h, state_c])
    xx = Concatenate(axis = -1)([x1,x2])
    xx = Dropout(0.5)(xx)
    x1,state_h, state_c  = rnn_lay(300,activation = 'tanh',return_state = True,return_sequences = True)(xx)
    x2 = rnn_lay(300,activation = 'tanh',return_state = False,return_sequences = True,go_backwards = True)(xx,[state_h, state_c])
    xx = Concatenate(axis = -1)([x1,x2])
    
    xx = Dropout(0.5)(xx)
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(xx)

    model = Model(input = first_input,output = result)
    plot_model(model, to_file='../model.png',show_shapes = True)
    #model.load_weights('../checkpoints/simple.18-1.65.model')

    #reduce y
    y = y[:,2:-2,:]
    s_mat = np.ones(y.shape[0:2],dtype = np.float32)
    for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j,-1] == 1:
                    #s_mat[i,j] = 5
                    s_mat[i,j:] = 0
                    break
                elif y[i,j,37] == 1: #sil
                    s_mat[i,j] = 0.5

    opt = Adam(lr = 0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    model.compile(loss=loss_with_mask, optimizer=opt,metrics=[acc_with_mask],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    cks = ModelCheckpoint('../checkpoints/comwithmask.{epoch:02d}-{val_loss:.2f}.cks',save_best_only=True,period = 2)

    model.fit(x,y,batch_size = 30,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    print 'Done'
