from keras.models import *
from keras.layers import *
import cnn,rnn

import myinput
import dic_processing
from keras.utils import plot_model
from keras.callbacks import *
from keras.optimizers import *


if __name__ == '__main__':
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
        dic_processing.pad_dic(dic1,max_len,max_out_len,num_classes)
    else:
        dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

    dic_processing.catogorate_dic(dic1,num_classes+1)
    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]

    x = x.reshape(num,max_len,features_count,1)
    #(777,39) -> (777,39,1)


    first_input = Input(shape=(max_len,features_count,1))
    #cnn_input = BatchNormalization(axis = -2) (first_input)         #the axis of nomaliztion is -2 (3696,777,39,1)
    cnn_input = first_input
    cnn_output = cnn.output(cnn_input,kernel_size =(1,5),depth = 1,filters = 10,padding ='valid',normalization = False,use_bias = False)
    cnn_output = cnn.output(cnn_output,kernel_size =(1,35),depth = 1,filters = 50,padding ='valid',normalization = False,use_bias = False)
    print cnn_output.shape
    seq_input = Reshape((777,50))(cnn_output)
    #
    seq_input = Masking()(seq_input)

    seq_output,st = SimpleRNN(200,return_state = True,activation='tanh')(seq_input) 
    seq_output = RepeatVector(max_out_len)(seq_output)
    seq_output = SimpleRNN(200,return_sequences = True,activation='tanh')(seq_output,initial_state = st) 
    #
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(seq_output)

    model = Model(input = first_input,output = result)

    plot_model(model, to_file='../model.png',show_shapes = True)

    #
    s_mat = np.ones(y.shape[0:2],dtype = np.float32)
    for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j,-1] == 1:
                    s_mat[i,j:] = 0
                    break
    #
    sgd_opt = SGD(lr = 0.04)
    model.compile(loss='categorical_crossentropy', optimizer=sgd_opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    cks = ModelCheckpoint('../checkpoints/seq_test.{epoch:02d}-{val_loss:.2f}.model',save_best_only=True,period = 2)


    model.fit(x,y,batch_size = 100,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    print 'Done'
