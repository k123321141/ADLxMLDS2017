from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import plot_model
import myinput
import numpy as np
import dic_processing
import cnn 
import auto_decoder_encoder

max_len = 777
num_classes = 48
features_count = 39
max_out_len = 80
seq = True

#dic init setting,reshape
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




#model

x = x.reshape(num,max_len,features_count,1)


cnn_model = load_model('../models/cnn.model')

cnn_input = Input(shape=(max_len,features_count,1))
cnn_output = cnn_model(cnn_input)

seq_input = Input(shape=(max_len,num_classes+1))

#seq_output = auto_decoder_encoder.seq_output(cnn_output,max_out_len = max_out_len,num_classes = num_classes+1)
seq_output = auto_decoder_encoder.seq_output(seq_input,max_out_len = max_out_len,num_classes = num_classes+1)


model = Model(input = seq_input,output = seq_output)

plot_model(model, to_file='../model.png',show_shapes = True)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(z,y,batch_size = 10,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)
print 'Done'
model.save('../models/seq2seq.model')
from keras.models import *
from keras.layers import *
import cnn,rnn

num_classes = 48
features_count = 39


if __name__ == '__main__':
    import myinput
    import dic_processing
    from keras.utils import plot_model
    from keras.callbacks import *
    from keras.optimizers import *
#dic init setting,reshape
    max_len = 777
    dic1 = myinput.load_input('mfcc')
    dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

    dic_processing.catogorate_dic(dic1,num_classes+1)

    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]




#model

    x = x.reshape(num,max_len,features_count,1)
    #(777,39) -> (777,39,1)
    y = y[:,1:-1,:]
    #(3696,777,49) -> (3696,775,49)
    print y.shape 


    first_input = Input(shape=(max_len,features_count,1))
    cnn_input = BatchNormalization(axis = -2) (first_input)         #the axis of nomaliztion is -2 (3696,777,39,1)
    cnn_output = cnn.output(cnn_input,kernel_size =(3,5),depth = 1,filters = 10,padding ='valid')
    #(777,39,1) -> (775,35,10)
    rnn_input = Reshape((775,35*10))(cnn_output)
    #
    rnn_output = rnn.output(rnn_input,bidirect = True,depth = 2,)
    #
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(rnn_output)

    model = Model(input = first_input,output = result)

    plot_model(model, to_file='../model.png',show_shapes = True)

    #
    s_mat = np.zeros(y.shape[0:2],dtype = np.float32)
    np.place(s_mat,s_mat == 0,1)
    for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j,-1] == 1:
                    s_mat[i,j:] = 0
                    break
    #
    sgd_opt = SGD(lr = 0.01)

    model.compile(loss='categorical_crossentropy', optimizer=sgd_opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    cks = ModelCheckpoint('../checkpoints/combine.{epoch:02d}-{val_loss:.2f}.model',save_best_only=True,period = 5)
    model.fit(x,y,batch_size =100,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    print 'Done'
    model.save('../models/cnn+rnn.model')
