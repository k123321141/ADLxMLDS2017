from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.utils import plot_model
import myinput
import numpy as np
import dic_processing
import cnn,rnn 
import auto_decoder_encoder
import split_by_sil
max_len = 777
num_classes = 48
features_count = 39
max_out_len = 80
seq = False

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

#max_len,x,y = split_by_sil.split_by_sil(x,y)

num = x.shape[0]

#model

model = Sequential()
model.add(BatchNormalization(input_shape = (777,39)))



model.add(TimeDistributed(Dense(256,activation = 'sigmoid',input_dim = 39)))
model.add(Dropout(0.10))
model.add(Dense(num_classes+1,activation = 'softmax'))


plot_model(model, to_file='../model.png',show_shapes = True)

#construct sample matrix
s_mat = np.zeros((num,777),dtype = np.float32)
np.place(s_mat,s_mat == 0,1)

for i in range(num):
        for j in range(777):
            if y[i,j,48] == 1:
                len_of_sample = j
                break
        s_mat[i,len_of_sample] = 0
#
sgd_opt = SGD(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer = sgd_opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x,y,batch_size =400,epochs = 2000,callbacks=[early_stopping],validation_split = 0.05,sample_weight = s_mat)
print 'Done'
x

