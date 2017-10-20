from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import plot_model
import myinput
import numpy as np
import dic_processing
from seq_model import seq_model
num_classes = 48
features_count = 39

max_len = 777
max_out_len = 80




#dic init setting,reshape
dic1 = myinput.load_input('mfcc')
#change y with seq_y
seq_y = myinput.read_seq_Y('../data/train.lab')
for k in dic1.keys():
    x,y = dic1[k]
    y2 = myinput.dic2ndarray( seq_y[k] )
    y2 = y2.reshape(y2.shape[0],1)
    dic1[k] = x,y2
#
dic2 = myinput.load_input('mfcc')
dic_processing.pad_dic(dic2,max_len,max_len,num_classes)
dic_processing.catogorate_dic(dic2,num_classes+1)
#
dic_processing.pad_dic(dic1,max_len,max_out_len,num_classes)
dic_processing.catogorate_dic(dic1,num_classes+1)

x,y = dic_processing.toXY(dic1)
x2,y2 = dic_processing.toXY(dic2)

num = x.shape[0]
x = x.reshape(num,max_len,features_count,1)

#cnn
cnn_input = Input(shape=(max_len,features_count,1))
'''
cnn_out = Conv2D(20,input_shape = (max_len,features_count),kernel_size = (1,5),padding='valid',activation = 'relu',data_format = 'channels_last')(cnn_input)
#(777,35,20)

for i in range(3):
    #40 -> 80 -> 160
    cnn_out = Conv2D(40*(2**i),kernel_size = (1,3),padding='valid',activation = 'relu',data_format = 'channels_last')(cnn_out) 
#(777,29,160)

cnn_out = Conv2D(320,kernel_size = (1,29),padding='valid',activation = 'relu',data_format = 'channels_last')(cnn_out)
#(777,1,320)
cnn_out = Reshape((max_len,320))(cnn_out)
#(777,320)
'''

cnn_out = Conv2D(40,input_shape = (max_len,features_count),kernel_size = (1,5),padding='valid',activation = 'relu',data_format = 'channels_last')(cnn_input)
cnn_out = Conv2D(200,kernel_size = (1,35),padding='valid',activation = 'relu',data_format = 'channels_last')(cnn_out)
#(777,1,120)
cnn_out = Reshape((max_len,200))(cnn_out)
#(777,120)


#seq_model
seq_input = Masking()(cnn_out)

seq_out = seq_model(200,seq_input,max_out_len,num_classes+1,depth = 3,rnn_lay = GRU)

model = Model(input = cnn_input,output = seq_out)

plot_model(model, to_file='../model.png',show_shapes = True)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x,y,batch_size = 5,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)

model.save('../models/cnn_seq.model')
