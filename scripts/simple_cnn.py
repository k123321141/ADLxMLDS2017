from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import plot_model
import myinput
import numpy as np
import dic_processing
num_classes = 48
features_count = 39

max_len = 777
max_out_len = 80




#dic init setting,reshape
dic1 = myinput.load_input('mfcc')
dic_processing.pad_dic(dic1,max_len,max_len,num_classes)
dic_processing.catogorate_dic(dic1,num_classes+1)

x,y = dic_processing.toXY(dic1)
num = x.shape[0]
x = x.reshape(num,max_len,features_count,1)

first_input = Input(shape=(max_len,features_count,1))
'''
xx = Conv2D(30,input_shape = (max_len,features_count),kernel_size = (1,features_count),padding='valid',activation = 'relu',data_format = 'channels_last')(first_input) 
'''
xx = Conv2D(30,input_shape = (max_len,features_count),kernel_size = (1,5),padding='valid',activation = 'relu',data_format = 'channels_last')(first_input) 

#777,35,30
xx = Conv2D(60,kernel_size = (1,5),padding='valid',activation = 'relu',data_format = 'channels_last')(xx) 
xx = Conv2D(90,kernel_size = (1,31),padding='valid',activation = 'relu',data_format = 'channels_last')(xx)
#(777,1,120)
#xx = Conv2D(30,kernel_size = (1,3),padding='valid',activation = 'relu',data_format = 'channels_last')(xx)
#(777,35,30)
xx = Reshape((max_len,90))(xx)
xx = Masking(mask_value=0)(xx)
rnn_lay = GRU
xx = rnn_lay(60,activation='tanh',return_sequences=True,implementation=1)(xx)
xx = rnn_lay(40,activation='tanh',return_sequences=True,implementation=1)(xx)
xx = rnn_lay(num_classes+1,activation='softmax',return_sequences=True,implementation=1)(xx)


#model = Model(input = cnn_input,output = seq_output)
model = Model(input = first_input,output = xx)

plot_model(model, to_file='../model.png',show_shapes = True)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x,y,batch_size = 10,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)

model.save('../models/simple_k.model')
