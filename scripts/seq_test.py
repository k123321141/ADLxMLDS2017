import myinput
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.callbacks import EarlyStopping
import numpy as np
import random
import dic_processing
num_classes = 48
features_count = 108

max_len = 777
max_out_len = 80




#dic init setting,reshape
dic1 = myinput.load_input('mfcc')
dic2 = myinput.load_input('fbank')
dic3 = myinput.stack_x(dic1,dic2)
#change y with seq_y
seq_y = myinput.read_seq_Y('../data/train.lab')
for k in dic3.keys():
    x,y = dic3[k]
    y2 = myinput.dic2ndarray( seq_y[k] )
    y2 = y2.reshape(y2.shape[0],1)
    dic3[k] = x,y2
#    print type(x),type(y2)
#
dic_processing.pad_dic(dic3,max_len,max_out_len,num_classes)
dic_processing.catogorate_dic(dic3,num_classes+1)

x,y = dic_processing.toXY(dic3)

print 'lol'


import seq2seq
from seq2seq.models import SimpleSeq2Seq
model = Sequential()
#model.add(Masking(mask_value=0, input_shape=(max_len, features_count)))
#model = SimpleSeq2Seq(input_dim=features_count, hidden_dim=features_count, output_length=max_out_len, output_dim=num_classes)
model.add(SimpleSeq2Seq(input_dim=features_count, hidden_dim=features_count, output_length=max_out_len, output_dim=num_classes+1))


model.add(TimeDistributed(Dense(num_classes+1,activation='softmax')))
#model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#training loop
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x,y,batch_size = 100,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)

model.save('../models/seq2seq.model')
