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


#dic init setting,reshape
dic1 = myinput.load_input('fbank')
dic2 = myinput.load_input('mfcc')
dic3 = myinput.stack_x(dic1,dic2)

dic_processing.pad_dic(dic3,max_len)
dic_processing.catogorate_dic(dic3,num_classes)

x,y = dic_processing.toXY(dic3)



#model setting
model = Sequential()

model.add(LSTM(features_count,input_dim = features_count, activation='tanh',return_sequences=False,implementation=1))
#model.add(Dense(features_count, activation="tanh"))
model.add(RepeatVector(max_len))
model.add(LSTM(features_count, activation='tanh',return_sequences=True,implementation=1))


#model.add(LSTM(features_count, activation='tanh',return_sequences=False,implementation=1))

#
model.add(TimeDistributed(Dense(num_classes,activation='softmax')))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#training loop
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x,y,batch_size = 100,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)

model.save('../models/seq2seq.model')
