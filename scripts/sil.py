import myinput
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.callbacks import EarlyStopping
import numpy as np
import random
import dic_processing
num_classes = 48
features_count = 39

max_len = 777
max_out_len = 80




#dic init setting,reshape
dic1 = myinput.load_input('mfcc')
#change y with seq_y
for k in dic1.keys():
    x,y = dic1[k]
    np.place(y, y==37, 1)
    np.place(y, y!=37, 0)

    dic1[k] = x,y
#
dic_processing.pad_dic(dic1,max_len,max_len,0)
dic_processing.catogorate_dic(dic1,2)

x,y = dic_processing.toXY(dic1)



model = Sequential()
model.add(Masking(mask_value=0, input_shape=(max_len, features_count)))
model.add(SimpleRNN(30,input_dim = features_count, activation='tanh',return_sequences=True,implementation=1))
model.add(SimpleRNN(20, activation='tanh',return_sequences=True,implementation=1))
model.add(TimeDistributed(Dense(2,activation='sigmoid')))


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x,y,batch_size = 1000,epochs = 20,callbacks=[early_stopping],validation_split = 0.05)

model.save('../models/sil.model')
