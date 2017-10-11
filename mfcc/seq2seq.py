import myinput
from keras.models import *
from keras.layers.recurrent import SimpleRNN
from keras.layers import *
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import EarlyStopping

import numpy as np
import random

hidden_dim = 39
features_count = 39
num_classes = 48
validation_rate = 0.05


epochs = 200

max_len = 777
def init_dic(dic):
    x_buf = []
    y_buf = []
    for sentence_id in dic.keys():
        x,y = dic[sentence_id]
        num = x.shape[0]
        assert x.shape[0] == y.shape[0] and features_count == x.shape[1]
        x = np.lib.pad(x,((0,777-num),(0,0)),'constant', constant_values=(0, 0))
        x_buf.append(x)
        
        
        y = np.lib.pad(y,((0,777-num),(0,0)),'constant', constant_values=(0, num_classes))
        y = to_categorical(y,num_classes+1)
        y_buf.append(y)

        

    sample_num = len(dic.keys())
    x = np.asarray(x_buf)
    y = np.asarray(y_buf)
    
    return x,y


#dic init setting,reshape
dic = myinput.load_input()
x,y = init_dic(dic)

print x.shape,y.shape





#model setting
model = Sequential()

#model.add(Embedding(input_dim = features_count,output_dim=features_count))

#decoder
model.add(LSTM(features_count, input_dim = features_count,return_sequences=True))
model.add(Dense(features_count, activation="relu"))
#model.add(RepeatVector(max_len))
model.add(LSTM(features_count, return_sequences=True))

model.add(TimeDistributed(Dense(num_classes+1,activation='softmax')))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#training loop
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x,y,batch_size=400,epochs=epochs,validation_split=0.05,callbacks=[early_stopping])

model.save('./seq2seq.model')
