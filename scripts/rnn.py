import myinput
from keras.models import *
from keras.layers.recurrent import SimpleRNN
from keras.layers import *
from keras.utils import *
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import numpy as np
import random

hidden_dim = 39
num_classes = 48
validation_rate = 0.05
features_count = 108



def split_dic_validation(dic,rate):
    sample_count = len(dic.keys())
    vali_count = int(np.ceil(sample_count*rate))
    vali_dic = {}
    train_dic = {}
    
    
    i = 0
    shffle_keys = dic.keys()
    random.shuffle(shffle_keys)
    for key in shffle_keys:
        if i < vali_count:
            vali_dic[key] = dic[key]
        else:
            train_dic[key] = dic[key]
        i += 1
    print 'split to train_dic:%d    vali_dic:%d' % (sample_count - vali_count,vali_count)
    return train_dic,vali_dic

#make x,y in dic to fit the input shape of (1,timestep,features_count)
#make y to ont-hot vector for categorical_crossentropy
def init_dic(dic):
    for sentence_id in dic.keys():
        x,y = dic[sentence_id]
        num = x.shape[0]
        assert x.shape[0] == y.shape[0] and features_count == x.shape[1]
        x = x.reshape(1,num,features_count)
        y = ( to_categorical(y,num_classes) ).reshape(1,num,num_classes)
        dic[sentence_id] = (x,y)

def dic2generator(dic):
    while True:
        shffle_keys = dic.keys()
        random.shuffle(shffle_keys)
        for key in shffle_keys:
            yield dic[key]


#dic init setting,reshape
dic1 = myinput.load_input('fbank')
dic2 = myinput.load_input('mfcc')
dic3 = myinput.stack_x(dic1,dic2)

init_dic(dic3)



training_dic,validation_dic = split_dic_validation(dic3,validation_rate)

training_generator = dic2generator(training_dic)
validation_generator = dic2generator(validation_dic)


validation_steps = len(validation_dic.keys())
steps_per_epoch = len(training_dic.keys())


#model setting
model = Sequential()

model.add(SimpleRNN(features_count,input_dim = features_count, activation='relu',return_sequences=True))
model.add(SimpleRNN(features_count, activation='relu',return_sequences=True))
#
model.add(TimeDistributed(Dense(num_classes,activation='softmax')))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#training loop
early_stopping = EarlyStopping(monitor='val_loss', patience=2)


model.fit_generator(training_generator,steps_per_epoch = steps_per_epoch,epochs = 200,validation_data = validation_generator,validation_steps=validation_steps,callbacks=[early_stopping])
model.save('./rnn.model')
