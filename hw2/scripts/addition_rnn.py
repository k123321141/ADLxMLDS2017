# -*- coding: utf-8 -*-
from keras.callbacks import *
import keras
import numpy as np
import my_model
import myinput
import config
import HW2_config
import os
from os.path import join
from keras.models import *

vocab_map = myinput.init_vocabulary_map()
BATCH_SIZE,VALIDATION_PERCENT,CKS_PATH,VERBOSE,PRE_MODEL,SAVE_ITERATION = config.trainging_config()

decode_map = {vocab_map[k]:k for k in vocab_map.keys()}
def decode(y):
    output_len = y.shape[0]
    y = np.argmax(y,axis = -1)
    s = ''
    for j in range(output_len):
        vocab_idx = y[j]
        if vocab_idx != 0:      #<pad>
            s += decode_map[vocab_idx] + ' '
    s = s.strip() + '.' 
    return s.encode('utf-8')
if __name__ == '__main__':

    x = myinput.read_x()
    y_generator = myinput.load_y_generator()

    #testing
    test_x = myinput.read_x('../data/testing_data/feat/')
    test_y_generator = myinput.load_y_generator('../data/testing_label.json')
    

    epoch_idx = 0
    if os.path.isfile(PRE_MODEL):
        print('loading PRE-MODEL : ',PRE_MODEL)
        model = load_model(PRE_MODEL)
        epoch_idx = int( PRE_MODEL.slpit('_')[:-4] )
    else:
        model = my_model.model(HW2_config.input_len,HW2_config.input_dim,HW2_config.output_len,HW2_config.vocab_dim)
   
    print 'start training' 
    for epoch_idx in range(2000000):
        #train by labels
        train_cheat = np.repeat(myinput.caption_one_hot('<bos>'),HW2_config.video_num,axis = 0)
        for caption_idx in range(HW2_config.caption_list_mean):
            y = y_generator.next()
            print train_cheat.shape,y.shape
            np.copyto(train_cheat[:,1:,:],y[:,:-1,:])

            model.fit(x=[x,train_cheat], y=y,
                      batch_size=BATCH_SIZE,verbose=VERBOSE,
                      epochs=1)

        #test_y just for testing,no need for iter as a whole epoch 
        test_y = test_y_generator.next()
        #after a epoch
        if epoch_idx % SAVE_ITERATION == 0:
            model.save(join(CKS_PATH,'%d.cks'%epoch_idx))
        # Select 2 samples from the test set at random so we can visualize errors.
        testing(model,test_x,test_y,2) 


    #
    '''
    print 'train cheat '
    print decode(train_cheat[0,:,:]) 
    print np.argmax(train_cheat[0,:,:],axis = -1)
    print 'train'
    print decode(y_train[0,:,:]) 
    print np.argmax(y_train[0,:,:],axis = -1)
    print 'val'
    print decode(val_cheat[0,:,:]) 
    print np.argmax(val_cheat[0,:,:],axis = -1)
    '''

def testing(model,x,y,test_num = 1):
    for _ in range(test_num):
        idx = np.random.randint(0, len(x))
        rowx, rowy = x[idx,:,:], y[idx,:,:]
        #
        preds = my_model.my_pred(model,rowx,input_len,output_len)
        correct = decode(rowy[0])
        guess = decode(preds[0])

        print('T', correct)
        print('G',guess)
        print('---')







