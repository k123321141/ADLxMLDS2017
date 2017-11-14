# -*- coding: utf-8 -*-

from keras.callbacks import *
import keras
import numpy as np
import my_model
import myinput
import config
import HW2_config
import os
from myinput import decode,batch_decode
from os.path import join
from keras.models import *

vocab_map = myinput.init_vocabulary_map()
decode_map = {vocab_map[k]:k for k in vocab_map.keys()}
def testing(model,x,y,test_x,test_y,test_num = 1):
    
    idx = np.random.choice(len(x),test_num)
    rowx, rowy = x[idx,:,:], y[idx,:,:]
    #
    preds = my_model.batch_pred(model,rowx,HW2_config.output_len)
    correct = batch_decode(decode_map,rowy)
    guess = batch_decode(decode_map,preds)
    for i,c in enumerate(correct):
        print('%20s : %s' % ('training set label',c))
        print('%20s : %s' % ('predict lable',guess[i)])
        print('---')
if __name__ == '__main__':

    x = myinput.read_x()
    y_generator = myinput.load_y_generator()

    #testing
    test_x = myinput.read_x('../data/testing_data/feat/')
    test_y_generator = myinput.load_y_generator('../data/testing_label.json')
    

    epoch_idx = 0
    if os.path.isfile(config.PRE_MODEL):
        print('loading PRE-MODEL : ',config.PRE_MODEL)
        model = load_model(config.PRE_MODEL)
        epoch_idx = int( config.PRE_MODEL.slpit('_')[:-4] )
    else:
        vocab_dim = len(myinput.init_vocabulary_map())
        model = my_model.model(HW2_config.input_len,HW2_config.input_dim,HW2_config.output_len,vocab_dim)
   
    print 'start training' 
    for epoch_idx in range(2000000):
        #train by labels
        train_cheat = np.repeat(myinput.caption_one_hot('<bos>'),HW2_config.video_num,axis = 0)
        for caption_idx in range(HW2_config.caption_list_mean):
            y = y_generator.next()
            np.copyto(train_cheat[:,1:,:],y[:,:-1,:])
            print('caption iteration : (%3d/%3d)' % (caption_idx,HW2_config.caption_list_mean))
            model.fit(x=[x,train_cheat], y=y,
                      batch_size=config.BATCH_SIZE,verbose=config.VERBOSE,
                      epochs=1)

        #test_y just for testing,no need for iter as a whole epoch 
        test_y = test_y_generator.next()
        #after a epoch
        if epoch_idx % config.SAVE_ITERATION == 0:
            model.save(join(config.CKS_PATH,'%d.cks'%epoch_idx))
        # Select 2 samples from the test set at random so we can visualize errors.
        testing(model,x,y,test_x,test_y,2) 


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







