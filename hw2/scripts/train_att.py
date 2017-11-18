# -*- coding: utf-8 -*-

from keras.callbacks import *
import keras
import numpy as np
import attention 
import myinput
import config
import HW2_config
import os
import sys
import utils
import bleu_eval
from myinput import decode,batch_decode
from os.path import join
from keras.models import *
import os
vocab_map = myinput.init_vocabulary_map()
decode_map = {vocab_map[k]:k for k in vocab_map.keys()}
def testing(model,x,y,test_x,test_y,test_num = 1):
    
    idx = np.random.choice(len(x),test_num)
    rowx, rowy = x[idx,:,:], y[idx,:,:]
    #training set
    preds = seq2seq.batch_pred(model,rowx,HW2_config.output_len)
    train_correct = batch_decode(decode_map,rowy)
    train_guess = batch_decode(decode_map,preds)
    #test set
    idx = np.random.choice(len(test_x),test_num)
    rowx, rowy = test_x[idx,:,:],test_y[idx,:,:]
    test_preds = seq2seq.batch_pred(model,rowx,HW2_config.output_len)
    test_correct = batch_decode(decode_map,rowy)
    test_guess = batch_decode(decode_map,test_preds)
    for i,c in enumerate(train_correct):
        print('---')
        print('%20s : %s' % ('training set label',c))
        print('%20s : %s' % ('predict lable',train_guess[i]))
        print('%20s : %s' % ('test label',test_correct[i]))
        print('%20s : %s' % ('predict lable',test_guess[i]))
        print('---')
if __name__ == '__main__':
    x = myinput.read_x()
    y_generator = myinput.load_y_generator()

    #testing
    test_x = myinput.read_x('../data/testing_data/feat/')
    test_y_generator = myinput.load_y_generator('../data/testing_label.json')

    model = attention.model(80,4096,50,len(vocab_map)) 

    from keras.utils import plot_model
    plot_model(model, to_file='./model.png',show_shapes=True)
    print('Done') 
   
    print 'start training' 
    model.compile(loss=utils.loss_with_mask,
                  optimizer='adam',
                  metrics=[utils.acc_with_mask],sample_weight_mode = 'temporal')
    for epoch_idx in range(2000000):

        for caption_idx in range(1):
            y = y_generator.next()
            num,output_len,vocab_dim = y.shape
            y2 = np.zeros([num,80,vocab_dim])
            np.copyto(y2[:,:50,:],y)
            
            #np.copyto(train_cheat[:,1:,:],y[:,:-1,:])
            his = model.fit(x=x, y=y2,
                      batch_size=config.BATCH_SIZE,verbose=1,
                      epochs=1)


                

