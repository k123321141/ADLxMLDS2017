# -*- coding: utf-8 -*-
from keras.callbacks import *
import keras
import numpy as np
import my_model
import myinput
import config
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

    x,y = myinput.read_input()
    # Shuffle (x, y) in unison as the later parts of x will almost all be larger
    # digits.
    print 'start training'
    '''
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices,:,:]
    y = y[indices,:,:]
    '''
    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // VALIDATION_PERCENT
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]
   
    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)
    

    input_len,input_dim = x_train.shape[1:]
    output_len,vocab_dim = y_train.shape[1:]


    model = my_model.model(input_len,input_dim,output_len,vocab_dim)
    if os.path.isfile(PRE_MODEL):
        model = load_model(PRE_MODEL)
   
    # Train the model each generation and show predictions against the validation
    # dataset.
    train_cheat = np.zeros(y_train.shape, dtype=np.bool)
    train_cheat[:,0,:] = myinput.caption_one_hot('<bos>')[0,0,:]
    train_cheat[:,1:,:] = y_train[:,:-1,:]
    val_cheat = np.zeros(y_val.shape, dtype=np.bool)
    val_cheat[:,0,:] = myinput.caption_one_hot('<bos>')[0,0,:]
    val_cheat[:,1:,:] = y_val[:,:-1,:]
    print 'start training'
    #
    print 'train cheat '
    print decode(train_cheat[0,:,:]) 
    print np.argmax(train_cheat[0,:,:],axis = -1)
    print 'train'
    print decode(y_train[0,:,:]) 
    print np.argmax(y_train[0,:,:],axis = -1)
    print 'val'
    print decode(val_cheat[0,:,:]) 
    print np.argmax(val_cheat[0,:,:],axis = -1)


    #
    for iteration in range(1, 20000):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        #check point
        #cks = ModelCheckpoint(filepath = ('../models/%d_val_{val_loss:.2f}.cks'%iteration),save_best_only=True,period = 1)
        #


        model.fit(x=[x_train,train_cheat], y=y_train,
                  batch_size=BATCH_SIZE,verbose=VERBOSE,
                  epochs=1,validation_data =([x_val,val_cheat],y_val))
        if iteration % SAVE_ITERATION == 0:
            model.save(join(CKS_PATH,'1024_%d.cks'%iteration))
        # Select 10 samples from the validation set at random so we can visualize
        # errors.
        for i in range(2):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            #
            preds = my_model.my_pred(model,rowx,input_len,output_len)
            print('shape',preds.shape)
            correct = decode(rowy[0])
            guess = decode(preds[0])

            print('T', correct)
            ''' 
            if correct == guess:
                print(colors.ok + '☑' + colors.close, end" ")
            else:
                print(colors.fail + '☒' + colors.close, end=" ")
            '''
            print('G',guess)
            print('---')







