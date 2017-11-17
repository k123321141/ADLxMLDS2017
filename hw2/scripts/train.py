# -*- coding: utf-8 -*-

from keras.callbacks import *
import keras
import numpy as np
import seq2seq
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
def valid_sample_weight(y):
    video_num,output_len,vocab_dim = y.shape 
    mat = np.zeros([video_num,output_len])
    for i in range(video_num):
        length = utils.none_zeros_length(y[i,:,:])
        mat[i,0:length] = 1
    return mat
def weighted_by_frequency(y):
    video_num,output_len,vocab_dim = y.shape

    #count frequency
    fre = {idx:0 for idx in decode_map.keys()}
    for i in range(video_num):
        V = np.argmax(y[i,:,:],axis = -1)
        for j in range(output_len):
            v_idx = V[j]
            fre[v_idx] += 1
    #devide by frequency 
    mat = np.ones([video_num,output_len])
    for i in range(video_num):
        V = np.argmax(y[i,:,:],axis = -1)
        for j in range(output_len):
            v_idx = V[j]
            mat[i,j] /= fre[v_idx]
    
    return mat
def compute_belu():
    print('load testing data.')
    test_dic = myinput.load_x_dic('../data/testing_data/feat/')
    print('load model.')
    output_path = './out.txt'
    model = load_model(config.PRE_MODEL,custom_objects={'loss_with_mask':utils.loss_with_mask,'acc_with_mask':utils.acc_with_mask})
    #
    print('init decode map')
    vocab_map = myinput.init_vocabulary_map()
    decode_map = myinput.init_decode_map(vocab_map)
    #
    print('start prdiction.')
    with open(output_path,'w') as f:
        num = len(test_dic.keys())
        buf_x = np.zeros([num,HW2_config.input_len,HW2_config.input_dim],dtype=np.float32)
        for i,k in enumerate(sorted(test_dic.keys())):
            buf_x[i,:,:] = test_dic[k]
        preds = seq2seq.batch_pred(model,buf_x,HW2_config.output_len)

        guess = batch_decode(decode_map,preds)
        for i,k in enumerate(sorted(test_dic.keys())):
            out = '%s,%s\n' % (k,guess[i].replace(' <eos>',''))
            f.write(out)
    #
    bleu_eval.main('./out.txt','../data/testing_label.json')

if __name__ == '__main__':
    x = myinput.read_x()
    y_generator = myinput.load_y_generator()

    #testing
    test_x = myinput.read_x('../data/testing_data/feat/')
    test_y_generator = myinput.load_y_generator('../data/testing_label.json')
    now_belu = 0

    epoch_idx = 0
    if os.path.isfile(config.PRE_MODEL):
        print('loading PRE_MODEL : ',config.PRE_MODEL)
        model = load_model(config.PRE_MODEL,
                custom_objects={'loss_with_mask':utils.loss_with_mask,'acc_with_mask':utils.acc_with_mask})
    else:
        vocab_dim = len(myinput.init_vocabulary_map())
        model = seq2seq.model(HW2_config.input_len,HW2_config.input_dim,HW2_config.output_len,vocab_dim)
   
    print 'start training' 
    model.compile(loss=utils.loss_with_mask,
                  optimizer='adam',
                  metrics=[utils.acc_with_mask],sample_weight_mode = 'temporal')
    for epoch_idx in range(2000000):
        #train by labels
        train_cheat = np.repeat(myinput.caption_one_hot('<bos>'),HW2_config.video_num,axis = 0)
        #record the loss and acc
        metric_history = {}

        for caption_idx in range(HW2_config.caption_list_mean):
            y = y_generator.next()
            np.copyto(train_cheat[:,1:,:],y[:,:-1,:])
            his = model.fit(x=[x,train_cheat], y=y,
                      batch_size=config.BATCH_SIZE,verbose=config.VERBOSE,
                      epochs=1,sample_weight = weighted_by_frequency(y))
            print('caption iteration : (%3d/%3d)' % (caption_idx+1,HW2_config.caption_list_mean))
            #record the loss and acc
            for metric,val in his.history.items():
                if metric not in metric_history:
                    metric_history[metric] = 0
                metric_history[metric] += val[0]
            sys.stdout.flush()


        loss = []
        #print history
        print('epoch_idx : %5d' % epoch_idx)
        for metric,val in metric_history.items():
            val /= HW2_config.caption_list_mean 
            print('%15s:%30f'%(metric,val))
        metric_history.clear()

        #after a epoch
        if epoch_idx % config.SAVE_ITERATION == 0:
            #model.save(join(config.CKS_PATH,'%d.cks'%epoch_idx))
            model.save(config.CKS_PATH)
            #test_y just for testing,no need for iter as a whole epoch 
            test_y = test_y_generator.next()
            # Select 2 samples from the test set at random so we can visualize errors.
            testing(model,x,y,test_x,test_y,2) 
            belu = compute_belu()
            if belu > now_belu:
                now_belu = belu
                print('new high bleu : ',bleu,'save model..')
                model.save(config.CKS_PATH+str(belu))
                

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
        preds = seq2seq.my_pred(model,rowx,input_len,output_len)
        correct = decode(rowy[0])
        guess = decode(preds[0])

        print('T', correct)
        print('G',guess)
        print('---')







