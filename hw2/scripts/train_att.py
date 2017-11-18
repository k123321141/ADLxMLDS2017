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
            his = model.fit(x=[x,y], y=y,
                      batch_size=config.BATCH_SIZE,verbose=1,
                      epochs=1)


                

# -*- coding: utf-8 -*-
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
def get_high_belu():
    #belu1
    belu1_header = 'belu1.'
    buffer_list = [f for f in os.listdir(config.BELU_PATH) if f.startswith(belu1_header)]
    high = 0
    for f in buffer_list:
        score = float(f.replace(belu1_header,'') )
        if score > high:
            high = score
    belu1 = high
    #belu2
    belu2_header = 'belu2.'
    buffer_list = [f for f in os.listdir(config.BELU_PATH) if f.startswith(belu2_header)]
    high = 0
    for f in buffer_list:
        score = float(f.replace(belu2_header,'') )
        if score > high:
            high = score
    belu2 = high

    return belu1,belu2
def compute_belu(model):
    test_dic = myinput.load_x_dic('../data/testing_data/feat/')
    output_path = './out.txt'
    #
    print('start prdition.')
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
    return bleu_eval.main('./out.txt','../data/testing_label.json')

if __name__ == '__main__':
    x = myinput.read_x()
    y_generator = myinput.load_y_generator()

    #testing
    test_x = myinput.read_x('../data/testing_data/feat/')
    test_y_generator = myinput.load_y_generator('../data/testing_label.json')
    belu1_high,belu2_high = get_high_belu()

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


            #save the high belu score model
            belu1,belu2 = compute_belu(model)
            if belu1 > belu1_high:
                belu1_high = belu1
                print('new high bleu original score : ',belu1,'save model..')
                buffer_list = [f for f in os.listdir(config.BELU_PATH) if f.startswith('belu1')]
                for f in buffer_list:
                    os.remove(join(config.BELU_PATH,f))
                model.save(config.BELU_PATH+'belu1.'+str(belu1))
            if belu2 > belu2_high:
                belu2_high = belu2
                print('new high bleu new modified score : ',belu2,'save model..')
                buffer_list = [f for f in os.listdir(config.BELU_PATH) if f.startswith('belu2.')]
                for f in buffer_list:
                    os.remove(join(config.BELU_PATH,f))
                model.save(config.BELU_PATH+'belu2.'+str(belu2))
                

    #
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
from os.path import join
from keras.models import *
import os

vocab_map = myinput.init_vocabulary_map()



if __name__ == '__main__':
    x = myinput.read_x()
    y_generator = myinput.load_y_generator()

    #testing
    test_x = myinput.read_x('../data/testing_data/feat/')
    test_y_generator = myinput.load_y_generator('../data/testing_label.json')
    belu1_high,belu2_high = utils.get_high_belu()

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
                      epochs=1,sample_weight = utils.weighted_by_frequency(y))
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


            #save the high belu score model
            belu1,belu2 = utils.compute_belu(model)
            if belu1 > belu1_high:
                belu1_high = belu1
                print('new high bleu original score : ',belu1,'save model..')
                buffer_list = [f for f in os.listdir(config.BELU_PATH) if f.startswith('belu1')]
                for f in buffer_list:
                    os.remove(join(config.BELU_PATH,f))
                model.save(config.BELU_PATH+'belu1.'+str(belu1))
            if belu2 > belu2_high:
                belu2_high = belu2
                print('new high bleu new modified score : ',belu2,'save model..')
                buffer_list = [f for f in os.listdir(config.BELU_PATH) if f.startswith('belu2.')]
                for f in buffer_list:
                    os.remove(join(config.BELU_PATH,f))
                model.save(config.BELU_PATH+'belu2.'+str(belu2))
                

    #
