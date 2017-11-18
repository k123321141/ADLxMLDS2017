from keras.models import *
from keras.layers import *
import config
import keras.backend as K
import tensorflow as tf
import numpy as np
import myinput
import config
import HW2_config
import bleu_eval
assert K._BACKEND == 'tensorflow'

from myinput import decode,batch_decode

vocab_map = myinput.init_vocabulary_map()
decode_map = {vocab_map[k]:k for k in vocab_map.keys()}

def loss_with_mask(y_true, y_pred):
    #(batch,50,6528)
    #assert <pad> in t_true are all zeros.
    
    mask = tf.sign(tf.reduce_sum(y_true, axis = -1)) #(batch,50) -> 0,1 matrix

    cross_entropy = y_true * tf.log(y_pred)#(batch,50,6528)
    cross_entropy = -tf.reduce_sum(cross_entropy,axis=-1)
   
    cross_entropy = cross_entropy * mask
    return tf.reduce_mean(cross_entropy) 
    
    return (correct) / tf.reduce_sum(mask)
def acc_with_mask(y_true, y_pred):
    #(batch,50,6528)
    #assert <pad> in t_true are all zeros.
    
    mask = tf.sign(tf.reduce_sum(y_true, axis = -1))  #(batch,50) -> 0,1 matrix


    correct = tf.equal(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)) #(batch,50) 

    
    mask = tf.cast(mask,tf.float32)
    correct = tf.cast(correct,tf.float32)
    
    correct = tf.reduce_sum(mask * correct)

    
    return (correct) / tf.reduce_sum(mask)

def mask_zeros(x,src):
    #not zero
    mask = tf.sign(tf.abs(src)) 
    x = tf.multiply(mask,x)
    return x
def none_zeros_length(x):
    #(50,6528)
    for i in range(len(x)):
        if (x[i,:] == 0).all():
            break
        length = i
    return length
def my_pred(model,x,output_len):
    #(80,4096)
    x = x.reshape([1,80,4096])
    y_pred = myinput.caption_one_hot('<bos>')
    y_pred[0,1:,:] = 0
    for i in range(1,output_len):
        '''
        print 'pred'
        print decode(y_pred[0,:,:]) 
        print np.argmax(y_pred[0,:,:],axis = -1)
        '''
        y = model.predict([x,y_pred])
        next_idx = np.argmax(y[:,i-1,:],axis = -1)[0]
        y_pred[0,i,next_idx] = 1
        #print 'next ',i,next_idx

    return y_pred[0,1:,:]
def batch_pred(model,x,output_len):
    num = x.shape[0]
    y_pred = np.repeat(myinput.caption_one_hot('<bos>'),num,axis = 0)
    y_pred = y_pred.astype(np.float32)
    y_pred[:,1:,:] = 0
    for i in range(1,output_len):
        '''
        print 'pred'
        print decode(y_pred[0,:,:]) 
        print np.argmax(y_pred[0,:,:],axis = -1)
        '''
        y = model.predict([x,y_pred])
        np.copyto(y_pred[:,i,:],y[:,i-1,:])
        #print 'next ',i,next_idx

    return y_pred[:,1:,:]
def valid_sample_weight(y):
    video_num,output_len,vocab_dim = y.shape 
    mat = np.zeros([video_num,output_len])
    for i in range(video_num):
        length = none_zeros_length(y[i,:,:])
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
        preds = batch_pred(model,buf_x,HW2_config.output_len)

        guess = batch_decode(decode_map,preds)
        for i,k in enumerate(sorted(test_dic.keys())):
            out = '%s,%s\n' % (k,guess[i].replace(' <eos>',''))
            f.write(out)
    #
    return bleu_eval.main('./out.txt','../data/testing_label.json')
def testing(model,x,y,test_x,test_y,test_num = 1):
    
    idx = np.random.choice(len(x),test_num)
    rowx, rowy = x[idx,:,:], y[idx,:,:]
    #training set
    preds = batch_pred(model,rowx,HW2_config.output_len)
    train_correct = batch_decode(decode_map,rowy)
    train_guess = batch_decode(decode_map,preds)
    #test set
    idx = np.random.choice(len(test_x),test_num)
    rowx, rowy = test_x[idx,:,:],test_y[idx,:,:]
    test_preds = batch_pred(model,rowx,HW2_config.output_len)
    test_correct = batch_decode(decode_map,rowy)
    test_guess = batch_decode(decode_map,test_preds)
    for i,c in enumerate(train_correct):
        print('---')
        print('%20s : %s' % ('training set label',c))
        print('%20s : %s' % ('predict lable',train_guess[i]))
        print('%20s : %s' % ('test label',test_correct[i]))
        print('%20s : %s' % ('predict lable',test_guess[i]))
        print('---')

