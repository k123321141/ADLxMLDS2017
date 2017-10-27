import numpy as np
from keras.models import load_model

from mapping import * 
import myinput
import dic_processing
import random as r
import sys
import copy
from os.path import join

import model_cnn,model_rnn
from configuration import max_len,num_classes
from keras.utils import to_categorical





def predict_output(model,test_path,output_path):
    
    sentence_dict = myinput.read_data(test_path)

    dic_processing.pad_dic(sentence_dict,max_len,0)
    x = dic_processing.vstack_dic(sentence_dict)
    
    print('start predict..')
    y = model.predict(x)
    print('output prediction to %s ..' % output_path)
   
    #
    keys = sorted(sentence_dict.keys())
    #

    with open(output_path,'w') as f:
        f.write('id,phone_sequence\n')
        
        for i in range(len(keys)):
            sentenceID = keys[i]

            frame_seq = [np.argmax(y[i,j,:]) for j in range(y.shape[1])]


            s = convert_label_sequence(frame_seq).replace(',','')
         
            f.write('%s,%s\n' % (sentenceID,s))
            
def compare_output(model):
    
    
    
    #
    x,y = myinput.load_input('mfcc')
    #random select sample
    sample_count = 5
    random_indices = list(range(x.shape[0]))
    #r.shuffle(random_indices)
    random_indices = random_indices[:sample_count]
    #
    sub_x = x[random_indices,:,:]
    sub_y = y[random_indices,:,:]
    #
      
    z = model.predict(sub_x)
    for i in range(sample_count):
        
        x = sub_x[i,:,:]
        y_true = sub_y[i,:,:]
        #
        y_pred = z[i,:,:]
    
        y_true_seq = [np.argmax(y_true[j,:]) for j in range(y_true.shape[0])]  #777
        y_pred_seq = [np.argmax(y_pred[j,:]) for j in range(y_pred.shape[0])]
    
        s_true = convert_label_sequence(y_true_seq)
        s_pred = convert_label_sequence(y_pred_seq)
        

        print( '(%d/%d)    \n' % (i+1,sample_count) )
        h = 80
        k = int ( np.ceil((len(s_pred) / float(h))) ) 
        for j in range(k):
            start = j*h
            end = (j+1)*h
            print ( 'result  :   %s\nsrc     :   %s\n' % (s_pred[start:end],s_true[start:end]) )
#        print( 'result  :   %s\nsrc     :   %s\n' % (s_pred[end:],s_true[end:]))

#assert input y is one-hot and with padding tensor
def to_seq_y(y_with_padding,max_len=80):
    y = y_with_padding
    assert max_len < y.shape[1]
    buf = []
    for i in range(y.shape[0]):
        y_true_seq = [np.argmax(y_true[i,j,:]) for j in range(y.shape[1])]  #777
        #eos trimming
        eos = (y.shape[-1] - 1)
        if eos in y_true_seq:
            y_true_seq = y_true_seq[0: y_true_seq.index(eos)]
        c_arr = y_true_seq
        
        #reversed from index to char
        c_arr = mapping(c_arr,'48_reverse')
            
        #48 -> 39   
        c_arr = mapping(c_arr,'48_39')
       
        #trimming
        c_arr = trim_sil(c_arr)
        c_arr = trim_repeat(c_arr)

        #back to index
        c_arr = mapping(c_arr,'48_int')
        #padding eos
        if(len(c_arr) < max_len):
            c_arr.extend( [eos]*( max_len-len(c_arr) ) )
        #to ndarray
        y_ndarr = np.asarray(c_arr).reshape(1,max_len)
        #to one hot
        y_ndarr = to_categorical(y_ndarr,eos+1)
        buf.append(y_ndarr)
    return np.vstack(buf)
#y is a list
def convert_label_sequence(label_seq):
    #eof trimming
    if num_classes in label_seq:
        label_seq = label_seq[0: label_seq.index(num_classes)]
    c_arr = label_seq
    
    #reversed from index to char
    c_arr = mapping(c_arr,'48_reverse')
        
    #48 -> 39   
    c_arr = mapping(c_arr,'48_39')
   
    #threshold
    #c_arr = phoneme_threshold(c_arr,3)
    
    #trimming
    c_arr = trim_sil(c_arr)
    c_arr = trim_repeat(c_arr)

    #index -> char
    c_arr = mapping(c_arr,'48_char')

    #to string
    s = ''
    for c in c_arr:
        s += c +','
    return s
def phoneme_threshold(c_arr,thresd_hold = 3):
    t = 0
    pre = 'start'
    buf = copy.copy(c_arr)
    for i in range(len(buf)):
        c = buf[i]
        if c == pre:
            t += 1
        elif c != pre:
            if t < thresd_hold:
                buf[i-t:i] = ['del']*t
            t = 1
            pre = c
    result = []
    for c in buf:
        if c != 'del':
            result.append(c)
    return result




#python <model_path>   ->   output random 10 training predict with correct labels to stdout
#python <model_path> <test_path> <output_path> ->   output test predict to <output_path>
def main(argv,data_dir):
    assert len(argv) == 2 or len(argv) == 4
    model_path = argv[1]
    if 'cnn' in model_path:
        model = model_cnn.init_model()
    elif 'rnn' in model_path:
        model = model_rnn.init_model()
    elif 'best' in model_path:
        model = model_cnn.init_model()
    else:
        print('error with model path : %s' % model_path)
        sys.exit(1)
    '''
    import loss
    import tensorflow as tf
    if 'rnn' not in model_path:
        loss.mask_vector = tf.constant([48]*773,tf.int64)
    model = load_model(model_path,custom_objects = {'loss_with_mask' : loss.loss_with_mask,'acc_with_mask':loss.acc_with_mask})
    ''' 
    model.load_weights(model_path)
    #mapping init
    init(data_dir) 
    if len(argv) == 2:
        compare_output(model)
    else:
        test_path = argv[2]
        output_path = argv[3] 
        predict_output(model,test_path = test_path,output_path = output_path)


if __name__ == '__main__':
    main(sys.argv,data_dir ='../data/')
