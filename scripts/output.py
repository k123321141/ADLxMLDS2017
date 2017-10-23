import numpy as np
from keras.models import *
import mapping
import myinput
import dic_processing
import random as r

#assert sentenceID & frameID follow the numeric order
#return sentence_dict
max_len = 777
num_classes = 48

map_48_int_dict,map_48_reverse,map_48_char_dict,map_48_39_dict = mapping.read_maps()


def predict_output(model,input_path,output_path):
    
    dic_x = myinput.load_test(input_path)
    
    

    i = 1
    total = len(dic_x.keys())
    with open(output_path,'w') as f:
        f.write('id,phone_sequence\n')
        
        for setenceID in dic_x.keys():
            x = dic_x[setenceID]
            frame_len,feature_num = x.shape
#            padding for model
            x = np.pad(x,((0,max_len-frame_len),(0,0)),'constant', constant_values=0)
            x = x.reshape(1,max_len,feature_num,1)
            #
            y = model.predict(x)
            frame_seq = [np.argmax(y[0,j,:]) for j in range(frame_len)]

            s = convert_label_sequence(frame_seq)
            
            f.write('%s,%s\n' % (setenceID,s))
            
            print '%d/%d    %s,%s\n' % (i,total,setenceID,s)
            i+=1
def compare_output(model):
    
    
    
    #
    src = myinput.load_input('mfcc')
    keys = src.keys()
    r.shuffle(keys)
    #sub_keys = keys[0:10]
    sub_keys = keys
#    sub_keys = ['fadg0_si1279']
    sub_x ={k:(src[k]) for k in sub_keys }
    
    dic_x = sub_x

    i = 1
    total = len(dic_x.keys())
        
    for setenceID in dic_x.keys():
        x,src_y = dic_x[setenceID]
        frame_len,feature_num = x.shape
#            padding for model
        x = np.pad(x,((0,max_len-frame_len),(0,0)),'constant', constant_values=0)
        x = x.reshape(1,max_len,feature_num,1)
        #
        y = model.predict(x)
        frame_seq = [np.argmax(y[0,j,:]) for j in range(frame_len)]

        s = convert_label_sequence(frame_seq)
        

        src_y = src_y.reshape(frame_len)
        src_y = src_y.astype(np.int32)
        y_seq = src_y.tolist()
        s2 = convert_label_sequence(y_seq)[:-1]

        print '(%d/%d)  %s    \n' % (i,total,setenceID)
        h = 80
        k = (len(s) / h)
#            for j in range(k):
#                start = j*h
#                end = (j+1)*h
#                print 'result  :   %s\nsrc     :   %s\n' % (s[start:end],s2[start:end])
#            print 'result  :   %s\nsrc     :   %s\n' % (s[end:],s2[end:])

        i+=1
def generate_seq_y(output_path):
    
    dic_x = myinput.load_input('mfcc')
    
    i = 1
    total = len(dic_x.keys())
    with open(output_path,'w') as f:
         
        for setenceID in dic_x.keys():
            x,src_y = dic_x[setenceID]
            frame_len,feature_num = x.shape
            

            src_y = src_y.reshape(frame_len)
            src_y = src_y.astype(np.int32)
            y_seq = src_y.tolist()
            s2 = convert_label_sequence(y_seq)[:-1]

            print '(%d/%d)  %s    \n' % (i,total,setenceID)
            f.write('%s,%s\n' % (setenceID,s2))
            i+=1
#y is a list
def convert_label_sequence(label_seq):
    #eof trimming
    if num_classes in label_seq:
        label_seq = label_seq[0: label_seq.index(num_classes)]
    #reversed from index to char
    c_arr = [map_48_reverse[y] for y in label_seq]
        
            
    c_arr = [map_48_39_dict[c] for c in c_arr]
    #trimming
    c_arr = trim_sil(c_arr)
    c_arr = trim_repeat(c_arr)
    #
#    c_arr = [map_48_char_dict[c] for c in c_arr]

    s = ''
    for c in c_arr:
        s += c +','
    return s



def trim_sil(arr):
    #trim sil
    for i in range(len(arr)):
        a = arr[i]
        if a != 'sil':
            start_index = i
            break
    arr = arr[start_index:]

    for i in range(len(arr)-1,-1,-1):
        a = arr[i]
        if a != 'sil':
            end_index = i
            break
    arr = arr[:end_index+1]
    
    return arr

def trim_repeat(arr):
    #remove repeat
    pre = 'start'
    for i in range(len(arr)):
        a = arr[i]
        if a == pre:
            arr[i] = 'repeat'
        else:
            pre = a
    #
    result = []
    for a in arr:
        if a != 'repeat':
            result.append(a)
    return result



def map48_39(map_file_path):

    #read map file
    with open(map_file_path,'r') as f:
        lines = f.readlines()
    result = {}
    for l in lines:
        l = l.strip()
        src,dst = l.split('\t')
        result[src] = dst
    return result
def reverse_dic(dic):
    inv_map = {v: k for k, v in dic.iteritems()}
    return inv_map
def map_phone_char(map_file_path,to_char = False):
    #read map file
    with open(map_file_path,'r') as f:
        lines = f.readlines()
    result = {}
    for l in lines:
        l = l.strip()
        src,dst_index,dst_char = l.split('\t')
        if to_char:
            result[src] = dst_char
        else:
            result[src] = int(dst_index)

    return result


#read and save
if __name__ == '__main__':
    output_path = '../data/output.csv'
    model_path = '../models/simple_k.model'
    test1_path = '../data/mfcc/test.ark'
    test2_path = '../data/fbank/test.ark'
    
    
    
    model = load_model(model_path)
    
#    predict_output(model,input_path = '../data/mfcc/test.ark',output_path = output_path)
#    compare_output(model)
    generate_seq_y(output_path='../data/mfcc/seq_y.ark')
    print 'Done'




