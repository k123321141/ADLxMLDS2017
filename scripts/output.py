import numpy as np
from keras.models import *
import mapping
import myinput
#assert sentenceID & frameID follow the numeric order
#return sentence_dict
max_len = 777
features_count = 39
num_classes = 48

map_48_int_dict,map_48_reverse,map_48_char_dict,map_48_39_dict = mapping.read_maps()


def predict_output(model,input_path,output_path,map_48_int_dict,map_48_reverse,map_48_char_dict,map_48_39_dict):
    
    dic1 = myinput.load_test(input_path)
    x,fake_y = dic_processing.toXY(dic1)




    #y in shape (None,777,48)
    y = model.predict(x)
#    sentence_dict = dic3
    sentence_dict = dic1

    #for counting
    keys = sorted(sentence_dict.keys())
    total = len(keys)
    #the output is follow the sorted(keys)
    


    with open(output_path,'w') as f:
        f.write('id,phone_sequence\n')
        for i in range(total):
            sentenceID = keys[i]
            buf = y[i,:,:]
#            output a label sequence correspond to an sentence
            frame_seq = [argmax(buf[j,:]) for j in range(max_len)]
        
            s = convert_label_sequence(frame_seq)
            
            f.write('%s,%s\n' % (sentence_id,s))
            print '%d/%d    %s,%s\n' % (i,total,sentenceID,s)
#y is a list
def convert_label_sequence(label_seq):
    #eof trimming
    if num_classes in y_arr:
        y_arr = y_arr[0: y_arr.index(num_classes)]
    #reversed from index to char
    c_arr = [rev_dic[y] for y in y_arr]
        
            
    c_arr = [map_48_39_dict[c] for c in c_arr]
    #trimming
    c_arr = trim_sil(c_arr)
    c_arr = trim_repeat(c_arr)
    #
    c_arr = [map_48_char_dict[c] for c in c_arr]

    s = ''
    for c in c_arr:
        s += c
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
    
    predict_output(model,input_path = '../data/mfcc/test.ark',output_path)

    print 'Done'




