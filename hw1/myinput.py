import numpy as np
import sys
import mapping
import dic_processing
import configuration
from os.path import join
#return a dict which contain
#keys = {maeb0_si1411,maeb0_si2250,...} ->  sentenceID  ->  string
def read_data(data_path):
    sentence_dict = {}
    print('read X ',data_path)
    #training data
    with open(data_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip().split(' ')
        #fadg0_si1279_3
        speaker,sentence,frame = l[0].split('_')
        #fadg0_si1279
        key = speaker + '_' + sentence
        
        vals = [ float(x) for x in l[1:] ]
        if key not in sentence_dict:
            sentence_dict[key] = {}
        frame_dict = sentence_dict[key]
        frame_dict[int(frame)] = vals

    #to ndarray
    for sentenceID in sentence_dict.keys():
        frame_dict = sentence_dict[sentenceID]
        x_data_list =[frame_dict[index] for index in sorted(frame_dict.keys()) ] 
            
        #to numpy array,in shape(sentence_len,1)
        x_ndarr = np.asarray(x_data_list,np.float32)
        
        sentence_dict[sentenceID] = x_ndarr 

    return sentence_dict

#return dict[]
def read_label(label_path):
    sentence_dict = {}
    #label
    with open(label_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        buf,label = l.strip().split(',')
        
        #fadg0_si1279_3
        speaker,sentence,frame = buf.split('_')
        #fadg0_si1279
        key = speaker + '_' + sentence

        
        #new sentence during processing
        if key not in sentence_dict:
            sentence_dict[key] = {}
        #set y to specied frame id
        frame_dict = sentence_dict[key]
        
        frame_dict[int(frame)] = label
    #to ndarray
    for sentenceID in sentence_dict.keys():
        frame_dict = sentence_dict[sentenceID]
        y_labels =[frame_dict[index] for index in sorted(frame_dict.keys()) ] 
        #mapping
        #from char to int,sil -> 37,range[0,47]
        y_labels = mapping.mapping(y_labels,'48_int')
            
        #to numpy array,in shape(sentence_len,1)
        y_ndarr = np.asarray(y_labels,np.float32)
        y_ndarr = y_ndarr.reshape(y_ndarr.shape[0],1)
        sentence_dict[sentenceID] = y_ndarr 
    return sentence_dict

def read_x(path,padding_len,padding_val):
    sentence_dict = read_data(path)

    dic_processing.pad_dic(sentence_dict,padding_len,padding_val)
    x = dic_processing.vstack_dic(sentence_dict)
    return x
def read_y(path,padding_len,padding_val,ont_hot = True):
    sentence_dict = read_label(path)
    dic_processing.pad_dic(sentence_dict,padding_len,padding_val)
    if ont_hot:
        dic_processing.catogorate_dic(sentence_dict,configuration.num_classes+1)#include the padding symbol
    y = dic_processing.vstack_dic(sentence_dict)
    return y
def write_npz(x,y,path):
    print ('saving ' , path)
    np.savez_compressed(path,x=x,y=y)
    print ('saving done.')
def read_npz(path):
    print ('reading npz' , path)
    loaded = np.load(path)
    x = loaded['x']
    y = loaded['y']
    return x,y    

#read and save
def init_npz(data_dir,cur_dir):
    mfcc_path = join(data_dir , 'mfcc/train.ark')
    fbank_path = join(data_dir ,'fbank/train.ark')
    label_path = join((data_dir , 'label/train.lab')

    npz_path = join(cur_dir , 'mfcc.npz')
    print('start init')
    mapping.init(data_dir)
    x = read_x(mfcc_path,padding_len = configuration.max_len,padding_val = 0)
    y = read_y(label_path,padding_len = configuration.max_len,padding_val = configuration.num_classes)

    write_npz(x,y,npz_path)
    
def load_input(npz_path):

    return read_npz(npz_path)

if __name__ == '__main__':
    init_npz()
    print('Done')







