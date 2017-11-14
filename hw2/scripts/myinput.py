import numpy as np
import os
import json
import random 
import sys
import config
from HW2_config import *
from os.path import join
'''
max caption length = 40
'''
data_path,label_path,FETCH_NUM = config.input_config()

def init_vocabulary_map(label_path = label_path):
    test = json.load(open(label_path,'r'))
    vocabulary_set = set()
    for test_json in test:
        caption_list = test_json['caption']
        for caption in caption_list:
            caption = caption[:-1] if caption[-1] == '.' else caption    #trim last . 'Hello.' -> 'Hello'
            caption = caption.lower()
            for v in caption.split(' '):
                vocabulary_set.add(v)
    #add <bos> <eos>
    vocabulary_set.add('<bos>')
    vocabulary_set.add('<eos>')

    #one hot encoding
    vocab_list = sorted(vocabulary_set)
    vocab_map = { v:(i+1) for i,v in enumerate(vocab_list)}
    vocab_map['<pad>'] = 0

    return vocab_map


vocab_map = init_vocabulary_map()
def load_x(data_path=data_path):
    print('load data from : ',data_path)
    fl = os.listdir(data_path)
    
    file_name_list = [f for f in fl if f.endswith('.npy')]
    #sort for pairing with label
    dic = {}
    for f in file_name_list:
        dic[f[:-4]] = np.load(join(data_path,f)).reshape([1,80,4096])
    return dic
def load_y(label_path = label_path):
    test = json.load(open(label_path,'r'))
    dic = {}
    for test_json in test:
        caption_list = test_json['caption']
        movie_id = test_json['id']
        dic[movie_id] = caption_list
       
    return dic
def load_y_generator(label_path = label_path):
    dic = load_y(label_path)
    #iter loop index
    #every video has different number of captions.
    #so iter each caption with same index,some video with small number caption will iterate more time.
    video_num = len(dic)
    vocab_dim = len(vocab_map.keys())
    idx = 0
    while True:
        y = np.zeros(video_num,output_len,vocab_dim)
        for i,video_name in enumerate(sorted(dic.keys())):
            caption_list = dic[video_name]
            caption_idx = idx % len(caption_list)
            caption = caption_list[caption_idx]
            y[i,:,:] = caption_one_hot(caption)
        idx += 1
        yield y
def read_input(data_path=data_path,label_path = label_path):
    print ('read label from : ',label_path)
    test = json.load(open(label_path,'r'))
    dic = {}
    for test_json in test:
        caption_list = test_json['caption']
        movie_id = test_json['id']
        dic[movie_id] = caption_list
        
    print('load data from : ',data_path)
    fl = os.listdir(data_path)
    

    file_name_list = [f for f in fl if f.endswith('.npy')]
    #
    x_num = len(file_name_list)
    caption_num = int(sum([len(caption_list) for caption_list in dic.values()]))
    vocab_dim = len(vocab_map.keys())
    print (caption_num,vocab_dim)
    buf_x = np.zeros([x_num*FETCH_NUM,80,4096],dtype=np.float32)
    buf_y = np.zeros([x_num*FETCH_NUM,50,vocab_dim],dtype=np.float32)
    #sort for pairing with label
    #pair 5 random caption for each x
    for i,f in enumerate(sorted(file_name_list)):
        feats = np.load(join(data_path,f)).reshape([1,80,4096])
        movie_id = f[:-4]               #replace('.npy','')
        caption_list = dic[movie_id]
        #random 5
        indices = range(len(caption_list))
        random.shuffle(indices)
        indices = indices[:FETCH_NUM]
        #
        for j,idx in enumerate(indices):
            caption = caption_list[idx]
            buf_x[i*FETCH_NUM+j,:,:] = feats[:,:,:]
            buf_y[i*FETCH_NUM+j,:,:] = caption_one_hot(caption)
            #print i*FETCH_NUM+j,i,j
        '''
        for i,caption in enumerate(caption_list):
            buf_x[i,:,:] = feats[:,:,:]
            buf_y[i,:,:] = caption_one_hot(caption)
        '''
    #total 1450 movie on training,each movie has 14-19 caption
    #the output will be

    return buf_x,buf_y
def caption_one_hot(caption,pad_len = 50):
    #trim caption
    caption = caption[:-1] if caption[-1] == '.' else caption    #trim last . 'Hello.' -> 'Hello'
    caption = caption + ' <eos>'
    caption = caption.lower()
    #
    vocab_dim = len(vocab_map.keys())
    buf = np.zeros([1,pad_len,vocab_dim],np.bool)
    buf[0,:,0] = 1  #<pad>
    for i,v in enumerate(caption.split(' ')):
        vocab_idx = vocab_map[v]
        buf[0,i,0] = 0
        buf[0,i,vocab_idx] = 1
    return buf
if __name__ == '__main__':

    dic = load_x()
    x = dic['7_XASfcYdBk_3_13.avi']
    print x[0,24,:]
    '''
    buf = []
    for l in dic.values():
        buf.append(len(l))
    print sorted(buf)
    print np.mean(buf),np.var(buf)
    '''
'''
x,y = read_input() 
nx = x.nbytes
ny = y.nbytes
def gb(x):
    x = float(x)
    x /= (1024**3)
    return '%.2f' % (x)
print x.shape,y.shape
print gb(nx),gb(ny)
'''
#np.savez('traing.npz',x=x,y=y)
