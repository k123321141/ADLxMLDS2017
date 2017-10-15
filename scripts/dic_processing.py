from keras.utils import to_categorical
import numpy as np


#just for test data,only x
def pad_test_dic(dic,len):


def pad_dic(dic,len):
    for sentence_id in dic.keys():
        x,y = dic[sentence_id]
        num = x.shape[0]
        assert x.shape[0] == y.shape[0]
        x = np.pad(x,((0,777-num),(0,0)),'constant', constant_values=0)
        
        #the length of y doesn't matter,just pad for alignment.
        y = np.pad(y,((0,777-num),(0,0)),'constant', constant_values=0)
        dic[sentence_id] = (x,y)

def catogorate_dic(dic,num_classes):
    for sentence_id in dic.keys():
        x,y = dic[sentence_id]
        num = x.shape[0]
        assert x.shape[0] == y.shape[0]
        y = ( to_categorical(y,num_classes) )
        dic[sentence_id] = (x,y)


def toXY(dic):
    buf_x = []
    buf_y = []
    for sentence_id in dic.keys():
        x,y = dic[sentence_id]
        num,feature_dim = x.shape
        
        x = x.reshape(1,num,feature_dim)
        y = y.reshape(1,num,y.shape[1])
    
        buf_x.append(x)
        buf_y.append(y)

    X = np.vstack(buf_x)
    Y = np.vstack(buf_y)
    return X,Y

def split_dic_validation(dic,rate):
    sample_count = len(dic.keys())
    vali_count = int(np.ceil(sample_count*rate))
    vali_dic = {}
    train_dic = {}
    
    
    i = 0
    shffle_keys = dic.keys()
    random.shuffle(shffle_keys)
    for key in shffle_keys:
        if i < vali_count:
            vali_dic[key] = dic[key]
        else:
            train_dic[key] = dic[key]
        i += 1
    print 'split to train_dic:%d    vali_dic:%d' % (sample_count - vali_count,vali_count)
    return train_dic,vali_dic

def dic2generator(dic):
    while True:
        shffle_keys = dic.keys()
        random.shuffle(shffle_keys)
        for key in shffle_keys:
            yield dic[key]

