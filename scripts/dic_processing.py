from keras.utils import to_categorical
import numpy as np




def pad_dic(dic,max_len,padding_val):
    for sentence_id in dic.keys():
        #x is a ndarray in shape (sentence_len,feature number) ,or (sentence_len,1) for labels data
        x = dic[sentence_id]
        num_x = x.shape[0]
        
        x = np.pad(x,((0,max_len-num_x),(0,0)),'constant', constant_values=padding_val)
        
        dic[sentence_id] = x

def catogorate_dic(dic):
    for sentence_id in dic.keys():
        x = dic[sentence_id]
        x = ( to_categorical(x) )
        dic[sentence_id] = x

def vstack_dic(dic):
    buf = []
    for sentence_id in sorted(dic.keys()):
        #assert each x has same shape for (777,39) or (777,1)
        x = dic[sentence_id]
        length,feature_dim = x.shape 
        
        x = x.reshape(1,length,feature_dim)
        a,b,c = x.shape

        print (x.shape)
        assert a == 1 and b == 777  and (c == 49 or c == 39)

        buf.append(x)


    X = np.vstack(buf)

    return X

