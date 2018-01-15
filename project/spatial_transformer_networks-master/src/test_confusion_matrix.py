
# coding: utf-8

import numpy as np
import argparse
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import * 
from keras.layers import *
from keras.utils import np_utils

import keras.backend as K
from spatial_transformer import SpatialTransformer

labels = ['off', 'Red', 'Yellow', 'Green','GreenStraightRight', 'GreenStraightLeft', 'GreenStraight', 'RedStraightLeft', 'GreenRight', 'RedStraight', 'GreenLeft', 'RedRight', 'RedLeft']
def parse():
    parser = argparse.ArgumentParser(description="for test fps")
    parser.add_argument('-m','--model', default='./model.h5', help='model path')
    parser.add_argument('-t','--type', default='stn', help='model type, for testing.')
    parser.add_argument('-b','--batch', default=256, type=int, help='testing prdict batch size')
    args = parser.parse_args()
    return args
def main():
    args = parse()
    
    #
    batch_size = args.batch 

    DIM = 60
    dep = 3
    nb_classes = 13
    data = '../datasets/train.npz'


# In[2]:


    data = np.load(data)
    X_test, y_test = data['x_test'], data['y_test']
    # reshape for convolutions
    X_test = X_test.reshape((X_test.shape[0], DIM, DIM, dep))
    y_test = np_utils.to_categorical(y_test, nb_classes)


    if args.type == 'stn':
        model = stn_model(DIM, dep, nb_classes)
    elif args.type == 'mlp':
        model = mlp_model(DIM, dep, nb_classes)
    else:
        import sys
        print('error with wrong type name %s ' % args.type)
        sys.exit()
    
    model.load_weights(args.model)
    #
    confusion_matrix(model, X_test, y_test, labels, args)
    
def locnet(input_shape):
    # initial weights
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]
    
    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
    locnet.add(Convolution2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Convolution2D(20, (5, 5)))

    locnet.add(Flatten())
    locnet.add(Dense(50))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights ,name='trans_mat'))
    #locnet.add(Activation('sigmoid'))
    return locnet

def stn_model(DIM, dep, nb_classes):
# In[6]:

    model = Sequential()
    
    input_shape = (DIM, DIM, dep)

    model.add(SpatialTransformer(localization_net=locnet(input_shape),
                                 output_size=(DIM/2,DIM/2), input_shape=input_shape))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model
def mlp_model(DIM, dep, nb_classes):
    model = Sequential()
    model.add(Flatten(input_shape = (DIM, DIM, dep)))
    model.add(Dense(1024, activation='sigmoid')) 
    model.add(Dense(1024, activation='sigmoid')) 
    model.add(Dense(1024, activation='sigmoid')) 
    model.add(Dense(512, activation='sigmoid')) 
    model.add(Dense(256, activation='sigmoid')) 

    model.add(Dense(nb_classes, activation='softmax'))
    
    return model

def confusion_matrix(model,x, y_true, labels, args):
    y_pred = np.argmax(model.predict(x), axis=-1)
    #print 'y_ture shape', y_true.shape
    y_true = np.argmax(y_true, axis=-1)
    num = y_true.shape[0]
    #count labels
    count = [0] * len(labels)
    #True Positive
    tp = [0.] * len(labels)
    #True Negative
    tn = [0.] * len(labels)
    #False Positive
    fp = [0.] * len(labels)
    #False Negative
    fn = [0.] * len(labels)
    
    for i in range(num):
        y_p,y_t, = y_pred[i], y_true[i]
        count[y_t] += 1
        if y_p == y_t:
            tp[y_t] += 1
            for i in range(len(labels)):
                if i != y_t:
                    tn[i] += 1
        else:
            fn[y_t] += 1
            fp[y_p] += 1     
    print 'Model type : %s' % args.type
    #for i,label in enumerate(labels):
    for i,label in enumerate(labels[:4]):

        tpr = float(tp[i]) / (tp[i]+fn[i]) if tp[i]+fn[i] != 0  else 0
        ppv = float(tp[i]) / (tp[i]+fp[i]) if tp[i]+fp[i] != 0  else 0

        print 'label : %10s' % label

        print '-----%8s---%8s-------%3s--------%3s' % ('positive', 'negative', 'TPR', 'PPV')
        print 'true  : %5d  ,  %5d  |  %.5f  ,  %.5f' % (tp[i], tn[i], tpr, ppv)
        print 'false : %5d  ,  %5d  |' % (fp[i], fn[i])

        print '-'*27
        print ''


if __name__ == '__main__':
    main()
