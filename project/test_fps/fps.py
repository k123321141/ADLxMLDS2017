
# coding: utf-8

import numpy as np
import argparse
np.random.seed(1337)  # for reproducibility
from scipy.misc import imresize
from scipy.ndimage import imread
from keras.datasets import mnist
from keras.models import * 
from keras.layers import *
from keras.utils import np_utils
from preprocess import normalize_img
import keras.backend as K
import time
import os
from os.path import join
from spatial_transformer import SpatialTransformer
def parse():
    parser = argparse.ArgumentParser(description="for test fps")
    parser.add_argument('-m','--model', default='./model.h5', help='model path')
    parser.add_argument('-t','--type', default='stn', help='model type, for testing.')
    parser.add_argument('-p','--path', default='./test_imgs/', help='testing images file path')
    parser.add_argument('-b','--batch', default=256, type=int, help='testing prdict batch size')
    args = parser.parse_args()
    return args

def main(args):
    #
    batch_size = args.batch 

    DIM = 60
    dep = 3
    nb_classes = 13


# In[2]:



    if args.type == 'stn':
        model = stn_model(DIM, dep, nb_classes)
    elif args.type == 'mlp':
        model = mlp_model(DIM, dep, nb_classes)
    else:
        import sys
        print('error with wrong type name %s ' % args.type)
        sys.exit()
    
    #loading files & model
    model.load_weights(args.model)
    imgs = read_dir(args.path)

    #start preprocessing testing images
    start_t = time.time()
    X = prepro(imgs)
    model.predict(X, batch_size=batch_size)
    end_t = time.time()
    
    t = end_t - start_t
    print('cost time %f on %d testing sample, average time %f.'  % (t, len(X), t/len(X)) )

    return t
    

def read_dir(dir_path):
    file_list = [join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.png')] 
    print('read directory %s with %d files.' % (dir_path, len(file_list)) )
    #data
    data_buf = []
    imgs = []
    for f in file_list:
        img = imread(f)
        assert len(img.shape) == 3
        imgs.append(img)

    return imgs
def prepro(imgs):
    buf = []
    for img in imgs:
        img = normalize_img(img)
        h,w,dep = img.shape
        img = img.reshape([1,h,w,dep])
        buf.append(img)
    
    X = np.vstack(buf).astype(np.uint8)
    X = X/255
    return X
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

if __name__ == '__main__':
    args = parse()
    #for stn network
    args.model = './model_stn.h5'
    args.type = 'stn'
    cost_t = main(args)
    #for mlp network
    args.model = './model_mlp.h5'
    args.type = 'mlp'
    cost_t = main(args)
