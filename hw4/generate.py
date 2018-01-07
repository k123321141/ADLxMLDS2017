# -*- coding: utf-8 -*-
from keras.models import *
from keras.layers import *
import numpy as np
from scipy.misc import imsave
from os.path import join
import argparse, re, os
np.random.seed(0413)




def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    # this will be our label
    eyes_class = Input(shape=(1,), dtype='int32',name='eyes_tag')
    hair_class = Input(shape=(1,), dtype='int32',name='hair_tag')

    cls1 = Flatten()(Embedding(12, latent_size,
                              embeddings_initializer='glorot_normal')(eyes_class))
    cls2 = Flatten()(Embedding(12, latent_size,
                              embeddings_initializer='glorot_normal')(hair_class))

    # hadamard product between z-space and a class conditional embedding
    latent = Input(shape=(latent_size, ),name='normal_nosie')
    h = Multiply()([latent, cls1, cls2])
    cnn = h

    cnn = Dense(3 * 3 * 512, input_dim=latent_size, activation='relu')(cnn)
    cnn = Reshape([3, 3, 512])(cnn)

    # upsample to (7, 7, ...)
    cnn = Conv2DTranspose(256, 5, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal')(cnn)

    # upsample to (14, 14, ...)
    cnn = Conv2DTranspose(128, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal')(cnn)
    # upsample to (16, 16, ...)
    cnn = Conv2DTranspose(128, 3, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal')(cnn)

    # upsample to (32, 32, ...)
    cnn = Conv2DTranspose(64, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal')(cnn)
    # upsample to (64, 64, ...)
    cnn = Conv2DTranspose(3, 5, strides=2, padding='same',
                            activation='tanh',
                            kernel_initializer='glorot_normal')(cnn)

    # this is the z space commonly referred to in GAN papers


    fake_image = cnn

    return Model([latent, eyes_class, hair_class], fake_image)



def main(test_txt_path):    
    # batch and latent size taken from the paper
    latent_size = 100


    # build the generator
    generator = build_generator(latent_size)
    generator.load_weights('./trained_model')
    with open(test_txt_path,'r') as f:
        lines = f.readlines()


    # text processing
    colors = ['<unk>','green', 'white', 'blue', 'aqua', 'gray', 'purple', 'red', 'pink', 'yellow', 'brown', 'black']
    parts = ['eyes', 'hair']
    # configure eyes pattern
    patt = ''
    for c in colors:
        patt += c + ' eyes|'
    patt = patt[:-1]
    eyes_patten = re.compile(patt)

    # configure hair pattern
    patt = ''
    for c in colors:
        patt += c + ' hair|'
    patt = patt[:-1]
    hair_patten = re.compile(patt)
    # 
    for line in lines:
        idx, line = line.split(',')
        tags = line.strip().lower()
        tags = tags.replace('eye','eyes')

        #eyes
        e_m = eyes_patten.findall(tags)
        if len(e_m) == 0:
            y1 = 0
        else:
            y1 = colors.index( e_m[0].replace(' eyes',''))
            y1 = np.array([1,y1])
        #hair
        h_m = hair_patten.findall(tags)
        if len(h_m) == 0:
            y2 = 0
        else:
            y2 = colors.index( h_m[0].replace(' hair',''))
            y2 = np.array([1,y2])
        for i in range(5):
            noise = np.random.normal(0, 1, (1, latent_size))
            img = generator.predict([noise, y1, y2]).reshape([64,64,3])
            path = join('./', 'sample', 'sample_%s_%d.jpg' % (idx, i+1))
            imsave(path, img)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hw4 bash parser')
    parser.add_argument('test_txt_path',default=None, help='txt files')
    args = parser.parse_args()
    if not os.path.isdir(join('./','sample')):
        os.mkdir(join('./','sample'))
    
    main(args.test_txt_path)

