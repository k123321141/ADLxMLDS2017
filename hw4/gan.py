
# coding: utf-8

import numpy as np
import argparse
import sys
np.random.seed(0413)  # for reproducibility
from scipy.misc import imresize, imsave
from keras.models import *
from keras.callbacks import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.regularizers import L1L2
import random


def parse():
    parser = argparse.ArgumentParser(description="hw4")
    parser.add_argument('-s','--summary', default='./summary/default', help='summary path')
    parser.add_argument('-m','--model', default='./model.h5', help='model path')
    parser.add_argument('-b','--batch', default=128, type=int, help='batch size')
    args = parser.parse_args()
    return args
def main():
    args = parse()
    batch_size = args.batch
    nb_epochs = 10000000 # you probably want to go longer than this

    npz = np.load('./train.npz')
    
    img = npz['x']
    text = npz['y']
    sample_num = 64
    #print img[0,:8,:8,:]
    #wrong_text = npz['wrong_y']
    

    img_dim = 64
    c_dim = 22
    n_dim = 100

    generator, discriminator, model = gan_model(img_dim, c_dim, n_dim)
    print 'generator summary'
    generator.summary()
    print 'discriminator summary'
    discriminator.summary()
    print 'end2end model summary'
    model.summary()
    '''
    import os
    path = './models/keep.h5'
    if os.path.isfile(path):
        model.load_weights(path)
    '''
    num = img.shape[0]
    sample_range = len(text)
    for e in range(1,nb_epochs,1):
        print 'Start training on iters : %5d' % e
        half_bat = sample_num/2
        idxs = np.random.randint(0,sample_range, half_bat)
        #train discriminator
        #prepare discriminator data
        
        #real img, right text
        x = img[idxs,:,:,:]
        y = np.ones([half_bat, 1]
                )
        #xx = x[:2,:,:,:]
        #z1 = discriminator.predict(xx)
        discriminator.train_on_batch(x=x, y=y)
        #z2 = discriminator.predict(xx)
        #print (z1==z2).all()

        #fake img, right text 
        n = noise_sample(n_dim, half_bat)
        fake_img = generator.predict(n)
        x = fake_img
        y = np.zeros([half_bat, 1])
        discriminator.train_on_batch(x, y)

        
        #train generator
        y = np.ones([sample_num,1])
        n = noise_sample(n_dim, sample_num)
        
        #z1 = discriminator.predict(xx)
        model.train_on_batch(n, y)
        #z2 = discriminator.predict(xx)
        #print (z1==z2).all()
        
        if e % 100 == 0:
            if e % 10000 == 0:
                model.save_weights('./models/model_%d.h5' % e)
            #generate img
            img_num = 12
            test_n = noise_sample(n_dim, img_num*img_num)
            merge_img = np.zeros([img_dim * img_num, img_dim * img_num, 3])
            gen_img = generator.predict(test_n)
            for i in range(img_num):
                for j in range(img_num):
                    merge_img[i*img_dim: (i+1)*img_dim, j*img_dim: (j+1)*img_dim, :] = gen_img[i + img_num*j, :, :, :]
            print merge_img[:4,:4,:]
            print merge_img[:4,:4,:]

            merge_img = (merge_img+1.) * 127.5
            merge_img = merge_img.astype(np.uint8)
            print merge_img[:4,:4,:]
            print merge_img[img_dim:img_dim+4, img_dim:img_dim+4,:]
            imsave('./gen_img/epoch-%d.png' % e, merge_img)
            model.save_weights('./models/keep.h5')

        #50973
def noise_sample(n_dim, num):
    return np.random.normal(0.,1.,size=(num, n_dim))
def generator_model(noise_dim):
    #c_dim = code dimension

    n_input = Input(shape=(noise_dim,))
    

    #input_shape = (122,)
    nch = 256
    dep = 3

    gen_model = Sequential(name='generator')
    gen_model.add(Dense(4 * 4 * nch, input_dim=noise_dim))
    gen_model.add(BatchNormalization( axis=-1))
    gen_model.add(Activation('relu'))
    
    gen_model.add(Reshape([4, 4, nch]))
    for i in range(dep):
        f = nch / (2** (i+1) )
        #k = max(min(16, 2**i)-1 , 1)
        k = 5
        gen_model.add(Conv2DTranspose(f, kernel_size=(k, k), strides=(2,2), padding='same', data_format='channels_last'))
        #gen_model.add(BatchNormalization( axis=-1))
        gen_model.add(Activation('relu'))
    gen_model.add(Conv2DTranspose(3, kernel_size=(5, 5), strides=(2,2), padding='same', data_format='channels_last'))
    gen_model.add(Activation('tanh'))
    
    gen_out = gen_model(n_input)
    model = Model(inputs=n_input, outputs=gen_out, name='generator')
    
    
    
    return model
def discriminator_model(img_dim):
    #c_dim = code dimension
    #input_shape = (122,)
    nch = 256
    #reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    dep = 3
   
    #
    img_input = Input(shape=(img_dim, img_dim, 3))
    x = img_input
    for i in range(dep):
        """
                                                #dim    64  32  16  8   4   2   1
        f = min(16 * (2 ** i), 512)             #       16  32  64  128 256 512 512
        k = max(3 - (2*i), 3)                   #       13  11  9   7   5   3   3 
        """
        f = 2 ** (i+5)
        k = 5
        s = 1
        x = Conv2D(f, kernel_size=(k, k), strides=(s,s), padding='same', data_format='channels_last') (x)
        x = MaxPooling2D((2,2))(x)
        #x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)
    '''
    x = Conv2D(128, kernel_size=(1, 1), strides=(1,1), padding='same', data_format='channels_last')(x)
    #x = BatchNormalization(mode=1,axis=-1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(1, kernel_size=(8, 8), strides=(1,1), padding='valid', data_format='channels_last')(x)
    x = Reshape([1,])(x)
    '''
    x = Flatten()(x)
    x = Dense(512, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)

    
    #consider condition
    
    model = Model(inputs=img_input, outputs=y, name='discriminator')
    return model
    
def gan_model(img_dim, c_dim, noise_dim):
    
    n_input = Input(shape=(noise_dim,)) 
    generator = generator_model(noise_dim)
    discriminator = discriminator_model(img_dim)
    
    opt = Adam(0.0002, 0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)

    x = generator(n_input)
    discriminator.trainable = False
    y = discriminator(x)
    
    model = Model(inputs=n_input, outputs=y, name='gan_end2end')
    
    model.compile(loss='binary_crossentropy', optimizer=opt)
    

    return generator, discriminator, model



if __name__ == '__main__':
    main()
