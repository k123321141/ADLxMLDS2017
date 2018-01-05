
# coding: utf-8

import numpy as np
import argparse
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
    print img.shape,text.shape
    x = img[18,:,:,:]
    x = x.reshape([-1])
    y = text[0,:]
    print x[100:200]
    print y
    import sys
    #sys.exit()
    sample_num = 64
    #wrong_text = npz['wrong_y']
    

    img_dim = 64
    c_dim = 22
    n_dim = 100

    generator, discriminator, model = gan_model(img_dim, c_dim, n_dim)
    generator.summary()
    discriminator.summary()
    model.summary()

    opt = Adam(0.0002, 0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)

    num = img.shape[0]
    sample_range = range(len(text))
    for e in range(1,nb_epochs,1):
        
        idxs = random.sample(sample_range, sample_num)
        idxs2 = random.sample(sample_range, sample_num)
        
        #train discriminator
        #prepare discriminator data
        x_buf = []
        c_buf = []
        #real img, right text
        x_buf.append(img[idxs, :, :, :])
        c_buf.append(text[idxs, :])
        #fake img, right text 
        #print 'predict fake'
        n = noise_sample(n_dim, sample_num)
        fake_img = generator.predict([n, text])
        x_buf.append(fake_img) 
        c_buf.append(text[idxs, :])

        #real img, wrong text
        #print 'prepare wrong text data'
        x_buf.append(img[idxs2, :, :, :])
        c_buf.append(text[idxs, :])
        #c_buf.append(wrong_text(text[idxs, :]) )
        #
        x = np.vstack(x_buf)
        c = np.vstack(c_buf)
        #prepare label
        y = np.zeros([x.shape[0], 1])
        for i in range(sample_num):
            y[i,0] = 1
        
        #train discriminator
        print 'Start training on iters : %5d' % e
        #print x.shape,c.shape,n.shape,y.shape, fake_img.shape
        discriminator.trainable = True
        discriminator.train_on_batch(x=[x,c], y=y)
        discriminator.trainable = False
        model.train_on_batch(x=[n, c[:sample_num, :] ],y=y[:sample_num])
        if e % 100 == 0:
            model.save_weights('./models/model_%d.h5' % e)
            #generate img
            img_num = 12
            test_n = noise_sample(n_dim, img_num*img_num)
            test_c = condition_seq()
            merge_img = np.zeros([img_dim * img_num, img_dim * img_num, 3])
            gen_img = generator.predict([test_n,test_c])
            for i in range(img_num):
                for j in range(img_num):
                    merge_img[i*img_dim: (i+1)*img_dim, j*img_dim: (j+1)*img_dim, :] = gen_img[i + img_num*j, :, :, :]

            merge_img = merge_img * 255
            imsave('./gen_img/epoch-%d.png' % e, merge_img)

        #50973
def condition_seq():
    c = np.zeros([12,12, 22]) 
    #hair
    for i in range(11):
        c[i+1,:,i] = 1
    #eyes
    for i in range(11):
        c[:,i+1,i + 11] = 1
    return c.reshape([12*12,22])
def wrong_text(y):
    
    wrong_y = np.zeros(y.shape)
    for i in range(y.shape[0]):
        arr = []
        for j in range(y.shape[1]):
            if y[i,j] == 0:
                arr.append(j)
        wrong_y[i,random.sample(arr, 4)] = 1
    return wrong_y
def noise_sample(n_dim, num):
    return np.random.normal(size=(num, n_dim))
def generator_model(noise_dim, c_dim):
    #c_dim = code dimension

    n_input = Input(shape=(noise_dim,))
    c_input = Input(shape=(c_dim,))
    
    gen_input = Concatenate(axis = -1) ([n_input, c_input]) 

    #input_shape = (122,)
    nch = 256
    dep = 3
    #reg = lambda: L1L2(l1=1e-7, l2=1e-7)

    gen_model = Sequential(name='generator')
    gen_model.add(Dense(4 * 4 * nch, input_dim=noise_dim+c_dim))
    gen_model.add(BatchNormalization(axis=-1))
    gen_model.add(Activation('relu'))
    gen_model.add(Reshape([4, 4, nch]))
    for i in range(dep):
        f = nch / (2** (i+1) )
        #k = max(min(16, 2**i)-1 , 1)
        k = 5
        gen_model.add(Conv2D(f, kernel_size=(k, k), padding='same', data_format='channels_last'))
        gen_model.add(UpSampling2D(size=(2, 2), data_format='channels_last'))
        gen_model.add(BatchNormalization( axis=-1))
        gen_model.add(Activation('relu'))

    gen_model.add(Conv2D(3, kernel_size=(5, 5), padding='same', data_format='channels_last'))
    gen_model.add(UpSampling2D(size=(2, 2), data_format='channels_last'))
    gen_model.add(Activation('sigmoid'))
    
    gen_out = gen_model(gen_input)
    model = Model(inputs=(n_input, c_input), outputs=gen_out, name='generator')
    
    
    
    return model
def discriminator_model(img_dim, c_dim):
    #c_dim = code dimension
    #input_shape = (122,)
    nch = 256
    #reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    dep = 4
   
    #
    
    img_model = Sequential()
    for i in range(dep):
        """
                                                #dim    64  32  16  8   4   2   1
        f = min(16 * (2 ** i), 512)             #       16  32  64  128 256 512 512
        k = max(3 - (2*i), 3)                   #       13  11  9   7   5   3   3 
        """
        f = 2 ** (i+5)
        k = 5

        if i == 0:
            img_model.add(Conv2D(f, kernel_size=(k, k), strides=(2,2), padding='same',
                                            input_shape=(img_dim, img_dim, 3), data_format='channels_last'))
        else:
            img_model.add(Conv2D(f, kernel_size=(k, k), strides=(2,2), padding='same', data_format='channels_last'))
        img_model.add(BatchNormalization( axis=-1))
        img_model.add(LeakyReLU())
        
        #img_model.add(MaxPooling2D((2, 2), data_format='channels_last'))
    

    img_model.add(Flatten())
    
    #consider condition
    img_input = Input(shape=(img_dim, img_dim, 3))
    c_input = Input(shape=(c_dim,))
    
    img_out = img_model(img_input)

    x = Concatenate(axis = -1) ([img_out, c_input]) 

    x = Dense(512, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=(img_input, c_input), outputs=y, name='discriminator')

    return model
    
def gan_model(img_dim, c_dim, noise_dim):
    
    n_input = Input(shape=(noise_dim,)) 
    c_input = Input(shape=(c_dim,))
    generator = generator_model(noise_dim, c_dim)
    discriminator = discriminator_model(img_dim, c_dim)

    x = Concatenate(axis=-1)([n_input, c_input])
    x = generator([n_input, c_input])
    y = discriminator([x, c_input])
    
    model = Model(inputs=[n_input, c_input], outputs=y, name='gan_end2end')

    return generator, discriminator, model



if __name__ == '__main__':
    main()
