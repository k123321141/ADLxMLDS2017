# -*- coding: utf-8 -*-
from keras import layers
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
from keras.utils import plot_model

np.random.seed(0413)

color_classes = 12
img_dim = 64
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
    h = layers.multiply([latent, cls1, cls2])
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


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper


    image = Input(shape=(img_dim, img_dim, 3))

    cnn = image
    cnn = Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(img_dim, img_dim, 3))(cnn)
    cnn = LeakyReLU(0.2)(cnn)
    cnn = Dropout(0.3)(cnn)

    cnn = Conv2D(64, 3, padding='same', strides=1)(cnn)
    cnn = LeakyReLU(0.2)(cnn)
    cnn = Dropout(0.3)(cnn)

    cnn = Conv2D(128, 3, padding='same', strides=2)(cnn)
    cnn = LeakyReLU(0.2)(cnn)
    cnn = Dropout(0.3)(cnn)

    cnn = Conv2D(256, 3, padding='same', strides=1)(cnn)
    cnn = LeakyReLU(0.2)(cnn)
    cnn = Dropout(0.3)(cnn)

    cnn = Flatten()(cnn)
    features = cnn
    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux1 = Dense(color_classes, activation='softmax', name='aux_eyes')(features)
    aux2 = Dense(color_classes, activation='softmax', name='aux_hair')(features)

    return Model(image, [fake, aux1, aux2])

def main():    
    discriminator = build_discriminator()
    discriminator.summary()
    plot_model(discriminator, to_file='./d_model.png',show_shapes = True)

    # build the generator
    generator = build_generator(100)
    generator.summary()
    plot_model(generator, to_file='./g_model.png',show_shapes = True)

if __name__ == '__main__':
    main()
