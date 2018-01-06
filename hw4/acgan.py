# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.

You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating,
as the compilation time can be a blocker using Theano.

Timings:

Hardware           | Backend | Time / Epoch
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min

Consult https://github.com/lukedeo/keras-acgan for more information and
example output
"""
from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

np.random.seed(0413)

npz = np.load('./train.npz')

color_classes = 12
img_dim = 64
def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    cnn = Sequential()

    cnn.add(Dense(3 * 3 * 512, input_dim=latent_size, activation='relu'))
    cnn.add(Reshape((3, 3, 512)))

    # upsample to (7, 7, ...)
    cnn.add(Conv2DTranspose(256, 5, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal'))

    # upsample to (14, 14, ...)
    cnn.add(Conv2DTranspose(128, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    # upsample to (16, 16, ...)
    cnn.add(Conv2DTranspose(128, 3, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal'))

    # upsample to (32, 32, ...)
    cnn.add(Conv2DTranspose(64, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    # upsample to (64, 64, ...)
    cnn.add(Conv2DTranspose(3, 5, strides=2, padding='same',
                            activation='tanh',
                            kernel_initializer='glorot_normal'))

    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    eyes_class = Input(shape=(1,), dtype='int32')
    hair_class = Input(shape=(1,), dtype='int32')

    cls1 = Flatten()(Embedding(12, latent_size,
                              embeddings_initializer='glorot_normal')(eyes_class))
    cls2 = Flatten()(Embedding(12, latent_size,
                              embeddings_initializer='glorot_normal')(hair_class))

    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls1, cls2])

    fake_image = cnn(h)

    return Model([latent, eyes_class, hair_class], fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(img_dim, img_dim, 3)))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(img_dim, img_dim, 3))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux1 = Dense(color_classes, activation='softmax', name='aux_eyes')(features)
    aux2 = Dense(color_classes, activation='softmax', name='aux_hair')(features)

    return Model(image, [fake, aux1, aux2])

if __name__ == '__main__':

    # batch and latent size taken from the paper
    epochs = 50
    batch_size = 100
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']
    )
    discriminator.summary()

    # build the generator
    generator = build_generator(latent_size)

    latent = Input(shape=(latent_size, ))
    eyes_class = Input(shape=(1,), dtype='int32')
    hair_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, eyes_class, hair_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux1, aux2 = discriminator(fake)
    combined = Model([latent, eyes_class, hair_class], [fake, aux1, aux2])

    print('Combined model:')
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']
    )
    combined.summary()

    # get our mnist data, and force it to be of shape (..., 28, 28, 1) with
    # range [-1, 1]
    npz = np.load('./train.npz')
    
    x_train = npz['x'][:16900]
    y1 = npz['eyes'][:16900]
    y2 = npz['hair'][:16900]
    y1 = y1.reshape([-1])
    y2 = y2.reshape([-1])
    print(x_train.shape,y1.shape)
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5


    num_train = x_train.shape[0]

    train_history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(x_train.shape[0] / batch_size)
        progress_bar = Progbar(target=num_batches)

        # we don't want the discriminator to also maximize the classification
        # accuracy of the auxiliary classifier on generated images, so we
        # don't train discriminator to produce class labels for generated
        # images (see https://openreview.net/forum?id=rJXTf9Bxg).
        # To preserve sum of sample weights for the auxiliary classifier,
        # we assign sample weight of 2 to the real images.
        disc_sample_weight = [np.ones(2 * batch_size),
                              np.concatenate((np.ones(batch_size) * 2,
                                              np.zeros(batch_size))),
                              np.concatenate((np.ones(batch_size) * 2,
                                              np.zeros(batch_size))),
                              ]

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            eyes_batch = y1[index * batch_size:(index + 1) * batch_size]
            hair_batch = y2[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_eyes = np.random.randint(0, color_classes, batch_size)
            sampled_hair = np.random.randint(0, color_classes, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_eyes.reshape((-1, 1)), sampled_hair.reshape((-1, 1))], verbose=0)

            x = np.concatenate((image_batch, generated_images))

            # use soft real/fake labels
            soft_zero, soft_one = 0.1, 0.9
            y = np.array([soft_one] * batch_size + [soft_zero] * batch_size)
            aux_y1 = np.concatenate((eyes_batch, sampled_eyes), axis=0)
            aux_y2 = np.concatenate((hair_batch, sampled_hair), axis=0)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(
                x, [y, aux_y1, aux_y2], sample_weight=disc_sample_weight))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_eyes = np.random.randint(0, color_classes, 2 * batch_size)
            sampled_hair = np.random.randint(0, color_classes, 2 * batch_size)

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size) * soft_one
            
            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_eyes.reshape((-1, 1)), sampled_hair.reshape((-1, 1))],

                [trick, sampled_eyes, sampled_hair]))

            progress_bar.update(index + 1)


        # see if the discriminator can figure itself out...
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            './models/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            './models/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        num_rows = color_classes
        noise = np.tile(np.random.uniform(-1, 1, (num_rows, latent_size)),
                        (color_classes, 1))
        #generate img
        img_num = color_classes
        noise = np.random.uniform(-1, 1, (color_classes**2, latent_size))
        sampled_eyes = np.zeros([color_classes**2, 1])
        for i in range(color_classes):
            sampled_eyes[i*color_classes : (i+1)*color_classes, 0] = i 
        sampled_hair = np.zeros([color_classes**2, 1])
        for i in range(color_classes):
            sampled_hair[i*color_classes : (i+1)*color_classes, 0] = i 

        merge_img = np.zeros([img_dim * img_num, img_dim * img_num, 3])
        #print(noise.shape,sampled_eyes.shape,sampled_hair.shape)
        gen_img = generator.predict(
                [noise, sampled_eyes, sampled_hair], verbose=0)
        for i in range(img_num):
            for j in range(img_num):
                merge_img[i*img_dim: (i+1)*img_dim, j*img_dim: (j+1)*img_dim, :] = gen_img[i + img_num*j, :, :, :]
        #print(merge_img[:8,:8,:])

        merge_img = (merge_img+1.) * 127.5
        merge_img = merge_img.astype(np.uint8)
        Image.fromarray(merge_img).save(
            './gen_img/plot_epoch_{0:03d}_generated.png'.format(epoch))
        """
        num_rows = color_classes
        noise = np.tile(np.random.uniform(-1, 1, (num_rows, latent_size)),
                        (color_classes, 1))

        sampled_eyes = np.array([
            [i] * num_rows for i in range(color_classes)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)

        # prepare real images sorted by class label
        real_labels = y_train[(epoch - 1) * num_rows * num_classes:
                              epoch * num_rows * num_classes]
        indices = np.argsort(real_labels, axis=0)
        real_images = x_train[(epoch - 1) * num_rows * num_classes:
                              epoch * num_rows * num_classes][indices]

        # display generated images, white separator, real images
        img = np.concatenate(
            (generated_images,
             np.repeat(np.ones_like(x_train[:1]), num_rows, axis=0),
             real_images))

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(img, 2 * num_classes + 1)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            'plot_epoch_{0:03d}_generated.png'.format(epoch))
        """
    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))
