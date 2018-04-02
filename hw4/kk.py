# -*- coding: utf-8 -*-

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import numpy as np

np.random.seed(0413)

npz = np.load('./train.npz')
colors = ['<unk>','green', 'white', 'blue', 'aqua', 'gray', 'purple', 'red', 'pink', 'yellow', 'brown', 'black']

color_classes = len(colors)#12
img_dim = 64

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    # this will be our label
    skip_thoughts_vector = Input(shape=(2400,), name='skip_thoughts_vector_input')
    skv = Dense(100, activation='linear')(skip_thoughts_vector)

    # hadamard product between z-space and a class conditional embedding
    latent = Input(shape=(latent_size, ),name='normal_nosie')
    condiction = Concatenate(axis=-1)([skv, latent])

    cnn = Dense(3 * 3 * 512, activation='relu')(condiction)
    cnn = Reshape([3, 3, 512])(cnn)

    # upsample to (7, 7, ...)
    cnn = Conv2DTranspose(256, 5, strides=1,padding='valid',
                            kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation('relu')(cnn)

    # upsample to (14, 14, ...)
    cnn = Conv2DTranspose(128, 5, strides=2, padding='same',
                            kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation(LeakyReLU(0.2))(cnn)

    # upsample to (16, 16, ...)
    cnn = Conv2DTranspose(128, 3, strides=1, padding='valid',
                            kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation('relu')(cnn)

    # upsample to (32, 32, ...)
    cnn = Conv2DTranspose(64, 5, strides=2, padding='same',
                            kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation(LeakyReLU(0.2))(cnn)
    
    # upsample to (64, 64, ...)
    cnn = Conv2DTranspose(32, 5, strides=2, padding='same',
                            kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = Activation(LeakyReLU(0.2))(cnn)
    
    cnn = Conv2DTranspose(3, 3, strides=1, padding='same',
                            kernel_initializer='glorot_normal')(cnn)
    cnn = Activation('tanh')(cnn)

    # this is the z space commonly referred to in GAN papers


    fake_image = cnn

    return Model([skip_thoughts_vector, latent], fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper


    image = Input(shape=(img_dim, img_dim, 3))
    skip_thoughts_vector = Input(shape=(2400,), name='skip_thoughts_vector_input')
    skv = Dense(100, activation='linear')(skip_thoughts_vector)


    cnn = image
    cnn = Conv2D(32, 3, padding='same', strides=2)(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = LeakyReLU(0.2)(cnn)

    cnn = Conv2D(64, 3, padding='same', strides=2)(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = LeakyReLU(0.2)(cnn)


    cnn = Conv2D(128, 3, padding='same', strides=2)(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = LeakyReLU(0.2)(cnn)
    
    cnn = Conv2D(64, 3, padding='same', strides=2)(cnn)
    cnn = BatchNormalization(axis=-1)(cnn)
    cnn = LeakyReLU(0.2)(cnn)

    skv = RepeatVector(16)(skv)
    skv = Reshape([4,4, -1])(skv)
    cnn = Concatenate(axis=-1)([cnn, skv])
    

    cnn = Conv2D(100, 3, padding='same', strides=1)(cnn)
    cnn = LeakyReLU(0.2)(cnn)

    cnn = Flatten()(cnn)
    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(cnn)
    return Model(inputs=[image, skip_thoughts_vector], outputs=fake)

def main():    
    # batch and latent size taken from the paper
    batch_size = 64
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy']
    )
    discriminator.summary()

    # build the generator
    generator = build_generator(latent_size)

    latent = Input(shape=(latent_size, ))
    skip_thoughts_vector = Input(shape=(2400,))

    # get a fake image
    fake = generator([skip_thoughts_vector, latent])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake = discriminator([fake, skip_thoughts_vector])
    combined = Model([skip_thoughts_vector, latent], [fake])

    print('Combined model:')
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy']
    )
    combined.summary()

    # get our mnist data, and force it to be of shape (..., 28, 28, 1) with
    # range [-1, 1]
    npz = np.load('./train.npz')
    
    x_train = npz['x']
    text = npz['text']
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5


    num_train = x_train.shape[0]

    train_history = defaultdict(list)

    #skip-thoughts enncoder
    encoder = load_skip_thoughts()

    for iters in range(10000000000):
        print('Iters {}'.format(iters))

        num_batches = 100 
        progress_bar = Progbar(target=num_batches)

        disc_sample_weight = [np.concatenate((np.ones(batch_size) * 2,
                                              np.ones(batch_size*2))),]

        iters_gen_loss = []
        iters_disc_loss = []

        for index in range(num_batches):
            # generate a new batch of noise
            noise = np.random.normal(0, 1, (batch_size, latent_size))
            idxs = np.random.choice(x_train.shape[0], batch_size)
            # get a batch of real images
            image_batch = x_train[idxs]
            text_batch = encode_text(encoder, text[idxs])

            # sample some wrong text from different index
            fake_idxs = np.random.choice(x_train.shape[0], batch_size)
            fake_text_batch = encode_text(encoder, text[idxs])

            # generate a batch of fake images, with the right text.
            generated_images = generator.predict(
                [text_batch, noise])

            x = np.concatenate([image_batch, generated_images, image_batch])

            # use soft real/fake labels and flip noise
            # true : normal distribution from 0.9 with interval 0.2, and clip by 0.7 1.2
            # fake : normal distribution from 0.1 with interval 0.1, and clip by 0.1 0.3
            soft_true = np.random.normal(0.9, 0.2,[batch_size, 1])
            soft_true = np.clip(soft_true, 0.7, 1.2)
            soft_fake = np.random.normal(0.1, 0.1,[batch_size*2, 1])
            soft_fake = np.clip(soft_fake, 0, 0.3)
            # true image, right text
            # fake image, right text
            # true image, wrong text
            y = np.concatenate([soft_true, soft_fake], axis=0)
            text_x = np.concatenate([text_batch, text_batch, fake_text_batch], axis=0)

            # see if the discriminator can figure itself out...
            iters_disc_loss.append(discriminator.train_on_batch(
                [x, text_x], y, sample_weight=disc_sample_weight))
            
            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.normal(0, 1, (4 * batch_size, latent_size))
            fake_idxs = np.random.choice(x_train.shape[0], batch_size*4)
            fake_text_batch = encode_text(encoder, text[fake_idxs])


            # we want to train the generator to trick the discriminator.
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.random.normal(0.9, 0.2,[batch_size*4, 1])
            trick = np.clip(trick, 0.7, 1.0)
            iters_gen_loss.append(combined.train_on_batch(
                [fake_text_batch, noise],

                [trick]))

            progress_bar.update(index + 1)


        # see if the discriminator can figure itself out...
        discriminator_train_loss = np.mean(np.array(iters_disc_loss), axis=0)
        # make new noise
        generator_train_loss = np.mean(np.array(iters_gen_loss), axis=0)

        # generate an iters report on performance
        '''
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        print('{0:<22s} | {1:7s} '.format(
            'component', *discriminator.metrics_names))
        print('-' * 100)

        ROW_FMT = '{0:<22s} | {1:<7.2f} '
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                            * train_history['discriminator'][-1]))
        '''
        # save weights every iters
        if iters % 10 == 0:
            generator.save_weights(
                './models/params_generator_iters_{0:03d}.hdf5'.format(iters), True)
            discriminator.save_weights(
                './models/params_discriminator_iters_{0:03d}.hdf5'.format(iters), True)

        # generate some digits to display
        num_rows = color_classes
        noise = np.tile(np.random.normal(0, 1, (num_rows, latent_size)),
                        (color_classes, 1))
        #generate img
        img_num = color_classes
        sample_text = np.zeros([color_classes**2, 1], dtype=object)
        for i in range(img_num):
            for j in range(img_num):
                s = '%s eyes , %s hair .' % (colors[i], colors[j])
                sample_text[i*color_classes +j, 0] = s

        sample_skip_thoughts = encode_text(encoder, sample_text)
        merge_img = np.zeros([img_dim * img_num, img_dim * img_num, 3])
        #print(noise.shape,sampled_eyes.shape,sampled_hair.shape)
        gen_img = generator.predict(
                [sample_skip_thoughts, noise], verbose=0)
        for i in range(img_num):
            for j in range(img_num):
                merge_img[i*img_dim: (i+1)*img_dim, j*img_dim: (j+1)*img_dim, :] = gen_img[i + img_num*j, :, :, :]
        #print(merge_img[:8,:8,:])

        merge_img = (merge_img+1.) * 127.5
        merge_img = merge_img.astype(np.uint8)
        Image.fromarray(merge_img).save(
            './gen_img/plot_iters_{0:03d}_generated.png'.format(iters))

def encode_text(encoder, text_batch):
    data = text_batch.flatten().tolist()
    encodings = encoder.encode(data)
    return encodings
    

def load_skip_thoughts():
    VOCAB_FILE = "./pretrained/skip_thoughts_uni_2017_02_02/vocab.txt"
    EMBEDDING_MATRIX_FILE = "./pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy"
    CHECKPOINT_PATH = "./pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424"
    # The following directory should contain files rt-polarity.neg and
    # rt-polarity.pos.
    MR_DATA_DIR = "/dir/containing/mr/data"

    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(),
                           vocabulary_file=VOCAB_FILE,
                              embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                             checkpoint_path=CHECKPOINT_PATH)
    return encoder


def get_nn(ind, encodingsm, num=2):
    encoding = encodings[ind]
    scores = sd.cdist([encoding], encodings, "cosine")[0]
    sorted_ids = np.argsort(scores)
    print("Sentence:")
    print("", data[ind])
    print("\nNearest neighbors:")
    for i in range(1, num + 1):
        print(" %d. %s (%.3f)" % (i, data[sorted_ids[i]], scores[sorted_ids[i]]))

if __name__ == '__main__':
    main()

