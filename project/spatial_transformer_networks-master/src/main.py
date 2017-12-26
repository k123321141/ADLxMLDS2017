# coding: utf-8


import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD

import keras.backend as K
from spatial_transformer import SpatialTransformer



MODEL_PATH = './model.h5'

def build_model(DIM, DEP, nb_classes):


    # In[29]:
    # initial weights
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]


    # In[47]:
    input_shape = (DIM, DIM, DEP)
    

    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
    locnet.add(Convolution2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Convolution2D(20, (5, 5)))

    locnet.add(Flatten())
    locnet.add(Dense(50))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights))
    #locnet.add(Activation('sigmoid'))


    # In[48]:


    model = Sequential()

    model.add(SpatialTransformer(localization_net=locnet,
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

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


    # In[49]:


    XX = model.input
    YY = model.layers[0].output
    F = K.function([XX], [YY])


    # In[50]:


    return model, F

# In[ ]:

def main():
    
    
    DIM = 60
    DEP = 3
    nb_classes = 13
    mnist_cluttered = '../datasets/train.npz'
    
    model, F = build_model(DIM, DEP, nb_classes)
    
    data = np.load(mnist_cluttered)
    X_train, y_train = data['x_train'], data['y_train']
    X_valid, y_valid = data['x_valid'], data['y_valid']
    X_test, y_test = data['x_test'], data['y_test']

    # reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], DIM, DIM, DEP))
    X_valid = X_valid.reshape((X_valid.shape[0], DIM, DIM, DEP))
    X_test = X_test.reshape((X_test.shape[0], DIM, DIM, DEP))
    print y_valid.shape
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_valid = np_utils.to_categorical(y_valid, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print y_valid.shape
    print("Train samples: {}".format(X_train.shape))
    print("Validation samples: {}".format(X_valid.shape))
    print("Test samples: {}".format(X_test.shape))


    input_shape =  np.squeeze(X_train.shape[1:])
    input_shape = (DIM, DIM, DEP)
    print("Input shape:",input_shape)


    #training
    
    nb_epochs = 1000 
    batch_size = 256
    try:
        for e in range(nb_epochs):
            print('-'*40)
            #progbar = generic_utils.Progbar(X_train.shape[0])
    #         for b in range(150):
    #             #print(b)
    #             f = b * batch_size
    #             l = (b+1) * batch_size
    #             X_batch = X_train[f:l].astype('float32')
    #             y_batch = y_train[f:l].astype('float32')
    #             loss = model.train_on_batch(X_batch, y_batch)
    #             #print(loss)
    #             #progbar.add(X_batch.shape[0], values=[("train loss", loss)])
    #         scorev = model.evaluate(X_valid, y_valid, verbose=1)
    #         scoret = model.evaluate(X_test, y_test, verbose=1)
            model.fit(X_train, y_train, epochs=10000, batch_size=batch_size, validation_data=(X_test, y_test))
    #         print('Epoch: {0} | Valid: {1} | Test: {2}'.format(e, scorev, scoret))

    except KeyboardInterrupt:
        print('save weights to %s' % MODEL_PATH)
        model.save_weights(MODEL_PATH)
        pass
if __name__ == '__main__':
    main()





