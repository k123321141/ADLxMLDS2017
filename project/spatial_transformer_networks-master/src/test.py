# coding: utf-8


import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD

import keras.backend as K




def build_model(DIM, DEP, nb_classes):


    # In[29]:


    # In[47]:
    input_shape = (DIM, DIM, DEP)
    


    # In[48]:


    model = Sequential()


    model.add(Convolution2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3)))
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
            model.fit(X_train, y_train, epochs=10000, batch_size=batch_size, validation_data=(X_test, y_test))

    except KeyboardInterrupt:
        #print('save weights to %s' % MODEL_PATH)
        #model.save_weights(MODEL_PATH)
        pass
if __name__ == '__main__':
    main()





