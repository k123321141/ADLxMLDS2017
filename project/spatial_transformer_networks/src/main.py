# coding: utf-8


# In[1]:

#get_ipython().magic(u'matplotlib inline')

import numpy as np
np.random.seed(1337)  # for reproducibility

from scipy.misc import imresize
from keras.models import * 
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD

import keras.backend as K
from spatial_transformer import SpatialTransformer

batch_size = 128
nb_classes = 10
nb_epoch = 12

DIM = 60
npz = np.load('./train.npz')


# In[2]:
x_train = npz['x_train']
y_train = npz['y_train']
x_valid = npz['x_valid']
y_valid = npz['y_valid']
x_test = npz['x_test']
y_test = npz['y_test']


print("Train samples: {}".format(x_train.shape))
print("Validation samples: {}".format(x_valid.shape))
print("Test samples: {}".format(x_test.shape))


input_shape = (50,50,3)
print("Input shape:",input_shape)




# In[4]:

# initial weights
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]


# In[5]:

locnet = Sequential()
locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
locnet.add(Convolution2D(20, (5, 5)))
locnet.add(MaxPooling2D(pool_size=(2,2)))
locnet.add(Convolution2D(40, (5, 5)))

locnet.add(Flatten())
locnet.add(Dense(50))
locnet.add(Activation('relu'))
locnet.add(Dense(6, weights=weights))
#locnet.add(Activation('sigmoid'))


# In[6]:

model = Sequential()

model.add(SpatialTransformer(localization_net=locnet,
                             output_size=(30,30), input_shape=input_shape))

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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[7]:

XX = model.input
YY = model.layers[0].output
F = K.function([XX], [YY])


# In[8]:

print(x_train.shape[0]/batch_size)


# In[ ]:

nb_epochs = 100 # you probably want to go longer than this
batch_size = 256
try:
    model.fit(x_train, y_train, batch_size = 50, epochs = 2000, callbacks=[], validation_data=(x_valid, y_valid))
    model.fit()
    '''
    for e in range(nb_epochs):
        print('-'*40)
        #progbar = generic_utils.Progbar(x_train.shape[0])
        for b in range(150):
            #print(b)
            f = b * batch_size
            l = (b+1) * batch_size
            X_batch = x_train[f:l].astype('float32')
            y_batch = y_train[f:l].astype('float32')
            loss = model.train_on_batch(X_batch, y_batch)
            #print(loss)
            #progbar.add(X_batch.shape[0], values=[("train loss", loss)])
        scorev = model.evaluate(x_valid, y_valid, verbose=0)
        scoret = model.evaluate(x_test, y_test, verbose=0)
        print('Epoch: {0} | Valid: {1} | Test: {2}'.format(e, scorev, scoret))
        
    '''
except KeyboardInterrupt:
    pass







