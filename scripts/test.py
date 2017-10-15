import myinput
from keras.models import *
from keras.layers import *
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random


features_count = 5    #108
num_classes = 48
validation_rate = 0.05
max_len = 777

v = np.asarray([[1,2,3,4,5]])

def f(x):
    m = x*v
    num = x.shape[0]
    buf = []
    
    for i in range(num):
        s = sum(m[i,:])
        if s == 0:
            y = 0
        else:
            y = 1 if sum(m[i,:]) < 30 else 2
        buf.append(y)
    
    Y = np.asarray(buf)
    Y = Y.reshape(num,1)

    return Y
def padding(x,len):
    num_x = x.shape[0]
    
    return np.pad(x,((0,len-num_x),(0,0)),'constant', constant_values=0)

num = 10
max_len = 7
def rand_sentence(num,dim,a=1,b=10):
    buf = []
    for i in range(num):
        x = []
        for j in range(dim):
            x.append(random.randint(a,b))
        buf.append(x)
    X = np.asarray(buf)
    return X

x1 = padding(rand_sentence(3,5),max_len).reshape(1,7,5)
x2 = padding(rand_sentence(4,5),max_len).reshape(1,7,5)
x3 = padding(rand_sentence(5,5),max_len).reshape(1,7,5)

y1 = f(x1[0,:,:]).reshape(1,7,1)
y2 = f(x2[0,:,:]).reshape(1,7,1)
y3 = f(x3[0,:,:]).reshape(1,7,1)


x = np.vstack([x1,x2,x3])
y = np.vstack([y1,y2,y3])

#x = x.reshape(21,5)
#y = y.reshape(21,1)
y = to_categorical(y,3)

print 'X,Y shape:' ,x.shape,y.shape
print 'X : \n',x
print 'Y : \n',y




print x.shape,y.shape



print 'test'
model = Sequential()
#model.add(Embedding(input_dim = 10,output_dim=5,mask_zero = True,input_length = 5))
model.add(Masking(mask_value=0., input_shape=(7, 5)))

#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(x,y,epochs = 10)

z = model.predict(x)
print 'Z shape',z.shape
print 'Z',z

#model.add(Dense(num_classes+1, activation="softmax"))

#model.compile(loss='mse', optimizer='rmsprop')

#training loop

#model.fit(x,y,batch_size=150,epochs=epochs,validation_split=0.05)

