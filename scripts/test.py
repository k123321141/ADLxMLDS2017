from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import plot_model



def sp(x):
    length = x.shape[0]


model = Sequential()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x,y,batch_size = 400,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)

