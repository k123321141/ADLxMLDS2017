from d2 import *
from keras.models import *
from keras.layers.recurrent import SimpleRNN
from keras.layers import *
from keras.utils import *
from keras.utils import plot_model
x,y = test2()

num_classes = 48                       #lable dimension
timestep = 3
input_shape = [timestep,39]     #MFCC features number
hidden_dim = 39

y = to_categorical(y,num_classes)

#x = np.repeat(x,timestep,axis = 0)
#y = np.repeat(y,timestep,axis = 0)

#x = x.reshape(1124823,timestep,39)
#y = y.reshape(1124823,timestep,48)
x = x.reshape(1124823/timestep,timestep,39)
y = y.reshape(1124823/timestep,timestep,48)
bat_size = 300
x = x[0:372000,:,:]
y = y[0:372000,:,:]

print 'x ' ,x.shape
print 'y ' ,y.shape
#setting model
model = Sequential()
rnn_lay = SimpleRNN(hidden_dim,batch_input_shape =[bat_size,timestep,39], activation='relu',return_sequences=True,stateful=True, use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)


model.add(rnn_lay)
model.add(TimeDistributed(Dense(num_classes,activation='softmax')))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
plot_model(model, to_file='../../model.png')
model.fit(x=x,y=y,batch_size=bat_size,epochs=200,validation_split=0.05)
