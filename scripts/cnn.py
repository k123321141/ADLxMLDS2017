from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import plot_model
import MyInputData
#parameters setting
#convolution
filter_num = 25
kernel_width = 12
kernel_height = 12
kernel_shape = (kernel_width,kernel_height) 
#max pooling
pool_width = 2
pool_height = 2
pool_shape = (pool_width,pool_height)
#
x,y,test_x,test_y = MyInputData.read_data_sets(one_hot=True)
print x.shape,y.shape


num,h,w,dep = x.shape
num_classes = 10
label_length = 6

x = np.swapaxes(x,1,3)          # (3,200,60)
#w =200 ,h = 60 ,dep = 3 or 1
print x.shape
cnn_model = Sequential()
#model.add(Conv2D(filter_num,kernel_shape,activation = 'relu',input_shape  = x.shape[1:]))
cnn_model.add(Conv2D(filters = 16,kernel_size = 3,padding= 'same',data_format = 'channels_first',activation = 'relu',input_shape  = x.shape[1:]))
cnn_model.add(Conv2D(filters = 16,kernel_size = 3,padding= 'same',data_format = 'channels_first',activation = 'relu'))
#(16,200,60)
cnn_model.add(MaxPooling2D(2,strides = 2,data_format = 'channels_first'))
#(16,100,30)
cnn_model.add(Conv2D(filters = 32,kernel_size = 3,padding= 'same',data_format = 'channels_first',activation = 'relu'))
cnn_model.add(Conv2D(filters = 32,kernel_size = 3,padding= 'same',data_format = 'channels_first',activation = 'relu'))
#(32,100,30)
cnn_model.add(MaxPooling2D(2,strides = 2,data_format = 'channels_first'))
#(32,50,15)
cnn_model.add(Conv2D(filters = 64,kernel_size = 3,padding= 'same',data_format = 'channels_first',activation = 'relu'))
cnn_model.add(Conv2D(filters = 64,kernel_size = 3,padding= 'same',data_format = 'channels_first',activation = 'relu'))
cnn_model.add(MaxPooling2D(2,strides = 2,data_format = 'channels_first'))
#(64,25,7)
cnn_model.add(Conv2D(filters = 128,kernel_size = 3,padding= 'same',data_format = 'channels_first',activation = 'relu'))
cnn_model.add(Conv2D(filters = 128,kernel_size = 3,padding= 'same',data_format = 'channels_first',activation = 'relu'))
cnn_model.add(MaxPooling2D(2,strides = 2,data_format = 'channels_first'))
#(128,12,3)
cnn_model.add(Flatten())
cnn_model.add(Dropout(0.25))
#4608

for l in cnn_model.layers:
    print l.input_shape,' -> ' , l.output_shape 
#each output dense
image_input = Input(shape=(dep, w, h))
encoded_image = cnn_model(image_input)

#each ouput digit
dim = 1000
dense1 = [Dense(1000, activation="relu")(encoded_image) for i in range(label_length)]
dense2 = [Dense(100, activation="relu")(out) for out in dense1]
dense3 = [Dense(num_classes+1, activation="softmax")(out) for out in dense2]

dense4 = [Dense(num_classes+1, activation="softmax")(out) for out in dense1]
seq_output = Concatenate(axis=-1)(dense3)
seq_output = Reshape((6,11))(seq_output)

model = Model(input = image_input,output = seq_output)
plot_model(model, to_file='../model.png',show_shapes = True)
plot_model(cnn_model, to_file='../cnn_model.png',show_shapes = True)



model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
model.fit(x,y,batch_size = 100,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)
model.save('./cnn.model')
print 'save done'
