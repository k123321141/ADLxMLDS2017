from keras.models import *
from keras.layers import *

import myinput
import dic_processing
from keras.utils import plot_model
from keras.callbacks import *
from keras.optimizers import *

if __name__ == '__main__':

    max_len = 777
    num_classes = 48
    features_count = 39



    dic1 = myinput.load_input('mfcc')
    dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

    dic_processing.catogorate_dic(dic1,num_classes+1)

    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]




    #model

    x = x.reshape(num,max_len,features_count,1)
    #(777,39) -> (777,39,1)
    y = y[:,1:-1,:]
    #(3696,777,49) -> (3696,775,49)
    print y.shape 

    model = load_model('../checkpoints/combine.84-1.41.model')
    z = model.evaluate(x,y)
    print z
    print model.metrics_names
    #model.fit(x,y,batch_size = 10,epochs = 2000,validation_split = 0.05)
    print 'Done'


