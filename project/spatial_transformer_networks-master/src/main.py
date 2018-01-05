
# coding: utf-8

import numpy as np
import argparse
np.random.seed(1337)  # for reproducibility
from scipy.misc import imresize
from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import *
from keras.layers import *
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD

import keras.backend as K
from spatial_transformer import SpatialTransformer

def parse():
    parser = argparse.ArgumentParser(description="spatial tranformer network")
    parser.add_argument('-s','--summary', default='./summary/default', help='summary path')
    parser.add_argument('-m','--model', default='./model.h5', help='model path')
    parser.add_argument('-t','--type', default='stn', help='model type, for testing.')
    parser.add_argument('-b','--batch', default=256, type=int, help='batch size')
    parser.add_argument('-w','--class_weight', default=None, help='class weight')
    args = parser.parse_args()
    return args
def main():
    args = parse()

    nb_epochs = 100 # you probably want to go longer than this
    batch_size = args.batch 

    DIM = 60
    dep = 3
    nb_classes = 13
    mnist_cluttered = '../datasets/train.npz'


# In[2]:


    data = np.load(mnist_cluttered)
    X_train, y_train = data['x_train'], data['y_train']
    X_valid, y_valid = data['x_valid'], data['y_valid']
    X_test, y_test = data['x_test'], data['y_test']
# reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], DIM, DIM, dep))
    X_valid = X_valid.reshape((X_valid.shape[0], DIM, DIM, dep))
    X_test = X_test.reshape((X_test.shape[0], DIM, DIM, dep))
    print y_valid.shape
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_valid = np_utils.to_categorical(y_valid, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print y_valid.shape
    print("Train samples: {}".format(X_train.shape))
    print("Validation samples: {}".format(X_valid.shape))
    print("Test samples: {}".format(X_test.shape))


    if args.type == 'stn':
        model = stn_model(DIM, dep, nb_classes)
    elif args.type == 'multi_stn':
        model = multi_stn_model(DIM, dep, nb_classes)
    elif args.type == 'simple':
        model = simple_model(DIM, dep, nb_classes)
    elif args.type == 'mlp':
        model = mlp_model(DIM, dep, nb_classes)
    elif args.type == 'global_pooling':
        model = global_pooling_model(DIM, dep, nb_classes)
    else:
        import sys
        print('error with wrong type name %s ' % args.type)
        sys.exit()



    XX = model.input
#model.load_weights('./model.h5')
    YY = model.layers[0].output
    F = K.function([XX], [YY])
    labels = ['off', 'Red', 'Yellow', 'Green','GreenStraightRight', 'GreenStraightLeft', 'GreenStraight', 'RedStraightLeft', 'GreenRight', 'RedStraight', 'GreenLeft', 'RedRight', 'RedLeft']
#summary
    sess = tf.InteractiveSession()
    K.set_session(sess)



    summary_placeholders, update_ops, summary_op =             setup_summary(labels)
    summary_writer = tf.summary.FileWriter(
                args.summary  , sess.graph)
    sess.run(tf.global_variables_initializer())
            
            



# In[13]:



        
    tb_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: class_acc(model, X_test, y_test, labels, epoch, 
            sess, summary_placeholders, summary_op, update_ops, summary_writer)

    )


        
        
        
    model.summary()
    reset_callback = LambdaCallback(
        on_batch_end=lambda batch, logs: reset_trans(model)
    )



    try:
        
        if args.class_weight is None:
            model.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                          callbacks=[reset_callback, tb_callback],verbose=2)
        elif args.class_weight == 'train':
            model.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                          callbacks=[reset_callback, tb_callback],verbose=2,
                          class_weight = equal_class_weight(y_train, nb_classes))
        elif args.class_weight == 'test':
            model.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                          callbacks=[reset_callback, tb_callback],verbose=2,
                          class_weight = equal_class_weight(y_test, nb_classes))
        
            
    except KeyboardInterrupt:
        #confusion matrix
        model.save_weights(args.model)
        confusion_matrix(model, X_test, y_test, labels)
    model.save_weights(args.model)
    confusion_matrix(model, X_test, y_test, labels)
def locnet(input_shape):
    # initial weights
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]
    
    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
    locnet.add(Convolution2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Convolution2D(20, (5, 5)))

    locnet.add(Flatten())
    locnet.add(Dense(50))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights ,name='trans_mat'))
    #locnet.add(Activation('sigmoid'))
    return locnet

def stn_model(DIM, dep, nb_classes):
# In[6]:

    model = Sequential()
    
    input_shape = (DIM, DIM, dep)

    model.add(SpatialTransformer(localization_net=locnet(input_shape),
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
    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

    return model
def multi_stn_model(DIM, dep, nb_classes):
    model = Sequential()
    
    input_shape = (DIM, DIM, dep)

    model.add(SpatialTransformer(localization_net=locnet(input_shape),
                                 output_size=(DIM,DIM), input_shape=input_shape))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    input_shape = (DIM/2, DIM/2, 32)
    
    model.add(SpatialTransformer(localization_net=locnet(input_shape),
                                 output_size=(DIM/4,DIM/4), input_shape=input_shape))
    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()
    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    
    return model
def simple_model(DIM, dep, nb_classes):
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), padding='same', input_shape = (DIM, DIM, dep)))
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
    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    
    return model
def mlp_model(DIM, dep, nb_classes):
    model = Sequential()
    model.add(Flatten(input_shape = (DIM, DIM, dep)))
    model.add(Dense(1024, activation='sigmoid')) 
    model.add(Dense(1024, activation='sigmoid')) 
    model.add(Dense(1024, activation='sigmoid')) 
    model.add(Dense(512, activation='sigmoid')) 
    model.add(Dense(256, activation='sigmoid')) 

    model.add(Dense(nb_classes, activation='softmax'))
    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    
    return model
def global_pooling_model(DIM, dep, nb_classes):
    model = Sequential()
    #input shape = (60,60,3)
    model.add(Convolution2D(256, (5, 5), padding='same', input_shape = (DIM, DIM, dep)))
    model.add(LeakyReLU())
    model.add(Convolution2D(256, (5, 5), padding='same'))
    model.add(LeakyReLU())
    model.add(Convolution2D(256, (5, 5), padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(DIM, DIM)))


    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    
    return model

def confusion_matrix(model,x, y_true, labels):
    y_pred = np.argmax(model.predict(x), axis=-1)
    #print 'y_ture shape', y_true.shape
    y_true = np.argmax(y_true, axis=-1)
    num = y_true.shape[0]
    #count labels
    count = [0] * len(labels)
    #True Positive
    tp = [0.] * len(labels)
    #True Negative
    tn = [0.] * len(labels)
    #False Positive
    fp = [0.] * len(labels)
    #False Negative
    fn = [0.] * len(labels)
    
    for i in range(num):
        y_p,y_t, = y_pred[i], y_true[i]
        count[y_t] += 1
        if y_p == y_t:
            tp[y_t] += 1
            for i in range(len(labels)):
                if i != y_t:
                    tn[i] += 1
        else:
            fn[y_t] += 1
            fp[y_p] += 1     
    for i,label in enumerate(labels):
        print 'label : %10s' % label

        print '-----%8s---%8s----' % ('Positive', 'Negative')
        print 'True  : %5d  ,  %5d' % (tp[i], tn[i])
        print 'False : %5d  ,  %5d' % (fp[i], fn[i])
        print '-'*27
        print ''

def ceil(x):
    a = int(x)
    return a+1 if x > a else a
        
def show_trans_lay(lay):
    idx  = [0,2,4,5]
    idx = range(6)
    k = lay.get_weights()[6]
    b = lay.get_weights()[7]
    print k[:,idx]
    print b[idx]
def reset_trans(model):
    lays = [lay for lay in model.layers if 'spatial_transformer' in lay.name ]
    for lay in lays: 
        reset_trans_lay(lay)
def reset_trans_lay(lay):
    ws = lay.get_weights()
    k = ws[6]
    b = ws[7]
    k[:,1] = 0
    k[:,3] = 0
    b[1] = 0
    b[3] = 0
    lay.set_weights(ws)
def class_acc(model,x, y_true, labels, epoch, sess, summary_placeholders, summary_op, update_ops, summary_writer):
    y_pred = np.argmax(model.predict(x), axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    num = y_true.shape[0]
    #count labels
    count = [0] * len(labels)
    #True Positive
    tp = [0.] * len(labels)
    #True Negative
    tn = [0.] * len(labels)
    #False Positive
    fp = [0.] * len(labels)
    #False Negative
    fn = [0.] * len(labels)
    total_acc = 0.
    
    for i in range(num):
        y_p,y_t, = y_pred[i], y_true[i]
        count[y_t] += 1
        if y_p == y_t:
            total_acc += 1
            tp[y_t] += 1
            for i in range(len(labels)):
                if i != y_t:
                    tn[i] += 1
        else:
            fn[y_t] += 1
            fp[y_p] += 1     
    
    total_acc = total_acc / num
    #
    acc = [0.] * len(labels)
    tpr = [0.] * len(labels)
    tnr = [0.] * len(labels)
    fpr = [0.] * len(labels)
    fnr = [0.] * len(labels)
    for i in range(len(labels)):
        acc[i] = float(tp[i]+tn[i]) / (tp[i]+ tn[i]+ fp[i]+ fn[i])
        tpr[i] = float(tp[i]) / (tp[i] + fn[i]) if (tp[i] + fn[i]) != 0 else 0
        tnr[i] = float(tn[i]) / (tn[i] + fp[i]) if (tn[i] + fp[i]) != 0 else 0
        fpr[i] = 1. - tnr[i] 
        fnr[i] = 1. - tpr[i] 
        
    result = [0.] * (9*len(labels) + 1)
    
    result[0] = total_acc
    for i,label in enumerate(labels):
        idx = i*9+1
        result[idx] = acc[i]
        result[idx+1] = tpr[i]
        result[idx+2] = tnr[i]
        result[idx+3] = fpr[i]
        result[idx+4] = fnr[i]
        result[idx+5] = tp[i]
        result[idx+6] = tn[i]
        result[idx+7] = fp[i]
        result[idx+8] = fn[i]
    
    
    #summary
    for i,result in enumerate(result):
        sess.run(update_ops[i], feed_dict={
            summary_placeholders[i]: result
        })
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, epoch)

def setup_summary(labels):
    summary_vars = [tf.Variable(0.) for i in range(len(labels) *9 +1 )] 
    summary_placeholders = [tf.Variable(0.) for var in summary_vars ]
    
    tf.summary.scalar('Total_Accuracy/Epoch', summary_vars[0])
        
    for i,label in enumerate(labels): 
        idx = 9*i + 1
        tf.summary.scalar('%s_%s/Epoch' % (label, 'Accuracy'), summary_vars[idx])
        tf.summary.scalar('%s_%s/Epoch' % (label, 'True_Positive_Rate'), summary_vars[idx+1])
        tf.summary.scalar('%s_%s/Epoch' % (label, 'True_Negative_Rate'), summary_vars[idx+2])
        tf.summary.scalar('%s_%s/Epoch' % (label, 'False_Positive_Rate'), summary_vars[idx+3])
        tf.summary.scalar('%s_%s/Epoch' % (label, 'False_Negative_Rate'), summary_vars[idx+4])
        tf.summary.scalar('%s_%s/Epoch' % (label, 'True_Positive'), summary_vars[idx+5])
        tf.summary.scalar('%s_%s/Epoch' % (label, 'True_Negative'), summary_vars[idx+6])
        tf.summary.scalar('%s_%s/Epoch' % (label, 'False_Positive'), summary_vars[idx+7])
        tf.summary.scalar('%s_%s/Epoch' % (label, 'False_Negative'), summary_vars[idx+8])
    
    
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                  range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op

def equal_class_weight(y_train, nb_classes):
    num = y_train.shape[0]
    count = np.zeros([nb_classes,])
    y_train = np.argmax(y_train, axis=-1).astype(np.uint8)

    for i in range(num):
        y = int(y_train[i])
        count[y] += 1
    w = count / num
    return w

if __name__ == '__main__':
    main()
