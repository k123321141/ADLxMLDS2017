
# coding: utf-8

import numpy as np
import argparse
np.random.seed(1337)  # for reproducibility
from scipy.misc import imresize
from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import *
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD

import keras.backend as K
from spatial_transformer import SpatialTransformer

def parse():
    parser = argparse.ArgumentParser(description="spatial tranformer network")
    parser.add_argument('-s','--summary', default='./summary/default', help='summary path')
    parser.add_argument('-m','--model', default='./model.h5', help='model path')
    parser.add_argument('-t','--type', default='original', help='model type, for testing.')
    args = parser.parse_args()
    return args
def main():
    args = parse()

    nb_epochs = 100 # you probably want to go longer than this
    batch_size = 256

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


    if args.type == 'original':
        model = original_model(DIM, dep, nb_classes)
    elif args.type == 'complex':
        model = complex_model(DIM, dep, nb_classes)
    elif args.type == 'simple':
        model = simple_model(DIM, dep, nb_classes)
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


        
        
        
    trans_lay = model.layers[0]

    reset_callback = LambdaCallback(
        on_batch_end=lambda batch, logs: reset_trans(model)
    )


    X_train = X_train.reshape([X_train.shape[0],-1])
    X_test = X_test.reshape([X_test.shape[0],-1])
    try:
        
#        for e in range(nb_epochs): 
#            print('-'*40)
            #progbar = generic_utils.Progbar(X_train.shape[0])
#         for b in range( ceil(float(X_train.shape[0]) / batch_size) ):
#             print('%7d / %7d' % (f, X_train.shape[0]))

#             f = b * batch_size
#             l = min(X_train.shape[0]+1, (b+1) * batch_size)
#             X_batch = X_train[f:l]
#             y_batch = y_train[f:l]
#             loss = model.train_on_batch(X_batch, y_batch)
#             reset_trans_lay(trans_lay)
            
                
#             #print(loss)
#             #progbar.add(X_batch.shape[0], values=[("train loss", loss)])
#         scorev = model.evaluate(X_valid, y_valid, verbose=1)
#         scoret = model.evaluate(X_test, y_test, verbose=1)
        '''
        model.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                      callbacks=[reset_callback, tb_callback],verbose=2,
                      class_weight = equal_class_weight(y_test, nb_classes))
        '''
        model.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                      callbacks=[reset_callback, tb_callback],verbose=1,
                      class_weight = equal_class_weight(y_train, nb_classes))
        #         print('Epoch: {0} | Valid: {1} | Test: {2}'.format(e, scorev, scoret))
        
            
    except KeyboardInterrupt:
        #confusion matrix
        model.save_weights(args.model)
        confusion_matrix(model, X_test, y_test, labels)
    model.save_weights(args.model)
    confusion_matrix(model, X_test, y_test, labels)
def simple_model(DIM, dep, nb_classes):
    model = Sequential()


    model.add(Dense(4096, activation='sigmoid', input_shape=(DIM* DIM* dep,)))
    model.add(Dense(2048, activation='sigmoid'))
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(nb_classes, activation='softmax'))
    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    
    return model

def confusion_matrix(model,x, y_true, labels):
    y_pred = np.argmax(model.predict(x), axis=-1)
    print 'y_ture shape', y_true.shape
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
                    fn[i] += 1
        else:
            tn[y_t] += 1
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
                    fn[i] += 1
        else:
            tn[y_t] += 1
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
