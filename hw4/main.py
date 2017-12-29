
# coding: utf-8

import numpy as np
import argparse
np.random.seed(1337)  # for reproducibility
from scipy.misc import imresize
from keras.models import *
from keras.callbacks import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.regularizers import L1L2
import random


def parse():
    parser = argparse.ArgumentParser(description="hw4")
    '''
    parser.add_argument('-s','--summary', default='./summary/default', help='summary path')
    parser.add_argument('-m','--model', default='./model.h5', help='model path')
    parser.add_argument('-t','--type', default='stn', help='model type, for testing.')
    parser.add_argument('-b','--batch', default=256, type=int, help='batch size')
    parser.add_argument('-w','--class_weight', default=None, help='class weight')
    '''
    args = parser.parse_args()
    return args
def main():
    args = parse()

    nb_epochs = 100 # you probably want to go longer than this

    npz = np.load('./train.npz')
    
    img = npz['x']
    text = npz['y']
    #wrong_text = npz['wrong_y']
    
    '''
    batch_size = args.batch 


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
    '''

    img_dim = 64
    c_dim = 22
    n_dim = 100

    generator, discriminator, model = gan_model(img_dim, c_dim, n_dim)
    generator.summary()
    discriminator.summary()
    model.summary()

    opt = Adam(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)

    num = img.shape[0]
    for i in range(nb_epochs):
        #train discriminator
        #prepare discriminator data
        x_buf = []
        c_buf = []
        #real img, right text
        x_buf.append(img)
        c_buf.append(text)
        #fake img, right text 
        print 'predict fake'
        n_sample = noise_sample(n_dim, num)
        #n_sample = noise_sample(n_dim, 100)
        g_input  = np.concatenate([n_sample, text], axis=-1)
        #g_input  = np.concatenate([n_sample, text[:100]], axis=-1)
        fake_img = generator.predict(g_input)
        x_buf.append(fake_img) 
        #c_buf.append(text)
        c_buf.append(text[:100])

        #real img, wrong text
        print 'prepare wrong text data'
        x_buf.append(img)
        c_buf.append(wrong_text(text) )

        #
        print 'stack'
        for x in x_buf:
            print 'x' , x.shape
        x = np.vstack(x_buf)
        c = np.vstack(c_buf)
        #prepare label
        y = np.zeros([x.shape[0], 1])
        for i in range(img.shape[0]):
            y[i,0] = 1
        print x.shape,c.shape,y.shape
        '''
        x = x[16600:17200] 
        c = c[16600:17200] 
        y = y[16600:17200] 
        '''
        n = noise_sample(n_dim, 600)
        #train discriminator
        discriminator.trainable = True
        discriminator.fit(x=[x,c],y=y, epochs=1, batch_size=32, verbose=1)
        discriminator.trainable = False
        model.fit(x=[n,c],y=y, epochs=1, batch_size=32, verbose=1)


        #50973

def wrong_text(y):
    
    wrong_y = np.zeros(y.shape)
    for i in range(y.shape[0]):
        arr = []
        for j in range(y.shape[1]):
            if y[i,j] == 0:
                arr.append(j)
        wrong_y[i,random.sample(arr, 4)] = 1
    return wrong_y
def noise_sample(n_dim, num):
    return np.random.normal(size=(num, n_dim))
def generator_model(noise_dim, c_dim):
    #c_dim = code dimension
    #input_shape = (122,)
    nch = 256
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    w = 8 
    model = Sequential(name='generator')
    model.add(Dense(nch * w * w, input_dim=noise_dim+c_dim, kernel_regularizer=reg()))
    #model.add(BatchNormalization())
    model.add(Reshape([w, w, nch]))
    model.add(Conv2D(int(nch / 2), kernel_size=(3, 3), padding='same', kernel_regularizer=reg(), data_format='channels_last'))
    model.add(BatchNormalization( axis=-1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2), data_format='channels_last'))

    model.add(Conv2D(int(nch / 2), kernel_size=(5, 5), padding='same', kernel_regularizer=reg(), data_format='channels_last'))
    model.add(BatchNormalization( axis=-1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2), data_format='channels_last'))

    model.add(Conv2D(int(nch / 4), kernel_size=(5, 5), padding='same', kernel_regularizer=reg(), data_format='channels_last'))
    model.add(BatchNormalization( axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2), data_format='channels_last'))

    model.add(Conv2D(3, kernel_size=(5, 5), padding='same', kernel_regularizer=reg(), data_format='channels_last'))
    model.add(Activation('sigmoid'))
    
    
    return model
def discriminator_model(img_dim, c_dim):
    #c_dim = code dimension
    #input_shape = (122,)
    nch = 256
    h = 5
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
   
    #
    
    img_model = Sequential()
    img_model.add(Conv2D(int(nch / 4), kernel_size=(5, 5), padding='same', kernel_regularizer=reg(),
                                            input_shape=(img_dim, img_dim, 3), data_format='channels_last'))
    img_model.add(BatchNormalization( axis=-1))
    img_model.add(LeakyReLU(0.2))
    img_model.add(MaxPooling2D((2, 2), data_format='channels_last'))

    img_model.add(Conv2D(int(nch / 2), kernel_size=(5, 5), padding='same', kernel_regularizer=reg(), data_format='channels_last'))
    img_model.add(BatchNormalization( axis=-1))
    img_model.add(LeakyReLU(0.2))
    img_model.add(MaxPooling2D((2, 2), data_format='channels_last'))

    img_model.add(Conv2D(int(nch / 4), kernel_size=(2, 2), padding='same', kernel_regularizer=reg(), data_format='channels_last'))
    img_model.add(BatchNormalization( axis=1))
    img_model.add(LeakyReLU(0.2))

    img_model.add(Flatten())
    
    #consider condition
    img_input = Input(shape=(img_dim, img_dim, 3))
    c_input = Input(shape=(c_dim,))
    
    img_out = img_model(img_input)

    x = Concatenate(axis = -1) ([img_out, c_input]) 
   
    x = Dense(128, activation='sigmoid')(x)
    x = Dense(64, activation='sigmoid')(x)
    x = Dense(32, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=(img_input, c_input), outputs=y, name='discriminator')

    return model
    
def gan_model(img_dim, c_dim, noise_dim):
    
    n_input = Input(shape=(noise_dim,)) 
    c_input = Input(shape=(c_dim,))
    generator = generator_model(noise_dim, c_dim)
    discriminator = discriminator_model(img_dim, c_dim)

    x = Concatenate(axis=-1)([n_input, c_input])
    x = generator(x)
    y = discriminator([x, c_input])
    
    model = Model(inputs=[n_input, c_input], outputs=y, name='gan_end2end')

    return generator, discriminator, model

def global_pooling_model(DIM, dep, nb_classes):
    model = Sequential()
    #input shape = (60,60,3)
    model.add(Conv2D(256, (5, 5), padding='same', input_shape = (DIM, DIM, dep)))
    model.add(LeakyReLU())
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(256, (5, 5), padding='same'))
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
