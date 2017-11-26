from keras.models import *
from keras.layers import *
import config
import keras.backend as K
import tensorflow as tf
import numpy as np
import myinput
import config
import utils
import custom_recurrents
from keras.callbacks import *
import keras
import numpy as np
import myinput
import config
import HW2_config
import utils
import bleu_eval
from os.path import join
from keras.models import *
from keras.optimizers import Adam
import os,sys
assert K._BACKEND == 'tensorflow'

def model(input_len,input_dim,output_len,vocab_dim):
    print('Build model...')
    # Try replacing GRU, or SimpleRNN.
    #
    data = Input(shape=(input_len,input_dim))
    label = Input(shape=(output_len,vocab_dim))
    #in this application, input dim = vocabulary dim
    #masking
    x = data
    #scaling data
    x = BatchNormalization()(x)

    #encoder, bidirectional RNN
    for _ in range(config.DEPTH):
        #forward RNN
        x = Bidirectional(config.RNN(config.HIDDEN_SIZE,return_sequences = True))(x)
        x = Dropout(config.DROPOUT)(x)

    #word embedding
    y = TimeDistributed(Dense(config.EMBEDDING_DIM,activation = 'linear',use_bias = False,name='word_embedding'))(label)
    #decoder
    print('build attention')
    hidden_states = config.RNN(config.HIDDEN_SIZE,return_sequences = True)(y)
    context = custom_recurrents.AttentionLayer()([x,hidden_states])
    combine = Concatenate(axis = -1)([context,hidden_states])
    
    pred = config.RNN(config.HIDDEN_SIZE,return_sequences = True)(combine)
    pred = TimeDistributed(Dense(vocab_dim,activation ='softmax',use_bias = False))(pred) 
    
    model = Model(inputs = [data,label],output=pred)  
    model.summary()
    return model
def set_train_by_label(model,train_by_label):
    decoder_lay = 'None'
    for lay in model.layers:
        if lay.name == 'decoder':
            decoder_lay = lay
    assert decoder_lay != 'None'
    decoder_lay.train_by_label = train_by_label


vocab_map = myinput.init_vocabulary_map()



if __name__ == '__main__':
    x = myinput.read_x()
    y_generator = myinput.load_y_generator()

    #testing
    test_x = myinput.read_x('../data/testing_data/feat/')
    test_y_generator = myinput.load_y_generator('../data/testing_label.json')
    belu1_high,belu2_high = utils.get_high_belu()

    epoch_idx = 0
    vocab_dim = len(myinput.init_vocabulary_map())
    model = model(HW2_config.input_len,HW2_config.input_dim,HW2_config.output_len,vocab_dim)
   
    print 'start training' 
    '''
    from keras.utils import plot_model
    plot_model(model, to_file='./model.png',show_shapes = True)
    '''
    opt = Adam(lr = config.LR)
    model.compile(loss=utils.loss_with_mask,
                  optimizer=opt,
                  metrics=[utils.acc_with_mask],sample_weight_mode = 'temporal')
    '''
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'],sample_weight_mode = 'temporal')
    '''
    for epoch_idx in range(2000000):
        #train by labels
        train_cheat = np.repeat(myinput.caption_one_hot('<bos>'),HW2_config.video_num,axis = 0)
        #record the loss and acc
        metric_history = {}

        for caption_idx in range(HW2_config.caption_list_mean):
            y = y_generator.next()
            np.copyto(train_cheat[:,1:,:],y[:,:-1,:])
            #sample weight
            s_mat = utils.weighted_by_frequency(y,config.DIVIDE_BY_FREQUENCY,config.INVERSE_RATE) #* utils.valid_sample_weight(y)
            #
            his = model.fit(x=[x,train_cheat], y=y,
                      batch_size=config.BATCH_SIZE,verbose=config.VERBOSE,
                      epochs=1,sample_weight = s_mat) 
            print('caption iteration : (%3d/%3d)' % (caption_idx+1,HW2_config.caption_list_mean))
            #record the loss and acc
            for metric,val in his.history.items():
                if metric not in metric_history:
                    metric_history[metric] = 0
                metric_history[metric] += val[0]
            sys.stdout.flush()


        loss = []
        #print history
        print('epoch_idx : %5d' % epoch_idx)
        for metric,val in metric_history.items():
            val /= HW2_config.caption_list_mean 
            print('%15s:%30f'%(metric,val))
        metric_history.clear()

        #after a epoch
        if epoch_idx % config.SAVE_ITERATION == 0:
            #model.save(join(config.CKS_PATH,'%d.cks'%epoch_idx))
            model.save(config.CKS_PATH)
            #test_y just for testing,no need for iter as a whole epoch 
            test_y = test_y_generator.next()
            # Select 2 samples from the test set at random so we can visualize errors.
            utils.testing(model,x,y,test_x,test_y,5)


            #save the high belu score model
            belu1,belu2 = utils.compute_belu(model)
            if belu1 > belu1_high:
                belu1_high = belu1
                print('new high bleu original score : ',belu1,'save model..')
                buffer_list = [f for f in os.listdir(config.BELU_PATH) if f.startswith('belu1')]
                for f in buffer_list:
                    os.remove(join(config.BELU_PATH,f))
                model.save(config.BELU_PATH+'belu1.'+str(belu1)+'.cks')
            if belu2 > belu2_high:
                belu2_high = belu2
                print('new high bleu new modified score : ',belu2,'save model..')
                buffer_list = [f for f in os.listdir(config.BELU_PATH) if f.startswith('belu2.')]
                for f in buffer_list:
                    os.remove(join(config.BELU_PATH,f))
                model.save(config.BELU_PATH+'belu2.'+str(belu2)+'.cks')
                

    #
