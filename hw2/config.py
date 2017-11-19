from keras.layers import *
#def model_config():
RNN = GRU
HIDDEN_SIZE = 128
DEPTH = 2
EMBEDDING_DIM = 1000
DROPOUT = 0.5

#def trainging_config():
MODEL_NAME = 'attention'
BATCH_SIZE = 128
VALIDATION_PERCENT = 10
CKS_PATH = '../checkpoints/attention.cks'
BELU_PATH = '../checkpoints/'
VERBOSE = 0
PRE_MODEL = '../checkpoints/attention.cks'
#PRE_MODEL = 'None'
SAVE_ITERATION = 1
#TRAIN_BY_LABEL = False
TRAIN_BY_LABEL = True 
LR = 0.01
INVERSE_RATE = True
#def input_config():
DATA_PATH = '../data/training_data/feat/'
LABEL_PATH = './training_label.json'
FETCH_NUM = 3

