

import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

VOCAB_FILE = "./pretrained/skip_thoughts_uni_2017_02_02/vocab.txt"
EMBEDDING_MATRIX_FILE = "./pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy"
CHECKPOINT_PATH = "./pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
MR_DATA_DIR = "/dir/containing/mr/data"

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                       vocabulary_file=VOCAB_FILE,
                          embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                         checkpoint_path=CHECKPOINT_PATH)
s = 'yellow eyes.'
data = [s,'bule hair.']
data = ['yellow eyes', 'bule eyes', 'orange eyes', 'red eyes', 'gray eyes', 'black eyes']
encodings = encoder.encode(data)

def get_nn(ind, num=2):
    encoding = encodings[ind]
    scores = sd.cdist([encoding], encodings, "cosine")[0]
    sorted_ids = np.argsort(scores)
    print("Sentence:")
    print("", data[ind])
    print("\nNearest neighbors:")
    for i in range(1, num + 1):
        print(" %d. %s (%.3f)" % (i, data[sorted_ids[i]], scores[sorted_ids[i]]))
for i,d in enumerate(data):
    get_nn(i)

