CUDA_VISIBLE_DEVICES="" python2 ./test_confusion_matrix.py -m ./models/model_mlp.h5 -t mlp
CUDA_VISIBLE_DEVICES="" python2 ./test_confusion_matrix.py -m ./models/model_stn.h5 -t stn

