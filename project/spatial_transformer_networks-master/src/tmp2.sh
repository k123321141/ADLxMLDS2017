#CUDA_VISIBLE_DEVICES="" python2 ./main.py
python2 -u ./main.py --model ./model_main.h5 -s ./summary/main_train -t 'original'
python2 -u ./main.py --model ./model_complex.h5 -s ./summary/complex_train -t 'complex'
python2 -u ./main.py --model ./model_simple.h5 -s ./summary/simple_train -t 'simple'
