#CUDA_VISIBLE_DEVICES="" python2 ./main.py
python2 ./main.py --model ./model_main.h5 -s ./summary/main
python2 ./complex.py --model ./model_com.h5 -s ./summary/complex
python2 ./test.py --model ./model_test.h5 -s ./summary/test
