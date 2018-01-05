#CUDA_VISIBLE_DEVICES="" python2 ./main.py
python2 -u ./main.py --model ./models/model_stn.h5 -s ./summary/stn -t 'stn'
#python2 -u ./main.py --models ./models/model_multi_stn.h5 -s ./summary/multi_stn -t 'multi_stn'
#python2 -u ./main.py --models ./models/model_simple.h5 -s ./summary/simple -t 'simple'
python2 -u ./main.py --model ./models/model_mlp.h5 -s ./summary/mlp -t 'mlp'
#python2 -u ./main.py --models ./models/model_global_pooling.h5 -s ./summary/global_pooling -t 'global_pooling' -b 16


#class weight
python2 -u ./main.py --model ./models/model_train_stn.h5 -s ./summary/stn_train_weight -t 'stn' -w 'train'
#python2 -u ./main.py --models ./models/model_train_multi_stn.h5 -s ./summary/multi_stn_train_weight -t 'multi_stn' -w 'train'
#python2 -u ./main.py --models ./models/model_train_simple.h5 -s ./summary/simple_train_weight -t 'simple' -w 'train'
python2 -u ./main.py --model ./models/model_train_mlp.h5 -s ./summary/mlp_train_weight -t 'mlp' -w 'train'
#python2 -u ./main.py --models ./models/model_train_global_pooling -s ./summary/global_pooling_train_weight -t 'global_pooling' -b 16 -w 'train'

#class weight
python2 -u ./main.py --model ./models/model_test_stn.h5 -s ./summary/stn_test_weight -t 'stn' -w 'test'
#python2 -u ./main.py --models ./models/model_test_multi_stn.h5 -s ./summary/multi_stn_test_weight -t 'multi_stn' -w 'test'
#python2 -u ./main.py --models ./models/model_test_simple.h5 -s ./summary/simple_test_weight -t 'simple' -w 'test'
python2 -u ./main.py --model ./models/model_test_mlp.h5 -s ./summary/mlp_test_weight -t 'mlp' -w 'test'
#python2 -u ./main.py --models ./models/model_test_global_pooling -s ./summary/global_pooling_test_weight -t 'global_pooling' -b 16 -w 'test'
