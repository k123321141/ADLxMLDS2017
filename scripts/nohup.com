<<<<<<< HEAD
Traceback (most recent call last):
  File "combine_cnn_rnn.py", line 1, in <module>
    from keras.models import *
ImportError: No module named keras.models
=======
Using TensorFlow backend.
combine_cnn_rnn.py:47: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=Tensor("ti..., inputs=Tensor("in...)`
  model = Model(input = first_input,output = result)
2017-10-25 17:09:10.568321: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 17:09:10.568362: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 17:09:10.568370: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 17:09:10.568377: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 17:09:10.568384: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 17:09:10.646608: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-10-25 17:09:10.646857: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 970
major: 5 minor: 2 memoryClockRate (GHz) 1.253
pciBusID 0000:01:00.0
Total memory: 3.94GiB
Free memory: 3.82GiB
2017-10-25 17:09:10.646874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-10-25 17:09:10.646883: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-10-25 17:09:10.646892: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0)
reading  ../data/mfcc.npz
reading done
Train on 3511 samples, validate on 185 samples
Epoch 1/2000

Epoch 2/2000

Epoch 3/2000

Epoch 4/2000

Epoch 5/2000

Epoch 6/2000

Epoch 7/2000

Epoch 8/2000

Epoch 9/2000

Epoch 10/2000

Epoch 11/2000

Epoch 12/2000

Epoch 13/2000

Epoch 14/2000

Epoch 15/2000

Epoch 16/2000

Epoch 17/2000

Epoch 18/2000

Epoch 19/2000

Epoch 20/2000

Epoch 21/2000

Epoch 22/2000

Epoch 23/2000

Epoch 24/2000

Epoch 25/2000

Epoch 26/2000

Epoch 27/2000

Epoch 28/2000

Epoch 29/2000

Epoch 30/2000

Epoch 31/2000

Epoch 32/2000

Epoch 33/2000

Epoch 34/2000

Epoch 35/2000

Epoch 36/2000

Epoch 37/2000

Epoch 38/2000

Epoch 39/2000

Epoch 40/2000

Epoch 41/2000

Epoch 42/2000

Epoch 43/2000

Epoch 44/2000

Epoch 45/2000

Epoch 46/2000

Epoch 47/2000

Epoch 48/2000

Epoch 49/2000

Epoch 50/2000

Epoch 51/2000

Epoch 52/2000

Epoch 53/2000

Epoch 54/2000

Epoch 55/2000

Epoch 56/2000

Epoch 57/2000

Epoch 58/2000

Epoch 59/2000

Epoch 60/2000

Epoch 61/2000

Epoch 62/2000

Epoch 63/2000

Epoch 64/2000

Epoch 65/2000

Epoch 66/2000

Epoch 67/2000

Epoch 68/2000

Epoch 69/2000

Epoch 70/2000

Epoch 71/2000

Epoch 72/2000

Epoch 73/2000

Epoch 74/2000

Epoch 75/2000

Epoch 76/2000

Epoch 77/2000

Epoch 78/2000

Epoch 79/2000

Epoch 80/2000

Epoch 81/2000

Epoch 82/2000

Epoch 83/2000

Epoch 84/2000

Epoch 85/2000

Epoch 86/2000

Epoch 87/2000

Epoch 88/2000

Epoch 89/2000

Epoch 90/2000

Epoch 91/2000

Epoch 92/2000

Epoch 93/2000

Epoch 94/2000

Epoch 95/2000

Epoch 96/2000

Epoch 97/2000

Epoch 98/2000

Epoch 99/2000

Epoch 100/2000

Epoch 101/2000

Epoch 102/2000

Epoch 103/2000

Epoch 104/2000

Epoch 105/2000

Epoch 106/2000

Epoch 107/2000

Epoch 108/2000

Epoch 109/2000

Epoch 110/2000

Epoch 111/2000

Epoch 112/2000

Epoch 113/2000

Epoch 114/2000

Epoch 115/2000

Epoch 116/2000

Epoch 117/2000

Epoch 118/2000

Epoch 119/2000

Epoch 120/2000

Epoch 121/2000

Epoch 122/2000

Epoch 123/2000

Epoch 124/2000

Epoch 125/2000

Epoch 126/2000

Epoch 127/2000

Epoch 128/2000

Epoch 129/2000

Epoch 130/2000

Epoch 131/2000

Epoch 132/2000

Epoch 133/2000

Epoch 134/2000

Epoch 135/2000

Epoch 136/2000

Epoch 137/2000

Epoch 138/2000

Epoch 139/2000

Epoch 140/2000

Epoch 141/2000

Epoch 142/2000

Epoch 143/2000

Epoch 144/2000

Epoch 145/2000

Epoch 146/2000

Epoch 147/2000

Epoch 148/2000

Epoch 149/2000

Epoch 150/2000

Epoch 151/2000

Epoch 152/2000

Epoch 153/2000

Epoch 154/2000

Epoch 155/2000

Epoch 156/2000

Epoch 157/2000

Epoch 158/2000

Epoch 159/2000

Epoch 160/2000

Epoch 161/2000

Epoch 162/2000

Epoch 163/2000

Epoch 164/2000

Epoch 165/2000

Epoch 166/2000

Epoch 167/2000

Epoch 168/2000

Epoch 169/2000

Epoch 170/2000
<<<<<<< HEAD

>>>>>>> 90f4d4437d75ed74056f4d32d1fa7ccc7b5e3400
=======

Epoch 171/2000

Epoch 172/2000

Epoch 173/2000

Epoch 174/2000

Epoch 175/2000

Epoch 176/2000

Epoch 177/2000

Epoch 178/2000

Epoch 179/2000

Epoch 180/2000

Epoch 181/2000

Epoch 182/2000






