Using TensorFlow backend.
combine_cnn_rnn.py:81: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=Tensor("ti..., inputs=Tensor("in...)`
  model = Model(input = first_input,output = result)
2017-10-27 10:51:41.377424: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-27 10:51:41.377466: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-27 10:51:41.377475: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-27 10:51:41.377481: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-27 10:51:41.377488: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-10-27 10:51:41.456470: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-10-27 10:51:41.456710: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 970
major: 5 minor: 2 memoryClockRate (GHz) 1.253
pciBusID 0000:01:00.0
Total memory: 3.94GiB
Free memory: 3.82GiB
2017-10-27 10:51:41.456726: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-10-27 10:51:41.456734: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-10-27 10:51:41.456743: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0)
reading  ../data/mfcc.npz
reading done
(3696, 777, 39)
(?, 773, 31, 30)
Train on 3511 samples, validate on 185 samples
Epoch 1/2000
  30/3511 [..............................] - ETA: 747s - loss: 1.4236 - acc_with_mask: 0.0189