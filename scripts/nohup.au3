Using TensorFlow backend.
/home/k123/ADLxMLDS2017/scripts/rnn.py:12: UserWarning: Update your `SimpleRNN` call to the Keras 2 API: `SimpleRNN(128, implementation=1, return_sequences=True, activation="tanh")`
  xx = Bidirectional(rnn_lay(hidden_dim,activation=activation,return_sequences=True,consume_less ='mem'))(xx)
auto_decoder_encoder.py:76: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=Tensor("ti..., inputs=Tensor("in...)`
  model = Model(input = first_input,output = result)
2017-10-25 21:28:46.900293: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 21:28:46.900326: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 21:28:46.900334: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 21:28:46.900340: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 21:28:46.900347: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-10-25 21:28:46.998238: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-10-25 21:28:46.998479: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 970
major: 5 minor: 2 memoryClockRate (GHz) 1.253
pciBusID 0000:01:00.0
Total memory: 3.94GiB
Free memory: 3.82GiB
2017-10-25 21:28:46.998494: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-10-25 21:28:46.998502: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-10-25 21:28:46.998511: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0)
reading  ../data/mfcc.npz
reading done
(3696, 777, 39, 1) (3696, 80, 49)
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

Done