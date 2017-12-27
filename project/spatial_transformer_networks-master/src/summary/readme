總共有5種model
stn 
對原始輸入三維rgb圖像做平移縮放後接CNN(限制旋轉的參數)

multi_stn 
承襲上述stn 在後面的CNN feature map也做平移縮放

simple 
直接CNN pooling做classification

mlp
flatten後做mlp classification

global_pooling
咨翰做的 接三層CNN後 global pooling 

詳細在main.py的五種model裡有keras的code

每種model會有13個類別的confusion matrix
由於樣本數的差異
所以我有另外補上True Positive Rate, etc.
tp -> tpr
加上各類別的accuracy
每個總共有9種scalar(4+4+1) 圖表 附在tensorboard

另外每個模型我都另外train了額外兩種版本
1.
是對training set做class weight
已出現的次數為反比 希望能夠對個別class做平均的訓練
2.
對testing set做class weight
希望能夠比對的三者各個class的confusion matrix的變化

references:
https://en.wikipedia.org/wiki/Confusion_matrix
https://hci.iwr.uni-heidelberg.de/node/6132
https://github.com/oarriaga/spatial_transformer_networks/blob/master/src/mnist_cluttered_example.ipynb

