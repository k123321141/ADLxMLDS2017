目錄裡沒有其他前處理的檔案
模型在/hw1/models/ 沒有另外上傳空間
以下
bash_parser.py使用來處理bash script用的
loss.py跟configuration.py是訓練模型上的一些常數及函式定義
剩下都是做輸入輸出處理而已

用keras寫成 後端是指定tensorflow
我在bash script裡面加入了KERAS_BACKEND=tensorflow
各個python script路徑關係是由bash $0做判斷

成績最好的那個沒有來得及推上kaggle
我把成果寫在報告裡
model_cnn就是model_best




# ADLxMLDS2017
*.sh              ->  bash scripts for TAs.
model_*.py        ->  traing code for each models.
bash_parser.py    ->  使用python做各個python script的funciton call，達成bash script目的
loss.py           ->  定義了loss function跟accuracy with mask，估計不算padding的準確率
myinput.py        ->  用來處理TIMIT的資料parser
dic_processing.py ->  配合muinput處理TIMIT 資料
configuration.py  ->  定義模型用的常數
mapping.py        ->  處理各個輸入輸出的對應符號
output.py         ->  處理模型的輸出
mfcc.npz          ->  為了方便使用numpy，根據TIMIT資料做壓縮

