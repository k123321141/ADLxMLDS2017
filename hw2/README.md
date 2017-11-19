目錄裡沒有其他前處理的檔案
有一個training_label_json方便我計算vocabulary set dimension
模型在dropbox https://www.dropbox.com/s/avchxo5dilcic51/best_model.cks

以下
bash_parser.py使用來處理bash script用的
config.py跟utils.py是訓練模型上的一些常數及函式定義

用keras寫成 後端是指定tensorflow

嘗試改寫了keras.layers.recurrent去做attention model
所以這次load_model有一些custom object



# ADLxMLDS2017

hw2_seq2seq.sh    ->  bash scripts for TAs.


seq2seq.py        ->  traing code for S2VT model.

attention.py      ->  traing code for attention model.

bash_parser.py    ->  使用python做各個python script的funciton call，達成bash script目的

utils.py           ->  定義了loss function跟accuracy with mask，以及模型的IO

myinput.py        ->  用來處理TIMIT的資料parser

config.py  ->  定義模型用的常數
