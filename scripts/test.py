import myinput
import dic_processing
from configuration import max_len,num_classes,max_out_len,features_count

if __name__ == '__main__':
    #read input from pre-proccessing npz
    #x = myinput.read_x('../data/mfcc/train.ark',padding_len = max_len,padding_val = 0)
    #y = myinput.read_y('../data/train.lab',padding_len = max_len,padding_val = num_classes)
    x,y = myinput.load_input('mfcc') 
    print(x.shape,y.shape)
