import sys
import myinput
import output
from os.path import join,exists
if __name__ == '__main__':

    print((sys.argv)) 
    assert len(sys.argv) == 5
    cur_dir = sys.argv[1]
    data_dir = sys.argv[2]
    output_path = sys.argv[3]
    model_name = sys.argv[4]
    
    
    print('cur_dir         :  %s\ndata_dir       :   %s\noutput_path     :   %s\n' % (cur_dir,data_dir,output_path))
     
    #if not exists( join(cur_dir,'mfcc.npz') ): 
    #    myinput.init_npz(data_dir,cur_dir)
    if model_name == 'rnn':
        #prediction of rnn
        print('predict rnn model')
        argv = ['',join(cur_dir,'rnn.hdf5') , join(data_dir,'mfcc','test.ark'),output_path]
        output.main(argv,data_dir)
    elif model_name == 'cnn':
        #prediction of cnn + rnn
        print('predict cnn + rnn model')
        argv = ['',join(cur_dir,'cnn.hdf5') , join(data_dir,'mfcc','test.ark'),output_path]
        output.main(argv,data_dir)
    elif model_name == 'best':
        #prediction of cnn + rnn
        print('predict cnn + rnn model')
        argv = ['',join(cur_dir,'cnn.hdf5') , join(data_dir,'mfcc','test.ark'),output_path]
        output.main(argv,data_dir)
    else:
        print('error')

