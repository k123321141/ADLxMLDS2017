import sys
if __name__ == '__main__':

    print(sys.argv) 
    assert len(sys.argv) == 4

    data_dir,test_output_path,peer_review_path = sys.argv[1:]
    '''
    if model_name == 'rnn':
        #prediction of rnn
        print('predict rnn model')
        argv = ['',join(cur_dir,'models','rnn.hdf5') , join(data_dir,'mfcc','test.ark'),output_path]
        output.main(argv,data_dir)
    '''
