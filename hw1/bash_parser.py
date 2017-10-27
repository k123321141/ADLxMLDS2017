import sys
import myinput
if __name__ == '__main__':
    print((sys.argv)) 
    assert len(sys.argv) == 4
    cur_dir = sys.argv[1]
    data_dir = sys.argv[2]
    output_path = sys.argv[3]
    
    
    print('cur_dir       :  %s\ndata_dir    :   %s\noutput_path     :   %s\n' % (cur_dir,data_dir,output_path))
    
    myinput.init_npz(data_dir,cur_dir)


