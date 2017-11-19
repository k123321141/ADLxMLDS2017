import sys
import evaluate
from os.path import join



def read_id(data_dir,id_path):
    ids = []
    with open(path,'r') as f:
        while not done:
            line = f.readline().strip()
            ids.append(line)
            if(line == ''):
                done = True
    file_name_list = [join(data_dir,id+'.npy') for id in ids]
    return file_name_list
if __name__ == '__main__':

    print(sys.argv) 
    assert len(sys.argv) == 4

    data_dir,test_output_path,peer_review_path = sys.argv[1:]

    #testing 
    testing_feat_dir = join(data_dir,
            'testing_data',
            'feat')

    testing_ids = read_id(data_dir,
            join(data_dir,'testing_id.txt'))

    evaluate.main(testing_feat_dir,
            testing_ids,
            test_output_path)
    #peer review
    peer_feat_dir = join(data_dir,
            'peer_review',
            'feat')

    peer_ids = read_id(data_dir,
            join(data_dir,'peer_review_id.txt'))

    evaluate.main(peer_feat_dir,
            peer_ids,
            peer_review_path)
    print('Done')
