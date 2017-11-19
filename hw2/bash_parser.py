import sys
import evaluate
from os.path import join,isfile
import os
from keras.models import load_model
import utils
import custom_recurrents

def read_id(data_dir,id_path):
    ids = []
    with open(id_path,'r') as f:
        while True:
            line = f.readline().strip()
            if(line == ''):
                break
            ids.append(line)
    file_name_list = [id+'.npy' for id in ids]
    return file_name_list
if __name__ == '__main__':

    print(sys.argv) 
    assert len(sys.argv) == 4
    #check model
    model_path = './best_model.cks'
    if not isfile(model_path):
        wget_script =('wget https://www.dropbox.com/s/avchxo5dilcic51/best_model.cks')
        os.system(wget_script)
    
    model = load_model(model_path,
                custom_objects={'loss_with_mask':utils.loss_with_mask,
                    'acc_with_mask':utils.acc_with_mask,
                    'AttentionDecoder':custom_recurrents.AttentionDecoder})
    data_dir,test_output_path,peer_review_path = sys.argv[1:]

    #testing 
    testing_feat_dir = join(data_dir,
            'testing_data',
            'feat')

    testing_ids = read_id(data_dir,
            join(data_dir,'testing_id.txt'))

    evaluate.main(model,
            testing_feat_dir,
            testing_ids,
            test_output_path)
    #peer review
    peer_feat_dir = join(data_dir,
            'peer_review',
            'feat')

    peer_ids = read_id(data_dir,
            join(data_dir,'peer_review_id.txt'))

    evaluate.main(model,
            peer_feat_dir,
            peer_ids,
            peer_review_path)
    print('Done')
