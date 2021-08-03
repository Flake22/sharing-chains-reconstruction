import os
import scipy.io as sio
import numpy as np
import argparse
import h5py

from src.results import get_labels

def get_histograms(db, db_config, split):
    features_file = 'features/dct_coef_' + db + '_' + str(db_config) + '_'+ split +'.mat'
    mat_content = sio.loadmat(features_file)
    Hist = mat_content['Features']
    file_path = mat_content['file_path']
    features_file = 'features/addi_features_' + db + '_' + str(db_config) + '_'+ split +'.mat'
    mat_content = sio.loadmat(features_file)
    Addi = mat_content['Features']

    return Hist, Addi, file_path

def get_header(db, db_config, split):

    features_file = 'features/header_features_' + db + '_' + str(db_config) + '_'+ split +'.mat'
    mat_content = sio.loadmat(features_file)
    Features = mat_content['Features']
    Features = np.asarray([f[:8] for f in Features])
    return Features

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation of several features and configurations')
    parser.add_argument('-db', required=True, help='''one of datasets: [controlled, uncontrolled, iplab, iplab_irene, isima, ...]''')
    parser.add_argument('-db_config', required=True, type=int, help='integer: how to select and label images')
    args = parser.parse_args()

    db = args.db
    db_config = args.db_config

    # get classes
    classes = get_labels(db, db_config)

    # init file
    out = h5py.File(os.path.join('features',db+'-conf'+str(db_config)+'.hdf5'),'w')

    for split in ['train', 'validation', 'test']:
        try: # this is added to prevent errors for datasets for which we only have test features
            # Read DCT and META features
            DCT, META, file_path = get_histograms(db, db_config, split)

            # Header featrues
            Header = get_header(db, db_config, split)

            # Get labels and sort DeepDCT
            nFiles = DCT.shape[0]
            labels = []
            for i in range(nFiles):
                fp = file_path[i][0][0]

                for cl in classes:
                    if fp.find(os.sep+cl+os.sep) != -1:
                        labels.append(cl)
                        break
            
            # Write features to h5py
            out[split+'/labels'] = np.array(labels, dtype='S')

            out[split+'/features/dct'] = DCT
            out[split+'/features/meta'] = META
            out[split+'/features/header'] = Header
        except:
            pass

    out.close()