import os
import scipy.io as sio
import numpy as np
import numpy.random as random
import h5py
from threading import Event, Thread
import time

split_ratio = [0.6, 0.2, 0.2]

'''
Controlled dataset of DCT coefficients
'''
class Controlled_Dataset:
    def __init__(self, mapping_file, args):
        self.patch_size = args.patch_size
        self.db_config = args.db_config
        # --- split training, validation, test --- #
        # --- 1 sharing --- #
        if self.db_config == 1:
            self.categories = ['FB', 'FL', 'TW']
            QFs = ['QF-50','QF-60','QF-70','QF-80','QF-90','QF-100']
            self.categories = [c+os.sep+qf for c in self.categories for qf in QFs]
        # --- 3 sharing --- #
        elif self.db_config == 2:
            platforms = ['FB', 'FL', 'TW']
            self.categories = platforms + [p1+'-'+ p2 for p1 in platforms for p2 in platforms]
            QFs = ['QF-50','QF-60','QF-70','QF-80','QF-90','QF-100']
            self.categories = [c+os.sep+qf for c in self.categories for qf in QFs]
        elif self.db_config == 3:
            platforms = ['FB', 'FL', 'TW']
            self.categories = platforms
            self.categories = [p1+'-'+ p2 for p1 in platforms for p2 in platforms]
            self.categories =  platforms + self.categories + [p1+'-'+ p2 for p1 in self.categories for p2 in platforms]
            QFs = ['QF-50','QF-60','QF-70','QF-80','QF-90','QF-100']
            self.categories = [c+os.sep+qf for c in self.categories for qf in QFs]

    def get_len_train_val_test(self, mapping_file):
        self.mapping = sio.loadmat(mapping_file)['mapping'] # *.jpeg <-> *.hdf5
        print(self.mapping)
        self.n_images = self.mapping.shape[0]
        dict_categories = {k:[] for k in self.categories}
        for i in range(self.n_images):
            for k in self.categories:
                if os.sep+k+os.sep in self.mapping[i,0][0]:
                    dict_categories[k].append(self.mapping[i,1][0])
                    break
        self.training_files = []
        self.validation_files = []
        self.test_files = []
        self.training_labels = []
        self.validation_labels = []
        self.test_labels = []
        split_ratio = [0.6, 0.2, 0.2]
        for j,k in enumerate(self.categories):
            n_elements = len(dict_categories[k])
            dict_categories[k].sort() # sort to make sure train,validation,test are scene disjoint
            start = 0
            label = self.get_label(os.sep+k+os.sep)
            self.training_files.extend(dict_categories[k][start:start+int(n_elements*split_ratio[0])])
            self.training_labels.extend([label]*int(n_elements*split_ratio[0]))
            start = start+int(n_elements*split_ratio[0])
            self.validation_files.extend(dict_categories[k][start:start+int(n_elements*split_ratio[1])])
            self.validation_labels.extend([label]*int(n_elements*split_ratio[1]))
            start = start+int(n_elements*split_ratio[1])
            self.test_files.extend(dict_categories[k][start:])
            self.test_labels.extend([label]*len(dict_categories[k][start:]))
        self.n_training_files = len(self.training_labels)
        self.n_validation_files = len(self.validation_labels)
        self.n_test_files = len(self.test_labels)

        
    def dump(self, path):
        # --- select files --- #
        dict_categories = {}
        for k in self.categories:
            folderName = os.path.join('dataset', 'controlled', k)
            dict_categories[k] = [os.path.join(folderName, f) for f in os.listdir(folderName) if f!='.DS_Store']

        self.training_files = []
        self.validation_files = []
        self.test_files = []
        self.training_labels = []
        self.validation_labels = []
        self.test_labels = []

        for j,k in enumerate(self.categories):
            n_elements = len(dict_categories[k])
            dict_categories[k].sort() # sort to make sure train,validation,test are scene disjoint
            start = 0
            label = self.get_label(os.sep+k+os.sep)
            self.training_files.extend(dict_categories[k][start:start+int(n_elements*split_ratio[0])])
            self.training_labels.extend([label]*int(n_elements*split_ratio[0]))
            start = start+int(n_elements*split_ratio[0])
            self.validation_files.extend(dict_categories[k][start:start+int(n_elements*split_ratio[1])])
            self.validation_labels.extend([label]*int(n_elements*split_ratio[1]))
            start = start+int(n_elements*split_ratio[1])
            self.test_files.extend(dict_categories[k][start:])
            self.test_labels.extend([label]*len(dict_categories[k][start:]))
        self.n_training_files = len(self.training_labels)
        self.n_validation_files = len(self.validation_labels)
        self.n_test_files = len(self.test_labels)

        # --- save to file --- #
        with open(os.path.join(path, 'controlled_config_%d_train.dat' % self.db_config), 'w') as fout:
            for file in self.training_files:
                fout.write(file + '\n')

        with open(os.path.join(path, 'controlled_config_%d_validation.dat' % self.db_config), 'w') as fout:
            for file in self.validation_files:
                fout.write(file + '\n')

        with open(os.path.join(path, 'controlled_config_%d_test.dat' % self.db_config), 'w') as fout:
            for file in self.test_files:
                fout.write(file + '\n')


    def get_label(self, name):
        # --- 1 sharing --- #
        if self.db_config == 1:
            categories = ['FB', 'FL', 'TW']
            for l,k in enumerate(categories):
                if os.sep+k+os.sep in name:
                    return l
        # --- 2 sharing --- #
        elif self.db_config == 2:
            platforms = ['FB', 'FL', 'TW']
            categories = platforms + [p1+'-'+ p2 for p1 in platforms for p2 in platforms]
            for l,k in enumerate(categories):
                if os.sep+k+os.sep in name:
                    return l
        # --- 3 sharing --- #
        elif self.db_config == 3:
            platforms = ['FB', 'FL', 'TW']
            categories = [p1+'-'+ p2 for p1 in platforms for p2 in platforms]
            categories =  platforms + categories + [p1+'-'+ p2 for p1 in categories for p2 in platforms]
            for l,k in enumerate(categories):
                if os.sep+k+os.sep in name:
                    return l

    def get_patches(self, dct_coeff_map, top_dense=None):
    # --- split the map into multiple patches ---- #
    # --- format: [num_patches, patch_size, patch_size] --- #
        h,w = dct_coeff_map.shape
        nr = np.arange(0, h-h%self.patch_size, self.patch_size)
        nc = np.arange(0, w-w%self.patch_size, self.patch_size)
        patches = np.zeros((nr.shape[0]*nc.shape[0], self.patch_size, self.patch_size), dtype=np.float32)
        sparse_ness = np.zeros(nr.shape[0]*nc.shape[0])
        EPS = 0.1
        for i in range(nr.shape[0]):
            for j in range(nc.shape[0]):
                patches[i*nc.shape[0]+j,:,:] = dct_coeff_map[nr[i]:nr[i]+self.patch_size, nc[j]:nc[j]+self.patch_size]
                if top_dense is not None:
                    sparse_ness[i*nc.shape[0]+j] = np.sum(np.abs(patches[i*nc.shape[0]+j,:,:]) > EPS)
        if top_dense is not None and top_dense <= nr.shape[0]*nc.shape[0]:
            srt_inds = sparse_ness.argsort()[::-1][:top_dense]
            return patches[srt_inds,:,:]
            
        return patches

    def read_hdf5(self, file_path):
    # --- reads hdf5 file and return the map of DCT coefficients --- #
        try:
            if not os.path.isdir(file_path):
                file_path = file_path.replace('qnapshare', 'working')
            f = h5py.File(file_path, 'r')
            keys = list(f.keys())
            map = np.asarray(f[keys[0]], dtype=np.float32).T
            return map
        except Exception as e:
            print(e)
    
    def read_mat(self, file_path):
    # --- reads mat file and return the map of DCT coefficients --- #
        try:
        
            #if not os.path.isdir(file_path):
            #    file_path = file_path.replace('qnapshare', 'working')
            map = np.asarray(sio.loadmat(file_path)['dct_coef_map'], dtype=np.float32)
            return map
        except Exception as e:
            print(e)

    def start_reading_threads(self, n_threads, batch_size, top_dense=None):
        # --- start reading threads -- #
        self.x_buffer = [None]*n_threads
        self.y_buffer = [None]*n_threads
        self.reading_from = 0
        self.reading_ready = np.zeros(n_threads)
        self.threads = []
        self.stop_reading = 0
        self.events = []
        for i in range(n_threads):
            t = Thread(target=self.thread_read, args=(i, batch_size, top_dense))
            self.events.append(Event())
            self.threads.append(t)
            t.start()
    
    def start_reading_fullsize_threads(self, n_threads):
        # --- start reading threads -- #
        self.x_buffer = [None]*n_threads
        self.y_buffer = [None]*n_threads
        self.reading_from = 0
        self.reading_ready = np.zeros(n_threads)
        self.threads = []
        self.stop_reading = 0
        self.events = []
        for i in range(n_threads):
            t = Thread(target=self.thread_read_fullsize, args=[i])
            self.events.append(Event())
            self.threads.append(t)
            t.start()

    def stop_reading_threads(self):
        self.stop_reading = 1
        for t in self.threads:
            t.join()

    def thread_read(self, thread_id, batch_size, top_dense=None):
        EPS = 0.1
        while self.stop_reading == 0:
            if self.reading_ready[thread_id] == 0:
                x,y = self.training_samples(batch_size, top_dense)
                # --- write to buffer --- #
                self.x_buffer[thread_id] = self.reshape_x(x)
                self.y_buffer[thread_id] = y
                self.reading_ready[thread_id] = 1
                self.events[thread_id].set()
            # else:
            #     time.sleep(0.1)

    def thread_read_fullsize(self, thread_id):
        while self.stop_reading == 0:
            if self.reading_ready[thread_id] == 0:
                x,y = self.training_samples(1, full_size=True)
                # --- write to buffer --- #
                self.x_buffer[thread_id] = self.reshape_x(x) 
                self.y_buffer[thread_id] = y
                self.reading_ready[thread_id] = 1
                self.events[thread_id].set()
            #else:
            #    time.sleep(0.1)
    

    def fetch_training_samples(self, batch_size, top_dense=None):
        # while self.reading_ready[self.reading_from] == 0:
        #     pass
        self.events[self.reading_from].wait()
        x = self.x_buffer[self.reading_from]
        y = self.y_buffer[self.reading_from]
        self.reading_ready[self.reading_from] = 0
        self.events[self.reading_from].clear()
        self.reading_from = (self.reading_from + 1) % len(self.threads)
        return x,y

    def reshape_x(self, x):
        # --- reshape x to [batch_size, patch_size/8, patch_size/8, 63] --- #
        batch_size = x.shape[0]
        nr = int(self.patch_size/8)
        nc = int(self.patch_size/8)
        x_ = np.zeros((batch_size, nr, nc, 63))
        for i in range(nr):
            for j in range(nc):
                block = x[:,i*8:i*8+8,j*8:j*8+8]
                block = block.reshape([-1,64])
                x_[:,i,j,:] = block[:,1:] # remove DC component
        return x_

    def training_samples(self, batch_size, top_dense=None, full_size=False):
        assert batch_size <= self.n_training_files, 'not enough train.data'
        assert (full_size==False) or (full_size==True and batch_size==1), 'only one full-size map is allowed'
        # --- shuffle --- #
        perm = np.random.permutation(self.n_training_files)
        perm = perm[:batch_size].tolist()
        # --- read into patches --- #
        data = None
        labels = []
        for i,fi in enumerate(perm):
            dct_coef_map = self.read_hdf5(self.training_files[fi])
            if full_size == False: 
                patches = self.get_patches(dct_coef_map, top_dense)
            else:
                patches = dct_coef_map[np.newaxis, :, :]
            n = patches.shape[0]
            if i == 0:
                data = patches
            else:
                data = np.concatenate((data, patches), axis=0)
            labels.extend([self.training_labels[fi]]*n)
        # --- data format: [batch_size*num_patches, patch_size, patch_size]
        perm = np.random.permutation(data.shape[0])
        labels = np.asarray(labels, dtype=np.int32)
        return data[perm,:,:], labels[perm]

    def training_sample_fullsize(self, offset):
        assert offset + 1 <= self.n_training_files, 'not enough train.data'
        data = self.read_hdf5(self.training_files[offset])
        data = np.expand_dims(data, axis=0)
        label = [self.training_labels[offset]]
        return data, np.asarray(label,dtype=np.int32)
    
    def training_sample_sequentially(self, offset):
        assert offset + 1 <= self.n_training_files, 'not enough train.data'
        data = self.read_hdf5(self.training_files[offset])
        data = self.get_patches(data)
        label = [self.training_labels[offset]]*data.shape[0]
        return data, np.asarray(label,dtype=np.int32)
    
    def validation_samples(self, offset, batch_size=1, top_dense=None):
        assert offset + batch_size <= self.n_validation_files, 'not enough validation data'
        # --- read into patches --- #
        data = None
        labels = []
        labels_mask = [] # to keep track patch and file
        for i in range(offset,offset+batch_size):
            dct_coef_map = self.read_hdf5(self.validation_files[i])
            patches = self.get_patches(dct_coef_map, top_dense)
            n = patches.shape[0]
            if i == offset:
                data = patches
            else:
                data = np.concatenate((data, patches), axis=0)
            labels.extend([self.validation_labels[i]]*n)
            labels_mask.extend([i]*n)
        # --- data format: [batch_size*num_patches, patch_size, patch_size]
        return data, np.asarray(labels,dtype=np.int32), np.asarray(labels_mask,dtype=np.int32)
    
    def validation_sample_fullsize(self, offset):
        assert offset + 1 <= self.n_validation_files, 'not enough validation data'
        data = self.read_hdf5(self.validation_files[offset])
        data = np.expand_dims(data, axis=0)
        label = [self.validation_labels[offset]]
        return data, np.asarray(label,dtype=np.int32)

    def test_samples(self, offset, batch_size=1, top_dense=None):
        assert offset + batch_size <= self.n_test_files, 'not enough test data'
        # --- read into patches --- #
        data = None
        labels = []
        labels_mask = [] # to keep track patch and file
        for i in range(offset,offset+batch_size):
            dct_coef_map = self.read_hdf5(self.test_files[i])
            patches = self.get_patches(dct_coef_map, top_dense)
            n = patches.shape[0]
            if i == offset:
                data = patches
            else:
                data = np.concatenate((data, patches), axis=0)
            labels.extend([self.test_labels[i]]*n)
            labels_mask.extend([i]*n)
        # --- data format: [batch_size*num_patches, patch_size, patch_size]
        return data, np.asarray(labels,dtype=np.int32), np.asarray(labels_mask,dtype=np.int32)

    def test_sample_fullsize(self, offset):
        assert offset + 1 <= self.n_test_files, 'not enough test data'
        data = self.read_hdf5(self.test_files[offset])
        data = np.expand_dims(data, axis=0)
        label = [self.test_labels[offset]]
        return data, np.asarray(label,dtype=np.int32)
    
    def test_sample_sequentially(self, offset):
        assert offset + 1 <= self.n_test_files, 'not enough test data'
        data = self.read_hdf5(self.test_files[offset])
        data = self.get_patches(data)
        label = [self.test_labels[offset]]*data.shape[0]
        return data, np.asarray(label,dtype=np.int32)

class Uncontrolled_Dataset(Controlled_Dataset):
    def __init__(self, mapping_file, args):
        self.patch_size = args.patch_size
        self.db_config = args.db_config
        # --- split training, validation, test --- #
        # --- 1 sharing --- #
        if self.db_config == 1:
            self.categories = ['FB', 'FL', 'TW']
        # --- 2 sharing --- #
        elif self.db_config == 2:
            platforms = ['FB', 'FL', 'TW']
            self.categories = platforms + [p1+'-'+ p2 for p1 in platforms for p2 in platforms]
        # --- 3 sharing --- #
        elif self.db_config == 3:
            platforms = ['FB', 'FL', 'TW']
            self.categories = platforms
            self.categories = [p1+'-'+ p2 for p1 in platforms for p2 in platforms]
            self.categories =  platforms + self.categories + [p1+'-'+ p2 for p1 in self.categories for p2 in platforms]
        
    def dump(self, path):
        # --- select files --- #
        dict_categories = {}
        for k in self.categories:
            folderName = os.path.join('dataset', 'uncontrolled', k)
            dict_categories[k] = [os.path.join(folderName, f) for f in os.listdir(folderName) if f!='.DS_Store']

        self.training_files = []
        self.validation_files = []
        self.test_files = []
        self.training_labels = []
        self.validation_labels = []
        self.test_labels = []

        for j,k in enumerate(self.categories):
            n_elements = len(dict_categories[k])
            dict_categories[k].sort() # sort to make sure train,validation,test are scene disjoint
            start = 0
            label = self.get_label(os.sep+k+os.sep)
            self.training_files.extend(dict_categories[k][start:start+int(n_elements*split_ratio[0])])
            self.training_labels.extend([label]*int(n_elements*split_ratio[0]))
            start = start+int(n_elements*split_ratio[0])
            self.validation_files.extend(dict_categories[k][start:start+int(n_elements*split_ratio[1])])
            self.validation_labels.extend([label]*int(n_elements*split_ratio[1]))
            start = start+int(n_elements*split_ratio[1])
            self.test_files.extend(dict_categories[k][start:])
            self.test_labels.extend([label]*len(dict_categories[k][start:]))
        self.n_training_files = len(self.training_labels)
        self.n_validation_files = len(self.validation_labels)
        self.n_test_files = len(self.test_labels)

        # --- save to file --- #
        with open(os.path.join(path, 'uncontrolled_config_%d_train.dat' % self.db_config), 'w') as fout:
            for file in self.training_files:
                fout.write(file + '\n')

        with open(os.path.join(path, 'uncontrolled_config_%d_validation.dat' % self.db_config), 'w') as fout:
            for file in self.validation_files:
                fout.write(file + '\n')

        with open(os.path.join(path, 'uncontrolled_config_%d_test.dat' % self.db_config), 'w') as fout:
            for file in self.test_files:
                fout.write(file + '\n')

