import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import os.path as osp
import numpy as np
import json

import provider
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILES_MODELNET = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES_MODELNET = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

MODELNET10_TRAIN_FILE = 'data/ModelNet10/trainShuffled_Relabel.h5'
MODELNET10_TEST_FILE = 'data/ModelNet10/testShuffled_Relabel.h5'

CHAIR_PATH = 'data/Chair'
KEYPOINT_CHAIR_PATH = 'data/Chair/keypts_chair.mat'
CHAIR_FILES =  os.listdir(CHAIR_PATH)
TRAIN_CHAIR_FILES = [osp.join(CHAIR_PATH,f) for f in CHAIR_FILES if 'train' in f]
VAL_CHAIR_FILES = [osp.join(CHAIR_PATH,f) for f in CHAIR_FILES if 'val' in f]
TEST_CHAIR_FILES = [osp.join(CHAIR_PATH,f) for f in CHAIR_FILES if 'test' in f]

KEYPOINTNET_PATH = "/media/tianxing/Samsung 1T/ShapeNetCore/"

def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    return pc

def get_pointcloud(dataset, NUM_POINT=2048, shuffle=True):
    """
    Load the dataset into memory
    """
    if dataset == 'modelnet':
        train_file_idxs = np.arange(0, len(TRAIN_FILES_MODELNET))
        data_train = []
        label_train = []
        for fn in range(len(TRAIN_FILES_MODELNET)):
            print('----' + str(fn) + '-----')
            current_data, current_label = provider.loadDataFile(TRAIN_FILES_MODELNET[fn])
            current_data = current_data[:,0:NUM_POINT,:]
            current_label = np.squeeze(current_label)
            data_train.append(current_data)
            label_train.append(current_label)
        result_train = np.vstack(data_train)
        label_train = np.concatenate(label_train, axis=None)
        if shuffle:
            X_train, y_train, _ = provider.shuffle_data(result_train, np.squeeze(label_train)) 
        else:
            X_train, y_train = result_train, np.squeeze(label_train)
        
        data_test = []
        label_test = []
        for fn in range(len(TEST_FILES_MODELNET)):
            print('----' + str(fn) + '-----')
            current_data, current_label = provider.loadDataFile(TEST_FILES_MODELNET[fn])
            current_data = current_data[:,0:NUM_POINT,:]
            current_label = np.squeeze(current_label)
            data_test.append(current_data)
            label_test.append(current_label)
        result_test = np.vstack(data_test)
        label_test = np.concatenate(label_test, axis=None)
        if shuffle:
            X_test, y_test, _ = provider.shuffle_data(result_test, np.squeeze(label_test))
        else:
            X_test, y_test = result_test, np.squeeze(label_test)
    elif dataset == 'shapenet':
        shapenet_data, shapenet_label = provider.get_shapenet_data()
        shapenet_data = shapenet_data[:,0:NUM_POINT,:]
        X_train, X_test, y_train, y_test = train_test_split(shapenet_data, shapenet_label, test_size=0.2, random_state=42, shuffle=shuffle)
    elif dataset == 'shapenet_chair':
        shapenet_data, shapenet_label = provider.get_shapenet_data()
        shapenet_data = shapenet_data[:,0:NUM_POINT,:]
        shapenet_data, shapenet_label = shapenet_data[shapenet_label==17], shapenet_label[shapenet_label==17]
        X_train, X_test, y_train, y_test = train_test_split(shapenet_data, shapenet_label, test_size=0.2, random_state=42, shuffle=shuffle)
    elif dataset == 'modelnet10':
        current_data, current_label = provider.loadDataFile(MODELNET10_TRAIN_FILE)
        current_data = current_data[:,0:NUM_POINT,:]
        if shuffle:
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        X_train, y_train = current_data, current_label

        current_data, current_label = provider.loadDataFile(MODELNET10_TEST_FILE)
        current_data = current_data[:,0:NUM_POINT,:]
        if shuffle:
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        X_test, y_test = current_data, current_label
    elif dataset == 'keypoint':
        current_data, current_label = provider.load_mat_keypts(TRAIN_CHAIR_FILES, KEYPOINT_CHAIR_PATH, NUM_POINT)
        if shuffle:
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
            for i in range(current_data.shape[0]):  # shuffle order of points in a single model, otherwise keypoints are always at the end
                idx = np.arange(current_data.shape[1])
                np.random.shuffle(idx)
                current_data = current_data[:, idx, :]
                current_label = current_label[:, idx]
        current_label = np.squeeze(current_label)
        X_train, y_train = current_data, current_label

        current_data, current_label = provider.load_mat_keypts(TEST_CHAIR_FILES, KEYPOINT_CHAIR_PATH, NUM_POINT)
        if shuffle:
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
            for i in range(current_data.shape[0]):
                idx = np.arange(current_data.shape[1])
                np.random.shuffle(idx)
                current_data = current_data[:, idx, :]
                current_label = current_label[:, idx]
        current_label = np.squeeze(current_label)
        X_test, y_test = current_data, current_label
    elif dataset == 'keypoint_10class':
        current_data, current_label = provider.load_mat_keypts(TRAIN_CHAIR_FILES, KEYPOINT_CHAIR_PATH, NUM_POINT)
        current_label[:, -10:] = np.arange(1, 11)
        if shuffle:
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
            for i in range(current_data.shape[0]):  # shuffle order of points in a single model, otherwise keypoints are always at the end
                idx = np.arange(current_data.shape[1])
                np.random.shuffle(idx)
                current_data = current_data[:, idx, :]
                current_label = current_label[:, idx]
        current_label = np.squeeze(current_label)
        X_train, y_train = current_data, current_label

        current_data, current_label = provider.load_mat_keypts(TEST_CHAIR_FILES, KEYPOINT_CHAIR_PATH, NUM_POINT)
        current_label[:, -10:] = np.arange(1, 11)
        if shuffle:
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
            for i in range(current_data.shape[0]):
                idx = np.arange(current_data.shape[1])
                np.random.shuffle(idx)
                current_data = current_data[:, idx, :]
                current_label = current_label[:, idx]
        current_label = np.squeeze(current_label)
        X_test, y_test = current_data, current_label
    elif dataset == "keypointnet":
        json_path = osp.join(KEYPOINTNET_PATH, "annotations/all.json")
        annots = json.load(open(json_path))
        X = []
        y = []
        for annot in annots:
            class_id = annot["class_id"]
            model_id = annot["model_id"]
            kpts = []
            for kpt in annot["keypoints"]:
                kpts.append(kpt["xyz"])
            pcd_path = osp.join(KEYPOINTNET_PATH, f"pcds/{class_id}/{model_id}.pcd")
            if os.path.exists(pcd_path):
                pcd = naive_read_pcd(pcd_path)
                pcd = pcd[0:NUM_POINT, :]
            else:
                continue
            if len(kpts) != 10:
                continue
            pcd = np.concatenate((pcd[:-10], kpts))
            label = np.zeros(NUM_POINT-10)
            label = np.concatenate((label, np.ones(10)))
            X.append(pcd)
            y.append(label)
        current_data = np.array(X)
        current_label = np.array(y)
        if False and shuffle:
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
            for i in range(current_data.shape[0]):  # shuffle order of points in a single model, otherwise keypoints are always at the end
                idx = np.arange(current_data.shape[1])
                np.random.shuffle(idx)
                current_data = current_data[:, idx, :]
                current_label = current_label[:, idx]
            current_label = np.squeeze(current_label)
        X_train, X_test, y_train, y_test = train_test_split(current_data, current_label, test_size=0.2, random_state=42, shuffle=shuffle)
    else:
        raise NotImplementedError()
    print(f'Dataset name: {dataset}')
    print(f'X_train: {X_train.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'y_test: {y_test.shape}')
    return X_train, X_test, y_train, y_test

# debug
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_pointcloud('keypointnet')
    print(X_train)
    print(np.sum(y_train))