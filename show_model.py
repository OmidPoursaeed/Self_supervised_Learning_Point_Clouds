"""
Scripts for testing point cloud rotation
"""

import os
import sys
import numpy as np
import skimage.io
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
from provider import rotate_point_cloud_by_angle_xyz


NUM_CLASSES = 40
BATCH_SIZE = 4
NUM_POINT = 1024
ROTATE_POINT_CLOUD = False
ROTATE_TENSOR = False
ROTATE_POINT_CLOUD_BY_AXIS_ANGEL = True
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

U_THREE_AXIS = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]

os.makedirs('images', exist_ok=True)
count = 0
sess = tf.Session()
for fn in range(len(TEST_FILES)):
    print('----'+str(fn)+'----')
    current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
    current_data = current_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]

    for file_idx in range(0, file_size, BATCH_SIZE):
        if file_idx + BATCH_SIZE > file_size:
            break
        data = current_data[file_idx: file_idx+BATCH_SIZE, :, :]

        for idx in range(BATCH_SIZE):
            img_filename = f"images/test_{count}_idx_{idx}_front.jpg"
            output_img = pc_util.point_cloud_three_views(np.squeeze(data[idx]))
            skimage.io.imsave(img_filename, output_img)
        
        if ROTATE_POINT_CLOUD:
            for k in range(6):
                rotated_data = provider.rotate_point_by_label(data, [k]*BATCH_SIZE)
                for idx in range(BATCH_SIZE):
                    img_filename = f"images/test_{count}_idx_{idx}_rot_{k}.jpg"
                    output_img = pc_util.point_cloud_three_views(np.squeeze(rotated_data[idx]))
                    skimage.io.imsave(img_filename, output_img)
        
        if ROTATE_TENSOR:
            w = tf.constant(data)
            for k in range(6):
                rotated_data = provider.rotate_point_by_label(w, [k]*BATCH_SIZE, use_tensor=True)
                result = sess.run(rotated_data)
                for idx in range(BATCH_SIZE):
                    img_filename = f"images/test_{count}_idx_{idx}_tensor_rot_{k}.jpg"
                    output_img = pc_util.point_cloud_three_views(np.squeeze(result[idx]))
                    skimage.io.imsave(img_filename, output_img)
        
        if ROTATE_POINT_CLOUD_BY_AXIS_ANGEL:
            for k in range(6):
                u = [U_THREE_AXIS[k]] * BATCH_SIZE
                theta = [np.pi/10] * BATCH_SIZE
                rotated_data = provider.rotate_point_cloud_by_axis_angle(data, u, theta)
                for idx in range(BATCH_SIZE):
                    img_filename = f"images/test_{count}_idx_{idx}_rotaxis_{k}.jpg"
                    output_img = pc_util.point_cloud_three_views(np.squeeze(rotated_data[idx]))
                    skimage.io.imsave(img_filename, output_img)
                

        # img_filename = f"images/test_{count}_x90.jpg"
        # data = current_data[file_idx, :, :]
        # data = rotate_point_cloud_by_angle_xyz(data, angle_x=np.pi/2)
        # output_img = pc_util.point_cloud_three_views(np.squeeze(data))
        # scipy.misc.imsave(img_filename, output_img)
        
        # img_filename = f"images/test_{count}_y90.jpg"
        # data = current_data[file_idx, :, :]
        # data = rotate_point_cloud_by_angle_xyz(data, angle_y=np.pi/2)            
        # output_img = pc_util.point_cloud_three_views(np.squeeze(data))
        # scipy.misc.imsave(img_filename, output_img)
        
        # img_filename = f"images/test_{count}_z90.jpg"
        # data = current_data[file_idx, :, :]
        # data = rotate_point_cloud_by_angle_xyz(data, angle_z=np.pi/2)
        # output_img = pc_util.point_cloud_three_views(np.squeeze(data))
        # scipy.misc.imsave(img_filename, output_img)
        
        count += 1
        break