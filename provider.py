import os
import sys
import socket
import numpy as np
import h5py
import scipy
from scipy.io import loadmat
sys.path.append('./latent_3d_points_py3/')
from latent_3d_points_py3.src import in_out
from latent_3d_points_py3.src.general_utils import plot_3d_point_cloud

from functools import partial
import tqdm

import tensorflow as tf
import tensorflow.math as tm
import multiprocessing
import torch
# from numpy import pi, cos, sin, arccos, arange
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if socket.gethostname == 'tianxing-GS73-7RE':
    SHAPENET_DIR = '/media/tianxing/Data/datasets/shape_net_core_uniform_samples_2048/'
else:
    SHAPENET_DIR = './data/shape_net_core_uniform_samples_2048/'
scratch_shapenet_dir = '/scratch/shape_net_core_uniform_samples_2048'
if os.path.exists(scratch_shapenet_dir):
    SHAPENET_DIR = scratch_shapenet_dir
    print(f'Loading shapenet from {SHAPENET_DIR}')

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))



def get_shapenet_data():
    labels_lst = list(in_out.snc_category_to_synth_id().keys())
    data = []
    labels = []
    for label in tqdm.tqdm(labels_lst, desc='loading data'):
        syn_id = in_out.snc_category_to_synth_id()[label]
        class_dir = os.path.join(SHAPENET_DIR , syn_id)
        pc = in_out.load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='*.ply', verbose=False)
        cur_data, _, _ = pc.full_epoch_data(shuffle=False)
        data.append(cur_data)
        labels.append([labels_lst.index(label)] * cur_data.shape[0])
    current_data = np.concatenate(data, axis=0)
    current_label = np.concatenate(labels, axis=0)
    print(current_data.shape)
    print(current_label.shape)
    
    current_data, current_label, _ = shuffle_data(current_data, np.squeeze(current_label))            
    current_label = np.squeeze(current_label)
    return current_data, current_label

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
          Bx1 array, rotated angle for all point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    angles = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        angles[k, ...] = rotation_angle
    return rotated_data, angles

def rotate_tensor_point_cloud(point_cloud):
    """ Randomly rotate one tensor point cloud
          Nx3 tensor, original point cloud
        Return:
          Nx3 tensor, rotated point clouds
    """
    random_angles = np.random.uniform(size = 3) * 2 * np.pi
    angle_x, angle_y, angle_z = random_angles[0], random_angles[1], random_angles[2]
    rotated_tensor = rotate_tensor_by_angle_xyz(point_cloud, angle_x, angle_y, angle_z)
    return rotated_tensor

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_list(batch_data, rotation_angles):
    """ Rotate the point cloud along up direction with certain angle list.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angles[k])
        sinval = np.sin(rotation_angles[k])
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_xyz(batch_data, angle_x=0, angle_y=0, angle_z=0):
    """ Rotate the point cloud along up direction with certain angle.
        Rotate in the order of x, y and then z.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = batch_data.reshape((-1, 3))
    
    cosval = np.cos(angle_x)
    sinval = np.sin(angle_x)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    rotated_data = np.dot(rotated_data, rotation_matrix)

    cosval = np.cos(angle_y)
    sinval = np.sin(angle_y)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(rotated_data, rotation_matrix)

    cosval = np.cos(angle_z)
    sinval = np.sin(angle_z)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(rotated_data, rotation_matrix)
    
    return rotated_data.reshape(batch_data.shape)

def rotate_point_cloud_by_angle_xyz_cuda(batch_data, angle_x=0, angle_y=0, angle_z=0):
    """ Same interface as rotate_point_cloud_by_angle_xyz, but use gpu to compute
    """
    rotated_data = torch.Tensor(batch_data).cuda()
    rotated_data = rotated_data.view((-1, 3))

    cosval = np.cos(angle_x)
    sinval = np.sin(angle_x)
    rotation_matrix = torch.Tensor([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    rotation_matrix = rotation_matrix.cuda()
    rotated_data = torch.mm(rotated_data, rotation_matrix)

    cosval = np.cos(angle_y)
    sinval = np.sin(angle_y)
    rotation_matrix = torch.Tensor([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotation_matrix = rotation_matrix.cuda()
    rotated_data = torch.mm(rotated_data, rotation_matrix)

    cosval = np.cos(angle_z)
    sinval = np.sin(angle_z)
    rotation_matrix = torch.Tensor([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    rotation_matrix = rotation_matrix.cuda()
    rotated_data = torch.mm(rotated_data, rotation_matrix)
    rotated_data = rotated_data.view(batch_data.shape)
    
    return rotated_data.cpu().numpy()


def rotate_point_cloud_by_axis_angle(batch_data, u, theta):
    """ Rotate the point cloud around the axis u by angle theta

        u is a list of tuples, B * (x,y,z), consisting of unit vectors of the axis
        theta is a list of angles, B length array, representing the angle
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    u = np.squeeze(u)
    theta = np.squeeze(theta)
    eps = 1e-6
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        x, y, z = u[k]
        assert 1-eps < x*x+y*y+z*z < 1+eps

        cosval = np.cos(theta[k])
        sinval = np.sin(theta[k])

        rotation_matrix = np.array([[cosval + x*x*(1-cosval), x*y*(1-cosval) - z*sinval, x*z*(1-cosval) + y*sinval],
                                    [y*x*(1-cosval) + z*sinval, cosval + y*y*(1-cosval), y*z*(1-cosval) - x*sinval],
                                    [z*x*(1-cosval) - y*sinval, z*y*(1-cosval) + x*sinval, cosval + z*z*(1-cosval)]])
        

        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    
    return rotated_data


def tflt(value):
    return tf.constant(value, dtype=tf.float32)

def rotate_tensor_by_angle_xyz(input_data, graph, angle_x=0, angle_y=0, angle_z=0):
    """ Rotate the point cloud along up direction with certain angle.
        Rotate in the order of x, y and then z.
        Input:
          input_data: Nx3 tensor, original batch of point clouds
          angle_x: tf constant
          angle_y: tf constant
          angle_z: tf constant
        Return:
          Nx3 tensor, rotated batch of point clouds
    """
    with graph.as_default():
        # print('angle_x', angle_x)
        if angle_x == 0:
            angle_x = tflt(0)
        if angle_y == 0:
            angle_y = tflt(0)
        if angle_z == 0:
            angle_z = tflt(0)
    
        input_data = tf.cast(input_data, tf.float32)
        rotation_matrix1 = tf.stack([1, 0, 0,
                                0, tm.cos(angle_x), -tm.sin(angle_x),  
                                0, tm.sin(angle_x), tm.cos(angle_x)])
        rotation_matrix1 = tf.reshape(rotation_matrix1, (3,3))
        rotation_matrix1 = tf.cast(rotation_matrix1, tf.float32)
        shape_pc = tf.matmul(input_data, rotation_matrix1)

        rotation_matrix2 = tf.stack([tm.cos(angle_y), 0, tm.sin(angle_y),
                                    0, 1, 0,
                                    -tm.sin(angle_y), 0, tm.cos(angle_y)])
        rotation_matrix2 = tf.reshape(rotation_matrix2, (3,3))
        rotation_matrix2 = tf.cast(rotation_matrix2, tf.float32)
        shape_pc = tf.matmul(shape_pc, rotation_matrix2)

        rotation_matrix3 = tf.stack([tm.cos(angle_z), -tm.sin(angle_z), 0,
                                    tm.sin(angle_z), tm.cos(angle_z), 0,
                                    0, 0, 1])
        rotation_matrix3 = tf.reshape(rotation_matrix3, (3,3))    
        rotation_matrix3 = tf.cast(rotation_matrix3, tf.float32)
        shape_pc = tf.matmul(shape_pc, rotation_matrix3)
        return shape_pc

def rotate_tensor_by_batch(batch_data, label):
    batch_size = batch_data.get_shape().as_list()[0]
    phi, theta = get_sunflower_angle(batch_size, label)
    rotate_fun = partial(rotate_tensor_by_angle_xyz,  angle_x=theta, angle_z=phi)
    rotated_data = tf.map_fn(rotate_fun, batch_data)
  
    return rotated_data

def get_sunflower_angle(num_points, label):
    pts = sunflower_distri(num_points)
    pt = pts[label]
    phi = compute_angle([pt[0],pt[1],0],[1,0,0])
    theta = compute_angle(pt,[0,0,1])
    return phi, theta

def nested_tf_cond(conditions, callables):
    cond, to_call = conditions[0], callables[0]
    if len(conditions) == 1:
        return tf.cond(cond, lambda: to_call, lambda: to_call)
    else:
        return tf.cond(cond, lambda: to_call, lambda: nested_tf_cond(conditions[1:], callables[1:]))

def rotate_tensor_by_label(batch_data, label, graph):
    """ Rotate a batch of points by [label] to 6 or 18 possible angles
        [label] is a tensor
        Input:
          BxNx3 array
        Return:
          BxNx3 array
        
    """
    with graph.as_default():
        batch_size = batch_data.get_shape().as_list()[0]
        rotated_data = [None] * batch_size
        splits = tf.split(batch_data, batch_size, 0)
        label = tf.dtypes.cast(label, dtype=tf.float32)
        label = tf.split(label, batch_size, 0)
        label = [tf.squeeze(l) for l in label]
        for k in range(batch_size):
            shape_pc = splits[k] 
            l = label[k]
            cond0 = tm.equal(tflt(0), l)
            cond1 = tm.logical_and(tm.less_equal(tflt(1), l), tm.less_equal(l, tflt(3)))
            cond2 = tm.logical_and(tm.less_equal(tflt(4), l), tm.less_equal(l, tflt(5)))
            cond3 = tm.logical_and(tm.less_equal(tflt(6), l), tm.less_equal(l, tflt(9)))
            cond4 = tm.logical_and(tm.less_equal(tflt(10), l), tm.less_equal(l, tflt(13)))
            cond5 = tm.logical_and(tm.less_equal(tflt(14), l), tm.less_equal(l, tflt(17)))

            call0= rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=0)
            call1= rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=(l)*np.pi/4)
            call2 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_z=(l*2-7)*np.pi/2)
            call3 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=(l*2-11)*np.pi/4)
            call4 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_z=(l*2-19)*np.pi/4)
            call5 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=np.pi/2, angle_z=(l*2-27)*np.pi/4)

            conditions = [cond0, cond1, cond2, cond3, cond4, cond5]
            callables = [call0, call1, call2, call3, call4, call5]
            shape_pc = nested_tf_cond(conditions, callables)
   
            rotated_data[k] = shape_pc

        rotated_data = tf.squeeze(tf.stack(rotated_data, axis=0))
        return rotated_data

def rotate_point_by_label(batch_data, label):
    """ Rotate a batch of points by the label
        Input:
          BxNx3 array
        Return:
          BxNx3 array
        
    """
    rotate_func = rotate_point_cloud_by_angle_xyz
    batch_size = batch_data.shape[0]
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_size):
        shape_pc = batch_data[k, ...]
        l = label[k]
        if l==0:
            pass
        elif 1<=l<=3:
            shape_pc = rotate_func(shape_pc, angle_x=l*np.pi/2)
        elif 4<=l<=5:
            shape_pc = rotate_func(shape_pc, angle_z=(l*2-7)*np.pi/2)
        elif 6<=l<=9:
            shape_pc = rotate_func(shape_pc, angle_x=(l*2-11)*np.pi/4)
        elif 10<=l<=13:
            shape_pc = rotate_func(shape_pc, angle_z=(l*2-19)*np.pi/4)
        else: #l == 14 ~ 17
            shape_pc = rotate_func(shape_pc, angle_x=np.pi/2, angle_z=(l*2-27)*np.pi/4)
        rotated_data[k, ...] = shape_pc

    return rotated_data

def rotate_tensor_by_label_32(batch_data, label, graph):
    """ Rotate a batch of points by the label
        32 possible directions:
            vertices of a regular icosahedron
            vertices of a regular dodecahedron
        Input:
          BxNx3 array
        Return:
          BxNx3 array
    """
    with graph.as_default():
        batch_size = batch_data.get_shape().as_list()[0]
        rotated_data = [None] * batch_size
        splits = tf.split(batch_data, batch_size, 0)
        label = tf.dtypes.cast(label, dtype=tf.float32)
        label = tf.split(label, batch_size, 0)
        label = [tf.squeeze(l) for l in label]
        for k in range(batch_size):
            shape_pc = splits[k] 
            l = label[k]
            cond0 = tm.equal(tflt(0), l)
            cond1 = tm.logical_and(tm.less_equal(tflt(1), l), tm.less_equal(l, tflt(5)))
            cond2 = tm.logical_and(tm.less_equal(tflt(6), l), tm.less_equal(l, tflt(10)))
            cond3 = tm.logical_and(tm.less_equal(tflt(11), l), tm.less_equal(l, tflt(20)))
            cond4 = tm.logical_and(tm.less_equal(tflt(21), l), tm.less_equal(l, tflt(25)))
            cond5 = tm.logical_and(tm.less_equal(tflt(26), l), tm.less_equal(l, tflt(30)))
            cond6 = tm.equal(tflt(31), l)

            call0= rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=0)
            call1= rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=0.646, angle_y=(l-1)*np.pi/2.5)
            call2 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=1.108, angle_y=(l*2-11)*np.pi/5)
            call3 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=np.pi/2, angle_y=(l-11)*np.pi/5)
            call4 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=2.033, angle_y=(l*2-11)*np.pi/5)
            call5 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=2.496, angle_y=(l-26)*np.pi/2.5)
            call6 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=np.pi)

            conditions = [cond0, cond1, cond2, cond3, cond4, cond5, cond6]
            callables = [call0, call1, call2, call3, call4, call5, call6]
            shape_pc = nested_tf_cond(conditions, callables)

            rotated_data[k] = shape_pc

        rotated_data = tf.squeeze(tf.stack(rotated_data, axis=0))
        return rotated_data

def rotate_point_by_label_32(batch_data, label, use_tensor=False):
    """ Rotate a batch of points by the label
        32 possible directions:
            vertices of a regular icosahedron
            vertices of a regular dodecahedron
        Input:
          BxNx3 array
        Return:
          BxNx3 array
    """
    if use_tensor:
        rotate_func = rotate_tensor_by_angle_xyz
        batch_size = batch_data.get_shape().as_list()[0]
        rotated_data = [None] * batch_size
        splits = tf.split(batch_data, batch_size, 0)

    else:
        rotate_func = rotate_point_cloud_by_angle_xyz
        batch_size = batch_data.shape[0]
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    
    for k in range(batch_size):
        shape_pc = splits[k] if use_tensor else batch_data[k, ...]
        l = label[k]
        if l==0:
            pass  # not rotate, upright
        elif 0 < l <= 5:
            # 1 - 5 is the upper layer
            # rotate down 37 degree, then 72 degree along gravity axis
            shape_pc = rotate_func(shape_pc, angle_x=0.646, angle_y=(l-1)*np.pi/2.5)
        elif 6 <= l <= 10:
            # 6 - 10 is the second layer
            # rotate down 63.5 degree, then 72 degree along gravity axis
            shape_pc = rotate_func(shape_pc, angle_x=1.108, angle_y=(l*2-11)*np.pi/5)
        elif 11 <= l <= 20:
            # 11 - 20 is the horizontal layer
            # rotate down 90 degree, then 36 degree along gravity axis
            shape_pc = rotate_func(shape_pc, angle_x=np.pi/2, angle_y=(l-11)*np.pi/5)
        elif 21 <= l <= 25:
            # 21 - 25 is symmetrical to 6 - 10
            # rotate down 180 - 63.5 degree, then 72 degree along gravity axis
            shape_pc = rotate_func(shape_pc, angle_x=2.033, angle_y=(l*2-11)*np.pi/5)
        elif 26 <= l <= 30:
            # 26 - 30 is symmetrical to 1 - 5
            # rotate down 180 - 37 degree, then 72 degree along gravity axis
            shape_pc = rotate_func(shape_pc, angle_x=2.496, angle_y=(l-26)*np.pi/2.5)
        else:
            # downwards
            shape_pc = rotate_func(shape_pc, angle_x=np.pi)    
       
        if use_tensor:
            rotated_data[k] = shape_pc
        else:
            rotated_data[k, ...] = shape_pc    
    if use_tensor: 
        rotated_data = tf.squeeze(tf.stack(rotated_data, axis=0))
    return rotated_data

def rotate_tensor_by_label_54(batch_data, label, graph):
    """ Rotate a batch of points by the label
        54 possible directions:
            vertices of a regular icosahedron
            vertices of a regular dodecahedron
        Input:
          BxNx3 array
        Return:
          BxNx3 array
    """
    with graph.as_default():
        batch_size = batch_data.get_shape().as_list()[0]
        rotated_data = [None] * batch_size
        splits = tf.split(batch_data, batch_size, 0)
        label = tf.dtypes.cast(label, dtype=tf.float32)
        label = tf.split(label, batch_size, 0)
        label = [tf.squeeze(l) for l in label]
        for k in range(batch_size):
            shape_pc = splits[k] 
            l = label[k]
            cond0 = tm.logical_or(tm.equal(tflt(0), l), tm.equal(tflt(1), l))
            cond1 = tm.logical_and(tm.logical_and(tm.less_equal(tflt(2), l), tm.less_equal(l, tflt(15))), tm.equal(tflt(0), tm.mod(l, tflt(2))))
            cond2 = tm.logical_and(tm.logical_and(tm.less_equal(tflt(2), l), tm.less_equal(l, tflt(15))), tm.equal(tflt(1), tm.mod(l, tflt(2))))
            cond3 = tm.logical_and(tm.logical_and(tm.less_equal(tflt(16), l), tm.less_equal(l, tflt(39))), tm.equal(tflt(0), tm.mod(l, tflt(2))))
            cond4 = tm.logical_and(tm.logical_and(tm.less_equal(tflt(16), l), tm.less_equal(l, tflt(39))), tm.equal(tflt(1), tm.mod(l, tflt(2))))
            cond5 = tm.logical_and(tm.less_equal(tflt(40), l), tm.less_equal(l, tflt(53)))

            call0= rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=np.pi*l)
            call1= rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=np.pi/6, angle_z=2*(l-2)*np.pi/7)
            call2 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=5*np.pi/6, angle_z=2*(l-2)*np.pi/7)
            call3 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=np.pi/3, angle_z=2*(l-16)*np.pi/12)
            call4 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=2*np.pi/3, angle_z=2*(l-16)*np.pi/12)
            call5 = rotate_tensor_by_angle_xyz(shape_pc, graph, angle_x=np.pi/2, angle_z=2*(l-40)*np.pi/14)

            conditions = [cond0, cond1, cond2, cond3, cond4, cond5]
            callables = [call0, call1, call2, call3, call4, call5]
            shape_pc = nested_tf_cond(shape_pc, conditions, callables)

            rotated_data[k] = shape_pc

        rotated_data = tf.squeeze(tf.stack(rotated_data, axis=0))
        return rotated_data

def rotate_point_by_label_54(batch_data, label, use_tensor=False):
    """ Rotate a batch of points by the label
        54 possible directions:
            2 points at poles
            14 points at 30 degrees from the z-axis (7 points on each hemisphere)
            24 points at 60 degrees from the z-axis (12 points on each hemisphere)
            14 points on the equator
        Input:
          BxNx3 array
        Return:
          BxNx3 array
    """
    if use_tensor: #only rotate one tensor
        rotate_func = rotate_tensor_by_angle_xyz
        batch_size = batch_data.get_shape().as_list()[0]
        rotated_data = [None] * batch_size
        splits = tf.split(batch_data, batch_size, 0)

    else: 
        rotate_func = rotate_point_cloud_by_angle_xyz
        batch_size = batch_data.shape[0]
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)

    for k in range(batch_size):
        shape_pc = splits[k] if use_tensor else batch_data[k, ...]
        l = label[k] # l = 0, ..., 53
        if l==0 or l==1:
            shape_pc = rotate_func(shape_pc, angle_x=np.pi*l)  
        elif 2<=l<=15:
            shape_pc = rotate_func(shape_pc, angle_z=2*(l-2)*np.pi/7)  
            if l%2==0:
                shape_pc = rotate_func(shape_pc, angle_x=np.pi/6)  
            else:
                shape_pc = rotate_func(shape_pc, angle_x=5*np.pi/6)  
        elif 16<=l<=39:
            shape_pc = rotate_func(shape_pc, angle_z=2*(l-16)*np.pi/12)  
            if l%2==0:
                shape_pc = rotate_func(shape_pc, angle_x=np.pi/3)  
            else:
                shape_pc = rotate_func(shape_pc, angle_x=2*np.pi/3)   
        else:         
            shape_pc = rotate_func(shape_pc, angle_x=np.pi/2, angle_z=2*(l-40)*np.pi/14)  
        if use_tensor:
            rotated_data[k] = shape_pc
        else:
            rotated_data[k, ...] = shape_pc

    if use_tensor: 
        rotated_data = tf.squeeze(tf.stack(rotated_data, axis=0))
    return rotated_data


def rotate_point_by_label_n(batch_data, label, graph, n, use_tensor=False, distri='sunflower'):
    """ Rotate a batch of points by the label
        n possible directions:
            n points forms a sunflower spiral distribution 
            with golden ratio (1 + sqrt(5))/2
        Input:
          BxNx3 array
        Return:
          BxNx3 array
    """
    if use_tensor: #only rotate one tensor
        rotate_func = rotate_tensor_by_angle_xyz
        batch_size = batch_data.get_shape().as_list()[0]
        rotated_data = [None] * batch_size
        splits = tf.split(batch_data, batch_size, 0)
        # label = tf.dtypes.cast(label, dtype=tf.float32)
        # label = tf.split(label, batch_size, 0)
        # label = [tf.squeeze(l) for l in label]
        label = np.random.randint(0, n, size=batch_size)

    else: 
        rotate_func = rotate_point_cloud_by_angle_xyz
        batch_size = batch_data.shape[0]
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)

    if distri == 'sunflower':
        pts = sunflower_distri(n)
    elif distri == 'normal':
        pts = normal_distri(n)

    np.random.shuffle(pts)
    for k in range(batch_data.shape[0]):
        shape_pc = splits[k] if use_tensor else batch_data[k, ...]
        l = label[k] # l = 0, ..., n-1
        pt = pts[l]
        phi = compute_angle([pt[0],pt[1],0],[1,0,0])
        theta = compute_angle(pt,[0,0,1])
        if use_tensor:
            shape_pc = rotate_func(shape_pc, graph, angle_x=theta, angle_z=phi)
            rotated_data[k] = shape_pc
        else:
            shape_pc = rotate_func(shape_pc, angle_x=theta, angle_z=phi)
            rotated_data[k, ...] = shape_pc

    if use_tensor: 
        rotated_data = tf.squeeze(tf.stack(rotated_data, axis=0))
    return rotated_data

def compute_angle(pt1, pt2):
    """ Compute the angle between pt1 and pt2 with respect to the origin
        Input:
          pt1: 1x3 list
          pt2: 1x3 list
    """
    a=np.array(pt1)
    c=np.array(pt2)
    b = np.array([0, 0, 0])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

def compute_angle_tensor(pts1, pts2):
    """ Compute the angle between pt1 and pt2 with respect to the origin
        Input:
          pts1: batch_size x 1 x 3 tensor
          pts2: batch_size x 1 x 3 tensor
    """
    b = tf.constant([0.,0.,0.])
    angle_diff = []
    for pt1, pt2 in zip(pts1, pts2):
        ba = tf.subtract(pt1, b)
        bc = tf.subtract(pt2, b)
        cosine_angle = tm.divide(tf.tensordot(ba, bc, 1), tm.multiply(tf.norm(ba), tf.norm(bc)))
        angle = tm.acos(cosine_angle)
        angle_diff.append(tf.cast(angle, tf.float32))
    return tf.stack(angle_diff, axis=0)

def sunflower_distri(num_pts): 
    """ Compute [num_pts] of points that conforms a sunflower distribution
    with golden ratio (1 + sqrt(5))/2
        Input:
          num_pts: number of points, int
    """
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    pts=np.concatenate((np.expand_dims(x, axis=1), np.expand_dims(y, axis=1), np.expand_dims(z, axis=1)), axis=1)
    return pts

def normal_distri(num_pts):
    """ Compute [num_pts] of points following https://stats.stackexchange.com/users/919/whuber
    with x, y, z conforming to normal distributions
        Input:
          num_pts: number of points, int
    """
    pts = np.random.normal(size = (num_pts, 3))
    pts_normed = sklearn.preprocessing.normalize(pts, axis=1, norm='l2')
    return pts_normed

def re(x, y, z):
    return tf.stack([x,y,z], axis=0)

def map_label_to_angle_18(label):
    """
        return a list of tensors, each is the rotation angle corresponding to label [l]
    """
    batch_size = label.shape[0]
    label = tf.dtypes.cast(label, dtype=tf.float32)
    label = tf.split(label, batch_size, 0)
    label = [tf.squeeze(l) for l in label]
    angles = []
    for l in label:
        cond0 = tm.equal(tflt(0), l)
        cond1 = tm.logical_and(tm.less_equal(tflt(1), l), tm.less_equal(l, tflt(3)))
        cond2 = tm.logical_and(tm.less_equal(tflt(4), l), tm.less_equal(l, tflt(5)))
        cond3 = tm.logical_and(tm.less_equal(tflt(6), l), tm.less_equal(l, tflt(9)))
        cond4 = tm.logical_and(tm.less_equal(tflt(10), l), tm.less_equal(l, tflt(13)))
        cond5 = tm.logical_and(tm.less_equal(tflt(14), l), tm.less_equal(l, tflt(17)))

        call0= re(0., 0., 0.)
        call1= re(l*np.pi/2, 0, 0)
        call2 = re(0, 0, (l*2-7)*np.pi/2)
        call3 = re((l*2-11)*np.pi/4, 0, 0)
        call4 = re(0, 0, (l*2-19)*np.pi/4)
        call5 = re(np.pi/2, 0, (l*2-27)*np.pi/4)

        conditions = [cond0, cond1, cond2, cond3, cond4, cond5]
        callables = [call0, call1, call2, call3, call4, call5]
        angle = nested_tf_cond(conditions, callables)
        angles.append(angle)
    return angles

def map_label_to_angle_32(label):
    """
        return a list of tensors, each is the rotation angle corresponding to label [l]
    """
    batch_size = label.shape[0]
    label = tf.dtypes.cast(label, dtype=tf.float32)
    label = tf.split(label, batch_size, 0)
    label = [tf.squeeze(l) for l in label]
    angles = []
    for l in label:
        cond0 = tm.equal(tflt(0), l)
        cond1 = tm.logical_and(tm.less_equal(tflt(1), l), tm.less_equal(l, tflt(5)))
        cond2 = tm.logical_and(tm.less_equal(tflt(6), l), tm.less_equal(l, tflt(10)))
        cond3 = tm.logical_and(tm.less_equal(tflt(11), l), tm.less_equal(l, tflt(20)))
        cond4 = tm.logical_and(tm.less_equal(tflt(21), l), tm.less_equal(l, tflt(25)))
        cond5 = tm.logical_and(tm.less_equal(tflt(26), l), tm.less_equal(l, tflt(30)))
        cond6 = tm.equal(tflt(31), l)

        call0= re(0., 0., 0.)
        call1= re(0.646, (l-1)*np.pi/2.5, 0)
        call2 = re(1.108, (l*2-11)*np.pi/5, 0)
        call3 = re(np.pi/2, (l-11)*np.pi/5, 0)
        call4 = re(2.033, (l*2-11)*np.pi/5, 0)
        call5 = re(2.496, (l-26)*np.pi/2.5, 0)
        call6 = re(np.pi, 0, 0)

        conditions = [cond0, cond1, cond2, cond3, cond4, cond5, cond6]
        callables = [call0, call1, call2, call3, call4, call5, call6]
        angle = nested_tf_cond(conditions, callables)
        angles.append(angle)
    return angles

def get_rotated_angle_diff(num_angles, label1, label2):
    """
        get the angle between two rotation directions, 
        represented as label1 and label2
        Input:
            label1:[batch_size] tensor
            label2:[batch_size] tensor
        Return:
            [batch_size] list of tensors
    """
    label1 = tf.to_int64(label1)
    label2 = tf.to_int64(label2)
    if num_angles <= 18:
        angles1 = map_label_to_angle_18(label1)
        angles2 = map_label_to_angle_18(label2)
    elif num_angles == 32:
        angle1 = map_label_to_angle_32(label1)
        angle2 = map_label_to_angle_32(label2)
    else:
        raise NotImplementedError
    return compute_angle_tensor(angles1, angles2)

def _rotation_multiprocessing_worker(func, batch_data, label, return_dict, idx, *args):
    result = func(batch_data, label, *args)
    return_dict[idx] = result

def rotation_multiprocessing_wrapper(func, batch_data, label, *args, num_workers=8):
    """
    A wrapper for doing rotation using multiprocessing
    Input:
        func: a function for rotating on batch data, e.g. rotate_point_by_label
        batch_data: B*N*3 numpy array
        label: B length numpy array
    Returns:
        B*N*3 numpy array
    """
    batch_size = batch_data.shape[0] // num_workers
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size if i < num_workers - 1 else batch_data.shape[0]
        # print(f"[INFO] batch {i}, start index {start_idx}, end index {end_idx}")
        cur_data = batch_data[start_idx: end_idx]
        cur_label = label[start_idx: end_idx]
        p = multiprocessing.Process(target=_rotation_multiprocessing_worker,
                                    args=(func, cur_data, cur_label, return_dict, i, *args))
        p.start()
        jobs.append(p)
    for proc in jobs:
        proc.join()
    result = np.concatenate([return_dict[i] for i in range(num_workers)])
    # print("[INFO] rotated dimension:", result.shape)
    return result

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def load_mat_keypts(files, keypts_path, NUM_POINT):
    keypts_dict = loadmat(keypts_path)
    keypts = keypts_dict['keypts']
    # print('\n')
    # print(keypts[1])
    # print('\n')
    # print(keypts[2])
    # print('\n')
    # print(keypts[3])
    names = [n[0][0] for n in keypts_dict['modelname']]
    data = None
    label = None
    data = []
    labels = []
    for f in files:
        x = loadmat(f)
        pts = x['pts']

        # pts = [pt[0][0:NUM_POINT, :] for pt in pts]
        # pts = [pt[0] for pt in pts]
        # pts = np.stack(pts, axis = 0)
        model_names = [n[0][0] for n in x['modelList']]
        indices = [names.index(n) for n in model_names]
        keypts_labels = np.asarray([keypts[i] for i in indices])
        
        for i in range(len(pts)):
            pt = pts[i][0]
            if pt.shape[0] < NUM_POINT:
                continue  # skip invalid object
            keypt = keypts_labels[i]
            t = sum([1 for ke in keypt if ke != []])
            if t < 10:
                continue  # skip objects with fewer than 10 keypts
            label = np.zeros(NUM_POINT-t)
            label = np.concatenate((label, np.ones(t)))
            for kpt in keypt:
                if kpt != [] and kpt[0] != []:
                    kpt_reshaped = np.reshape(kpt[0], (1, 3))

                    if True: 
                        # From the visualization, it seems keypoints and 3d models are not
                        # aligned. So we rotate along z-axis by -90 degrees to align them.
                        # Manually check keypoint_visualize.ipynb for results.
                        rotation_angle = -np.pi/2
                        cosval = np.cos(rotation_angle)
                        sinval = np.sin(rotation_angle)
                        rotation_matrix = np.array([[cosval, -sinval, 0],
                                                    [sinval, cosval, 0],
                                                    [0, 0, 1]])
                        kpt_reshaped = np.matmul(kpt_reshaped, rotation_matrix)
                    
                    pt = np.concatenate((pt, kpt_reshaped))
            pt = pt[-NUM_POINT:, :]
            data.append(pt)
            labels.append(label)
    data = np.asarray(data)
    labels = np.asarray(labels)
    
    return (data, labels)



    