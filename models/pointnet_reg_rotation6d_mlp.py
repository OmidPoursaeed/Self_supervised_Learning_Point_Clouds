import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
import provider
from transform_nets import input_transform_net, feature_transform_net

OUT_CHANNEL = 6

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 3, 3))  #TODO: change dimension of label
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None, use_input_trans=True, use_feature_trans=True):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    if use_input_trans:
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
    else:
        point_cloud_transformed = point_cloud
    input_image = tf.expand_dims(point_cloud_transformed, -1)
    with tf.variable_scope('pointnet_cls_rotation'):
        net = tf_util.conv2d(input_image, 64, [1,3],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv2', bn_decay=bn_decay)

    if use_feature_trans:
        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])
    else:
        net_transformed = net
    with tf.variable_scope('pointnet_cls_rotation'):
        net = tf_util.conv2d(net_transformed, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                padding='VALID', scope='maxpool')

        net = tf.reshape(net, [batch_size, -1])
        
        net = tf_util.fully_connected(net, 512, activation_fn=tf.nn.leaky_relu, scope='fc1')
        net = tf_util.fully_connected(net, OUT_CHANNEL, activation_fn=None, scope='fc3')
        
    return net, end_points


# u, v batch*n
def cross_product(u, v):
    batch_size = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = tf.concat((tf.reshape(i, (batch_size,1)),
                     tf.reshape(j, (batch_size,1)),
                     tf.reshape(k, (batch_size,1))),
                    1) #batch*3
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = tf.linalg.l2_normalize(x_raw, axis=1) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = tf.linalg.l2_normalize(z, axis=1)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = tf.reshape(x, (-1,3,1))
    y = tf.reshape(y, (-1,3,1))
    z = tf.reshape(z, (-1,3,1))
    matrix = tf.concat((x,y,z), 2) #batch*3*3
    return matrix

def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*6, in 6d representation
        label: B*3, in rotation matrix"""
    pred_rotation_matrix = compute_rotation_matrix_from_ortho6d(pred)  # B*3*3

    loss = tf.math.pow(label - pred_rotation_matrix, 2)
    loss = tf.math.reduce_mean(loss)

    return loss


def get_exp_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*4,
        label: B*4, """
    label_axis = label[:, 0:3]
    label_angles = label[:, 3:4]

    label_real = tf.math.cos(label_angles/2)
    label_img = tf.math.sin(label_angles/2)  * label_axis
    print(f'label_read shape {label_real.shape}')
    print(f'label_img shape {label_img.shape}')

    label_quat = tf.concat((label_real, label_img), axis=1)
    print(f'label_quat shape {label_quat.shape}')

    # normalize the prediction axis and angles
    pred_exp = tf.math.exp(pred)
    pred_quat = tf.math.l2_normalize(pred_exp, axis=-1)

    dot_product = tf.reduce_sum(tf.math.multiply(pred_quat, label_quat), axis=1, keepdims=True)
    dot_product = tf.reduce_mean(dot_product)
    regression_loss = 1 - dot_product

    tf.summary.scalar('regression loss', regression_loss)

    return regression_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
