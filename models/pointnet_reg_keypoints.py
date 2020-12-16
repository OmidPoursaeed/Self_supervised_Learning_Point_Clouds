import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 10, 3))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None, num_classes=None, use_input_trans=True, use_feature_trans=True):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    num_keypoints = 10
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
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                    scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                            scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                    scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                            scope='dp2')
        net = tf_util.fully_connected(net, 3 * num_keypoints, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001, use_trans_loss=True, use_angle_loss=False):
    """ pred: B * (num_kpts * 3),
        label: B * num_kpts * 3 """
    pred_reshaped = tf.reshape(pred, [pred.shape[0], -1, 3])
    print(f'pred shape {pred_reshaped.shape}')
    print(f'label shape {label.shape}')

    dist1, idx1, dist2, idx2 = nn_distance_cpu(pred_reshaped, label)
    loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
    return loss


def nn_distance_cpu(pc1, pc2):
    '''
    Input:
        pc1: float TF tensor in shape (B,N,C) the first point cloud
        pc2: float TF tensor in shape (B,M,C) the second point cloud
    Output:
        dist1: float TF tensor in shape (B,N) distance from first to second
        idx1: int32 TF tensor in shape (B,N) nearest neighbor from first to second
        dist2: float TF tensor in shape (B,M) distance from second to first
        idx2: int32 TF tensor in shape (B,M) nearest neighbor from second to first
    '''
    N = pc1.get_shape()[1].value
    M = pc2.get_shape()[1].value
    pc1_expand_tile = tf.tile(tf.expand_dims(pc1,2), [1,1,M,1])
    pc2_expand_tile = tf.tile(tf.expand_dims(pc2,1), [1,N,1,1])
    pc_diff = pc1_expand_tile - pc2_expand_tile # B,N,M,C
    pc_dist = tf.reduce_sum(pc_diff ** 2, axis=-1) # B,N,M
    dist1 = tf.reduce_min(pc_dist, axis=2) # B,N
    idx1 = tf.argmin(pc_dist, axis=2) # B,N
    dist2 = tf.reduce_min(pc_dist, axis=1) # B,M
    idx2 = tf.argmin(pc_dist, axis=1) # B,M
    return dist1, idx1, dist2, idx2


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
