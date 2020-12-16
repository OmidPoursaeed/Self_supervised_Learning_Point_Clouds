import tensorflow as tf
import numpy as np
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def get_transform_K(inputs, is_training, bn_decay=None, K=3):
    """ Transform Net, input is BxNx1xK gray image
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='tconv2', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K * K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [K * K], initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32) + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    # transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
    transform = tf.reshape(transform, [batch_size, K, K])
    return transform


def get_transform(point_cloud, is_training, bn_decay=None, K=3):
    """ Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1, 3], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='tconv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='tconv4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert (K == 3)
        weights = tf.get_variable('weights', [128, 3 * K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [3 * K], initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32) + tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    # transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
              batch_size, num_point, weight_decay, bn_decay=None, use_input_trans=False, use_feature_trans = False):
    """ ConvNet baseline, input is BxNx3 gray image """

    # segmentation network
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    if use_input_trans:
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
    else:
        point_cloud_transformed = point_cloud
    input_image = tf.expand_dims(point_cloud_transformed, -1)
    with tf.variable_scope('pointnet_cls_rotation'):
        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        out1 = net
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        out2 = net
    if use_feature_trans:
        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])
    else:
        net_transformed = net
    with tf.variable_scope('pointnet_cls_rotation'):
        net = tf_util.conv2d(net_transformed, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        out3 = net
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        out4 = net
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        out5 = net
        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='maxpool')

        out_max = net
    with tf.variable_scope('pointnet_part_seg') as sc:
        one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
        out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])

        expand = tf.tile(out_max, [1, num_point, 1, 1])
        concat = tf.concat(axis=3, values=[expand, out1, out2, out3, out4, out5])
        print(concat)
        net2 = tf_util.conv2d(concat, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                              bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay)
        net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp1')
        net2 = tf_util.conv2d(net2, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                              bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay)
        net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp2')
        net2 = tf_util.conv2d(net2, 128, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                              bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay)
        net2 = tf_util.conv2d(net2, part_num, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None,
                              bn=False, scope='seg/conv4', weight_decay=weight_decay)

        net2 = tf.reshape(net2, [batch_size, num_point, part_num])

    return tf.zeros((batch_size, cat_num)), net2, end_points


def get_loss(l_pred, seg_pred, label, seg, weight, end_points):
    #per_instance_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_pred, labels=label)
    #label_loss = tf.reduce_mean(per_instance_label_loss)
    per_instance_label_loss = tf.zeros(label.shape)
    label_loss = tf.reduce_mean(per_instance_label_loss)
    # size of seg_pred is batch_size x point_num x part_cat_num
    # size of seg is batch_size x point_num
    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg),
                                           axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    # Enforce the transformation as orthogonal matrix
    #transform = end_points['transform']  # BxKxK
    #K = transform.get_shape()[1].value
    #mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1])) - tf.constant(np.eye(K), dtype=tf.float32)
    #mat_diff_loss = tf.nn.l2_loss(mat_diff)

    #total_loss = weight * seg_loss + (1 - weight) * label_loss + mat_diff_loss * 1e-3
    total_loss = seg_loss
    return total_loss, label_loss, per_instance_label_loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res

if __name__=='__main__':
    with tf.Graph().as_default():
        in1 = tf.zeros((32,1024,1, 64))
        in2 = tf.zeros((32,1024,1, 64))
        in3 = tf.zeros((32,1024,1, 64))
        in4 = tf.zeros((32,1024,1, 128))
        in5 = tf.zeros((32,1024,1, 1024))
        in_max = tf.zeros((32,1,1, 1024))
        in_label = tf.ones((32,1,1, 16))
        train_true = tf.constant(True)
        #outputs = get_model(in1, in2, in3, in4, in5, in_max, in_label, train_true, 16, 15, 32, 1024, 0, 0.5)
        outputs = get_model(tf.zeros((32,1024,3)), in_label, train_true, 16, 15, 32, 1024, 0, 0.5)

        for v in tf.global_variables():
            print(v.name)
        print(outputs)
