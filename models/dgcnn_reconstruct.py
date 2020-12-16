import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util_reconstruct
from transform_nets_reconstruct import input_transform_net


def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None, weight_decay=0.0):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}
  k = 20

  adj = tf_util_reconstruct.pairwise_distance(point_cloud)
  nn_idx = tf_util_reconstruct.knn(adj, k=k)
  edge_feature = tf_util_reconstruct.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3, is_dist=True)
  point_cloud_transformed = tf.matmul(point_cloud, transform)

  input_image = tf.expand_dims(point_cloud_transformed, -1)
  adj = tf_util_reconstruct.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util_reconstruct.knn(adj, k=k)
  edge_feature_0 = tf_util_reconstruct.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

  out1 = tf_util_reconstruct.conv2d(edge_feature_0, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv1', bn_decay=bn_decay, is_dist=True)

  out2 = tf_util_reconstruct.conv2d(out1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

  net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)

  adj = tf_util_reconstruct.pairwise_distance(net_1)
  nn_idx = tf_util_reconstruct.knn(adj, k=k)
  edge_feature = tf_util_reconstruct.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

  out3 = tf_util_reconstruct.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

  out4 = tf_util_reconstruct.conv2d(out3, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv4', bn_decay=bn_decay, is_dist=True)

  net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)


  adj = tf_util_reconstruct.pairwise_distance(net_2)
  nn_idx = tf_util_reconstruct.knn(adj, k=k)
  edge_feature = tf_util_reconstruct.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

  out5 = tf_util_reconstruct.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

  net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)



  out7 = tf_util_reconstruct.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  net = tf_util_reconstruct.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')


  # MLP on global point cloud vector
  net = tf.reshape(net, [batch_size, -1])
  net = tf_util_reconstruct.fully_connected(net, 512, bn=True, is_training=is_training,
                                scope='fc1', bn_decay=bn_decay)
  net = tf_util_reconstruct.dropout(net, keep_prob=0.5, is_training=is_training,
                         scope='dp1')
  net = tf_util_reconstruct.fully_connected(net, 256, bn=True, is_training=is_training,
                                scope='fc2', bn_decay=bn_decay)
  net = tf_util_reconstruct.dropout(net, keep_prob=0.5, is_training=is_training,
                        scope='dp2')
  #net = tf_util_reconstruct.fully_connected(net, 40, activation_fn=None, scope='fc3')

  return net, end_points


def get_loss(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=40)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
