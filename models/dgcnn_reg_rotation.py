import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from provider import get_rotated_angle_diff
from transform_nets import edge_feature_transform_net, feature_transform_net


def placeholder_inputs(batch_size, num_point):
	pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
	labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 4))
	return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, num_angles=None, bn_decay=None, use_input_trans=False, use_feature_trans=False):
	""" Classification PointNet, input is BxNx3, output Bx40 """

	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value
	end_points = {}
	k = 20

	adj_matrix = tf_util.pairwise_distance(point_cloud)
	nn_idx = tf_util.knn(adj_matrix, k=k)
	edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

	if use_input_trans:
		with tf.variable_scope('transform_net1') as sc:
			transform = edge_feature_transform_net(edge_feature, is_training, bn_decay, K=3)
		point_cloud_transformed = tf.matmul(point_cloud, transform)
	else:
		point_cloud_transformed = point_cloud


	adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
	nn_idx = tf_util.knn(adj_matrix, k=k)
	edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

	net = tf_util.conv2d(edge_feature, 64, [1,1],
						padding='VALID', stride=[1,1],
						bn=True, is_training=is_training,
						scope='dgcnn1', bn_decay=bn_decay)
	net = tf.reduce_max(net, axis=-2, keep_dims=True)
	net1 = net

	adj_matrix = tf_util.pairwise_distance(net)
	nn_idx = tf_util.knn(adj_matrix, k=k)
	edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

	net = tf_util.conv2d(edge_feature, 64, [1,1],
						padding='VALID', stride=[1,1],
						bn=True, is_training=is_training,
						scope='dgcnn2', bn_decay=bn_decay)
	net = tf.reduce_max(net, axis=-2, keep_dims=True)
	net2 = net

	adj_matrix = tf_util.pairwise_distance(net)
	nn_idx = tf_util.knn(adj_matrix, k=k)
	edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  

	net = tf_util.conv2d(edge_feature, 64, [1,1],
						padding='VALID', stride=[1,1],
						bn=True, is_training=is_training,
						scope='dgcnn3', bn_decay=bn_decay)
	net = tf.reduce_max(net, axis=-2, keep_dims=True)
	net3 = net

	adj_matrix = tf_util.pairwise_distance(net)
	nn_idx = tf_util.knn(adj_matrix, k=k)
	edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
	
	net = tf_util.conv2d(edge_feature, 128, [1,1],
						padding='VALID', stride=[1,1],
						bn=True, is_training=is_training,
						scope='dgcnn4', bn_decay=bn_decay)
	net = tf.reduce_max(net, axis=-2, keep_dims=True)
	net4 = net
	with tf.variable_scope('dgcnn'):
		net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1], 
							padding='VALID', stride=[1,1],
							bn=True, is_training=is_training,
							scope='agg', bn_decay=bn_decay)
		
		net = tf.reduce_max(net, axis=1, keep_dims=True) 

		# MLP on global point cloud vector
		net = tf.reshape(net, [batch_size, -1]) 
		net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
										scope='fc1', bn_decay=bn_decay)
		net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
								scope='dp1')
		net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
										scope='fc2', bn_decay=bn_decay)
		net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
								scope='dp2')
		net = tf_util.fully_connected(net, 4, activation_fn=None, scope='fc3')

	return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
	""" pred: B*4,
        label: B*4 """
	pred_axis = pred[:, 0:3]
	pred_angles = pred[:, 3:4]
	label_axis = label[:, 0:3]
	label_angles = label[:, 3:4]

    # normalize the prediction axis and ./data/shape_net_core_uniform_samples_2048/angles
	pred_axis = tf.math.l2_normalize(pred_axis, axis=-1)
    # TODO: normalize the angles?
	
	dot_product = tf.reduce_sum(tf.math.multiply(pred_axis, label_axis), axis=1, keepdims=True)
	axis_loss = 1 - dot_product

    # axis_loss = tf.compat.v1.losses.mean_squared_error(labels=label_axis, predictions=pred_axis)
	axis_loss = tf.reduce_mean(axis_loss)
	tf.summary.scalar('axis loss', axis_loss)
	angles_loss = tf.compat.v1.losses.mean_squared_error(labels=label_angles, predictions=pred_angles)
	angles_loss = tf.reduce_mean(angles_loss)
	tf.summary.scalar('angles loss', angles_loss)
	
	axis_loss_weight = 0.5  # TODO: change this weight
	
	regression_loss = axis_loss * axis_loss_weight + angles_loss * (1-axis_loss_weight)
	tf.summary.scalar('regression loss', regression_loss)
	
	return regression_loss


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
		# loss = get_loss(logits, label_pl, None)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			feed_dict = {input_pl: input_feed, label_pl: label_feed}
			res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
