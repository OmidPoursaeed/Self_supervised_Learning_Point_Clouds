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
	labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 3, 3))
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
		net = tf_util.fully_connected(net, 6, activation_fn=None, scope='fc3')

	return net, end_points


# u, v batch*n
def cross_product(u, v):
    batch_size = u.shape[0]
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
        label: B*3*3, in rotation matrix"""
    pred_rotation_matrix = compute_rotation_matrix_from_ortho6d(pred)  # B*3*3

    loss = tf.math.pow(label - pred_rotation_matrix, 2)
    loss = tf.math.reduce_mean(loss)

    return loss



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
