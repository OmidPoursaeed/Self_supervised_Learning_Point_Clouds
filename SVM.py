import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import socket
import importlib
import os
import os.path as osp
import sys
from pathlib import Path
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parent = Path(BASE_DIR).parent
sys.path.append(str(parent))
# sys.path.append(os.path.join(parent, 'reconstruct_space'))
# sys.path.append(os.path.join(parent, 'reconstruct_space', 'models'))
# sys.path.append(os.path.join(parent, 'reconstruct_space', 'utils'))

sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'latent_3d_points_py3'))
import provider
import tf_util
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.random import sample_without_replacement
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_loader

from src.autoencoder import Configuration as Conf
from src.point_net_ae import PointNetAutoEncoder
from src.tf_utils import reset_tf_graph

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn_svm', help='Model name: pointnet_cls_rot_svm')
parser.add_argument('--log_dir', default='log_svm', help='Log dir')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--model_path', default='./log_rotation/dgcnn_modelnet_rot_angles_6_batch_32_opt_adam_lr_0.001_trans_loss_False_input_trans_False_feature_trans_False_y_False/model.ckpt', help='model checkpoint file path')
parser.add_argument('--svm_c', type=float, default=1.0, help='Penalty parameter C of the error term in SVM')
parser.add_argument('--dataset', type=str, choices=['shapenet', 'modelnet', 'modelnet10'], default='modelnet', help='dataset to train on [default: modelnet]')
parser.add_argument('--ae_feature', action='store_true',  help='concatenate features from AE from [latent_3d_points_py3] or [pointnet-autoencoder')
parser.add_argument('--ae_path', type=str, default='./log_ae/all_class_ae_10', help='path to AE model')
parser.add_argument('--add_fc', action='store_true', help='add two fc before SVM')
parser.add_argument('--ss_feature', action='store_true', help='add feature from this paper https://papers.nips.cc/paper/9455-self-supervised-deep-learning-on-point-clouds-by-reconstructing-space. ')
parser.add_argument('--ss_path', type=str, default='../reconstruct_space/log/240_model.ckpt', help='path to self supervised model model')

FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path
C = FLAGS.svm_c

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_svm.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
if FLAGS.dataset == 'modelnet':
    NUM_CLASSES = 40
elif FLAGS.dataset == 'shapenet':
    NUM_CLASSES = 57
elif FLAGS.dataset == 'modelnet10':
    NUM_CLASSES = 10
else:
    raise NotImplementedError()

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(200000)
BN_DECAY_CLIP = 0.99

USE_INPUT_TRANS = False if 'input_trans_False' in MODEL_PATH else True
USE_FEATURE_TRANS = False if 'feature_trans_False' in MODEL_PATH else True

PERCENTAGES = [1, 100]

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
MODELNET10_TRAIN_FILE = 'data/ModelNet10/trainShuffled_Relabel.h5'
MODELNET10_TEST_FILE = 'data/ModelNet10/testShuffled_Relabel.h5'


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)   

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    pt_graph = tf.Graph()
    with pt_graph.as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

             # Get model
            pred, end_points = MODEL.get_model(pointclouds_pl,
                                                is_training_pl,
                                                bn_decay=bn_decay,
                                                use_input_trans=USE_INPUT_TRANS,
                                                use_feature_trans=USE_FEATURE_TRANS,
                                                )
    if FLAGS.ae_feature:
        feature_graph = tf.Graph()
        with feature_graph.as_default():
            # with tf.device('/gpu:'+str(GPU_INDEX)):
            ae_configuration = osp.join(FLAGS.ae_path, 'configuration')
            ae_conf = Conf.load(ae_configuration)
            ae_conf.experiment_name = 'all_class_ae'
            ae_conf.encoder_args['verbose'] = False
            ae_conf.decoder_args['verbose'] = False
            ae = PointNetAutoEncoder(ae_conf.experiment_name, ae_conf)

    if FLAGS.ss_feature:
        feature_graph = tf.Graph()
        with feature_graph.as_default():
            with tf.device('/gpu:'+str(GPU_INDEX)):
                model_ss = importlib.import_module('dgcnn_reconstruct') # import network module
               
                pointclouds_pl_ss, labels_pl_ss = model_ss.placeholder_inputs(BATCH_SIZE, NUM_POINT)
                is_training_pl_ss = tf.placeholder(tf.bool, shape=())

                # simple model
                pred_ss, end_points_ss = model_ss.get_model(pointclouds_pl_ss, is_training_pl_ss)
                # loss = model_ss.get_loss(pred, labels_pl, end_points)
                # all_nodes = feature_graph.get_operations()
                # for n in all_nodes:
                #     print(n.name)


                
                
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    #sess = tf.Session(config=config)

    sess = tf.Session(graph=pt_graph, config=config)

    if FLAGS.ae_feature or FLAGS.ss_feature:
        feature_sess = tf.Session(graph=feature_graph, config=config)


    # To fix the bug introduced in TF 0.12.1 as in
    # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
    #sess.run(init)

    # Restore weights
    with sess.as_default():
        with pt_graph.as_default():
            # with tf.device('/gpu:'+str(GPU_INDEX)):
            saver = tf.train.Saver()
            saver.restore(sess, MODEL_PATH)
            print("Restored previous weights")
            ops = {'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'step': batch}
            sys.stdout.flush()
            

    if FLAGS.ae_feature or FLAGS.ss_feature:
        with feature_sess.as_default():
            with feature_graph.as_default():
                # with tf.device('/gpu:'+str(GPU_INDEX)):
                # Load pre-trained AE
                # if FLAGS.ae_feature == 'latent_gan':
                if FLAGS.ae_feature:
                    ae.restore_model(FLAGS.ae_path, 1000, verbose=True)
                    print("loaded AE model")

                elif FLAGS.ss_feature:
                    feature_saver = tf.train.Saver()
                    
                    # print('\n\n\n\n\n\n')
                    # print_tensors_in_checkpoint_file(file_name=FLAGS.ss_path, all_tensors=False, tensor_name='')
                    ops_ss = {'pointclouds_pl': pointclouds_pl_ss,
                            'labels_pl': labels_pl_ss,
                            'is_training_pl': is_training_pl_ss,
                            'pred': pred_ss
                            }
                    feature_saver.restore(feature_sess, FLAGS.ss_path)
                    print("part prediction model + dgcnn loaded")


    with sess.as_default():
        X_train, X_test, y_train, y_test = data_loader.get_pointcloud(FLAGS.dataset)
        if FLAGS.ae_feature:
            result, labels = get_feature_svm(sess, ops, X_train, y_train,  ae, feature_sess)
        elif FLAGS.ss_feature:
            result, labels = get_feature_svm(sess, ops, X_train, y_train,  None, feature_sess, ops_ss)
        else:
            result, labels = get_feature_svm(sess, ops, X_train, y_train)
        
    if FLAGS.add_fc:
        feature_train = result
        fc_graph = tf.Graph()
        with fc_graph.as_default():
            
            feature_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, feature_train.shape[1]))
            labels_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE, ))

            is_training_pl = tf.placeholder(tf.bool, shape=())

            train_prediction, loss, optimizer = fc_svm(feature_pl, labels_pl, is_training=is_training_pl, layer_dim = (512, 128))

            fc_sess = tf.Session(graph=fc_graph, config=config)
            
            def train_eval_svm(feature, labels, is_training):
                file_size = feature.shape[0]
                num_batches = file_size // BATCH_SIZE
                loss_total = 0.0
                acc = 0.0
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx+1) * BATCH_SIZE
                    f = feature[start_idx:end_idx, :]
                    l = labels[start_idx:end_idx]
                 
                    _, lo, predictions = fc_sess.run([optimizer, loss, train_prediction], {is_training_pl: is_training, feature_pl: f, labels_pl: l} )
                    loss_total += lo
                    acc += svm_accuracy(predictions, l)
                loss_total = loss_total/num_batches
                acc = acc/num_batches
                return loss_total, acc

            with fc_sess.as_default():
                tf.initialize_all_variables().run()
                for step in range(10001):
                    train_loss, train_acc = train_eval_svm(feature_train, y_train, True)

                    
                    if step % 500 == 0:
                        print('step:{} train loss:{:.6f} train accuracy: {:.2f}'.format(
                                step, train_loss, train_acc))

                        if FLAGS.ae_feature:
                            feature_test = get_feature_svm(sess, ops, X_test, y_test,  ae, feature_sess)
                        elif FLAGS.ss_feature:
                            result = get_feature_svm(sess, ops, X_train, y_train,  ae, feature_sess, ops_ss)
                        else:
                            feature_test = get_feature_svm(sess, ops, X_test, y_test)
                        test_loss, test_acc = train_eval_svm(feature_test, y_test, False)

                        print('step:{} test loss:{:.6f} test accuracy: {:.2f}'.format(
                                step, test_loss, test_acc))

    else:
        accuracies = []
        percentages = []
        for percentage in PERCENTAGES:
            indices = sample_without_replacement(result.shape[0], int(result.shape[0]*percentage/100))
            # pred = fc_svm(result[indices], labels[indices], is_training_pl)
            # print(f"Percentage of training set:{percentage}, Train accuracy: {svm_accuracy(result[indices], labels[indices])}"
            clf = LinearSVC(penalty='l2', C=C, dual=False, max_iter=10000)
            clf.fit(result[indices], labels[indices])
            print(f"Percentage of training set:{percentage}, Train accuracy: {clf.score(result[indices], labels[indices])}")
            percentages.append(percentage)
            if FLAGS.ae_feature:
                accuracies.append(eval_one_epoch(sess, ops, clf, X_test, y_test,  ae, feature_sess))
            elif FLAGS.ss_feature:
                accuracies.append(eval_one_epoch(sess, ops, clf, X_test, y_test,  None, feature_sess, ops_ss))
            else:
                accuracies.append(eval_one_epoch(sess, ops, clf, X_test, y_test))

        plt.rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(dpi=150, figsize=(8,6))
        ax.semilogx(percentages, accuracies, marker='.')
        ax.set_xlabel('% of Labeled Data Used')
        ax.set_ylabel('Classification Accuracy')
        # ax.set(xlabel='% of Labeled Data Used', ylabel='Classification Accuracy')
        ax.set_xlim([1, 100])
        ax.grid(True, which='both')
        # ax.set_ylim([0,1])
        fig_path = os.path.join(os.path.split(MODEL_PATH)[0], f"svm_accuracy_C_{C}.png")
        fig.savefig(fig_path)
        fig.savefig(fig_path[:-3]+'svg')
        print(f"Figure saved to {fig_path}")

def ae_concat(data, result, ae, ae_sess):
    with ae_sess.as_default():
       
        file_size = data.shape[0]
        num_batches = file_size // BATCH_SIZE
        total_objects = num_batches * BATCH_SIZE
        current_data = data[0:total_objects,:,:]
        latent_codes = ae.get_latent_codes(current_data)
        #concat pointnet features and AE features
        result = np.concatenate((result, latent_codes), axis=1)
        # print(f"concatenated result shape {result.shape}"
        
    return result


def get_feature_svm(sess, ops, data, label, ae=None, feature_sess=None, ops_ss=None):
    """ ops: dict mapping from string to tf ops """
    with sess.as_default():
        result, labels = get_network_output(ops, sess, data, label)
    if FLAGS.ae_feature:
        result = ae_concat(data, result, ae, feature_sess)
    elif FLAGS.ss_feature:
        ss_feature, ss_labels = get_network_output(ops, feature_sess, data, label, True, ops_ss)
        print('feature.shape', ss_feature.shape)
        result = np.concatenate((result, ss_feature), axis=1)
    return result, label

def eval_one_epoch(sess, ops, clf, data, label, ae=None, feature_sess=None, ops_ss=None):
    """ ops: dict mapping from string to tf ops """
    with sess.as_default():
        result, labels = get_network_output(ops, sess, data, label)
    if FLAGS.ae_feature:
        result = ae_concat(data, result, ae, feature_sess)
    elif FLAGS.ss_feature:
        ss_feature, ss_labels = get_network_output(ops, feature_sess, data, label, True, ops_ss)
        result = np.concatenate((result, ss_feature), axis=1)
    
    test_acc = clf.score(result, labels)
    print("Test accuracy:", test_acc)
    return test_acc

def get_network_output(ops, sess, data, label, ss_feature=False, ops_ss=None):
    is_training = False
    
    preds = []
    labels = []
    
    current_data = data
    current_label = label

    current_data = current_data[:,0:NUM_POINT,:]
    # current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        # Augment batched point clouds by rotation and jittering
        batch_data = current_data[start_idx:end_idx, :, :]
        # rotated_data, _ = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        # jittered_data = provider.jitter_point_cloud(rotated_data)
        if ss_feature:
            feed_dict = {ops_ss['pointclouds_pl']: batch_data,
                            ops_ss['labels_pl']: current_label[start_idx:end_idx],
                            ops_ss['is_training_pl']: is_training,}
            pred_val = sess.run([ops_ss['pred']], feed_dict=feed_dict)

        else:
            feed_dict = {ops['pointclouds_pl']: batch_data,
                                ops['labels_pl']: current_label[start_idx:end_idx],
                                ops['is_training_pl']: is_training,}
            step, pred_val = sess.run([ops['step'], ops['pred']], feed_dict=feed_dict)
        preds.append(pred_val)
        labels.append(np.reshape(current_label[start_idx:end_idx], (-1, 1)))

    
    result = np.vstack(preds)
    if ss_feature:
        result = np.reshape(result, (-1, result.shape[-1]))
    labels = np.concatenate(labels, axis=None)

    return result, labels

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay



def fc_svm(feature, labels, is_training, layer_dim = (512, 128)):
    bn_decay = get_bn_decay(tf.Variable(0))
    regulation_rate = 5e-4
    delta = C
    with tf.variable_scope('fc_svm'):
        feature = tf_util.fully_connected(feature, layer_dim[0], bn=True, is_training=is_training,
                                        scope='fc1', bn_decay=bn_decay)
        
        feature = tf_util.fully_connected(feature, layer_dim[1], bn=True, is_training=is_training,
                                        scope='fc2', bn_decay=bn_decay)
        weights = tf.Variable(tf.truncated_normal([layer_dim[1], NUM_CLASSES]))
        biases = tf.Variable(tf.zeros([NUM_CLASSES]))
        logits = tf.matmul(feature, weights) + biases
        one_hot_labels = tf.one_hot(labels, NUM_CLASSES)
        y = tf.reduce_sum(logits * one_hot_labels, 1, keep_dims=True)

        loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, logits - y + delta), 1)) - delta
        loss += regulation_rate * tf.nn.l2_loss(weights)
        loss += regulation_rate * tf.nn.l2_loss(weights)

        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
        train_prediction = tf.nn.softmax(logits)

        return train_prediction, loss, optimizer

def svm_accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == labels)
            / predictions.shape[0])
    # X = net
    # Y = labels
    # example_id = np.array([str(i) for i in range(len(Y))])

    # x_column_name = 'x'
    # example_id_column_name = 'example_id'

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={x_column_name: X, example_id_column_name: example_id},
    #     y=Y,
    #     num_epochs=None,
    #     shuffle=True)

    # svm = tf.contrib.learn.SVM(
    #     example_id_column=example_id_column_name,
    #     feature_columns=(tf.contrib.layers.real_valued_column(
    #         column_name=x_column_name, dimension=128),),
    #     l2_regularization=0.1)

    # svm.fit(input_fn=train_input_fn, steps=10)
    # return svm


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
