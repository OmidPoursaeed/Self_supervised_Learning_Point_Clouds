import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
#tf.debugging.set_log_device_placement(True)
import socket
import importlib
import os
import sys
from utils import pc_util
import skimage.io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
from provider import (rotation_multiprocessing_wrapper,
                      rotate_point_by_label,
                      rotate_point_by_label_32,
                      rotate_point_by_label_54,
                      rotate_point_by_label_n,
                      )
import tf_util
from sklearn.model_selection import train_test_split
import data_loader
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_seg', help='Model name [default: pointnet_cls_rotation]')
parser.add_argument('--log_dir', default='log_segmentation', help='Log dir [default: log_segmentation]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--no_transformation_loss', action='store_true', help='Disable transformation loss')
parser.add_argument('--no_input_transform', action='store_true', help='Disable input transformation layer')
parser.add_argument('--no_feature_transform', action='store_true', help='Disable feature transformation layer')
parser.add_argument('--dataset', type=str, choices=['shapenet', 'modelnet', 'keypoint', 'keypoint_10class'], default='keypoint', help='dataset to train on [default: modelnet]')
parser.add_argument('--use_regression', action='store_true', help='use regression')
parser.add_argument('--resume', type=str, default='', help='checkpoint to restore from')
parser.add_argument('--ten_class', action='store_true', help='treat the problem as 11 class classification')

FLAGS = parser.parse_args()

if FLAGS.no_feature_transform:
    FLAGS.no_transformation_loss = True


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
USE_TRANS_LOSS = not FLAGS.no_transformation_loss
USE_INPUT_TRANS = not FLAGS.no_input_transform
USE_FEATURE_TRANS = not FLAGS.no_feature_transform
USE_REGRESSION = FLAGS.use_regression
RESUME = FLAGS.resume
TEN_CLASS = FLAGS.ten_class

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
log_para_name = f"{FLAGS.model}_{FLAGS.dataset}_batch_{FLAGS.batch_size}_opt_{FLAGS.optimizer}_lr_{FLAGS.learning_rate}_trans_loss_{USE_TRANS_LOSS}_input_trans_{USE_INPUT_TRANS}_feature_trans_{USE_FEATURE_TRANS}" 
LOG_DIR = os.path.join(LOG_DIR, log_para_name)

# Find a directory not in use
while os.path.exists(LOG_DIR):
    idx = LOG_DIR.rfind('_')
    tail = LOG_DIR[idx+1:]
    if tail.isdigit():
        LOG_DIR = LOG_DIR[:idx+1] + str(int(tail)+1)
    else:
        LOG_DIR = LOG_DIR + '_1'
os.makedirs(LOG_DIR)

print("Logging to", LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_seg.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_rot.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

LOG_DIR_PCK = os.path.join(LOG_DIR, 'pck_output')
os.makedirs(LOG_DIR_PCK, exist_ok=True)
LOG_DIR_KPTS = os.path.join(LOG_DIR, 'keypoints')
os.makedirs(LOG_DIR_KPTS, exist_ok=True)

MAX_NUM_POINT = 2048
NUM_CLASSES = 11 if TEN_CLASS else 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

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
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, name='batch_num')
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, 
                                                is_training_pl, 
                                                bn_decay=bn_decay,
                                                num_classes=NUM_CLASSES,
                                                use_input_trans=USE_INPUT_TRANS,
                                                use_feature_trans=USE_FEATURE_TRANS,
                                            )
            loss = MODEL.get_loss(pred, labels_pl, end_points, use_angle_loss=False, use_trans_loss=USE_TRANS_LOSS)
            tf.summary.scalar('loss', loss)

            if not USE_REGRESSION:
                correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)


        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        if RESUME:
            variables = slim.get_variables_to_restore()
            variables_to_restore = [v for v in variables if ('fc3' not in v.name) and ('segmentation_branch' not in v.name) and ('batch_num' not in v.name)]
            saver_base = tf.train.Saver(variables_to_restore)
            saver_base.restore(sess, RESUME)
        
        # Start the actual training
        X_train, X_test, y_train, y_test = data_loader.get_pointcloud(dataset=FLAGS.dataset,
                                                           NUM_POINT=NUM_POINT)

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            # Train
            rotate_and_train(X_train, y_train, sess, ops, train_writer, is_training=True)
            
            # Eval
            rotate_and_eval(X_test, y_test, sess, ops, test_writer, is_training=False, epoch=epoch)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def pred_to_top10(pred):
    """
    Convert pred output to keypoints
    Input:
        pred: batch_size * NUM_POINT * NUM_CLASSES
    Output:
        batch_size * NUM_POINT, with 10 in each row marked 1, others 0
    """
    result = np.zeros((pred.shape[0], pred.shape[1]))
    if not TEN_CLASS:
        # argsort twice to obtain the ranking
        # See: https://stackoverflow.com/a/6266510
        order = np.argsort(np.argsort(-pred[:, :, 1], axis=1), axis=-1)
        result[order < 10] = 1
    else:
        for idx in range(pred.shape[0]):  # going through the batch
            for i in range(10):
                order = np.argsort(-pred[idx, :, i+1])
                j = 0
                while result[idx, order[j]] == 1:
                    j += 1
                result[idx, order[j]] = 1
    assert np.sum(result) == pred.shape[0] * 10
    return result


def visualize_keypoints(data, label_kpts, pred_kpts, epoch):
    """
    Plot and save predicted and ground truth keypoints
    """
    idx = np.random.randint(data.shape[0])
    img = pc_util.point_cloud_three_views_with_keypoint(data[idx], label_kpts[idx], pred_kpts[idx])
    fname = os.path.join(LOG_DIR_KPTS, 'Kpts_epoch_%04d_idx_%05d.png' % (epoch, idx))
    skimage.io.imsave(fname, img)


def get_nearest_keypoints(data, pred):
    """
    Find the nearest points of pred in data
    Input:
        data: batch_size * NUM_POINT * 3
        pred: batch_size * 10 * 3
    Output:
        batch_size * 10 * 3
    """
    pred_kpts = np.reshape(pred, (data.shape[0], -1, 3))
    for batch_idx in range(data.shape[0]):
        dist_m = cdist(pred_kpts[batch_idx, :, :], data[batch_idx, :, :])
        min_indices = np.argmin(dist_m, axis=1)
        pred_kpts[batch_idx] = data[batch_idx, min_indices, :]
    return pred_kpts


def calculate_pck(data, label, pred, epoch):
    """
    Calculate and plot the pck curve under various thresholds
    """
    print(f'data shape {data.shape}')
    print(f'label shape {label.shape}')
    print(f'pred shape {pred.shape}')

    if not USE_REGRESSION:
        pred_top10 = pred_to_top10(pred)
        print(f'pred_top10 shape {pred_top10.shape}')
        pred_kpts = np.reshape(data[pred_top10 == 1], (data.shape[0], -1, 3))
        label_kpts = np.reshape(data[label == 1], (data.shape[0], -1, 3))
    else:
        pred_kpts = get_nearest_keypoints(data, pred)
        label_kpts = np.reshape(label, (data.shape[0], -1, 3))

    visualize_keypoints(data, label_kpts, pred_kpts, epoch)

    print(f'pred_kpts shape {pred_kpts.shape}')
    print(f'label_kpts shape {label_kpts.shape}')

    thresholds = np.linspace(0.0001, 0.1, num=200)
    accuracies = []
    for threshold in thresholds:
        correct_num = 0
        incorrect_num = 0
        for batch_idx in range(label_kpts.shape[0]):
            dist_m = cdist(label_kpts[batch_idx, :, :], pred_kpts[batch_idx, :, :])
            min_dist = np.min(dist_m, axis=1)
            correct = np.sum(min_dist <= threshold)
            correct_num += correct
            incorrect_num += (label_kpts.shape[1] - correct)
        accuracies.append(correct_num/(correct_num+incorrect_num))
    fig, ax = plt.subplots(dpi=150, figsize=(8,6))
    ax.plot(thresholds, accuracies)
    ax.grid(True, which='both')
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('% Correspondence')
    fig_path = os.path.join(LOG_DIR_PCK, 'Epoch_%04d.png' % (epoch))
    fig.savefig(fig_path)
    plt.close()
    log_path = os.path.join(LOG_DIR_PCK, 'Epoch_%04d.csv' % (epoch))
    result = np.array([thresholds, accuracies]).T
    np.savetxt(log_path, result, delimiter=",")

        
def rotate_and_train(current_data, current_label, sess, ops, train_writer, is_training):
    # Randomly generate the label, and rotate the data
    # current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
    current_label = np.squeeze(current_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0

    if USE_REGRESSION:
        current_label = current_data[current_label == 1]
        current_label = np.reshape(current_label, (-1, 10, 3))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        # Augment batched point clouds by rotation and jittering
        rotated_data = current_data[start_idx:end_idx, :, :]  # provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        jittered_data = provider.jitter_point_cloud(rotated_data)
        feed_dict = {ops['pointclouds_pl']: jittered_data,
                        ops['labels_pl']: current_label[start_idx:end_idx],
                        ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        # print(f'pred val shape {pred_val.shape}')
        # print(f'label val shape {current_label.shape}')

        if not USE_REGRESSION:
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
    
    log_string('Train mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('Train accuracy: %f' % (total_correct / float(total_seen) / NUM_POINT))


def rotate_and_eval(current_data, current_label, sess, ops, test_writer, is_training, epoch=None):
    total_correct = 0
    total_seen = 0
    loss_sum = 0

    current_label = np.squeeze(current_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    if USE_REGRESSION:
        current_label = current_data[current_label == 1]
        current_label = np.reshape(current_label, (-1, 10, 3))

    preds = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                        ops['labels_pl']: current_label[start_idx:end_idx],
                        ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        preds.append(pred_val)
        if not USE_REGRESSION:
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)

    preds = np.concatenate(preds, axis=0)
    print(f'preds shape {preds.shape}')

    calculate_pck(current_data[:num_batches*BATCH_SIZE], current_label[:num_batches*BATCH_SIZE], preds, epoch)
    
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen) / NUM_POINT))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
