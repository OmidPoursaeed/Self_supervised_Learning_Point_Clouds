import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
import socket
import importlib
import os
import sys

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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn_scoped', help='Model name [default: pointnet_cls_rotation]')
parser.add_argument('--log_dir', default='log_rotation', help='Log dir [default: log_rotation]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--num_angles', type=int, choices=[6, 18, 24, 32, 54, 100], default=18, help="Number of angles for rotation prediction [default: 18]")
parser.add_argument('--no_transformation_loss', action='store_true', help='Disable transformation loss')
parser.add_argument('--no_input_transform', action='store_true', help='Disable input transformation layer')
parser.add_argument('--no_feature_transform', action='store_true', help='Disable feature transformation layer')
parser.add_argument('--dataset', type=str, choices=['shapenet', 'modelnet', 'modelnet10', 'shapenet_chair'], default='modelnet', help='dataset to train on [default: modelnet]')
parser.add_argument('--all_directions', action='store_true', help='Use all directions instead of randomly sampling')
parser.add_argument('--enable_y_axis', action='store_true', help='Use y rotation as a label')
parser.add_argument('--num_y_rotation_angles', type=int, default=4, help='Number of rotation angles along the y-axis')
parser.add_argument('--use_angle_loss', action='store_true', help='whether to penalize more when predicted angles are further away')

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
ALL_DIRECTIONS = FLAGS.all_directions
USE_ANGLE_LOSS = FLAGS.use_angle_loss

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
log_para_name = f"{FLAGS.model}_{FLAGS.dataset}_rot_angles_{FLAGS.num_angles}_batch_{FLAGS.batch_size}_opt_{FLAGS.optimizer}_lr_{FLAGS.learning_rate}_trans_loss_{USE_TRANS_LOSS}_input_trans_{USE_INPUT_TRANS}_feature_trans_{USE_FEATURE_TRANS}_y_{FLAGS.enable_y_axis}" 
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
os.system('cp train_rotation_prediction.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_rot.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES_WITHOUT_Y = FLAGS.num_angles
ENABLE_Y_AXIS = FLAGS.enable_y_axis
NUM_Y_ROTATION_ANGLES = FLAGS.num_y_rotation_angles
Y_ANGLE_INCREMENT = 2 * np.pi / NUM_Y_ROTATION_ANGLES 
if ENABLE_Y_AXIS:
    NUM_CLASSES = FLAGS.num_angles * NUM_Y_ROTATION_ANGLES
    print(f"Enabling rotation along y-axis as labels, the actual number of classes now is {NUM_CLASSES}")
else:
    NUM_CLASSES = FLAGS.num_angles

BATCH_SIZE = BATCH_SIZE * NUM_CLASSES if ALL_DIRECTIONS else BATCH_SIZE

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
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, 
                                                is_training_pl, 
                                                bn_decay=bn_decay,
                                                num_angles=NUM_CLASSES,
                                                use_input_trans=USE_INPUT_TRANS,
                                                use_feature_trans=USE_FEATURE_TRANS,
                                            )
            loss = MODEL.get_loss(pred, labels_pl, end_points, use_angle_loss = USE_ANGLE_LOSS, use_trans_loss=USE_TRANS_LOSS)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
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

        
        # Start the actual training
        X_train, X_test, _, _ = data_loader.get_pointcloud(dataset=FLAGS.dataset,
                                                           NUM_POINT=NUM_POINT)

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            # Train
            rotate_and_train(X_train, sess, ops, train_writer, is_training=True)
            
            # Eval
            rotate_and_eval(X_test, sess, ops, test_writer, is_training=False)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

        
def rotate_and_train(current_data, sess, ops, train_writer, is_training):
    # Randomly generate the label, and rotate the data

    if ALL_DIRECTIONS:  # repeat the data NUM_CLASSES times in the batch dimension
        current_data = np.repeat(current_data, NUM_CLASSES, axis=0)
        print(f'Expanded data shape: {current_data.shape}')

    if ALL_DIRECTIONS:
        current_label = np.tile(np.arange(NUM_CLASSES), int(current_data.shape[0]//NUM_CLASSES))
    else:
        current_label = np.random.randint(NUM_CLASSES, size=current_data.shape[0])

    if ENABLE_Y_AXIS:
        # If y-axis rotations are used as labels, we do the y-axis rotation first before feeding the point cloud
        # to the general rotations
        y_angles = (current_label // NUM_CLASSES_WITHOUT_Y) * Y_ANGLE_INCREMENT
        y_angles = np.squeeze(y_angles)
        print(f'y_angles shape: {y_angles.shape}')
        current_data = rotation_multiprocessing_wrapper(provider.rotate_point_cloud_by_angle_list, current_data, y_angles)
    
    # When enabling y-axis, (current_label %  NUM_CLASSES_WITHOUT_Y) is the general label,
    # (current_label // NUM_CLASSES_WITHOUT_Y) is the y-axis label

    current_label_without_y = current_label % NUM_CLASSES_WITHOUT_Y
    if NUM_CLASSES_WITHOUT_Y == 6:
        current_data = rotation_multiprocessing_wrapper(rotate_point_by_label, current_data, current_label_without_y)
    elif NUM_CLASSES_WITHOUT_Y == 18:
        current_data = rotation_multiprocessing_wrapper(provider.rotate_point_by_label, current_data, current_label_without_y)
    elif NUM_CLASSES_WITHOUT_Y == 32:
        current_data = rotation_multiprocessing_wrapper(rotate_point_by_label_32, current_data, current_label_without_y)
    elif NUM_CLASSES_WITHOUT_Y == 54:
        current_data = rotation_multiprocessing_wrapper(rotate_point_by_label_54, current_data, current_label_without_y)
    elif NUM_CLASSES_WITHOUT_Y: #sunflower distribution
        current_data = rotation_multiprocessing_wrapper(rotate_point_by_label_n, current_data, current_label_without_y, NUM_CLASSES_WITHOUT_Y)
    else:
        raise(NotImplementedError())

    current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
    current_label = np.squeeze(current_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
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
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
    
    log_string('Train mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('Train accuracy: %f' % (total_correct / float(total_seen)))


def rotate_and_eval(current_data, sess, ops, test_writer, is_training):
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    # Randomly generate the label, and rotate the data
    current_label = np.random.randint(NUM_CLASSES, size=current_data.shape[0])

    if ENABLE_Y_AXIS:
        # If y-axis rotations are used as labels, we do the y-axis rotation first before feeding the point cloud
        # to the general rotations
        y_angles = (current_label // NUM_CLASSES_WITHOUT_Y) * Y_ANGLE_INCREMENT
        current_data = rotation_multiprocessing_wrapper(provider.rotate_point_cloud_by_angle_list, current_data, y_angles)
    
    current_label_without_y = current_label % NUM_CLASSES_WITHOUT_Y
    if NUM_CLASSES_WITHOUT_Y == 6:
        current_data = rotation_multiprocessing_wrapper(rotate_point_by_label, current_data, current_label_without_y)
    elif NUM_CLASSES_WITHOUT_Y == 18:
        current_data = rotation_multiprocessing_wrapper(provider.rotate_point_by_label, current_data, current_label_without_y)
    elif NUM_CLASSES_WITHOUT_Y == 32:
        current_data = rotation_multiprocessing_wrapper(rotate_point_by_label_32, current_data, current_label_without_y)
    elif NUM_CLASSES_WITHOUT_Y == 54:
        current_data = rotation_multiprocessing_wrapper(rotate_point_by_label_54, current_data, current_label_without_y)
    elif NUM_CLASSES_WITHOUT_Y: #sunflower distribution
        current_data = rotation_multiprocessing_wrapper(rotate_point_by_label_n, current_data, current_label_without_y, NUM_CLASSES_WITHOUT_Y)
    else:
        raise(NotImplementedError())

    current_label = np.squeeze(current_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                        ops['labels_pl']: current_label[start_idx:end_idx],
                        ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
