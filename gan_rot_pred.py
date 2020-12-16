import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import os.path as osp
import matplotlib.pylab as plt
import importlib

# import tensorflow.contrib 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

from latent_3d_points_py3.src.autoencoder import Configuration as Conf
from latent_3d_points_py3.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder
from latent_3d_points_py3.src.general_utils import plot_3d_point_cloud
from latent_3d_points_py3.src.tf_utils import reset_tf_graph

from latent_3d_points_py3.src.vanilla_gan import Vanilla_GAN
from latent_3d_points_py3.src.w_gan_gp import W_GAN_GP
from latent_3d_points_py3.src.w_gan_gp_rot import W_GAN_GP_ROT
from latent_3d_points_py3.src.w_gan_gp_shared_weights import W_GAN_GP_SHARED_WEIGHTS
from latent_3d_points_py3.src.generators_discriminators import point_cloud_generator, mlp_discriminator, leaky_relu


from latent_3d_points_py3.src.evaluation_metrics import minimum_mathing_distance, jsd_between_point_cloud_sets, coverage

from latent_3d_points_py3.src.in_out import snc_category_to_synth_id, load_all_point_clouds_under_folder

parser = argparse.ArgumentParser(description='Arguments for GAN with rotation loss')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--experiment_name', default='gan', help='Model name [default: pointnet_cls_rotation]')
parser.add_argument('--top_out_dir', default='./log_gan_rot', help='Log dir [default: log_gan_rot]')
# parser.add_argument('--top_in_dir', default='./latent_3d_points_py3/data/shape_net_core_uniform_samples_2048/', help='Log dir [default: log_rotation]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--class_number', type=int, default=0, help='The index of class to train on, 0...39, see ./data/modelnet40_ply_hdf5_2048/shape_names.txt')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--init_lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--num_angles', type=int, choices=[8, 20, 32, 54, 100], default=8, help="Number of angles for rotation prediction [default: 18]")
parser.add_argument('--use_transformation_loss', action='store_true', help='Disable transformation loss')
parser.add_argument('--use_input_transform', action='store_true', help='Disable input transformation layer')
parser.add_argument('--use_feature_transform', action='store_true', help='Disable feature transformation layer')
parser.add_argument('--beta', type=float, default=0.5, help='ADAMs momentum')
parser.add_argument('--plot_train_curve', action='store_false', help='Whether to plot training G & D loss curve')
parser.add_argument('--use_wgan', action='store_false', help='Wasserstein with gradient penalty, or not?')
parser.add_argument('--save_gan_model', action='store_false', help='Save checkpoint for each epoch')
parser.add_argument('--save_synthetic_samples', action='store_false', help='If true, every saver_step epochs we produce & save synthetic pointclouds')
parser.add_argument('--add_rotation_loss', action='store_false', help='If true, add rotation loss when updating G and D')
parser.add_argument('--weight_rotation_loss_d', type=float, default=0.05, help='d_loss += rotation_loss * weight_rotation_loss_d')
parser.add_argument('--weight_rotation_loss_g', type=float, default=0.05, help='g_loss += rotation_loss * weight_rotation_loss_g')
parser.add_argument('--model', default='pointnet_cls_rotation', help='Model name [default: pointnet_cls_rotation]')
parser.add_argument('--lr_pred', type=float, default=0.001, help='Initial learning rate for rotation prediction[default: 0.001]')
parser.add_argument('--share_weights', action='store_true', help='Use a shared weights model between discriminator and rotation prediction')
parser.add_argument('--restore_model', action='store_true', help='Restore Model from pretrained GAN')
parser.add_argument('--pretrained_model', type=str, default='wgan_rotLoss_False_angles_8_rotWeightsDG_0.0_0.0', help='which pretrained model to restore')
parser.add_argument('--ms_task', action='store_false', help='whether to use the methdo introduced in https://arxiv.org/abs/1911.06997')

# python gan_rot_pred.py --weight_rotation_loss_d 0.2 --use_wgan --add_rotation_loss --save_synthetic_samples --plot_train_curve 

flags = parser.parse_args()

BATCH_SIZE = flags.batch_size
NUM_POINT = flags.num_point
MAX_EPOCH = flags.max_epoch
GPU_INDEX = flags.gpu

CLASS_NUMBER = flags.class_number
USE_WGAN = flags.use_wgan
INIT_LR = flags.init_lr
TOP_OUT_DIR = flags.top_out_dir
gan_prefix = 'w' if USE_WGAN else 'raw_'
BETA = flags.beta
SAVE_GAN_MODEL = flags.save_gan_model
PLOT_TRAIN_CURVE = flags.plot_train_curve
SAVE_SYNTHETIC_SAMPLES = flags.save_synthetic_samples
NUM_ANGLES = flags.num_angles
SHARE_WEIGHTS = False  # TODO: fix argparsing
ADD_ROTATION_LOSS = flags.add_rotation_loss
if ADD_ROTATION_LOSS:
    EXPERIMENT_NAME = f'{gan_prefix}{flags.experiment_name}_rotLoss_{ADD_ROTATION_LOSS}_angles_{NUM_ANGLES}_rotWeightsDG_{flags.weight_rotation_loss_d}_{flags.weight_rotation_loss_d}'
else:
    EXPERIMENT_NAME = f'{gan_prefix}{flags.experiment_name}_rotLoss_{ADD_ROTATION_LOSS}_angles_{NUM_ANGLES}'

MODEL_SAVER_ID = 'models.ckpt'


# Optimization parameters
noise_params = {'mu':0, 'sigma': 0.2}
noise_dim = 128
n_out = [NUM_POINT, 3] # Dimensionality of generated samples.


# Load point-clouds.
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


current_data, current_label = provider.loadDataFile(TRAIN_FILES[0]) 
for fn in range(1, len(TRAIN_FILES)):
    tmp_data, tmp_label = provider.loadDataFile(TRAIN_FILES[fn])
    current_data = np.concatenate((current_data, tmp_data), axis=0)
    current_label = np.concatenate((current_label, tmp_label), axis=0)
current_data = current_data[:,0:NUM_POINT,:]
current_label = np.squeeze(current_label)

#select indices of label==0
train_class_idxs = np.array([i for i, e in enumerate(current_label) if e == CLASS_NUMBER])
current_data = current_data[train_class_idxs, :, :] #(625, 1024, 3)
current_label = current_label[train_class_idxs] #(625, 1)
all_pc_data = PointCloudDataSet(point_clouds=current_data, labels=current_label, copy=True, init_shuffle=True)

discriminator = mlp_discriminator
generator = point_cloud_generator

if SAVE_GAN_MODEL:
    train_dir = osp.join(TOP_OUT_DIR, 'checkpoints', EXPERIMENT_NAME)
    create_dir(train_dir)


if SAVE_SYNTHETIC_SAMPLES:
    synthetic_data_out_dir = osp.join(TOP_OUT_DIR, 'synthetic_samples/', EXPERIMENT_NAME, 'npy_files')
    synthetic_img_save_dir = osp.join(TOP_OUT_DIR, 'synthetic_samples/', EXPERIMENT_NAME, 'imgs')
    create_dir(synthetic_data_out_dir)
    create_dir(synthetic_img_save_dir)

eventfile_dir = osp.join(TOP_OUT_DIR, 'eventfile', EXPERIMENT_NAME)
create_dir(eventfile_dir)

reset_tf_graph()

if USE_WGAN:
    disc_kwargs = {'b_norm': False}
    gan_kwargs = {'init_lr': INIT_LR, 
                'n_out': n_out, 
                'noise_dim': noise_dim, 
                'flags': flags, 
                'batch_size_value': BATCH_SIZE,
                'lam': 10,
                }
    if ADD_ROTATION_LOSS:
        gan = W_GAN_GP_ROT(EXPERIMENT_NAME, discriminator, generator, 
                        disc_kwargs=disc_kwargs, gan_kwargs = gan_kwargs)
    elif SHARE_WEIGHTS:
        gan = W_GAN_GP_SHARED_WEIGHTS(EXPERIMENT_NAME, discriminator, generator, 
                                    disc_kwargs=disc_kwargs, gan_kwargs = gan_kwargs)
    else:
        gan = W_GAN_GP(EXPERIMENT_NAME, discriminator, generator, 
                        disc_kwargs=disc_kwargs, gan_kwargs = gan_kwargs)
else:    
    leak = 0.2
    disc_kwargs = {'non_linearity': leaky_relu(leak), 'b_norm': False}
    gan = Vanilla_GAN(EXPERIMENT_NAME, INIT_LR, n_out, noise_dim,
                      discriminator, generator, beta=BETA, disc_kwargs=disc_kwargs)

if flags.restore_model:
    restore_path = osp.join(TOP_OUT_DIR, 'checkpoints', flags.pretrained_model)
    gan.restore_model(restore_model, 250)

accum_syn_data = []
train_stats = []


def train():
    with tf.device('/gpu:'+str(GPU_INDEX)):
        writer = tf.summary.FileWriter(eventfile_dir, gan.sess.graph)
        print(f"Logging to {eventfile_dir}")
        for _ in range(MAX_EPOCH):
            epoch = int(gan.sess.run(gan.increment_epoch))
            losses, duration = gan._single_epoch_train(all_pc_data, epoch, BATCH_SIZE, noise_params, writer=None)
            if SAVE_GAN_MODEL and not epoch%100:
                checkpoint_path = osp.join(train_dir, MODEL_SAVER_ID)
                gan.saver.save(gan.sess, checkpoint_path, global_step=gan.epoch)

            if SAVE_SYNTHETIC_SAMPLES and not epoch%10:
                syn_data = gan.generate(BATCH_SIZE, noise_params)
                # np.savez(osp.join(synthetic_data_out_dir, f'epoch_{epoch}'), syn_data)
                if ADD_ROTATION_LOSS:
                    title = f'epoch_{epoch}_fake \nlr: {INIT_LR} \n num_angles: {flags.num_angles} \nrot_loss_weights_dg: {flags.weight_rotation_loss_d}, {flags.weight_rotation_loss_g}'
                else:
                    title = f'epoch_{epoch}_fake \nlr: {INIT_LR} \n num_angles: {flags.num_angles}'

                plot_kwargs = {'epoch': epoch,
                               'in_u_sphere': True, 
                               'title': title, 
                               'save_dir': synthetic_img_save_dir, 
                              'file_name': f'epoch_{epoch}.png'
                            }
                plot_3d_point_cloud(syn_data, plot_kwargs)
 
            train_stats.append((epoch, losses))
            if PLOT_TRAIN_CURVE and not epoch%10 :
                plot(train_stats, epoch, flags)
                plot_acc(train_stats, epoch, flags)

def compute_pc_loss(ref_pcs, sample_pcs, use_EMD=True):
    batch_size = 100     # Find appropriate number that fits in GPU.
    normalize = True     # Matched distances are divided by the number of 
                        # points of thepoint-clouds.

    mmd, matched_dists = minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=use_EMD)
    cov, matched_ids = coverage(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=use_EMD)
    jsd = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)

    return mmd, cov, jsd

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def plot_acc(train_stats, epoch, flags):
    x = range(len(train_stats))
    real_rot_acc = [t[1]['acc'][0] for t in train_stats]
    fake_rot_acc = [t[1]['acc'][1] for t in train_stats]
    accuracies = [real_rot_acc, fake_rot_acc]
    colors = ['r', 'y']

    for i, (acc, color) in enumerate(zip(accuracies, colors)):
        plt.plot(x, acc, colors[i]+'-')

    # plt.title(s)
    plt.legend(['real_rot_acc', 'fake_rot_acc'], loc=0)
    plt.tick_params(axis='x', which='both', bottom='off', top='off')
    plt.tick_params(axis='y', which='both', left='off', right='off')
    
    plt.xlabel('Epochs.') 
    plt.ylabel('Accuracy.')
    train_curve_save_dir = osp.join(TOP_OUT_DIR, 'train_curve/', EXPERIMENT_NAME)
    create_dir(train_curve_save_dir)
    file_name = osp.join(train_curve_save_dir, f'epoch_{epoch}_acc_1.png')
    while osp.isfile(file_name):
        idx = file_name.rfind('.')
        tail = file_name[idx-1]
        file_name = file_name[:idx-1] + str(int(tail)+1) + ".png"

    print(f'Accuracy plot saved in {file_name}')
    plt.savefig(file_name)
    plt.clf()
    plt.close()


def plot(train_stats, epoch, flags):
    x = range(len(train_stats))
    if ADD_ROTATION_LOSS:
        # rot_loss = [t[1]['rot_loss'] for t in train_stats]
        real_rot_loss = [t[1]['d_losses'][0] for t in train_stats]
        d_loss = [t[1]['d_losses'][1] for t in train_stats]
        d_all_loss = [t[1]['d_losses'][2] for t in train_stats]

        fake_rot_loss = [t[1]['g_losses'][0] for t in train_stats]
        g_loss = [t[1]['g_losses'][1] for t in train_stats]
        g_all_loss = [t[1]['g_losses'][2] for t in train_stats]
        losses = [d_loss, g_loss, real_rot_loss, fake_rot_loss,]

        colors = ['r', 'y', 'b', 'g']
        s = f'lr: {INIT_LR}, num_angles: {flags.num_angles}, rot_loss_weights_dg: {flags.weight_rotation_loss_d}, {flags.weight_rotation_loss_g}'
    else:
        d_loss = [t[1][0] for t in train_stats]
        g_loss = [t[1][1] for t in train_stats]
        losses = [d_loss, g_loss]
        colors = ['r', 'y']
        s = f'no rotation loss, lr: {INIT_LR}'

    
    for i, (loss, color) in enumerate(zip(losses, colors)):
        plt.plot(x, loss, colors[i]+'-')

    plt.title(s)
    if ADD_ROTATION_LOSS:
        plt.legend(['d_loss', 'g_loss', 'real_pc_rot_loss','fake_pc_rot_loss'], loc=0)
    else:
        plt.legend(['d_loss', 'g_loss'], loc=0)
    plt.tick_params(axis='x', which='both', bottom='off', top='off')
    plt.tick_params(axis='y', which='both', left='off', right='off')
    
    plt.xlabel('Epochs.') 
    plt.ylabel('Loss.')
    train_curve_save_dir = osp.join(TOP_OUT_DIR, 'train_curve/', EXPERIMENT_NAME)
    create_dir(train_curve_save_dir)
    file_name = osp.join(train_curve_save_dir, f'epoch_{epoch}_1.png')
    while osp.isfile(file_name):
        idx = file_name.rfind('.')
        tail = file_name[idx-1]
        file_name = file_name[:idx-1] + str(int(tail)+1) + ".png"

    print(f'train plot saved in {file_name}')
    plt.savefig(file_name)
    plt.clf()
    plt.close()

train()