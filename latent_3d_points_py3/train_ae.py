import os
import os.path as osp
import matplotlib.pylab  as plt
from pathlib import Path
from src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from src.autoencoder import Configuration as Conf
from src.point_net_ae import PointNetAutoEncoder

from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder, load_all_shapenet_data

from src.tf_utils import reset_tf_graph
from src.general_utils import plot_3d_point_cloud_simple

top_out_dir = '../log_ae/'          # Use to save Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

experiment_name = 'all_class_ae'
n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'

SHAPENET_DIR = '../data/shape_net_core_uniform_samples_2048/'
all_pc_data = load_all_shapenet_data(SHAPENET_DIR, n_threads=8, file_ending='.ply', verbose=True)
train_params = default_train_params(single_class = False)
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)

Path(top_out_dir).mkdir(parents=True, exist_ok=True)
train_dir = osp.join(top_out_dir, experiment_name)
while os.path.exists(train_dir):
    idx = train_dir.rfind('_')
    tail = train_dir[idx+1:]
    if tail.isdigit():
        train_dir = train_dir[:idx+1] + str(int(tail)+1)
    else:
        train_dir = train_dir + '_1'
os.makedirs(train_dir)

conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5   # How often to evaluate/print out loss on 
                         # held_out data (if they are provided in ae.train() ).
conf.save(osp.join(train_dir, 'configuration'))

load_pre_trained_ae = False
restore_epoch = 500
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=restore_epoch)

reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(all_pc_data, conf, log_file=fout)
fout.close()
# train_dir = '../log_ae/all_class_ae_6'  
# conf = Conf.load(train_dir + '/configuration')
# reset_tf_graph()
# ae = PointNetAutoEncoder(conf.experiment_name, conf)
# ae.restore_model(conf.train_dir, epoch=restore_epoch)

feed_pc, feed_model_names, _ = all_pc_data.next_batch(10)
reconstructions = ae.reconstruct(feed_pc)[0]
latent_codes = ae.transform(feed_pc)

os.makedirs(osp.join(train_dir, "imgs"))
for i in range(10):
    fig = plot_3d_point_cloud_simple(reconstructions[i][:, 0], 
                    reconstructions[i][:, 1], 
                    reconstructions[i][:, 2], in_u_sphere=True);
    print(f'image saved at {train_dir}/fig{i}')
    plt.savefig((f"{train_dir}/fig{i}"))
    plt.close('all')