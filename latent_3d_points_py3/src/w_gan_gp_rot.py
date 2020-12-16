'''
Created on May 22, 2018

Author: Achlioptas Panos (Github ID: optas)
'''

import numpy as np
import time
import tensorflow as tf
import importlib
import os
import os.path as osp


from tflearn import *
from . gan import GAN
from functools import partial
# sys.path.append(BASE_DIR)

# from train_rotation_prediction import eval_one_epoch
import provider
import train_rotation_prediction
from . general_utils import plot_3d_point_cloud


class W_GAN_GP_ROT(GAN):
    '''Gradient Penalty.
    https://arxiv.org/abs/1704.00028
    '''

    def __init__(self, name, discriminator, generator, gen_kwargs={}, disc_kwargs={}, gan_kwargs = {}, graph=None, **kwargs):

        GAN.__init__(self, name, graph)
        
        self.step_count = 0
        self.n_output = gan_kwargs.get('n_out') #(1024, 3)
        self.flags = gan_kwargs.get('flags')
        self.batch_size = gan_kwargs.get('batch_size_value', 32)
        batch = tf.Variable(0)
        self.noise_dim = gan_kwargs.get('noise_dim')
        self.init_lr = gan_kwargs.get('init_lr')
        lam = gan_kwargs.get('lam')

        self.num_angles = self.flags.num_angles
        self.num_points = self.flags.num_point
        lr_pred = self.flags.lr_pred
        self.weight_rotation_loss_d = self.flags.weight_rotation_loss_d
        self.weight_rotation_loss_g = self.flags.weight_rotation_loss_g
        beta = self.flags.beta
        self.img_save_dir = osp.join(self.flags.top_out_dir, 'rotated_pc/', name)
        if not osp.exists(self.img_save_dir):
            os.makedirs(self.img_save_dir)
        self.visualize = False
        self.ms_task = self.flags.ms_task

        self.discriminator = discriminator
        self.generator = generator
        self.model_pred = importlib.import_module(self.flags.model) # import network module
                
        self.use_trans_loss = self.flags.use_transformation_loss
        self.use_input_trans = self.flags.use_input_transform
        self.use_feature_trans = self.flags.use_feature_transform
    
        self.is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training_pl')

        self.bn_decay = train_rotation_prediction.get_bn_decay(batch)

        self.get_pred = partial(self.model_pred.get_model, 
                                is_training=self.is_training_pl, 
                                bn_decay=self.bn_decay,
                                num_angles=self.num_angles,
                                use_input_trans=self.use_input_trans,
                                use_feature_trans=self.use_feature_trans)
        self.get_loss = partial(self.model_pred.get_loss, use_trans_loss=self.use_trans_loss)

        with tf.variable_scope(name):
            self.noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dim], name='noise')            # Noise vector.
            self.real_pc = tf.placeholder(tf.float32, shape=[self.batch_size] + self.n_output, name='real_pc')     # Ground-truth.
            with tf.variable_scope('rotation'):
                self.rot_label_pl = tf.placeholder(tf.int32, shape=self.batch_size, name='rot_label_pl')

                self.real_pc_rotated = self.rotate_n_angles(self.real_pc, self.rot_label_pl)
                self.real_pc_pred, real_pc_end_points = self.get_pred(self.real_pc_rotated)
                self.real_pc_rot_loss = self.get_loss(self.real_pc_pred, self.rot_label_pl, real_pc_end_points)

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output, **gen_kwargs)
                self.gen_out_rotated = self.rotate_n_angles(self.generator_out, self.rot_label_pl)
                self.gen_out_pred, gen_out_end_points = self.get_pred(self.gen_out_rotated)
                self.gen_out_rot_loss = self.get_loss(self.gen_out_pred, self.rot_label_pl, gen_out_end_points) #classification loss
                #need to fix
            if self.ms_task:
                with tf.variable_scope('mixed'):
                    #add fake pc as a rotation class
                    num_to_add = int(max(self.batch_size/self.num_angles, 1))
                    idx = tf.range(0, self.batch_size, 1)
                    idx = tf.random_shuffle(idx)[0:num_to_add]
                    self.fake_to_add = tf.gather(self.generator_out, idx)
                    self.mixed_pc = tf.concat([self.real_pc_rotated, self.fake_to_add], 0)
                    self.mixed_label = tf.concat([self.rot_label_pl, tf.constant(self.num_angles, shape = (num_to_add,))], axis = 0)
                    mixed_idx = tf.range(0, self.mixed_label.get_shape().as_list()[0], 1)
                    mixed_idx = tf.random_shuffle(mixed_idx)[0:self.batch_size]
                    self.mixed_pc = tf.gather(self.mixed_pc, mixed_idx)
                    self.mixed_label = tf.gather(self.mixed_label, mixed_idx)

                    self.mixed_pred, mixed_end_points = self.get_pred(self.mixed_pc)
                    self.mixed_loss = self.get_loss(self.mixed_pred, self.mixed_label, mixed_end_points)

            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.real_pc_rotated, scope=scope, **disc_kwargs)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.gen_out_rotated, reuse=True, scope=scope, **disc_kwargs)
            
            # Compute WGAN losses
            self.loss_d = tf.reduce_mean(self.synthetic_logit) - tf.reduce_mean(self.real_logit) # comparing rotated fake and real images
            self.loss_g = -tf.reduce_mean(self.synthetic_logit)

            # Add rotation loss
            if self.ms_task:
                self.g_ms_loss = tf.abs(self.gen_out_rot_loss - self.real_pc_rot_loss, name = 'abs')
                self.d_ms_loss = self.mixed_loss
                self.loss_d_rot = self.loss_d + self.weight_rotation_loss_d * self.d_ms_loss
                self.loss_g_rot = self.loss_g + self.weight_rotation_loss_g * self.g_ms_loss
            else:
                self.loss_d_rot = self.loss_d + self.weight_rotation_loss_d * self.real_pc_rot_loss
                self.loss_g_rot = self.loss_g + self.weight_rotation_loss_g * self.gen_out_rot_loss
            
            # Compute gradient penalty at interpolated points
            ndims = self.real_pc.get_shape().ndims #(1024, 3)
            alpha = tf.random_uniform(shape=[self.batch_size] + [1] * (ndims - 1), minval=0., maxval=1.)
            differences = self.generator_out - self.real_pc
            interpolates = self.real_pc + (alpha * differences)

            with tf.variable_scope('discriminator') as scope:
                gradients = tf.gradients(self.discriminator(interpolates, reuse=True, scope=scope, **disc_kwargs)[1], [interpolates])[0]

            # Reduce over all but the first dimension
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=list(range(1, ndims))))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.loss_d_rot += lam * gradient_penalty

            train_vars = tf.trainable_variables()
            d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]
            rot_params = [v for v in train_vars if '/rotation/' in v.name] #slightly suspecting that this part is incorrect
            self.opt_d = self.optimizer(self.init_lr, beta, self.loss_d_rot, d_params)
            self.opt_g = self.optimizer(self.init_lr, beta, self.loss_g_rot, g_params) #used loss_g + rot_loss to update
            self.opt_pred = self.optimizer(lr_pred, beta, self.real_pc_rot_loss, rot_params, batch) #only use real pics to update

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.init = tf.global_variables_initializer()

            #Launch the session
            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=self.graph)
            self.sess.run(self.init)


    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def _single_epoch_train(self, train_data, epoch, batch_size, noise_params, discriminator_boost=5, num_angles=8, rotation_boost=5, writer=None):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        n_examples = train_data.num_examples
        epoch_loss_d_rot = 0.
        epoch_loss_g_rot = 0.
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        epoch_rot_loss = 0.
        epoch_real_rot_loss = 0.
        epoch_fake_rot_loss = 0.

        n_batches = n_examples // batch_size
        start_time = time.time()

        iterations_for_epoch = n_batches // discriminator_boost

        is_training(True, session=self.sess)
        try:
            # Loop over all batches                
            for _ in range(iterations_for_epoch):
                #use real_pc to train rotation prediction model for a few iters
                for i in range(rotation_boost):
                    feed, _, _ = train_data.next_batch(batch_size)
                    rotation_label = np.random.randint(0, self.num_angles, size=self.batch_size)

                    feed_dict = {self.real_pc: feed,
                                 self.is_training_pl: True,
                                 self.rot_label_pl: rotation_label
                                }
                    _, rot_loss, real_pc_pred, real_pc_rotated = self.sess.run([self.opt_pred, self.real_pc_rot_loss, self.real_pc_pred, self.real_pc_rotated], feed_dict=feed_dict)
                    real_rot_acc = self.get_accuracy(real_pc_pred, rotation_label)
                    epoch_rot_loss += rot_loss
                    if self.visualize:
                        for j in range(self.batch_size):
                            title = f'epoch_{epoch}_real_pc \nlr: {self.init_lr} \n num_angles: {self.num_angles} \nrot_loss_weights_dg: {self.weight_rotation_loss_d}, {self.weight_rotation_loss_g}'
                            plot_kwargs = {'epoch': epoch,
                                'in_u_sphere': True, 
                                'ith': j,
                                'title': title, 
                                'save_dir': self.img_save_dir, 
                                'file_name': f'epoch{epoch}_rot{rotation_label[j]}_batch{j}.png'
                            }
                            plot_3d_point_cloud(real_pc_rotated, plot_kwargs) 
                            plot_kwargs_up = {'epoch': epoch,
                                'in_u_sphere': True, 
                                'ith': j,
                                'title': title, 
                                'save_dir': self.img_save_dir, 
                                'file_name': f'epoch{epoch}_rot{rotation_label[j]}_batch{j}_up.png'
                            }
                            plot_3d_point_cloud(feed, plot_kwargs_up) 
            

                for _ in range(discriminator_boost):
                    feed, _, _ = train_data.next_batch(batch_size)
                    z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                    feed_dict = {self.real_pc: feed,
                                 self.noise: z,
                                 self.is_training_pl: True,
                                 self.rot_label_pl: rotation_label
                                }
                    _, _, loss_d, real_pc_rot_loss, loss_d_rot, real_pc_rotated = self.sess.run([self.opt_d, self.opt_pred, self.loss_d, self.real_pc_rot_loss, self.loss_d_rot, self.real_pc_rotated], feed_dict=feed_dict)
                    epoch_real_rot_loss += real_pc_rot_loss
                    epoch_loss_d += loss_d
                    epoch_loss_d_rot += loss_d_rot # sum of two losses above

                # Update generator.
                z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                feed_dict = {self.real_pc: feed,
                             self.noise: z,
                             self.is_training_pl: True,
                             self.rot_label_pl: rotation_label
                            }
                _, loss_g, gen_out_rot_loss, gen_out_pred, loss_g_rot, generator_out, gen_out_rotated = \
                self.sess.run([self.opt_g, self.loss_g, self.gen_out_rot_loss, self.gen_out_pred, self.loss_g_rot, self.generator_out, self.gen_out_rotated], feed_dict=feed_dict)
                fake_rot_acc = self.get_accuracy(gen_out_pred, rotation_label)
                epoch_fake_rot_loss += gen_out_rot_loss
                epoch_loss_g += loss_g
                epoch_loss_g_rot += loss_g_rot # sum of two losses above

            if writer:
                names = ['loss_d', 'real_pc_rot_loss', 'loss_d_rot', 'loss_g', 'gen_out_rot_loss', 'loss_g_rot']
                values = [loss_d, real_pc_rot_loss, loss_d_rot, loss_g, gen_out_rot_loss, loss_g_rot]
                self._add_summaries(writer, names, values)
                self.step_count += 1
                
            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        epoch_rot_loss /= (iterations_for_epoch * rotation_boost)

        epoch_real_rot_loss /= (iterations_for_epoch * discriminator_boost)
        epoch_loss_d /= (iterations_for_epoch * discriminator_boost)
        epoch_loss_d_rot /= (iterations_for_epoch * discriminator_boost)

        epoch_fake_rot_loss /= iterations_for_epoch 
        epoch_loss_g /= iterations_for_epoch
        epoch_loss_g_rot /= iterations_for_epoch
        duration = time.time() - start_time
        dict = {'rot_loss': epoch_rot_loss, \
        'd_losses': [epoch_real_rot_loss, epoch_loss_d, epoch_loss_d_rot], \
        'g_losses': [epoch_fake_rot_loss, epoch_loss_g, epoch_loss_g_rot], \
        'acc': [real_rot_acc, fake_rot_acc]}

        extra = False
        if extra:
            print(f'EPOCH: {epoch}, loss_d: {epoch_loss_d}, real_rot_loss: {epoch_real_rot_loss}, loss_d_rot: {epoch_loss_d_rot}; \nloss_g: {epoch_loss_g}, fake_rot_loss: {epoch_fake_rot_loss}, loss_g_rot: {epoch_loss_g_rot}')
        else:
            print(f'EPOCH: {epoch}, real_rot_loss: {round(epoch_real_rot_loss, 3)}, fake_rot_loss: {round(epoch_fake_rot_loss, 3)}, loss_d_rot: {round(epoch_loss_d, 3)}, loss_g_rot: {round(epoch_loss_g, 3)}, real_rot_acc: {round(real_rot_acc, 3)}, fake_rot_acc: {round(fake_rot_acc, 3)}')
        
        return dict, duration
    
    def _add_summaries(self, writer, names, values):
        for name, value in zip(names, values):
            summary = tf.Summary(value=[
                tf.Summary.Value(tag=name, simple_value=value),
            ])
            writer.add_summary(summary, self.step_count)

    def eval_rot(self, batch_data):
        feed_dict = {self.real_pc: batch_data, self.is_training_pl: False}
        _, rot_loss = self.sess.run([self.opt_pred, self.real_pc_rot_loss], feed_dict=feed_dict)
        return rot_loss

    def get_accuracy(self, pred, labels):
        pred_val = np.argmax(pred, 1)
        correct = np.sum(pred_val == labels)
        return correct/len(labels)

    def rotate_n_angles(self, current_data, current_label):
        '''batch_data: Bx1024x3 tensor'''
        # current_label = np.random.randint(0, self.num_angles, size=self.batch_size)
        if self.num_angles == 6:
            current_data = provider.rotate_tensor_by_label(current_data, current_label, self.graph)
        elif self.num_angles == 18:
            current_data = provider.rotate_tensor_by_label(current_data, current_label, self.graph)
        elif self.num_angles == 32:
            current_data = provider.rotate_tensor_by_label_32(current_data, current_label, self.graph)
        elif self.num_angles == 54:
            current_data = provider.rotate_tensor_by_label_54(current_data, current_label, self.graph)
        elif self.num_angles: #sunflower distribution
            current_data = provider.rotate_point_by_label_n(current_data, current_label, self.graph, self.num_angles, use_tensor=True)
        else:
            raise(NotImplementedError())
        current_data = tf.convert_to_tensor(current_data)
        return current_data

    