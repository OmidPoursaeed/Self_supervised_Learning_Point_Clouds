'''
Created on May 22, 2018

Author: Achlioptas Panos (Github ID: optas)
'''

import numpy as np
import time
import tensorflow as tf
import importlib


from tflearn import *
from . gan import GAN
from . w_gan_gp_rot import *
from functools import partial

from models import pointnet_cls_rotation_discriminator  # TODO: not hard code the module
import provider
import train_rotation_prediction


class W_GAN_GP_SHARED_WEIGHTS(GAN):
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
        init_lr = gan_kwargs.get('init_lr')
        lam = gan_kwargs.get('lam')

        self.num_angles = self.flags.num_angles
        self.num_points = self.flags.num_point

        weight_rotation_loss_d = self.flags.weight_rotation_loss_d
        weight_rotation_loss_g = self.flags.weight_rotation_loss_g
        beta = self.flags.beta

        self.discriminator = pointnet_cls_rotation_discriminator
        self.generator = generator
        
        self.use_trans_loss = self.flags.use_transformation_loss
        self.use_input_trans = self.flags.use_input_transform
        self.use_feature_trans = self.flags.use_feature_transform
    
        self.is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training_pl')
        self.labels_pl = tf.placeholder(tf.int32, shape=(self.batch_size), name='labels_pl')

        self.bn_decay = train_rotation_prediction.get_bn_decay(batch)

        self.get_d = partial(self.discriminator.get_model,
                                is_training=self.is_training_pl, 
                                bn_decay=self.bn_decay,
                                num_angles=self.num_angles,
                                use_input_trans=self.use_input_trans,
                                use_feature_trans=self.use_feature_trans)
        self.get_loss = partial(self.discriminator.get_loss, use_trans_loss=self.use_trans_loss)

        with tf.variable_scope(name):
            self.noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dim], name='noise')            # Noise vector.
            self.real_pc = tf.placeholder(tf.float32, shape=[self.batch_size] + self.n_output, name='real_pc')     # Ground-truth.
            with tf.variable_scope('rotation'):
                self.real_pc_rotated, self.real_pc_label = rotate_n_angles(self.real_pc)

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output, **gen_kwargs)
                with tf.variable_scope('rotation', reuse=True):
                    self.gen_out_rotated, self.gen_out_label = rotate_n_angles(self.generator_out)

            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
                with tf.variable_scope('real_pc_rotation', reuse=tf.AUTO_REUSE):
                    self.real_pc_pred, real_pc_end_points, self.real_prob, self.real_logit = self.get_d(self.real_pc_rotated)
                    self.real_pc_rot_loss = self.get_loss(self.real_pc_pred, self.real_pc_label, real_pc_end_points)

                self.gen_out_pred, gen_out_end_points, self.synthetic_prob, self.synthetic_logit = self.get_d(self.gen_out_rotated)
                self.gen_out_rot_loss= self.get_loss(self.gen_out_pred, self.gen_out_label, gen_out_end_points)

            # Compute WGAN losses
            self.loss_d = tf.reduce_mean(self.synthetic_logit) - tf.reduce_mean(self.real_logit) #comparing rotated fake and real images
            self.loss_g = -tf.reduce_mean(self.synthetic_logit)

            # Add rotation loss
            self.loss_d_rot = self.loss_d + weight_rotation_loss_d * self.real_pc_rot_loss
            self.loss_g_rot = self.loss_g + weight_rotation_loss_g * self.gen_out_rot_loss
            # Compute gradient penalty at interpolated points

            ndims = self.real_pc.get_shape().ndims #(1024, 3)
            alpha = tf.random_uniform(shape=[self.batch_size] + [1] * (ndims - 1), minval=0., maxval=1.)
            differences = self.generator_out - self.real_pc
            interpolates = self.real_pc + (alpha * differences)

            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
                _, _, _, d_prob = self.get_d(interpolates)
                gradients = tf.gradients(d_prob, [interpolates])[0]

            # Reduce over all but the first dimension
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=list(range(1, ndims))))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.loss_d_rot += lam * gradient_penalty

            train_vars = tf.trainable_variables()
            d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]
            rot_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/real_pc_rotation/')]
            self.opt_d = self.optimizer(init_lr, beta, self.loss_d_rot, d_params)
            self.opt_g = self.optimizer(init_lr, beta, self.loss_g_rot, g_params) #used loss_g + rot_loss to update
            self.opt_pred = self.optimizer(init_lr, beta, self.real_pc_rot_loss, rot_params, batch) #only use real pics to update

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.init = tf.global_variables_initializer()

            # Launch the session
            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def _single_epoch_train(self, train_data, batch_size, noise_params, discriminator_boost=5, num_angles=54, rotation_boost=20, writer=None):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        n_examples = train_data.num_examples
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        n_batches = n_examples // batch_size
        start_time = time.time()

        iterations_for_epoch = n_batches // discriminator_boost

        is_training(True, session=self.sess)
        try:
            # Loop over all batches                
            for _ in range(iterations_for_epoch):
                #use real_pc to train rotation prediction model for a few iters
                for _ in range(rotation_boost):
                    feed, _, _ = train_data.next_batch(batch_size)
                    feed_dict = {self.real_pc: feed, self.is_training_pl: True}
                    _, rot_loss = self.sess.run([self.opt_pred, self.real_pc_rot_loss], feed_dict=feed_dict)

                for _ in range(discriminator_boost):
                    feed, _, _ = train_data.next_batch(batch_size)
                    z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                    feed_dict = {self.real_pc: feed, self.noise: z, self.is_training_pl: True}
                    _, _, loss_d, real_pc_rot_loss, loss_d_rot = self.sess.run([self.opt_d, self.opt_pred, self.loss_d, self.real_pc_rot_loss, self.loss_d_rot], feed_dict=feed_dict)
                    epoch_loss_d += loss_d_rot

                # Update generator.
                z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                feed_dict = {self.real_pc: feed, self.noise: z, self.is_training_pl: True}
                _, loss_g, gen_out_rot_loss, loss_g_rot, generator_out = self.sess.run([self.opt_g, self.loss_g, self.gen_out_rot_loss, self.loss_g_rot, self.generator_out], feed_dict=feed_dict)
                epoch_loss_g += loss_g_rot 

                print(f'loss_d: {loss_d}, real_pc_rot_loss: {real_pc_rot_loss}, loss_d_rot: {loss_d_rot}; \nloss_g: {loss_g}, gen_out_rot_loss: {gen_out_rot_loss}, loss_g_rot: {loss_g_rot}')
                
                if writer:
                    self._add_summaries(writer, 'loss_d', loss_d)
                    self._add_summaries(writer, 'real_pc_rot_loss', real_pc_rot_loss)
                    self._add_summaries(writer, 'loss_d_rot', loss_d_rot)
                    self._add_summaries(writer, 'loss_g', loss_g)
                    self._add_summaries(writer, 'gen_out_rot_loss', gen_out_rot_loss)
                    self._add_summaries(writer, 'loss_g_rot', loss_g_rot)
                    self.step_count += 1

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        epoch_loss_d /= (iterations_for_epoch * discriminator_boost)
        epoch_loss_g /= iterations_for_epoch
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
    
    def _add_summaries(self, writer, name, value):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=name, simple_value=value),
        ])
        writer.add_summary(summary, self.step_count)

   