#!/usr/bin/env python3
'''
WGAN-GP toy example
'''
import os
import sys
import time
import json

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
from skimage.transform import resize
he_normal = tf.contrib.keras.initializers.he_normal

from collections import defaultdict
from logger import logger
from utils.utils import *

from utils.sampler import gaussian_mixture_circle, gaussian_mixture_double_circle, swiss_roll
from utils.plot import plot_kde, plot_scatter, plot_heatmap, plot_heatmap_fast
from progress.bar import IncrementalBar

model_name = '01-WGANGP'#.DGw_truncNorm_002.DG_Relu'
summary_dir = './summary-toy/'
train_sample_directory = '[result][toy]'
model_directory = './__models-toy__/'

'''
Flags
'''
args = tf.app.flags
args.DEFINE_boolean('train', True, 'True for training, False for testing [%(default)s]')
args.DEFINE_string('name', model_name, 'Model name [%(default)s]')
args.DEFINE_string('ckpath', None, 'Checkpoint path for restoring (ex: __models__/{}.ckpt-100) [%(default)s]'.format(model_name))
args.DEFINE_string('z_init',  'uniform', 'normal: Normal(mean=0, std=1), uniform: Uniform(-1,1) [%(default)s]')
args.DEFINE_string('w_init',  'henormal', '"henormal", "heuniform", "xavnormal", "xavuniform", "truncnormal:<STDDEV>" [%(default)s]')
args.DEFINE_string('x_mode',  'gaussian', 'Ground Truth Distribution. "gaussian": Gaussian Mixture, "swiss": Swiss Roll [%(default)s]')
args.DEFINE_integer('max_epoch',100000, '[%(default)s]')
args.DEFINE_integer('d_epoch',  5,      '[%(default)s]')
args.DEFINE_integer('bs',       256,    '[%(default)s]')
args.DEFINE_integer('z_size',   100,    '[%(default)s]')
args.DEFINE_integer('n_mixture', 8,     'Number of Gaussian mixture [%(default)s]')
args.DEFINE_float  ('m_scale',  1.4,    'Gaussian mixture scale [%(default)s]')
args.DEFINE_float  ('m_std',    0.12,   'the standard deviation of each Gaussian [%(default)s]')
args.DEFINE_float  ('glr',      0.0001, '[%(default)s]')
args.DEFINE_float  ('dlr',      0.0001, '[%(default)s]')
args.DEFINE_float  ('beta1',    0.5,    '[%(default)s]')
args.DEFINE_float  ('beta2',    0.9,    '[%(default)s]')
# args.DEFINE_float  ('gamma',    0.5,    'degeneration coefficient [%(default)s]')
# args.DEFINE_float  ('d_thresh', 0.8,    '[%(default)s]')
# args.DEFINE_float  ('leak',     0.2,    '[%(default)s]')
args.DEFINE_boolean('decay',    False,  'Learning rate decay? [%(default)s]')
# args.DEFINE_boolean('Dp',       False,  'Update D again if D_acc < D_threshold? [%(default)s]')
# args.DEFINE_boolean('Gp',       False,  'Update G again if D_acc > D_threshold? [%(default)s]')
args.DEFINE_boolean('gradient_penalty', True, 'Add gradient penalty? [%(default)s]')
args.DEFINE_float('lmda', 0.1, '(for gradient penalty) the coefficient of interpolaters (0<..<1) [%(default)s]')
FLAGS = args.FLAGS

'''
Global Parameters
'''
n_epochs   = FLAGS.max_epoch
batch_size = FLAGS.bs
z_size     = FLAGS.z_size
g_lr       = FLAGS.glr
d_lr       = FLAGS.dlr
beta1      = FLAGS.beta1
beta2      = FLAGS.beta2
# gamma      = FLAGS.gamma
# d_thresh   = FLAGS.d_thresh
# leak_value = FLAGS.leak

num_mixture= FLAGS.n_mixture
scale      = FLAGS.m_scale
std        = FLAGS.m_std

def he_uniform(shape, dtype=tf.float32, partition_info=None):
    stdev = np.sqrt(2./shape[0])
    output = np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=shape
            )
    return tf.convert_to_tensor(output, dtype=dtype, name='he_uniform_variable')

'''
Write config
'''
content = {}
content['flags'] = FLAGS.__dict__['__flags']
content['model'] = {
        'G': {},
        'D': {
            'ground_truth': 'gaussian_mixture_circle(batch_size, num_mixture, scale={}, std={})'.format(scale, std),
            }
        }


weight_init = []
# --- [ MLP
def generator(z, phase_train=True, reuse=False, use_bias=False, name="gen"):
    net = z
    content['model']['G'] = {
            'weight_init_1': str(weight_init),
            'weight_init_2': str(weight_init),
            'weight_init_3': str(weight_init),
            }

    with tf.variable_scope(name, reuse=reuse):
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu, weights_initializer=weight_init, trainable=phase_train)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu, weights_initializer=weight_init, trainable=phase_train)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu, weights_initializer=weight_init, trainable=phase_train)
        net = slim.fully_connected(net, 2, activation_fn=None, weights_initializer=weight_init, trainable=phase_train) #output (x,y) of Gaussian
        print ("G output: ", net)
    return net

# ---
def discriminator(inputs, phase_train=True, reuse=False, use_bias=False, name="dis"):
    net = inputs
    with tf.variable_scope(name, reuse=reuse):
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu, weights_initializer=weight_init, trainable=phase_train)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu, weights_initializer=weight_init, trainable=phase_train)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu, weights_initializer=weight_init, trainable=phase_train)
        net = slim.fully_connected(net, 1, activation_fn=None, weights_initializer=weight_init, trainable=phase_train)
        print('D output: ', net)
    return net
# --- ]


def trainGAN(checkpoint=None, model_name=model_name):
    # Load Gaussian Mixture Distribution
    # samples_x = gaussian_mixture_circle(batch_size, num_mixture, scale=scale, std=0.5)
    print ("MODEL: ", model_name)

    global_step = tf.train.get_or_create_global_step()
    slim.add_model_variable(global_step)
    with tf.variable_scope('global_step_update', reuse=True):
        global_step_update = tf.assign_add(global_step, 1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    # ===[ Build generator & discriminator
    x_vector = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    z_vector = tf.placeholder(shape=[None, z_size], dtype=tf.float32)

    g_z = generator(z_vector, phase_train=True, reuse=False, name="gen")
    d_output_x = discriminator(x_vector, phase_train=True, reuse=False, name="dis")
    d_output_z = discriminator(g_z, phase_train=True, reuse=True, name="dis")

    # ===[ Crite loss
    d_loss = - (tf.reduce_mean(d_output_x) - tf.reduce_mean(d_output_z))
    g_loss = - tf.reduce_mean(d_output_z)
    # --- [ Gradient panelty
    if FLAGS.gradient_penalty:
        LAMBDA = FLAGS.lmda    #0.1 for toy task
        alpha = tf.placeholder(tf.float32, shape=[None, 1])
        interpolates = alpha * x_vector + ((1-alpha) * g_z)
        print (interpolates.shape)
        d_output_inp = discriminator(interpolates, phase_train=True, reuse=True, name="dis")
        gradients = tf.gradients(d_output_inp, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))    #L2-distance
        gradient_penalty = tf.reduce_mean(tf.square(slopes-1))
        d_loss += gradient_penalty


    print ("D_loss: ", d_loss)
    print ("G_loss: ", g_loss)

    para_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen")
    para_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dis")
    print_list(para_d, name="PARA_D")
    print_list(para_g, name="PARA_D")

    if FLAGS.decay:
        d_lr_decay = tf.train.exponential_decay(d_lr, global_step, 1000, 0.96, staircase=True)
        g_lr_decay = tf.train.exponential_decay(g_lr, global_step, 1000, 0.96, staircase=True)
    else:
        d_lr_decay = d_lr
        g_lr_decay = g_lr

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr_decay,beta1=beta1,beta2=beta2).minimize(d_loss,var_list=para_d, colocate_gradients_with_ops=True)
        optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr_decay,beta1=beta1,beta2=beta2).minimize(g_loss,var_list=para_g, colocate_gradients_with_ops=True)


    # Save summary
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    writer = tf.summary.FileWriter(summary_dir + model_name)
    summary_d_x_hist = tf.summary.histogram("d_output_x", d_output_x)
    summary_d_z_hist = tf.summary.histogram("d_output_z", d_output_z)
    summary_d_loss   = tf.summary.scalar("d_loss", d_loss)
    summary_g_loss   = tf.summary.scalar("g_loss", g_loss)
    d_summary_merge = tf.summary.merge([summary_d_loss,
                                        summary_d_x_hist,
                                        summary_d_z_hist,
                                        ])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer()) # DO NOT put it after ``saver.restore``
        saver = tf.train.Saver(slim.get_model_variables(), max_to_keep=1)
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
            global_step = tf.train.get_global_step(sess.graph)

        x_gen = generate_true_samples(batch_size)
        z_gen = generate_fake_samples(batch_size)

        # Plot Ground Truth Distribution
        x = np.concatenate([next(x_gen) for _ in range(100)])
        plot_scatter(x, train_sample_directory, 'scatter_groundtruth'.format(model_name))
        plot_kde(x, train_sample_directory, 'kde_groundtruth'.format(model_name))

        # Training G
        for t in range(sess.run(global_step), n_epochs):
            a = np.random.uniform(0,1,[batch_size,1])
            x = next(x_gen)
            z = next(z_gen)

            T_start = time.time()

            for _ in range(FLAGS.d_epoch):
                if FLAGS.gradient_penalty:
                    _, summary_d, disc_loss = sess.run([optimizer_op_d,d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x, alpha:a})
                else:
                    _, summary_d, disc_loss = sess.run([optimizer_op_d,d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
            _, summary_g, gen_loss = sess.run([optimizer_op_g,summary_g_loss,g_loss],feed_dict={z_vector:z})
            D_x, D_gz = sess.run([d_output_x, d_output_z], feed_dict={z_vector:z, x_vector:x})
            print (' [epoch {:>5d}] D_loss: {:<15.8e}  G_loss: {:<15.8e} | D(x) range: [{:< 6f}, {:< 8f}]  D(G(z)) range: [{:< 6f}, {:< 8f}] | [ETA] {}'.format(
                    t, disc_loss, gen_loss, D_x.min(), D_x.max(), D_gz.min(), D_gz.max(), time.time() - T_start))

            # output generated images
            if t % 100 == 0:
                samples_g = sess.run(g_z,feed_dict={z_vector:z}) #type=np.ndarray
                plot_scatter(samples_g, train_sample_directory, 'scatter_{}_'.format(model_name), suffix=t)
                plot_kde(samples_g, train_sample_directory, 'kde_{}_'.format(model_name), suffix=t)
                # --- [ Plot heatmap
                # samples_x = np.mgrid[-4:4:0.1, -4:4:0.1].transpose(1,2,0).reshape(-1,2)
                # predict = sess.run(d_output_x, feed_dict={x_vector:samples_x})
                # plot_heatmap(samples_x, predict, train_sample_directory, 'hmap_{}_D({})'.format(model_name, t), prob=True)
                # high = (predict > 0.7).reshape(-1)
                # plot_scatter(samples_x[high], train_sample_directory, 'hmap_{}_{}'.format(model_name, t))
                # --- [ Plot heatmap (fast)
                mesh = np.mgrid[-4:4:0.1, -4:4:0.1].transpose(1,2,0).reshape(-1,2)
                predict = sess.run(d_output_x, feed_dict={x_vector:mesh})
                predict = predict.reshape(80,80).transpose(1,0)
                predict = np.flip(predict, 0)
                plot_heatmap_fast(predict, train_sample_directory, 'hmap_{}_D({})'.format(model_name, t), prob=False)

            # save checkpoint
            # if t % 200 == 0:
            #     if not os.path.exists(model_directory):
            #         os.makedirs(model_directory)
            #     saver.save(sess, save_path = os.path.join(model_directory, '{}.ckpt'.format(model_name)), global_step=global_step)

            writer.add_summary(summary_d, t)
            writer.add_summary(summary_g, t)
            writer.flush()
            sess.run(global_step_update)

        writer.close()

def generate_true_samples(batchsize=1000):
    if FLAGS.x_mode.lower() in ['gaussian', 'g']:
        while True:
            yield gaussian_mixture_circle(batchsize, num_mixture, scale=scale, std=std)
    else:
        while True:
            yield swiss_roll(batchsize, scale=scale, std=std)

def generate_fake_samples(batchsize=1000):
    if FLAGS.z_init.lower() == 'normal':
        while True:
            yield np.random.normal(0, 1, size=[batchsize, z_size]).astype(np.float32)
    else:
        while True:
            yield np.random.uniform(-1, 1, size=[batchsize, z_size]).astype(np.float32)

def print_list(list, name='list'):
    print ("{}: ({})".format(name.upper(), len(list)))
    for i in list:
        print ("  ", i)


def testGAN(trained_model_path):
    name = trained_model_path.split('/')[-1].split('.ckpt')[0]
    z_vector = tf.placeholder(shape=[None, z_size],dtype=tf.float32)
    g_z = generator(z_vector, phase_train=False, reuse=True, name="gen")

    saver = tf.train.Saver(slim.get_model_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, trained_model_path)
        z_gen = generate_fake_samples(batch_size)

        t = 0
        while True:
            try:
                z = next(z_gen)
                samples_g = sess.run(g_z,feed_dict={z_vector:z}) #type=np.ndarray
                plot_scatter(samples_g, 'test', '{}_scatter'.format(name), suffix=t+'_test')
                plot_kde(samples_g, 'test', '{}_kde'.format(name), suffix=t+'_test')
                t+=1
                input("enter for next one...")
            except:
                break


def main(_):
    # ===[ Rename model name
    if FLAGS.w_init.lower().replace('_','') in ['henormal', 'henorm', 'hen', 'hn']:
        name = FLAGS.name + '.DGw_{}.DG_{}'.format('heNormal','Relu')
        w_initializer = he_normal()
    elif FLAGS.w_init.lower().replace('_','') in ['heuniform', 'heuni', 'heu', 'hu']:
        name = FLAGS.name + '.DGw_{}.DG_{}'.format('heUniform','Relu')
        w_initializer = he_uniform
    elif FLAGS.w_init.lower().replace('_','') in ['xavnormal', 'xaviernorm', 'xavnorm', 'xavn', 'xn']:
        name = FLAGS.name + '.DGw_{}.DG_{}'.format('xavNormal','Relu')
        w_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    elif FLAGS.w_init.lower().replace('_','') in ['xavuniform', 'xavieruni', 'xavuni', 'xavu', 'xavier', 'xu']:
        name = FLAGS.name + '.DGw_{}.DG_{}'.format('xavUniform','Relu')
        w_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
    elif FLAGS.w_init.lower().replace('_','').split(':')[0] in ['truncnormal', 'truncnorm', 'truncn', 'tn']:
        try:
            w_init_std = float(FLAGS.w_init.split(':')[-1])
        except:
            w_init_std = 0.02
        name = FLAGS.name + '.DGw_{}.DG_{}'.format('truncNorm_%s' % (str(w_init_std).replace('.','')),'Relu')
        w_initializer = tf.truncated_normal_initializer(stddev=w_init_std)
    else:
        logger.error('Cannot find "{}".'.format(FLAGS.w_init))
    global weight_init
    weight_init = w_initializer

    # ===[ Training phase
    if FLAGS.train:
        if FLAGS.x_mode.lower() == ['gaussian', 'g']:
            suffix = '-{}nGaussian-{}scale-{}std-{}bs-{}dlr-{}glr-{}Z'.format(FLAGS.n_mixture, FLAGS.m_scale, FLAGS.m_std, FLAGS.bs, FLAGS.dlr, FLAGS.glr, FLAGS.z_init)
        else:
            suffix = '-swiss-{}scale-{}std-{}bs-{}dlr-{}glr-{}Z'.format(FLAGS.m_scale, FLAGS.m_std, FLAGS.bs, FLAGS.dlr, FLAGS.glr, FLAGS.z_init)
        suffix = suffix + '-decay' if FLAGS.decay else suffix
        # suffix = suffix + '-D+' if FLAGS.Dp else suffix
        # suffix = suffix + '-G+' if FLAGS.Gp else suffix
        path = [x for x in os.listdir() if train_sample_directory+name+suffix in x]
        suffix += '-{}'.format(len(path))
        name = name + suffix

        global train_sample_directory
        train_sample_directory += name
        if not os.path.exists(train_sample_directory):
            os.makedirs(train_sample_directory)

        trainGAN(checkpoint=FLAGS.ckpath, model_name=name)
    # ===[ Testing phase
    else:
        if FLAGS.ckpath:
            testGAN(trained_model_path=FLAGS.ckpath, model_name=name)
        else:
            logger.error("Needs checkpoint path.")

if __name__ == '__main__':
    tf.app.run()

