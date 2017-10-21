#!/usr/bin/env python3
'''
regular GAN toy example
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

from utils.sampler import gaussian_mixture_circle, gaussian_mixture_double_circle
from utils.plot import plot_kde, plot_scatter, plot_heatmap, plot_heatmap_fast
from progress.bar import IncrementalBar

model_name = '00-GAN'#.DGw_heNormal.DG_Relu'
summary_dir = './summary-toy/'
train_sample_directory = '[result][toy]'
model_directory = './__models-toy__/'

'''
Flags
'''
args = tf.app.flags
args.DEFINE_boolean('train', True, 'True for training, False for testing [%(default)s]')
args.DEFINE_string('name', model_name, 'Model name [%(default)s]')
args.DEFINE_string('ckpath', None, 'Checkpoint path for restoring (ex: models/cifar10.ckpt-100) [%(default)s]')
args.DEFINE_string('z_init',  'uniform', '"normal": Normal(mean=0, std=1), "uniform": Uniform(-1,1) [%(default)s]')
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
# args.DEFINE_float  ('gamma',    0.5,    'degeneration coefficient [%(default)s]')
# args.DEFINE_float  ('d_thresh', 0.8,    '[%(default)s]')
# args.DEFINE_float  ('leak',     0.2,    '[%(default)s]')
args.DEFINE_boolean('decay',    False,  'Learning rate decay? [%(default)s]')
# args.DEFINE_boolean('Dp',       False,  'Update D again if D_acc < D_threshold? [%(default)s]')
# args.DEFINE_boolean('Gp',       False,  'Update G again if D_acc > D_threshold? [%(default)s]')
FLAGS = args.FLAGS

'''
Global Parameters
'''
n_epochs   = FLAGS.max_epoch
batch_size = FLAGS.bs
z_size     = FLAGS.z_size
g_lr       = FLAGS.glr
d_lr       = FLAGS.dlr
beta       = FLAGS.beta1
# gamma      = FLAGS.gamma
# d_thresh   = FLAGS.d_thresh
# leak_value = FLAGS.leak

num_mixture= FLAGS.n_mixture
scale      = FLAGS.m_scale
std        = FLAGS.m_std

'''
Write config
'''
content = {}
content['flags'] = FLAGS.__dict__['__flags']
content['model'] = {
        'G': {
            'net': [],
            'w_init': [],
            },
        'D': {
            'net': [],
            'w_init': [],
            'ground_truth': 'gaussian_mixture_circle(batch_size, num_mixture, scale={}, std={})'.format(scale, std),
            }
        }

def he_uniform(shape, dtype=tf.float32, partition_info=None):
    stdev = np.sqrt(2./shape[0])
    output = np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=shape
            )
    return tf.convert_to_tensor(output, dtype=dtype, name='he_uniform_variable')

# import random
# def gaussian_mixture_circle(batch_size, num_cluster=8, scale=1, std=1):
#     centers = [
#         (1,0),
#         (-1,0),
#         (0,1),
#         (0,-1),
#         (1./np.sqrt(2), 1./np.sqrt(2)),
#         (1./np.sqrt(2), -1./np.sqrt(2)),
#         (-1./np.sqrt(2), 1./np.sqrt(2)),
#         (-1./np.sqrt(2), -1./np.sqrt(2))
#     ]
#     centers = [(scale*x,scale*y) for x,y in centers]
#     while True:
#         dataset = []
#         for i in range(batch_size):
#             point = np.random.randn(2)*.02
#             center = random.choice(centers)
#             point[0] += center[0]
#             point[1] += center[1]
#             dataset.append(point)
#         dataset = np.array(dataset, dtype='float32')
#         dataset /= 1.414 # stdev
#         yield dataset


# --- [ MLP
def generator(z, phase_train=True, reuse=False, use_bias=False, name="gen"):
    # weight_init = tf.truncated_normal_initializer(stddev=0.3)  # Recommanded to train NN ; std=1 too much
    # weight_init = he_normal()
    net = z
    pack = (
            [512, tf.nn.relu, w_initializer],
            [512, tf.nn.relu, w_initializer],
            [512, tf.nn.relu, w_initializer],
            [2, None, w_initializer],
            )
    with tf.variable_scope(name, reuse=reuse):
        for (out_dim, nonlinear, weight_init) in pack:
            net = slim.fully_connected(net, out_dim, activation_fn=nonlinear, weights_initializer=weight_init, trainable=phase_train)
            content['model']['G']['net'].append(str(net))
            content['model']['G']['w_init'].append(str(weight_init))
            print ("G: ", net)
    return net  #output (x,y) of Gaussian

# ---
def discriminator(inputs, phase_train=True, reuse=False, use_bias=False, name="dis"):
    # weight_init = tf.truncated_normal_initializer(stddev=0.2)  # Recommanded to train NN
    # weight_init = he_normal()
    net = inputs
    pack = (
            [512, tf.nn.relu, w_initializer],
            [512, tf.nn.relu, w_initializer],
            [512, tf.nn.relu, w_initializer],
            [1, None, w_initializer]
            )
    with tf.variable_scope(name, reuse=reuse):
        for (out_dim, nonlinear, weight_init) in pack:
            net = slim.fully_connected(net, out_dim, activation_fn=nonlinear, weights_initializer=weight_init, trainable=phase_train)
            content['model']['D']['net'].append(str(net))
            content['model']['D']['w_init'].append(str(weight_init))
            print ("D: ", net)
        net_sigmoid = tf.nn.sigmoid(net)
        content['model']['D']['net'].append(str(net_sigmoid))
        print('D sigmoid: ', net_sigmoid)
    return net_sigmoid, net
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

    # Build generator
    x_vector = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    z_vector = tf.placeholder(shape=[None, z_size], dtype=tf.float32)

    g_z = generator(z_vector, phase_train=True, reuse=False, name="gen")
    d_output_ax, d_output_x = discriminator(x_vector, phase_train=True, reuse=False, name="dis")
    d_output_az, d_output_z = discriminator(g_z, phase_train=True, reuse=True, name="dis")

    # Compute the discriminator accuracy (probability > 0.5 => acc = 1)
    # `n_p_x` : the number of D(x) output which prob approximates 1
    # `n_p_z` : the number of D(G(z)) output which prob approximate 0
    d_output_ax = tf.reshape(d_output_ax, [-1])
    d_output_az = tf.reshape(d_output_az, [-1])
    n_p_x = tf.reduce_sum(tf.cast(d_output_ax > 0.5, tf.int32))  # hope all d_output_ax ~ 1
    n_p_z = tf.reduce_sum(tf.cast(d_output_az <= 0.5, tf.int32)) # hope all d_output_az ~ 0
    # d_acc = tf.divide( n_p_x + n_p_z, 2 * np.prod(d_output_ax.shape.as_list()) )

    # --- [ Sigmoid BCE (strongly)
    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=tf.ones_like(d_output_x))    # for discriminator
    d_loss+= tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.zeros_like(d_output_z))
    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.ones_like(d_output_z))    # for generator
             # + tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=tf.zeros_like(d_output_x))
    g_loss = tf.reduce_mean(g_loss)
    print ("D_loss: ", d_loss)
    print ("G_loss: ", g_loss)

    para_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen")
    para_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dis")
    content['model']['G']['params'] = [str(p) for p in para_g]
    content['model']['D']['params'] = [str(p) for p in para_d]
    print ("PARA_D: ")
    for p in para_d:
        print ("  ", p)
    print ("PARA_G: ")
    for p in para_g:
        print ("  ", p)

    if FLAGS.decay:
        d_lr_decay = tf.train.exponential_decay(d_lr, global_step, 1000, 0.96, staircase=True)
        g_lr_decay = tf.train.exponential_decay(g_lr, global_step, 1000, 0.96, staircase=True)
    else:
        d_lr_decay = d_lr
        g_lr_decay = g_lr

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr_decay,beta1=beta).minimize(d_loss,var_list=para_d, colocate_gradients_with_ops=True)
        optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr_decay,beta1=beta).minimize(g_loss,var_list=para_g, colocate_gradients_with_ops=True)


    # Save summary
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    writer = tf.summary.FileWriter(summary_dir + model_name)
    summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_ax)
    summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_az)
    summary_d_loss   = tf.summary.scalar("d_loss", d_loss)
    summary_g_loss   = tf.summary.scalar("g_loss", g_loss)
    summary_n_p_z    = tf.summary.scalar("n_p_z", n_p_z)
    summary_n_p_x    = tf.summary.scalar("n_p_x", n_p_x)
    # summary_d_acc    = tf.summary.scalar("d_acc", d_acc)
    d_summary_merge = tf.summary.merge([summary_d_loss,
                                        summary_d_x_hist,
                                        summary_d_z_hist,
                                        summary_n_p_x,
                                        summary_n_p_z,
                                        # summary_d_acc
                                        ])

    with open(os.path.join(train_sample_directory,'config.json'), 'w') as fp:
        json.dump(content, fp, indent=4, separators=(',', ': '))
    with open('CONFIG-' + model_name + '.json', 'w') as fp:
        json.dump(content, fp, indent=4, separators=(',', ': '))

    # Train
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer()) #Dont put it after ``saver.restore``
        saver = tf.train.Saver(slim.get_model_variables(), max_to_keep=1)
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
            global_step = tf.train.get_global_step(sess.graph)

        # Plot Ground Truth Distribution
        if FLAGS.x_mode.lower() == ['gaussian', 'g']:
            x = gaussian_mixture_circle(10000, num_mixture, scale=scale, std=std)
        else:
            x = swiss_roll(10000, scale=scale, std=std)
        plot_scatter(x, train_sample_directory, 'scatter_groundtruth'.format(model_name))
        plot_kde(x, train_sample_directory, 'kde_groundtruth'.format(model_name))

        # Training G
        for t in range(0, n_epochs):
            if FLAGS.x_mode.lower() == ['gaussian', 'g']:
                x = gaussian_mixture_circle(batch_size, num_mixture, scale=scale, std=std)
            else:
                x = swiss_roll(batch_size, scale=scale, std=std)
            if FLAGS.z_init.lower() == 'normal':
                z = np.random.normal(0, 1, size=[batch_size, z_size]).astype(np.float32)
            else:
                z = np.random.uniform(-1, 1, size=[batch_size, z_size]).astype(np.float32)

            T_start = time.time()

            for _ in range(FLAGS.d_epoch):
                _, summary_d, disc_loss = sess.run([optimizer_op_d,d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
            _, summary_g, gen_loss = sess.run([optimizer_op_g,summary_g_loss,g_loss],feed_dict={z_vector:z})
            # d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],feed_dict={z_vector:z, x_vector:x})
            n_x, n_z = sess.run([n_p_x, n_p_z],feed_dict={z_vector:z, x_vector:x})
            print (' [epoch {:>5d}] D_loss: {:<15.8e}  G_loss: {:<15.8e} | D(x)->1: {:<10d} D(G(z))->0: {:<10d} | [ETA] {}'.format(
                    t, disc_loss, gen_loss, n_x, n_z, time.time() - T_start))

            # output generated images
            if t % 500 == 0:
                samples_g = sess.run(g_z,feed_dict={z_vector:z}) #type=np.ndarray
                plot_scatter(samples_g, train_sample_directory, 'scatter_{}_'.format(model_name), suffix=t)
                plot_kde(samples_g, train_sample_directory, 'kde_{}_'.format(model_name), suffix=t)
                # --- [ Plot heatmap
                # samples_x = np.mgrid[-4:4:0.1, -4:4:0.1].transpose(1,2,0).reshape(-1,2)
                # predict = sess.run(d_output_ax, feed_dict={x_vector:samples_x})
                # plot_heatmap(samples_x, predict, train_sample_directory, 'hmap_{}_D({})'.format(model_name, t), prob=True)
                # high = (predict > 0.7).reshape(-1)
                # plot_scatter(samples_x[high], train_sample_directory, 'hmap_{}_{}'.format(model_name, t))
                # --- [ Plot heatmap (fast)
                mesh = np.mgrid[-4:4:0.1, -4:4:0.1].transpose(1,2,0).reshape(-1,2)
                predict = sess.run(d_output_ax, feed_dict={x_vector:mesh})
                predict = predict.reshape(80,80).transpose(1,0)
                predict = np.flip(predict, 0)
                plot_heatmap_fast(predict, train_sample_directory, 'hmap_{}_D({})'.format(model_name, t), prob=True)

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

def testGAN(trained_model_path=None):
    z_vector = tf.placeholder(shape=[batch_size, z_size],dtype=tf.float32)
    g_z = generator(z_vector, phase_train=False, reuse=True, name="gen")

    saver = tf.train.Saver(slim.get_model_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, trained_model_path)
        i = 0
        stddev = 0.33

        while True:
            i += 1
            try:
                next_sigma = float( input('Please enter the standard deviation of normal distribution [{}]: '.format(stddev)) or stddev )
                z_sample = np.random.normal(0, abs(next_sigma), size=[batch_size, z_size]).astype(np.float32)
                g_objects = sess.run(g_z, feed_dict={z_vector:z_sample})
                save_visualization(g_objects, save_path=os.path.join(train_sample_directory, '{}_test_{}_{}.jpg'.format(name, i, next_sigma)))
            except:
                break


def main(_):
    global w_initializer
    # Rename model name
    if FLAGS.w_init.lower().replace('_','') in ['henormal', 'henorm', 'hen', 'hn']:
        name = FLAGS.name + 'DGw_{}.DG_{}'.format('heNormal','Relu')
        w_initializer = he_normal()
    elif FLAGS.w_init.lower().replace('_','') in ['heuniform', 'heuni', 'heu', 'hu']:
        name = FLAGS.name + 'DGw_{}.DG_{}'.format('heUniform','Relu')
        w_initializer = he_uniform
    elif FLAGS.w_init.lower().replace('_','') in ['xavnormal', 'xaviernorm', 'xavnorm', 'xavn', 'xn']:
        name = FLAGS.name + 'DGw_{}.DG_{}'.format('xavNormal','Relu')
        w_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    elif FLAGS.w_init.lower().replace('_','') in ['xavuniform', 'xavieruni', 'xavuni', 'xavu', 'xavier', 'xu']:
        name = FLAGS.name + 'DGw_{}.DG_{}'.format('xavUniform','Relu')
        w_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
    elif FLAGS.w_init.lower().replace('_','').split(':')[0] in ['truncnormal', 'truncnorm', 'truncn', 'tn']:
        try:
            w_init_std = float(FLAGS.w_init.split(':')[-1])
        except:
            w_init_std = 0.02
        name = FLAGS.name + 'DGw_{}.DG_{}'.format('truncNorm_%s' % (str(w_init_std).replace('.','')),'Relu')
        w_initializer = tf.truncated_normal_initializer(stddev=w_init_std)
    else:
        logger.error('Cannot find "{}".'.format(FLAGS.w_init))

    # Training phase
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
    else:
        if FLAGS.ckpath:
            testGAN(trained_model_path=FLAGS.ckpath)
        else:
            logger.error("Needs checkpoint path.")

if __name__ == '__main__':
    tf.app.run()

