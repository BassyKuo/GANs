#!/usr/bin/env python3
'''
DCGAN
'''
import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
from skimage.transform import resize

from logger import logger
from utils.utils import *

from utils.load_data import load_imagenet
from utils.imagenet64_labels import labels as IMGNET_LABELS
from utils.benchmark import get_inception_score

model_name = '01-imagenet_DCGAN_slim'
summary_dir = './summary/'
train_sample_directory = './[result]' + model_name
model_directory = './__models__/'

'''
Flags
'''
args = tf.app.flags
args.DEFINE_boolean('train', True, 'True for training, False for testing [%(default)s]')
args.DEFINE_string('name', model_name, 'Model name [%(default)s]')
args.DEFINE_string('ckpath', None, 'Checkpoint path for restoring (ex: models/cifar10.ckpt-100) [%(default)s]')
# args.DEFINE_integer('f_dim', f_dim, 'Dimension of first layer in D [%(default)s]')
args.DEFINE_integer('max_epoch',100000, '[%(default)s]')
args.DEFINE_integer('bs',       256,    '[%(default)s]')
args.DEFINE_integer('z',        100,    '[%(default)s]')
args.DEFINE_float  ('glr',      0.0002, '[%(default)s]')
args.DEFINE_float  ('dlr',      0.0002, '[%(default)s]')
args.DEFINE_float  ('beta',     0.5,    '[%(default)s]')
args.DEFINE_float  ('d_thresh', 0.8,    '[%(default)s]')
args.DEFINE_float  ('leak',     0.2,    '[%(default)s]')
args.DEFINE_boolean('decay',    False,  'Learning rate decay? [%(default)s]')
args.DEFINE_boolean('Dp',       False,  'Update D again if D_acc < D_threshold? [%(default)s]')
args.DEFINE_boolean('Gp',       False,  'Update G again if D_acc > D_threshold? [%(default)s]')
FLAGS = args.FLAGS

'''
Global Parameters
'''
n_epochs   = FLAGS.max_epoch
batch_size = FLAGS.bs
z_size     = FLAGS.z
g_lr       = FLAGS.glr
d_lr       = FLAGS.dlr
beta       = FLAGS.beta
d_thresh   = FLAGS.d_thresh
leak_value = FLAGS.leak

img_len    = 64
img_channel= 3

f_dim = [1024, 512, 256, 128, img_channel]

# --- [ DCGAN (Radford rt al., 2015)
def generator(z, batch_size=batch_size, phase_train=True, reuse=False, use_bias=False, name="gen"):

    weight_init = tf.truncated_normal_initializer(stddev=0.02)  # Recommanded to train NN
    # weight_init = tf.random_uniform_initializer(minval=0.02*np.sqrt(3), maxval=0.02*np.sqrt(3), dtype=tf.float32)

    with tf.variable_scope(name, reuse=reuse):
        for i, dim in enumerate(f_dim):
            # FIRST LAYER
            if i == 0:
                net = slim.fully_connected(z, 4*4*dim, activation_fn=None, weights_initializer=weight_init, trainable=phase_train)
                net = tf.reshape(net, [batch_size, 4, 4, dim])
            # HIDDEN LAYER
            elif i < len(f_dim) - 1:
                net = slim.conv2d_transpose(net, dim, (5,5), stride=2, padding='SAME', weights_initializer=weight_init, trainable=phase_train,
                        activation_fn=None, normalizer_fn=slim.batch_norm, normalizer_params={'epsilon':1e-5, 'trainable':phase_train, 'fused': True})
                net = tf.nn.relu(net)
            # LAST LAYER
            else:
                net = slim.conv2d_transpose(net, dim, (5,5), stride=2, padding='SAME', weights_initializer=weight_init, trainable=phase_train, activation_fn=tf.nn.tanh)
            print ("G{}: ".format(i), net)
        net = tf.reshape(net, [batch_size, img_len, img_len, img_channel])
        print ("G output: ", net)

    return net

# ---
def discriminator(inputs, phase_train=True, reuse=False, use_bias=False, name="dis"):

    weight_init = tf.truncated_normal_initializer(stddev=0.02)  # Recommanded to train NN
    # weight_init = tf.random_uniform_initializer(minval=0.02*np.sqrt(3), maxval=0.02*np.sqrt(3), dtype=tf.float32)

    with tf.variable_scope(name, reuse=reuse):
        for i, dim in enumerate(f_dim[::-1][1:]):
            # FIRST LAYER
            if i == 0:
                net = slim.conv2d(inputs, dim, (5,5), stride=2, padding='SAME', weights_initializer=weight_init, trainable=phase_train, activation_fn=lrelu)
            # HIDDEN LAYER
            else:
                net = slim.conv2d(net, dim, (5,5), stride=2, padding='SAME', weights_initializer=weight_init, trainable=phase_train,
                        activation_fn=None, normalizer_fn=slim.batch_norm, normalizer_params={'epsilon':1e-5, 'trainable':phase_train, 'fused': True})
                net = lrelu(net)
            print ("D{}: ".format(i), net)
            if i == len(f_dim[::-1][1:]) -2:
                feature_map = net
        # LAST LAYER
        length = np.prod(net.shape.as_list()[1:])
        net = tf.reshape(net, shape=[-1, length])
        net = slim.fully_connected(net, 1, activation_fn=None, weights_initializer=weight_init, trainable=phase_train)
        net_sigmoid = tf.nn.sigmoid(net)
        print('D output: ', net)

    return net_sigmoid, net, feature_map
# --- ]


def trainGAN(checkpoint=None, model_name=model_name):
    # Load imagenet64 data
    imagenet = load_imagenet(label=IMGNET_LABELS)
    images = imagenet.train.data.reshape([-1, 64,64,3])
    labels = imagenet.train.labels
    print ("[!] IMAGENET64 count : ", len(images))
    print ("[!] IMAGENET64 images (max, min) : ", images.max(), images.min())
    train = images

    global_step = tf.train.get_or_create_global_step()
    slim.add_model_variable(global_step)
    with tf.variable_scope('global_step_update', reuse=True):
        global_step_update = tf.assign_add(global_step, 1)
    with tf.variable_scope('inception'):
        inp_score = tf.placeholder(tf.float32, name='score')
        # inp_score = tf.get_variable('score', shape=[], initializer=tf.zeros_initializer())
        # inp_std = tf.get_variable('std', shape=[], initializer=tf.zeros_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    # Build generator
    x_vector = tf.placeholder(shape=[batch_size, img_len, img_len, img_channel], dtype=tf.float32)
    z_vector = tf.placeholder(shape=[batch_size, z_size], dtype=tf.float32)

    g_z = generator(z_vector, phase_train=True, reuse=False, name="gen")
    d_output_ax, d_output_x, d_feature_x = discriminator(x_vector, phase_train=True, reuse=False, name="dis")
    d_output_az, d_output_z, d_feature_z = discriminator(g_z, phase_train=True, reuse=True, name="dis")

    # Compute the discriminator accuracy (probability > 0.5 => acc = 1)
    # `n_p_x` : the number of D(x) output which prob approximates 1
    # `n_p_z` : the number of D(G(z)) output which prob approximate 0
    d_output_ax = tf.reshape(d_output_ax, [-1])
    d_output_az = tf.reshape(d_output_az, [-1])
    n_p_x = tf.reduce_sum(tf.cast(d_output_ax > 0.5, tf.int32))  # hope all d_output_ax ~ 1
    n_p_z = tf.reduce_sum(tf.cast(d_output_az <= 0.5, tf.int32)) # hope all d_output_az ~ 0
    d_acc = tf.divide( n_p_x + n_p_z, 2 * np.prod(d_output_ax.shape.as_list()) )

    # Compute the discriminator and generator loss
    # --- [ Bad Accuracy
    # d_loss = -tf.reduce_mean(tf.log(d_output_ax) + tf.log(1-d_output_az))
    # g_loss = -tf.reduce_mean(tf.log(d_output_az))

    # --- [ Sigmoig BCE (strongly)
    # d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=tf.ones_like(d_output_x)) + \
    #          tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.zeros_like(d_output_z))
    # g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.ones_like(d_output_z))
    #          # tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=tf.zeros_like(d_output_x))
    # d_loss = tf.reduce_mean(d_loss)
    # g_loss = tf.reduce_mean(g_loss)

    # --- [ Trick: label smoothing
    d_output_x = tf.reshape(d_output_x, [-1])
    d_output_z = tf.reshape(d_output_z, [-1])
    d_output_shape = d_output_ax.get_shape().as_list()
    smooth_ones  = tf.random_uniform(d_output_shape, 0.7, 1.0)
    smooth_zeros = tf.random_uniform(d_output_shape, 0.0, 0.3)

    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=smooth_ones) + \
             tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=smooth_zeros)
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=smooth_ones) + \
             tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=smooth_zeros)
    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)

    # --- [ Softmax BCE (D output: tanh)
    # logits = tf.concat( (d_output_ax, d_output_az), axis=0 )
    # d_ground_truth = tf.concat( (tf.random_uniform([batch_size], 0.7, 1.0), -tf.random_uniform([batch_size], 0.7, 1.0)), axis=0 )
    # g_ground_truth = tf.concat( (-tf.random_uniform([batch_size], 0.7, 1.0), tf.random_uniform([batch_size], 0.7, 1.0)), axis=0 )

    # d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=d_ground_truth)
    # g_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=g_ground_truth)

    # --- [ Feature Loss (Salimans et al., 2016)
    # d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=tf.ones_like(d_output_x)) + \
    #          tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.zeros_like(d_output_z))
    # g_loss = tf.nn.l2_loss(tf.reduce_mean(d_feature_x, axis=0) - tf.reduce_mean(d_feature_z, axis=0)) + \
    #          tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.ones_like(d_output_z))
    # d_loss = tf.reduce_mean(d_loss)
    # g_loss = tf.reduce_mean(g_loss)

    print ("D_loss: ", d_loss)
    print ("G_loss: ", g_loss)

    para_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen")
    para_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dis")

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
    with tf.variable_scope(model_name, reuse=True):
        summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_ax)
        summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_az)
        summary_d_loss   = tf.summary.scalar("d_loss", d_loss)
        summary_g_loss   = tf.summary.scalar("g_loss", g_loss)
        summary_n_p_z    = tf.summary.scalar("n_p_z", n_p_z)
        summary_n_p_x    = tf.summary.scalar("n_p_x", n_p_x)
        summary_d_acc    = tf.summary.scalar("d_acc", d_acc)
        summary_inp_score = tf.summary.scalar("inp_score", inp_score)
        d_summary_merge = tf.summary.merge([summary_d_loss,
                                            summary_d_x_hist,
                                            summary_d_z_hist,
                                            summary_n_p_x,
                                            summary_n_p_z,
                                            summary_d_acc])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(slim.get_model_variables(), max_to_keep=1)
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
            global_step = tf.train.get_global_step(sess.graph)

        z_val = np.random.normal(0, 1, size=[batch_size, z_size]).astype(np.float32)
        # z_val = np.random.uniform(-1, 1, size=[batch_size, z_size]).astype(np.float32)

        tensors_prev = []
        for epoch in range(sess.run(global_step), n_epochs):
            idx = np.random.randint(len(train), size=batch_size)
            x = train[idx]
            z = np.random.normal(0, 1, size=[batch_size, z_size]).astype(np.float32)
            # z = np.random.uniform(-1, 1, size=[batch_size, z_size]).astype(np.float32)

            T_start = time.time()

            for _ in range(1):
                # _,  disc_loss = sess.run([optimizer_op_d,d_loss],feed_dict={z_vector:z, x_vector:x})
                _, summary_d, disc_loss = sess.run([optimizer_op_d,d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
            # _,  gen_loss, _ = sess.run([optimizer_op_g,g_loss,d_loss],feed_dict={z_vector:z, x_vector:x})
            _, summary_g, gen_loss = sess.run([optimizer_op_g,summary_g_loss,g_loss],feed_dict={z_vector:z, x_vector:x})
            d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],feed_dict={z_vector:z, x_vector:x})
            print (' [epoch {:>5d}] D_loss: {:<15.8e}  G_loss: {:<15.8e} | D(x)->1: {:<10d} D(G(z))->0: {:<10d} | D_acc: {:<10} | [ETA] {}'.format(
                    epoch, disc_loss, gen_loss, n_x, n_z, d_accuracy, time.time() - T_start))

            if FLAGS.Dp and d_accuracy < d_thresh:
                # _,  disc_loss, gen_loss = sess.run([optimizer_op_d,d_loss,g_loss],feed_dict={z_vector:z, x_vector:x})
                _, summary_d, disc_loss = sess.run([optimizer_op_d,d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
                d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],feed_dict={z_vector:z, x_vector:x})
                print ('D[epoch {:>5d}] D_loss: {:<15.8e}  G_loss: {:<15.8e} | D(x)->1: {:<10d} D(G(z))->0: {:<10d} | D_acc: {}'.format(
                    epoch, disc_loss, gen_loss, n_x, n_z, d_accuracy))
            if FLAGS.Gp and d_accuracy > d_thresh:
                # _,  gen_loss, disc_loss = sess.run([optimizer_op_g,g_loss,d_loss],feed_dict={z_vector:z, x_vector:x})
                _, summary_g, gen_loss = sess.run([optimizer_op_g,summary_g_loss,g_loss],feed_dict={z_vector:z, x_vector:x})
                d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],feed_dict={z_vector:z, x_vector:x})
                print ('G[epoch {:>5d}] D_loss: {:<15.8e}  G_loss: {:<15.8e} | D(x)->1: {:<10d} D(G(z))->0: {:<10d} | D_acc: {}'.format(
                    epoch, disc_loss, gen_loss, n_x, n_z, d_accuracy))

            # output generated images
            if epoch % 1000 == 0:
                g_train = sess.run(g_z,feed_dict={z_vector:z}) #type=np.ndarray
                g_val   = sess.run(g_z,feed_dict={z_vector:z_val}) #type=np.ndarray
                if not os.path.exists(train_sample_directory):
                    os.makedirs(train_sample_directory)
                save_visualization(g_train, save_path=os.path.join(train_sample_directory, '{}_{}.jpg'.format(model_name, epoch)))
                save_visualization(g_val, save_path=os.path.join(train_sample_directory, '{}_val_{}.jpg'.format(model_name, epoch)))
                save_visualization(x, save_path=os.path.join(train_sample_directory, '{}_real_{}.jpg'.format(model_name, epoch)))
                # --- [ in Python2, dump as a pickle file. Loading this file by `np.load(filename, encoding='latin1')` in Python3 ] ---
                # g_val.dump(os.path.join(train_sample_directory, '{}_{}.pkl'.format(model_name, epoch)))
                # ---------------------------------------------------------------------------------------------------------------------

            # save checkpoint
            if epoch % 1000 == 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, save_path = os.path.join(model_directory, '{}.ckpt'.format(model_name)), global_step=global_step)

            # Calculate ``Inception Score`` (T. Salimans, et al., 2016)
            if epoch % 10 == 0:
                g_train = sess.run(g_z,feed_dict={z_vector:z}) #type=np.ndarray
                g_train = (g_train + 1) * 127.5
                score, std = get_inception_score(list(g_train), splits=1)
                print ()
                print ("[*] Inception score [exp(KL(p(y|x) || p(y))]: ", score)
                # print ("[*] Difference in each split (standard deviation): ", std)
                print ()
                summary_inp = sess.run(summary_inp_score, feed_dict={inp_score: score})
                writer.add_summary(summary_inp, epoch)

            writer.add_summary(summary_d, epoch)
            writer.add_summary(summary_g, epoch)
            writer.flush()
            # tensors = tf.contrib.graph_editor.get_tensors(sess.graph)
            # print ('[*] tensors: ')
            # print (set(tensors) ^ set(tensors_prev))
            # print ('[*] count of tensors: ', len(tensors))
            # tensors_prev = tensors
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

def save_visualization(objs, save_path='./train_sample/sample.jpg'):
    size = (int(scipy.ceil(scipy.sqrt(len(objs)))), int(scipy.ceil(scipy.sqrt(len(objs)))))
    h, w = objs.shape[1], objs.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for n, obj in enumerate(objs):
        row = int(n / size[1])
        col = n % size[1]
        img[row*h:(row+1)*h, col*w:(col+1)*w, :] = obj

    scipy.misc.imsave(save_path, img)
    print ('[!] Save ', save_path)


def main(_):
    if FLAGS.train:
        name = '{}-{}bs-{}glr-{}dlr'.format(FLAGS.name, FLAGS.bs, FLAGS.glr, FLAGS.dlr)
        name = name + '-decay' if FLAGS.decay else name
        name = name + '-D+' if FLAGS.Dp else name
        name = name + '-G+' if FLAGS.Gp else name
        trainGAN(checkpoint=FLAGS.ckpath, model_name=name)
    else:
        if FLAGS.ckpath:
            testGAN(trained_model_path=FLAGS.ckpath)
        else:
            logger.error("Needs checkpoint path.")

if __name__ == '__main__':
    tf.app.run()

