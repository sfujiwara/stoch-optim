# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf


def build_weight():
    with tf.name_scope('output') as scope:
        weight = tf.Variable(
            tf.truncated_normal([1, dim]),
            name='weight'
        )
    return weight


def build_linear_model(weight, x_ph):
    logit = tf.matmul(weight, x_ph, transpose_b=True)
    return logit[0, :]


def build_optimizer(logit, x_ph, y_ph, lmd=1e-1, eta=1e-1):
    alpha = tf.minimum(
        tf.maximum((1 - y_ph * logit + eta * lmd) / (eta * tf.nn.l2_loss(x_ph)), 0),
        1
    )
    v = tf.reduce_mean(alpha * y_ph * tf.transpose(x_ph), 1)
    print v.get_shape()
    update = weight.assign((weight + eta * v) / (eta * lmd + 1))
    return update


if __name__ == '__main__':
    # Load data
    dataset = np.loadtxt('data/libsvm/adult/a1a.csv', delimiter=',', dtype=np.float32)
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape

    # Set constants
    # lmd = 1e-1
    # eta = 1e-1
    n_epoch = 10000
    mini_batch_size = 200

    # Placeholders
    x_ph = tf.placeholder(shape=[None, dim], dtype=tf.float32)
    y_ph = tf.placeholder(shape=[None], dtype=tf.float32)

    # Tensors
    weight = build_weight()
    logit = build_linear_model(weight, x_ph)

    # Build operation to update weight
    update = build_optimizer(logit, x_ph, y_ph)

    # For objective value using `all' data
    norm = tf.nn.l2_loss(weight, name='l2_regularizer')
    hinge = tf.reduce_mean(
        tf.maximum(0., 1 - y * tf.matmul(weight, x, transpose_b=True)),
        name='mean_of_hinge_loss'
    )

    # Create session
    sess = tf.Session()

    # Initialize variables
    init = tf.initialize_all_variables()
    sess.run(init)

    # Set summary writer
    summary_writer = tf.train.SummaryWriter('log', graph_def=sess.graph_def)
    tf.scalar_summary(norm.op.name, norm)
    tf.scalar_summary(hinge.op.name, hinge)
    summary_op = tf.merge_all_summaries()

    # Training iteration
    start = time.time()
    for i in xrange(n_epoch):
        # Select index at random
        # ind = np.random.randint(num)
        ind = np.random.choice(num, size=mini_batch_size, replace=False)
        fd = {
            x_ph: np.atleast_2d(x[ind]),
            y_ph: np.atleast_1d(y[ind])
        }
        sess.run(update, feed_dict=fd)
        if i % 100 == 0:
            print 'Epoch:', i
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, i)
    end = time.time()

    # Show result
    print 'comp.time:', end - start
    print 'training error:', sum(sess.run(logit, feed_dict={x_ph: x, y_ph: y}) * y < 0)
