import os, time, cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime

from utils import *

import matplotlib.pyplot as plt

from FC_DenseNet_Tiramisu import build_fc_densenet

from gen import Fib

def count_params():
    """
    Count total number of parameters in the model
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

def get_mask(mask):
    n_classes = 12
    msk_list = list()
    for cls in range(n_classes):
        cls_msk = tf.to_int32(tf.equal(mask, cls))
        msk_list.append(cls_msk)
    return tf.transpose(tf.stack(values=msk_list, axis=0, name='concat'), perm=[1, 2, 3, 0])


def predict(img_size=[128, 128]):
    class_dict = {0: "Negative",
                  1: "Positive"}

    n_classes = len(class_dict)

    print("Setting up training procedure ...")

    inp = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1], 3], name='input_images')

    network = build_fc_densenet(inp, preset_model='FC-DenseNet56', n_classes=2)
    labels = tf.argmax(network, axis=3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    print('Loaded latest model checkpoint')
    saver.restore(sess, "checkpoints/latest_model.ckpt")

    # graph = tf.Graph()
    # with graph.as_default():

    print("***** Begin prediction *****")
    for i, (images, _, file_list) in enumerate(Fib(img_pth='./soil/train/images',
                                           batch_size=8, shape=[128, 128], padding=[13, 14, 13, 14])):
        out = sess.run(labels, feed_dict={inp: images})

        for i, file_name in enumerate(file_list):
            cv2.imwrite(file_name+'.png', out[i, :, :]*65535)


if __name__ == '__main__':
    #train()
    predict()


def train(img_size=[128, 128]):
    num_epochs = 250

    class_dict = {0: "Negative",
                  1: "Positive"}

    n_classes = len(class_dict)

    print("Setting up training procedure ...")

    inp = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1], 3], name='input_images')
    labels = tf.placeholder(dtype=tf.int8, shape=[None, img_size[0], img_size[1]], name='labels')

    network = build_fc_densenet(inp, preset_model='FC-DenseNet56', n_classes=n_classes)
    ohe = get_mask(labels)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=ohe))
    opt = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.995).minimize(loss)
    tf_metric, tf_metric_update = tf.metrics.mean_iou(labels, tf.argmax(network, axis=3), name="iou", num_classes=n_classes)

    is_training = True
    continue_training = False
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    if continue_training or not is_training:
        print('Loaded latest model checkpoint')
        saver.restore(sess, "checkpoints/latest_model.ckpt")

    avg_scores_per_epoch = []

    #graph = tf.Graph()
    #with graph.as_default():

    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)


    print("***** Begin training *****")
    avg_loss_per_epoch = []
    # Do the training here
    for epoch in range(0, num_epochs):
        current_losses = []
        sess.run(running_vars_initializer)
        for i, (images, mask) in enumerate(Fib(img_pth='./soil/train/images', mask_pth='./soil/train/masks',
                                batch_size=8, shape=[128, 128], padding=[13, 14, 13, 14])):
            _, l = sess.run([opt, loss], feed_dict={inp: images, labels: mask})
            current_losses.append(l)
            print(i, '/500:', l)

            # estimate quality
            #sess.run(tf_metric_update, feed_dict={inp: images, labels: mask})
            #score = sess.run(tf_metric)
            #print("[TF] SCORE: ", score)

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)
        print(f'Epoch {epoch}: {mean_loss}')

        # Create directories if needed
        if not os.path.isdir("%s/%04d" % ("checkpoints", epoch)):
            os.makedirs("%s/%04d" % ("checkpoints", epoch))

        saver.save(sess, "%s/latest_model.ckpt" % "checkpoints")
        saver.save(sess, "%s/%04d/model.ckpt" % ("checkpoints", epoch))
        #@TODO add validation

