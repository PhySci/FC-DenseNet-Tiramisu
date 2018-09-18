import os, time, cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import tqdm

from utils import *

import matplotlib.pyplot as plt

from FC_DenseNet_Tiramisu import Tiramisu

from gen import Fib

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

def get_mask(mask, n_classes):
    """
    Return OHE mask
    :param mask:
    :param n_classes:
    :return:
    """
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

    o = Tiramisu(preset_model='FC-DenseNet56', num_classes=2)

    network = o.build_fc_densenet(inp)
    labels = tf.argmax(network, axis=3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    print('Loaded latest model checkpoint')
    saver.restore(sess, "checkpoints2/latest_model.ckpt")

    # graph = tf.Graph()
    # with graph.as_default():

    print("***** Begin prediction *****")
    for i, (images, _, file_list) in enumerate(Fib(img_pth='./soil/test/images',
                                           batch_size=1, shape=[128, 128], padding=[13, 14, 13, 14])):
        out = sess.run(labels, feed_dict={inp: images})

        for i, file_name in enumerate(file_list):
            cv2.imwrite(os.path.join('./soil/test/masks',file_name), out[i, 13:-14, 13:-14]*65535)


def train():
    o = Tiramisu(preset_model='FC-DenseNet56', num_classes=2, img_size=[128, 128])
    o.train(batch_size=8)

if __name__ == '__main__':
    train()
    #predict()



