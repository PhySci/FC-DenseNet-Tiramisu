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





def predict(img_size=[128, 128]):
    """

    :param img_size:
    :return:
    """
    print("Setting up training procedure ...")
    inp = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1], 3], name='input_images')
    o = Tiramisu(preset_model='FC-DenseNet56', num_classes=2, img_size=[128, 128])
    network = o.build_fc_densenet(inp)
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
    for i, (images, _, file_list) in enumerate(Fib(img_pth='../data/test/images',
                                           batch_size=8, shape=[128, 128], padding=[13, 14, 13, 14])):
        out = sess.run(labels, feed_dict={inp: images})

        for i, file_name in enumerate(file_list):
            cv2.imwrite(os.path.join('../data/test/masks', file_name), out[i, 13:-14, 13:-14]*65535)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    o = Tiramisu(preset_model='FC-DenseNet103', num_classes=2, img_size=[128, 128])
    o.train(batch_size=4, num_epochs=25)



