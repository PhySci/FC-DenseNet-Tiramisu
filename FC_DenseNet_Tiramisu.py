from __future__ import division
import os, time, cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from gen import Fib
import datetime
from tqdm import tqdm
import re
from collections import Counter

class Tiramisu:

    def __init__(self, num_classes=2, img_size=[256, 256], preset_model='FC-DenseNet56', dropout_p=0.2, lr=0.001):
        """
        Constructor
        :param num_classes:
        :param img_size:
        :param preset_model:
        :param dropout_p: dropout rate applied after each convolution (0. for not using)
        :return:
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.preset_model = preset_model
        self.dropout_p = dropout_p
        self.lr = lr
        self.batch_size = 8
        self.global_step = tf.Variable(0, dtype=tf.int8, name='global_step', trainable=False)
        self.is_training = tf.Variable(False, dtype=tf.bool, name='is_training', trainable=False)

        self.inp = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1], 1], name='input_images')
        self.labels = tf.placeholder(dtype=tf.int8, shape=[None, img_size[0], img_size[1]], name='labels')
        self.graph = self.get_graph()
        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer()
        self.metric = self.get_custom_metric()

    def get_optimizer(self):
        """
        Optimizer
        :return:
        """
        with tf.name_scope('optimizer'):
            learning_rate = tf.train.cosine_decay(
                                          learning_rate=self.lr,
                                          global_step=self.global_step,
                                          decay_steps=20000,
                                          alpha=0.1*self.lr,
                                          name='lr_decay')

            return tf.train.AdamOptimizer(learning_rate = self.lr).\
                            minimize(self.loss, global_step=self.global_step)
        
    def get_loss(self):
        """
        Loss
        :return:
        """
        with tf.name_scope('loss'):
            ohe = get_mask(self.labels, self.num_classes)
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.graph, labels=ohe))

    def get_graph(self):
        """
        Return fully-connected net
        :param inputs:
        :param preset_model:
        :param num_classes: number of classes
        :param n_filters_first_conv: number of filters for the first convolution applied
        :param n_pool: number of pooling layers = number of transition down = number of transition up
        :param growth_rate: number of new feature maps created by each layer in a dense block
        :param n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
        :param scope:
        :return:
        """
        n_filters_first_conv = 48

        if self.preset_model == 'FC-DenseNet56':
            n_pool = 5
            growth_rate = 12
            n_layers_per_block = 4
        elif self.preset_model == 'FC-DenseNet67':
            n_pool = 5
            growth_rate = 16
            n_layers_per_block = 5
        elif self.preset_model == 'FC-DenseNet103':
            n_pool = 5
            growth_rate = 16
            n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

        if type(n_layers_per_block) == list:
            assert (len(n_layers_per_block) == 2 * n_pool + 1)
        elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
        else:
            raise ValueError

        # convert mask
        with tf.variable_scope('Tiramisu') as sc:
            # First Convolution (we perform a first convolution).
            stack = slim.conv2d(self.inp, n_filters_first_conv, [3, 3], scope='first_conv')
            n_filters = n_filters_first_conv
            # Downsampling path
            skip_connection_dict = {}

            with tf.name_scope('downsampling'):      
                for i in range(n_pool):
                    # Dense Block
                    stack, _ = self.DenseBlock(stack, n_layers_per_block[i], growth_rate,
                                               scope='denseblock%d' % (i + 1))
                    n_filters += growth_rate * n_layers_per_block[i]
                    # At the end of the dense block, the current stack is stored in the skip_connections list
                    skip_connection_dict.update({i: stack})
                    # Transition Down
                    stack = self.TransitionDown(stack, n_filters, scope='transitiondown%d' % (i + 1))

            #     Bottleneck (Dense Block)
            with tf.name_scope('bottleneck'):
                stack, block_to_upsample = self.DenseBlock(stack, n_layers_per_block[n_pool], growth_rate,
                                                           scope='denseblock%d' % (n_pool + 1))
            #   Upsampling path   #
            with tf.name_scope('upsampling'):      
                for i in range(n_pool):
                    # Transition Up ( Upsampling + concatenation with the skip connection)
                    n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
                    stack = self.TransitionUp(block_to_upsample, skip_connection_dict.get(n_pool-i-1), n_filters_keep,
                                         scope='transitionup%d' % (n_pool + i + 1))

                    # Dense Block
                    # We will only upsample the new feature maps
                    stack, block_to_upsample = self.DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate,
                                                               scope='denseblock%d' % (n_pool + i + 2))

            #      Softmax
            net = slim.conv2d(stack, self.num_classes, [1, 1], scope='logits')
            return net

    def predict(self, test_pth=None, save_pth=None, to_img=False, to_rlc=True):
        """

        :param test_pth:
        :param save_pth:
        :param to_img:
        :param to_rlc:
        :return:
        """

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        print('Loaded latest model checkpoint')
        saver = tf.train.Saver(max_to_keep=1000)
        saver.restore(sess, "checkpoints/latest_model.ckpt")
        fid = open("Output.txt", "w")
        print('id,rle_mask', file=fid)
        print("***** Begin prediction *****")

        if to_rlc:
            fid = open("test-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + ".txt", "w")
            print('id,rle_mask', file=fid)

        for images, _, file_list in tqdm(Fib(img_pth=test_pth,
                                             batch_size=8,
                                             shape=[128, 128],
                                             padding=[13, 14, 13, 14],
                                             flip=False)):

            out = sess.run(self.graph, feed_dict={self.inp: images, self.is_training: False})
            
            answ = np.argmax(out, axis=3)

            for fi, file in enumerate(file_list):
                file_name, ext = os.path.splitext(file)
                if to_rlc:
                    s = file_name+','
                    arr = answ[fi, 13:-14, 13:-14]
                    arr = arr.flatten(order='F')
                    arr = np.insert(arr, 0, 0)
                    arr = np.append(arr, 0)
                    d = np.diff(np.int8(np.greater(arr, 0)))
                    starts = np.where(d == 1)[0]
                    ends = np.where(d == -1)[0]
                    len = ends - starts
                    assert (starts.shape == ends.shape)
                    v = np.stack((starts + 1, len), axis=1)
                    for id in range(v.shape[0]):
                        s += str(v[id, 0]) + ' ' + str(v[id, 1]) + ' '
                    print(s, file=fid)

    def get_custom_metric(self):
        with tf.name_scope('custom_metric'):      
            mask1 = tf.cast(self.labels, dtype=tf.bool, name='bool_mask1')
            mask2 = tf.cast(tf.argmax(self.graph, axis=3), dtype=tf.bool, name='bool_mask1')
            intersection = tf.logical_and(mask1, mask2, name='intersection')
            union = tf.logical_or(mask1, mask2, name='intersection')
            intersection_len = tf.count_nonzero(intersection, axis=[1, 2])
            union_len = tf.count_nonzero(union, axis=[1, 2])
            iou = tf.divide(intersection_len, union_len)
            iou = tf.where(tf.is_nan(iou), tf.ones_like(iou), iou)
            iou = tf.floor(iou*20)
            return iou/20.0

    def train(self, num_epochs=2, batch_size=2, repeat=False):
        """
        Train NN
        :param num_epochs:
        :return:
        """
        self.batch_size = batch_size
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver = tf.train.Saver(max_to_keep=100)

        tf_metric, tf_metric_update = tf.metrics.mean_iou(self.labels, tf.argmax(self.graph, axis=3),
                                                          name="iou", num_classes=self.num_classes)
        running_vars_initializer = tf.variables_initializer(var_list=
                                                            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="iou"))

        with tf.Session(config=config) as sess:
            if repeat:
                saver.restore(sess, "checkpoints/latest_model.ckpt")
            else:
                sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter('./train/'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"),
                                                 sess.graph)
             
            print("***** Begin training *****")
            avg_loss_per_epoch = []
            # Do the training here
            for epoch in range(0, num_epochs):
                count = Counter()
                print('Epoch: '+str(epoch))
                loss = []
                sess.run(running_vars_initializer)
                
                for images, mask, _ in tqdm(Fib(img_pth='./soil/train/images',
                                                mask_pth='./soil/train/masks',
                                                batch_size=batch_size,
                                                shape=[128, 128],
                                                padding=[13, 14, 13, 14],
                                                flip=True),
                                            ascii=True, unit=' batch'):
                    feed_dict = {self.inp: images,
                                 self.labels: mask,
                                 self.is_training: True}
                    _, l = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                    loss.append(l)

                for images, mask, _ in tqdm(Fib(img_pth='./soil/val/images', mask_pth='./soil/val/masks',
                                                batch_size=batch_size, shape=[128, 128],
                                                padding=[13, 14, 13, 14],
                                                flip=False),
                                            ascii=True, unit='batch'):
                    feed_dict = {self.inp: images,
                                 self.labels: mask,
                                 self.is_training: False}
                    _, cn = sess.run([tf_metric_update, self.metric], feed_dict=feed_dict)
                    
                    summary_op = tf.summary.image("plot", images)
                     
                    for el in list(cn):
                        count[el] +=1 
                                             
                iou_score, merge = sess.run([tf_metric, tf.summary.merge_all()], feed_dict=feed_dict)
                print("Mean IoU score: ", iou_score)
                print(count)
                       
                    
                    
                s = 0.0
                cnt = 0                              
                for key, value in count.items():
                    cnt += value
                    if key <0.5:
                        continue
                    s +=key*value
                try:    
                    s /= cnt
                except ZeroDivisionError:
                    s = 0.0
                print(s)
                stats = tf.Summary()
                stats.value.add(tag='metrics/loss_mean', simple_value=np.array(loss).mean())
                stats.value.add(tag='metrics/iou_mean', simple_value=iou_score)
                stats.value.add(tag='metrics/metric1', simple_value=s)
                train_writer.add_summary(stats, epoch)
                
                # Create directories if needed
                if not os.path.isdir("%s/%04d" % ("checkpoints", epoch)):
                    os.makedirs("%s/%04d" % ("checkpoints", epoch))

                saver.save(sess, "%s/latest_model.ckpt" % "checkpoints")
                saver.save(sess, "%s/%04d/model.ckpt" % ("checkpoints", epoch))

    def preact_conv(self, inputs, n_filters, filter_size=[3, 3], scope='preactivation'):
        """
        Basic pre-activation layer for DenseNets
        Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
        """
        with tf.name_scope('preactivation_'+scope) as sc:
            net = tf.nn.relu(slim.batch_norm(inputs, scope=sc))
            net = slim.conv2d(net, n_filters, filter_size, activation_fn=None, normalizer_fn=None, scope=sc)
            return slim.dropout(net, keep_prob=(1.0-self.dropout_p), scope=sc, is_training=self.is_training)

    def DenseBlock(self, stack, n_layers, growth_rate, scope=None):
        """
          DenseBlock for DenseNet and FC-DenseNet
        :param stack: input 4D tensor
        :param n_layers: number of internal layers
        :param growth_rate: number of feature maps per internal layer
        :param scope:
        :return: stack, new_features:  current stack of feature maps (4D tensor) and  4D tensor containing only the new feature maps generated
        """
        with tf.name_scope(scope) as sc:
            new_features = []
            for j in range(n_layers):
                with tf.name_scope('layer_' + str(j)) as sc1:
                    layer = self.preact_conv(stack, growth_rate, scope='layer_' + str(j))
                    new_features.append(layer)
                    stack = tf.concat([stack, layer], axis=-1)
            new_features = tf.concat(new_features, axis=-1)
            return stack, new_features

    def TransitionDown(self, inputs, n_filters, scope=None):
        """
        Transition Down (TD) for FC-DenseNet
        Apply 1x1 BN + ReLU + conv then 2x2 max pooling
        """
        with tf.name_scope(scope):
            l = self.preact_conv(inputs, n_filters, filter_size=[1, 1])
            l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
            return l

    def TransitionUp(self, block_to_upsample, skip_connection, n_filters_keep, scope=None):
        """
        Transition Up for FC-DenseNet
        Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection
        """
        with tf.name_scope(scope) as sc:
            l = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2],
                                      scope=sc)
            return tf.concat([l, skip_connection], axis=-1)

def get_mask(mask, n_classes):
    """
    Return OHE mask
    :param mask:
    :param n_classes:
    :return:
    """
    with tf.name_scope('get_mask'):
        msk_list = list()
        for cls in range(n_classes):
            cls_msk = tf.to_int32(tf.equal(mask, cls))
            msk_list.append(cls_msk)
        return tf.transpose(tf.stack(values=msk_list, axis=0, name='concat'), perm=[1, 2, 3, 0])




