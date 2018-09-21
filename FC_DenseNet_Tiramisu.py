from __future__ import division
import os, time, cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from gen import Fib
import datetime
from tqdm import tqdm
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

        self.inp = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1], 3], name='input_images')
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
        return tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.995).minimize(self.loss)

    def get_loss(self):
        """
        Loss
        :return:
        """
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
        :param dropout_p: dropout rate applied after each convolution (0. for not using)
        :param scope:
        :return:
        """
        n_filters_first_conv = 48
        dropout_p = 0.2
        scope = 'Tiramisu'

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
        with tf.variable_scope(scope, self.preset_model, [self.inp]) as sc:
            # First Convolution (we perform a first convolution).
            stack = slim.conv2d(self.inp, n_filters_first_conv, [3, 3], scope='first_conv')
            n_filters = n_filters_first_conv
            # Downsampling path
            skip_connection_list = []
            for i in range(n_pool):
                # Dense Block
                stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p,
                                      scope='denseblock%d' % (i + 1))
                n_filters += growth_rate * n_layers_per_block[i]
                # At the end of the dense block, the current stack is stored in the skip_connections list
                skip_connection_list.append(stack)
                # Transition Down
                stack = TransitionDown(stack, n_filters, dropout_p, scope='transitiondown%d' % (i + 1))

            skip_connection_list = skip_connection_list[::-1]
            #     Bottleneck (Dense Block)
            stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p,
                                                  scope='denseblock%d' % (n_pool + 1))
            #   Upsampling path   #
            for i in range(n_pool):
                # Transition Up ( Upsampling + concatenation with the skip connection)
                n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
                stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep,
                                     scope='transitionup%d' % (n_pool + i + 1))

                # Dense Block
                # We will only upsample the new feature maps
                stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p,
                                                      scope='denseblock%d' % (n_pool + i + 2))

            #      Softmax
            net = slim.conv2d(stack, self.num_classes, [1, 1], scope='logits')
            return net

    def predict(self, test_pth=None, save_pth=None):
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
        for images, _, file_list in tqdm(Fib(img_pth=test_pth,
                                             batch_size=8,
                                             shape=[128, 128],
                                             padding=[13, 14, 13, 14],
                                             flip=False)):
            
            out = sess.run(tf.argmax(self.graph, axis=3), feed_dict={self.inp: images})

            for fi, file_name in enumerate(file_list):
                cv2.imwrite(os.path.join(save_pth, file_name), out[fi, 13:-14, 13:-14]* 65535)
                
    def get_custom_metric(self):
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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver = tf.train.Saver(max_to_keep=1000)

        tf_metric, tf_metric_update = tf.metrics.mean_iou(self.labels, tf.argmax(self.graph, axis=3),
                                                          name="iou", num_classes=self.num_classes)
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="iou")
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)


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
                    _, l = sess.run([self.optimizer, self.loss], feed_dict={self.inp: images, self.labels: mask})
                    loss.append(l)

                for images, mask, _ in tqdm(Fib(img_pth='./soil/val/images', mask_pth='./soil/val/masks',
                                                batch_size=batch_size, shape=[128, 128],
                                                padding=[13, 14, 13, 14],
                                                flip=False),
                                            ascii=True, unit='batch'):
                    _, cn = sess.run([tf_metric_update, self.metric], feed_dict={self.inp: images, self.labels: mask})
                     
                    for el in list(cn):
                        count[el] +=1 
                                             
                iou_score = sess.run(tf_metric)
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

                summary = tf.Summary()
                summary.value.add(tag='metrics/loss_mean', simple_value=np.array(loss).mean())
                summary.value.add(tag='metrics/iou_mean', simple_value=iou_score)
                summary.value.add(tag='metrics/metric1', simple_value=s)
                train_writer.add_summary(summary, epoch)
                
                # Create directories if needed
                if not os.path.isdir("%s/%04d" % ("checkpoints", epoch)):
                    os.makedirs("%s/%04d" % ("checkpoints", epoch))

                saver.save(sess, "%s/latest_model.ckpt" % "checkpoints")
                saver.save(sess, "%s/%04d/model.ckpt" % ("checkpoints", epoch))


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

def preact_conv(inputs, n_filters, filter_size=[3, 3], dropout_p=0.2):
    """
    Basic pre-activation layer for DenseNets
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    Dropout (if dropout_p > 0) on the inputs
    """
    preact = tf.nn.relu(slim.batch_norm(inputs))
    conv = slim.conv2d(preact, n_filters, filter_size, activation_fn=None, normalizer_fn=None)
    if dropout_p != 0.0:
      conv = slim.dropout(conv, keep_prob=(1.0-dropout_p))
    return conv

def DenseBlock(stack, n_layers, growth_rate, dropout_p, scope=None):
  """
  DenseBlock for DenseNet and FC-DenseNet
  Args:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer
  Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
  """
  with tf.name_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
      # Compute new feature maps
      layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
      new_features.append(layer)
      # stack new layer
      stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features

def TransitionLayer(inputs, n_filters, dropout_p=0.2, compression=1.0, scope=None):
  """
  Transition layer for DenseNet
  Apply 1x1 BN  + conv then 2x2 max pooling
  """
  with tf.name_scope(scope) as sc:
    if compression < 1.0:
      n_filters = tf.to_int32(tf.floor(n_filters*compression))
    l = preact_conv(inputs, n_filters, filter_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='AVG')

    return l

def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None):
  """
  Transition Down (TD) for FC-DenseNet
  Apply 1x1 BN + ReLU + conv then 2x2 max pooling
  """
  with tf.name_scope(scope) as sc:
    l = preact_conv(inputs, n_filters, filter_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
    return l

def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None):
  """
  Transition Up for FC-DenseNet
  Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection
  """
  with tf.name_scope(scope) as sc:
    # Upsample
    l = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2])
    # Concatenate with skip connection
    l = tf.concat([l, skip_connection], axis=-1)
    return l