# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from faster_rcnn.lib.model.config import cfg

class Network(object):
  def __init__(self):
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}
    self._variables_to_fix = {}

  def _build_initial_network(self, is_training=False):
    is_training = False
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    net_conv = self._image_to_head(is_training)
    with tf.variable_scope(self._scope, self._scope):
      # region proposal network
      rpn_cls_prob, rpn_bbox_pred = self._region_proposal(net_conv, is_training, initializer)

    return net_conv, rpn_cls_prob, rpn_bbox_pred

  def _build_last_layer_network(self, pool5):
    # Run last layer network after doing pooling.
    return self._head_to_tail(pool5, is_training=False)

  def _build_inference_network(self, fc7):
    # Run inference network after doing last layer network, and reduce_mean [1, 2].
    raise NotImplementedError
    with tf.variable_scope(self._scope, self._scope):
      # region classification
      cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
                                                        initializer, initializer_bbox)

    return rois, cls_prob, bbox_pred

  def _region_proposal(self, net_conv, is_training, initializer):
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    self._act_summaries.append(rpn)
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')
    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

    return rpn_cls_prob, rpn_bbox_pred

  def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred

    return cls_prob, bbox_pred

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError

  def create_initial_architecture(self, num_classes, anchor_scales=(8, 16, 32, 64), anchor_ratios=(0.5, 1, 2)):
    self._image = tf.placeholder(tf.float32, shape=[1, 500, 375, 3])

    self._num_classes = num_classes
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)): 
      net_conv, rpn_cls_prob, rpn_bbox_pred = self._build_initial_network()

    return {
      'net_conv': net_conv,
      'rpn_cls_prob': rpn_cls_prob,
      'rpn_bbox_pred': rpn_bbox_pred
    }

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError
