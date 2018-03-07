#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os

from faster_rcnn.lib.nets.resnet_v1 import resnetv1
from faster_rcnn.lib.nets.mobilenet_v1 import mobilenetv1

def canonical_name(x):
    return x.name.split(":")[0]

def create_initial_graph(sess, net, model_checkpoint_filename, output_dir):
    _, name = os.path.split(model_checkpoint_filename)
    base_output_name = os.path.join(output_dir, name).rstrip('.ckpt') + '-initial'
    names = net.create_initial_architecture(num_classes=21)
    saver = tf.train.Saver()
    print("Restoring initial graph...")
    saver.restore(sess, model_checkpoint_filename)
    out_tensors = sorted(names.values(), key=lambda n: n.name)
    print("Freezing graph...")
    frozen_graphdef = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, map(canonical_name, out_tensors))
    frozen_graphdef_out = base_output_name + '.pb'
    print("Writing frozen graph...")
    with open(frozen_graphdef_out, 'w') as f:
        f.write(frozen_graphdef.SerializeToString())
    for i, n in enumerate(map(canonical_name, out_tensors)):
        print(" ==== Output {}: {}".format(i, n))
    print("Converting to tflite...")
    tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [net._image], out_tensors)
    tflite_model_out = base_output_name + '.tflite'
    print("Writing tflite model...")
    with open(tflite_model_out, 'w') as f:
        f.write(tflite_model)

def create_last_layer_graph(sess, net, model_checkpoint_filename, output_dir):
    _, name = os.path.split(model_checkpoint_filename)
    base_output_name = os.path.join(output_dir, name).rstrip('.ckpt') + '-last_layer'
    names = net.create_last_layer_architecture()
    saver = tf.train.Saver()
    print("Restoring initial graph...")
    saver.restore(sess, model_checkpoint_filename)
    out_tensors = sorted(names.values(), key=lambda n: n.name)
    print("Freezing graph...")
    frozen_graphdef = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, map(canonical_name, out_tensors))
    frozen_graphdef_out = base_output_name + '.pb'
    print("Writing frozen graph...")
    with open(frozen_graphdef_out, 'w') as f:
        f.write(frozen_graphdef.SerializeToString())
    for i, n in enumerate(map(canonical_name, out_tensors)):
        print(" ==== Output {}: {}".format(i, n))
    print("Converting to tflite...")
    tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [net._last_layer_input], out_tensors)
    tflite_model_out = base_output_name + '.tflite'
    print("Writing tflite model...")
    with open(tflite_model_out, 'w') as f:
        f.write(tflite_model)

def create_inference_graph(sess, net, model_checkpoint_filename, output_dir):
    _, name = os.path.split(model_checkpoint_filename)
    base_output_name = os.path.join(output_dir, name).rstrip('.ckpt') + '-inference'
    names = net.create_inference_architecture()
    saver = tf.train.Saver()
    print("Restoring initial graph...")
    saver.restore(sess, model_checkpoint_filename)
    out_tensors = sorted(names.values(), key=lambda n: n.name)
    print("Freezing graph...")
    frozen_graphdef = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, map(canonical_name, out_tensors))
    frozen_graphdef_out = base_output_name + '.pb'
    print("Writing frozen graph...")
    with open(frozen_graphdef_out, 'w') as f:
        f.write(frozen_graphdef.SerializeToString())
    for i, n in enumerate(map(canonical_name, out_tensors)):
        print(" ==== Output {}: {}".format(i, n))
    print("Converting to tflite...")
    tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [net._inference_input], out_tensors)
    tflite_model_out = base_output_name + '.tflite'
    print("Writing tflite model...")
    with open(tflite_model_out, 'w') as f:
        f.write(tflite_model)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Mobile Faster R-CNN demo')
    parser.add_argument('--net', required=True, help='Network to use [mobile]',
                        choices=['mobile'], default='mobile')
    parser.add_argument('--checkpoint', required=True, help='Path of the checkpoint')
    parser.add_argument('--output', required=True, help='output directory')
    args = parser.parse_args()

    return args

def get_net(net_name):
    if net_name == 'mobile':
        return mobilenetv1()
    else:
        raise NotImplementedError

def run_func(args, func):
    with tf.Session() as sess:
        net = get_net(args.net)
        func(sess, net, args.checkpoint, args.output)

def main():
    args = parse_args()
    for func in [create_initial_graph, create_last_layer_graph, create_inference_graph]:
        try:
            run_func(args, func)
        except RuntimeError:
            print(" --- Something errored...")

if __name__ == '__main__':
    main()
