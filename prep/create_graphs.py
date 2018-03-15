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

from faster_rcnn.nets.resnet_v1 import resnetv1
from faster_rcnn.nets.mobilenet_v1 import mobilenetv1

def canonical_name(x):
    return x.name.split(":")[0]

def create_graph(sess, net, model_checkpoint_filename, output_dir):
    _, name = os.path.split(model_checkpoint_filename)
    base_output_name = os.path.join(output_dir, name).rstrip('.ckpt') + '-complete'
    names = net.create_architecture(mode='TEST', tag='default', num_classes=81, anchor_scales=(8, 16, 32, 64))

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
    for i, v in enumerate([net._image, net._im_info]):
        print(" ===== Input {}: {}".format(i, v.name)) 
    for i, (k, v) in enumerate(names.items()):
        print(" ==== Output {}: {}, {}".format(i, k, v.name))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Mobile Faster R-CNN demo')
    parser.add_argument('--net', required=True, help='Network to use [mobile]',
                        choices=['mobile'], default='mobile')
    parser.add_argument('--checkpoint', required=True, help='Path of the checkpoint')
    parser.add_argument('--output', required=True, help='output directory')
    parser.add_argument('--image', default='../images/004545.jpg', help='Path of the input model trio')
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
    for func in [create_graph]:
        run_func(args, func)

if __name__ == '__main__':
    main()
