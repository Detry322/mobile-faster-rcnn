# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from faster_rcnn.model.config import cfg
from faster_rcnn.nms.py_cpu_nms import py_cpu_nms

def nms(dets, thresh, force_cpu=False):
  return py_cpu_nms(dets, thresh)
