import cv2
import argparse
import numpy as np
import tensorflow as tf

from models import load_all
from inference import run_inference

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def load_image(image_name):
    im = cv2.imread(image_name)
    fixed_im = im - np.array([[[102.9801, 115.9465, 122.7717]]])
    return fixed_im.reshape((1,) + fixed_im.shape)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Mobile Faster R-CNN python demo')
    parser.add_argument('--model', default='../../models/mobile_faster_rcnn_iter_1190000-complete.pb', help='Path of the input model trio')
    parser.add_argument('--image', default='../../images/004545.jpg', help='Path of the input model trio')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    im = load_image(args.image)
    model = load_graph(args.model)
    for i in model.get_operations():
        print i.name
    # run_inference(models, im)

if __name__ == "__main__":
    main()
