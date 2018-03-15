import cv2
import argparse
import numpy as np
import tensorflow as tf
import time

from models import load_all

from PIL import Image

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

def img_to_PIL(im):
    fixed_im = im + np.array([[[102.9801, 115.9465, 122.7717]]])
    fixed_im = fixed_im.astype(np.uint8)
    im = cv2.cvtColor(fixed_im, cv2.COLOR_BGR2RGB)
    return Image.fromarray(im)

def run_inference(sess, model, image):
    # return cls prob, bbox pred, and net_conv
    _input = model.get_tensor_by_name('prefix/image:0')
    _im_info = model.get_tensor_by_name('prefix/im_info:0')
    cls_score = model.get_tensor_by_name('prefix/MobilenetV1_4/cls_score/BiasAdd:0')
    cls_prob = model.get_tensor_by_name('prefix/MobilenetV1_4/cls_prob:0')
    bbox_pred = model.get_tensor_by_name('prefix/add:0')
    rois = model.get_tensor_by_name('prefix/MobilenetV1_2/rois/concat:0')

    return sess.run([cls_score, cls_prob, bbox_pred, rois], feed_dict={
        _input: image,
        _im_info: [image.shape[1], image.shape[2], 3]
    })

def draw(image, cls_score, cls_prob, bbox_pred, rois):
    print cls_score.shape
    print cls_prob.shape
    print bbox_pred.shape
    print rois.shape

    image = img_to_PIL(image[0])
    classes_probs = zip(range(len(cls_score)), np.argmax(cls_prob, axis=1), np.max(cls_prob, axis=1))
    non_background_classes = filter(lambda (ind, clss, prob): clss != 0, classes_probs)
    for ind, clss, prob in non_background_classes:
        bbox = rois[ind, 1:]
        if prob > 0.9:
            print bbox
            image.crop(bbox).show()
    image.show()


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
    with tf.Session(graph=model) as sess:
        start = time.clock()
        output = run_inference(sess, model, im)
        end = time.clock()
        print "{} seconds".format(end - start)
        print "{} proposals".format(len(output[0]))
    draw(im, *output)
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
