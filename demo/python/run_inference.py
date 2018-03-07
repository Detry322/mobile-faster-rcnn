import cv2
import argparse
import numpy as np

from models import load_all
from inference import run_inference

def load_image(image_name):
    im = cv2.imread(image_name)
    fixed_im = im - np.array([[[102.9801, 115.9465, 122.7717]]])
    return fixed_im.reshape((1,) + fixed_im.shape)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Mobile Faster R-CNN python demo')
    parser.add_argument('--model', required=True, help='Path of the input model trio')
    parser.add_argument('--image', required=True, help='Path of the input model trio')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    im = load_image(args.image)
    models = load_all(args.model)
    run_inference(models, im)

if __name__ == "__main__":
    main()
