import tensorflow as tf

from collections import namedtuple

ROI = namedtuple('ROI', ['cx', 'cy', 'w', 'h'])
Prediction = namedtuple('Prediction', ['category', 'prob'])
Result = namedtuple('Result', ['roi', 'prediction'])

def run_inference(models, image):
	rpn_cls_prob, rpn_bbox_pred, net_conv = part_1_inference(models, image)
	rois = part_1_get_rois(rpn_cls_prob, rpn_bbox_pred)
	best_rois = part_1_get_top_n_with_nms(rois)
	cropped_rois = part_1_crop_maxpool_rois(net_conv, best_rois)

	return [Result(roi, part_2_get_prediction(models, region)) for roi, region in cropped_rois]

def part_1_inference(models, image):
	# return cls prob, bbox pred, and net_conv
	return None, None, None

def part_1_get_rois(rpn_cls_prob, rpn_bbox_pred):
	# return a list of (centerx, centery, width, height) coordinates in normal image coords.
	pass

def part_1_get_top_n_with_nms(rois, n=100):
	# get top n
	return []

def part_1_crop_maxpool_rois(net_conv, top_rois):
	return [part_1_crop_maxpool_roi(net_conv, roi) for roi in top_rois]

def part_1_crop_maxpool_roi(net_conv, roi):
	pass


def part_2_get_prediction(models, region):
	roi_last_layer = part_2a_inference(models, region)
	combined = part_2a_combine(roi_last_layer)
	inferred = part_2b_inference(models, combined)
	return part_2b_get_prediction(inferred)


def part_2a_inference(models, cropped_roi):
	# return the processed roi
	pass

def part_2a_combine(roi_last_layer):
	# average pool [1, 2]
	pass

def part_2b_inference(models, fully_connected_input):
	# run inference for the max layer.
	pass

def part_2b_get_prediction(inference_output):
	# get (Category, probability) pair for most likely thing.
	pass

