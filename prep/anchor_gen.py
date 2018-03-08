import tensorflow as tf
import numpy as np

from faster_rcnn.layer_utils.generate_anchors import generate_anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length


def generate_anchors_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
	shift_x = tf.range(width) * feat_stride # width
	shift_y = tf.range(height) * feat_stride # height
	shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
	sx = tf.reshape(shift_x, shape=(-1,))
	sy = tf.reshape(shift_y, shape=(-1,))
	shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
	K = tf.multiply(width, height)
	shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))

	anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
	A = anchors.shape[0]
	anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

	length = K * A
	anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
	
	return tf.cast(anchors_tf, dtype=tf.float32), length





def run_generate_anchors(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
	with tf.Session() as sess:
		height_p = tf.placeholder(tf.int32)
		width_p = tf.placeholder(tf.int32)
		output = generate_anchors_tf(height_p, width_p, feat_stride, anchor_scales, anchor_ratios)
		result = sess.run(output, feed_dict={ height_p: height, width_p: width})
	return result


def main():
	height = 375
	width = 505
	feat_stride = 16
	anchor_scales = (8, 16, 32)
	anchor_ratios = (0.5, 1, 2)


	normal_anchors, normal_length = generate_anchors_pre(height, width, feat_stride, anchor_scales, anchor_ratios)
	tf_anchors, tf_length = run_generate_anchors(height, width, feat_stride, anchor_scales, anchor_ratios)

	assert normal_length == tf_length, "Lengths incorrect"
	assert tf_anchors.shape == normal_anchors.shape, "Shapes incorrect"
	# for i in range(len(tf_anchors)):
	# 	print tf_anchors[i]
	# 	print normal_anchors[i]
	# 	t = (tf_anchors[i] == normal_anchors[i]).all()
	# 	if not t:
	# 		break
	assert (tf_anchors == normal_anchors).all(), "Values incorrect"


if __name__ == "__main__":
	main()
