"""Inference example code.

Example code, demonstrating how the pre-trained model can be used to predict
the similarity of the spatial relations between two objects in two scenes.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from SpatialRelationCNN.model.model import SpatialRelationModel
import SpatialRelationCNN.model.utility as util


# Set this to the directory containing the pre-trained weights.
MODEL_CHECKPOINT_DIR = ""


def sphere_cloud(center, radius, num_points=3000):
    """Generate a sphere point cloud of randomly sampled points."""
    x = np.random.normal(size=(num_points, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True) * radius + center

# create the model
points = tf.placeholder(tf.float32, (None, 3), name="points")
segment_ids = tf.placeholder(tf.float32, None, name="ids")
model = SpatialRelationModel(cloud_tensor=points, id_tensor=segment_ids)

# create two scenes
# scene 1 is a small sphere on top of larger sphere
sphere1 = sphere_cloud((0, 0, 0), radius=1, num_points=4000)
sphere2 = sphere_cloud((0, 0, 1.5), radius=0.5, num_points=3000)
points1 = np.concatenate((sphere1, sphere2))
ids1 = np.concatenate((np.zeros(4000), np.ones(3000)))

# scene 2 is two spheres next to each other, i.e. dissimilar to scene 1.
points2 = np.concatenate((sphere1, sphere1 + np.array((0, 2, 0))))
ids2 = np.concatenate((np.zeros(4000), np.ones(4000)))

# scene 3 is an identical spheres on-top of another, i.e. similar to scene 1.
points3 = np.concatenate((sphere1, sphere1 + np.array((0, 0, 2))))
ids3 = np.concatenate((np.zeros(4000), np.ones(4000)))

# run the model
with tf.Session() as sess:
    # load the pre-trained model
    sess.run(tf.global_variables_initializer())
    if MODEL_CHECKPOINT_DIR:
        util.load_variables(model, sess, MODEL_CHECKPOINT_DIR)
    # Compute the distance between scene 1 and scene 2
    # and between scene 1 and scene 3.
    # Note that all points of each object/sphere need to have a unique ID.
    fd = {points: np.concatenate((points1, points2, points1, points3)),
          segment_ids: np.concatenate((ids1, ids2 + 2, ids1 + 4, ids3 + 6)),
          model.dropout_prob: 0.0}  # set dropout to 0 for inference
    # note that model.scene_distance is the distance between the embeddings
    # of the scenes divided by the moving mean distance of the training
    d = sess.run(model.scene_distance, feed_dict=fd)
    print("Distance is {}.".format(d))
