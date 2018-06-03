"""
Data augmentation methods.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from SpatialRelationCNN.utility import utility

import random
from math import pi
from copy import deepcopy
import numpy as np

import tensorflow as tf

import tfquaternion as tfq

# augmentation parameters
MAX_TRANSLATION_FACTOR = 2


@utility.scope_wrapper
def scene_description_triplet(scenes, dataset, more_augmentation=False):
    """Apply augmentation to the scene descriptions.

    Args:
        scenes: A list of three `data_io.relation_dataset.SceneDescription`s.

    Returns:
        A tuple, the augmented data_io.`relation_dataset.SceneDescription`s
        and a bool indicating if the instance was augmented with cloning.
    """
    # with a probability of 0.1: clone the reference scene and add a small
    # random transformation on the first clone and the same random
    # transformation times a factor on the second clone. We only do this for
    # classes with non-tilted objects to reduce the probability of creating
    # physically infeasible scenes
    assert len(scenes) == 3
    if (more_augmentation and random.uniform(0, 1) <= 0.3 and
            dataset.scene_to_label[scenes[0].name] in (0, 1, 2, 4)):
        scenes[2] = deepcopy(scenes[0])
        # get values in [-1, -0.5] | [0.5, 1]
        translation_delta = 0.5 - np.random.rand(3)
        translation_delta += np.sign(translation_delta) * 0.5
        translation_delta *= scenes[0].span / 10
        translation_delta += (0.5 * np.random.rand(3) *
                              np.sign(translation_delta))
        scenes[2].translations += np.stack(((0, 0, 0), translation_delta))
        return scenes, True
    # otherwise apply per scene augmentation (swap channels, random transform)
    return [scene_description(s, i == 2)
            for i, s in enumerate(scenes)], False


@utility.scope_wrapper
def scene_description(scene, random_transformation=False):
    """Apply augmentation to the scene description.

    Args:
        scene: A data_io.relation_dataset.SceneDescription.
        random_transformation: `bool`, if true, the scene's transformation
            is chosen randomly. (Used to create scenes that are physically
            infeasible with high probability.)

    Returns:
        The augmented data_io.relation_dataset.SceneDescription.
    """
    # with a probability of 0.2 sample a scene with random transformations
    scene = deepcopy(scene)
    if random_transformation and random.uniform(0, 1) <= 0.2:
        scene.translations += (scene.span * MAX_TRANSLATION_FACTOR *
                               (2 * np.random.rand(2, 3) - 1))
        scene.rotations += np.random.rand(2, 4)
    # swap channels (with a probability of 0.5)
    if random.uniform(0, 1) < 0.5:
        scene.translations = np.array([scene.translations[1],
                                       scene.translations[0]])
        scene.rotations = scene.rotations[1], scene.rotations[0]
        scene.cloud_ids = scene.cloud_ids[1], scene.cloud_ids[0]
        scene.obj1, scene.obj2 = scene.obj2, scene.obj1
    return scene


@utility.scope_wrapper
def pointcloud(points, segment_ids, batch_size, training_batch_size):
    """Augment the point cloud during training, rotate around the z axis."""
    def augment(points, segment_ids, training_batch_size):
        """Rotate a full scene around the z axis and scale on each axis."""
        scene_ids = segment_ids // 2
        bs = training_batch_size * 3
        rand_angle = tf.random_uniform([bs], maxval=(2 * pi))
        rot = tf.stack([[1.] * bs, [0.] * bs, [0.] * bs, rand_angle], axis=1)
        rotations = tfq.Quaternion(tf.gather(rot, scene_ids))
        points = tfq.rotate_vector_by_quaternion(rotations, points, 2, 2)
        return points

    # only augment during training
    return tf.cond(tf.equal(batch_size, 1),
                   lambda: tf.identity(points),
                   lambda: augment(points, segment_ids,
                                   training_batch_size))
