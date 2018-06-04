"""
Input functions for training and inference.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from SpatialRelationCNN.model import utility
from SpatialRelationCNN.model.generator_factory import GeneratorFactory
import SpatialRelationCNN.model.augmentation as augment

import numpy as np
import tensorflow as tf

import tfquaternion as tfq


class InputLayer(object):
    """The input pipeline base class, from RelationDataset to projection."""

    def __init__(self, dataset, more_augmentation=False):
        """The input pipeline, from RelationDataset to projection.

        Args:
            dataset: A `RelationDataset` object.
        """
        self.dataset = dataset
        self.generator_factory = GeneratorFactory(self.dataset,
                                                  more_augmentation)
        phases = ["train", "validation", "test"]
        self.generators = {n: None for n in phases}
        self.iterator = None
        self.iterator_init_ops = {n: None for n in phases}
        self.clouds_tensor, self.cloud_slice_indices = \
            self.create_cloud_constants()
        self.obj_ids_pl = tf.placeholder(tf.int32, shape=(None, 2),
                                         name="obj_ids")
        self.translations_pl = tf.placeholder(tf.float32, shape=(None, 2, 3),
                                              name="translations")
        self.rotations_pl = tf.placeholder(tf.float32, shape=(None, 2, 4),
                                           name="rotations")
        self.rotations = None  # stores the resulting rotations ...
        self.translations = None  # ... and  translations when generalizing
        self.translation_vars = []
        self.rotation_vars = []

    @utility.scope_wrapper
    def create_cloud_constants(self):
        """Create two `tf.constant`s of the obj point clouds and their ranges.

        The point clouds have differing numbers of points. To efficiently
        process them, all object point clouds are concatenated into one
        constant. To retrieve them afterwards, we create a second constant with
        shape (N+1), containing the start index for each point cloud with the
        length as an additional index. With this we can use slicing, which
        should be more efficient than using tf.where
        """
        np_clouds = [self.dataset.clouds[n] for n in self.dataset.cloud_names]
        # Create the slice indices as float32, as they'll only be used with
        # tf.gather which has no GPU kernel for integers.
        cloud_slice_indices = np.cumsum([0] + [len(c) for c in np_clouds],
                                        dtype=np.float32)
        tf_clouds = tf.constant(np.concatenate(np_clouds), dtype=tf.float32)
        return tf_clouds, cloud_slice_indices

    def switch_input(self, phase, sess):
        """Switch between test and training data."""
        if phase in self.iterator_init_ops:
            print("Switching input to {}.".format(phase))
            sess.run(self.iterator_init_ops[phase])
        else:
            raise Exception("Invalid phase name, must be one of {}."
                            "".format(self.iterator_init_ops))

    def _create_tf_datasets(self, split, batch_size):
        """Helper function that creates the train and test tf.data.Dataset."""
        out_types = (tf.int32, tf.float32, tf.float32, tf.int32, tf.bool)
        # out_shapes has an additional batch dim (None) and 3 or 1 scenes.
        out_shapes = ((None, None, 2), (None, None, 2, 3), (None, None, 2, 4),
                      (None, None), (None,))
        self.iterator = tf.data.Iterator.from_structure(out_types, out_shapes)
        for p in ["train", "validation", "test"]:
            # generator factory throws if there's no validation data
            try:
                self.generators[p] = self.generator_factory.scene_desc_generator(split, p)
            except ValueError:
                continue
            out_shapes = tuple([np.array(x).shape for x in next(self.generators[p]())])
            d = tf.data.Dataset.from_generator(self.generators[p], out_types,
                                               out_shapes)
            d = d.batch(batch_size if p == "train" else 1)
            # d = d.prefetch(3)
            self.iterator_init_ops[p] = self.iterator.make_initializer(d)

    @staticmethod
    def _repeat(a, repeats, batch_size, training_batch_size):
        """Repeat a[i] repeats[i] times."""
        return tf.cond(tf.equal(batch_size, 1),
                       lambda: utility.repeat(a, repeats, num_repeats=2),
                       lambda: utility.repeat(a, repeats, training_batch_size))

    @utility.scope_wrapper
    def _input_fn(self, obj_ids, translations, rotations, train_batch_size,
                  num_objs, do_augmentation):
        """The input function 's part that is shared.

        This function creates the scene point clouds from scene descriptions.

        Returns: Two tf.Tensors, the first contains all points of the
            objects in the batch with shape (N, 3) and the second contains the
            corresponding segment ids, the shape is (N,).
        """
        batch_size = tf.shape(obj_ids)[0]
        # flatten all inputs
        obj_ids = tf.reshape(obj_ids, (-1,))
        translations = tf.reshape(translations, (-1, 3))
        rotations = tf.reshape(rotations, (-1, 4))
        clouds_num_points = (self.cloud_slice_indices[1:] -
                             self.cloud_slice_indices[:-1])
        # vector with the number of points of each cloud
        num_points = tf.gather(clouds_num_points, obj_ids)
        # vector with a range where each number i is num_points[i] repeated
        segment_ids = self._repeat(tf.range(tf.shape(num_points)[0]),
                                   tf.to_int32(num_points), batch_size,
                                   num_objs)
        segment_ids = tf.to_int32(segment_ids)
        # repeat translations[i] and rotations[i] num_points[i] times
        translations = tf.gather(translations, segment_ids)
        rotations = tf.gather(rotations, segment_ids)
        rotations = tfq.Quaternion(rotations)
        obj_ids = tf.gather(tf.to_float(obj_ids), segment_ids)
        # indices of points consist of the start index plus range(num_points)
        start = tf.gather(self.cloud_slice_indices, tf.to_int32(obj_ids))
        ranges = tf.cond(tf.equal(batch_size, 1),
                         lambda: tf.concat([tf.range(num_points[i])
                                            for i in range(2)], axis=0),
                         lambda: tf.concat([tf.range(num_points[i])
                                            for i in range(num_objs)], axis=0))
        point_ids = tf.to_int32(start + ranges)
        points = tf.gather(self.clouds_tensor, point_ids)
        # Rotate objects. Note that the quaternions are relative to the object
        # clouds' origins, so no centering using the mean is required.
        points = tfq.rotate_vector_by_quaternion(rotations, points, 2, 2)
        points = tf.squeeze(points) + translations
        # if we're training, randomly rotate around the z_axis
        if do_augmentation:
            points = augment.pointcloud(points, segment_ids, batch_size,
                                        train_batch_size)
        return points, tf.to_float(segment_ids)

    def dataset_input_fn(self, train_batch_size, split):
        """The train input function using the tf.data.Dataset API.

        Args:
            train_batch_size: `int`, the batch size. Test batch size is always
                one.
            split: `int` in the interval [1, 15], the index of the split.
        """
        self._create_tf_datasets(split, train_batch_size)
        next_el = self.iterator.get_next()
        obj_ids, translations, rotations, labels, is_augmented = next_el
        points, segment_ids = self._input_fn(obj_ids, translations,
                                             rotations, train_batch_size,
                                             train_batch_size * 6, True)
        return (points, segment_ids, labels, is_augmented)

    def generalize_input_fn(self, trainable, disable_rotation=None):
        """Create the input function to use when running the generalization.

        This input function creates translation and rotation variables for
        each object, if trainable[i] is true or a constant if trainable[i]
        is false.

        Args:
            trainable: A list of `bool`s with one entry for each object that
                will be passed via self.obj_ids_pl.
                If trainable[i] is true, the translation and rotation for
                the i-th object in the batch will be trainable.
            disable_rotation: A list of `bool`s with one entry for each object
                that will be passed via self.obj_ids_pl. If trainable is set to
                true for this object and disable_rotation is set to true, only
                the translation of this object will be optimized
        """
        if self.translation_vars:
            raise ValueError("generalize_input_fn can only be called once per "
                             "input layer instance")
        for i, (t, no_rot) in enumerate(zip(trainable, disable_rotation)):
            tensor_t = tf.Variable if t else tf.constant
            self.translation_vars += [tensor_t([(0, 0, 0)], dtype=tf.float32,
                                               name="translation" + str(i))]
            if no_rot:
                tensor_t = tf.constant
            self.rotation_vars += [tensor_t([(0, 0, 0)], dtype=tf.float32,
                                            name="rotation" + str(i))]
        translation_delta = tf.reshape(self.translation_vars, (-1, 2, 3))
        rotation_delta = tf.reshape(self.rotation_vars, (-1, 2, 3))
        self.translations = self.translations_pl + translation_delta
        # don't optimize w of quaternion to prevent numerical instability
        rotation_delta = tf.pad(rotation_delta, [[0, 0], [0, 0], [1, 0]])
        self.rotations = self.rotations_pl + rotation_delta
        return self._input_fn(self.obj_ids_pl, self.translations,
                              self.rotations, None, len(trainable), False)

    def get_transform_vars(self):
        """Return all variables created to perform rotation and translation."""
        return [v for v in (self.rotation_vars + self.translation_vars)
                if isinstance(v, tf.Variable)]

    def reset_transform_vars(self, sess):
        """Reset translation and rotation to identity."""
        for v in self.get_transform_vars():
            sess.run(v.initializer)
