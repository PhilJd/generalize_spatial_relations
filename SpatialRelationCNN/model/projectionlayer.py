"""Depth projection of point clouds, including gradient approximation.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from SpatialRelationCNN.model.utility import scope_wrapper

import numpy as np
import tensorflow as tf


# _____________________________________________________________________________
#
class AxisMapping(object):
    """Base class to define properties of the {top, front, side} projection.

                     top 0^U
                   _____|___
                  /|    v  /|
                 /_|_____ / |  back side is 2^U
    side ^1U --> | |      | |
                 | |______|_|
                 | /      | /
        z|       |/_______|/
         |
         |_____ y
         /
      x /

        Img origin is top left.
                  x
          o------>
          |  ______
          | |      |
          v |      |
        y   |______|

    """

    def __init__(self):
        """Constructor, abstract class."""
        self.world_to_img_axis_name = None
        self.img_to_world_axis_name = None


class U0(AxisMapping):
    """The top projection 0^U."""

    def __init__(self):
        """Constructor, setting the axis mappings."""
        self.world_to_img_axis_name = {'x': 'y', 'y': 'x', 'z': '-d'}
        self.img_to_world_axis_name = {'x': 'y', 'y': 'x', 'd': '-z'}


class U1(AxisMapping):
    """The side projection 1^U."""

    def __init__(self):
        """Constructor, setting the axis mappings."""
        self.world_to_img_axis_name = {'x': 'x', 'y': 'd', 'z': '-y'}
        self.img_to_world_axis_name = {'x': 'x', 'y': '-z', 'd': 'y'}


class U2(AxisMapping):
    """The side projection 2^U."""

    def __init__(self):
        """Constructor, setting the axis mappings."""
        self.world_to_img_axis_name = {'x': 'd', 'y': '-x', 'z': '-y'}
        self.img_to_world_axis_name = {'x': '-y', 'y': '-z', 'd': 'x'}


def axis_index(axis_name):
    """Return the axis idx corresponding to axis_name."""
    axis_name = axis_name.replace("-", "")
    if axis_name == 'x':
        return 0
    elif axis_name == 'y':
        return 1
    elif axis_name == 'z':
        return 2
    else:
        raise ValueError("'{}' is not a valid axisname.".format(axis_name))


# _____________________________________________________________________________
#
class ProjectionGradient:
    """Gradient for an image with respect to d AND x-axis and y-axis."""

    def __init__(self):
        """Constructor for the projection gradient."""
        # the gradients with respect to the x- and y-axis
        self.grad_x = None
        self.grad_y = None

    @staticmethod
    @scope_wrapper
    def backward_pass_mod(x, registered_grad_name):
        """Overwrite the backward pass without touching the forward pass."""
        grad_map = {"Identity": registered_grad_name}
        with tf.get_default_graph().gradient_override_map(grad_map):
            return tf.identity(x)

    @staticmethod
    def get_sobel_kernels(kernel_size_grad_dir, kernel_size_orthogonal_dir):
        """Compute sobel kernels of arbitrary size.

        See
        http://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size
        for reference.

            Args:
                kernel_size_grad_dir: `int`, the kernel size in gradient
                    direction.
                kernel_size_orthogonal_dir: `int`, the kernel size in the
                    direction orthogonal to the gradient direction.

            Returns:
                A `tuple` of `np.array`, that is (g_x, g_y)
        """
        assert kernel_size_grad_dir % 2 == 1, "kernel size must be odd"
        assert kernel_size_orthogonal_dir % 2 == 1, "kernel size must be odd"
        g_x = np.zeros((kernel_size_orthogonal_dir, kernel_size_grad_dir),
                       dtype=np.float32)
        g_y = np.zeros((kernel_size_grad_dir, kernel_size_orthogonal_dir),
                       dtype=np.float32)
        center_x = kernel_size_grad_dir // 2  # center x index
        center_y = kernel_size_orthogonal_dir // 2
        for x in range(kernel_size_grad_dir):
            for y in range(kernel_size_orthogonal_dir):
                if x == center_x and y == center_y:
                    continue
                dist_x = x - center_x
                dist_y = y - center_y
                g_x[y][x] = dist_x / (dist_x * dist_x + dist_y * dist_y)
                g_y[x][y] = g_x[y][x]
        return (g_x, g_y)

    def apply_sobel(self, grad, sobel_size_graddim=7, sobel_size_orthdim=3):
        """Compute the image gradients of the gradients.

        Args:
            grad: The gradient of the three projections i.e. shape
                (N, num_objects, H, W)

        Returns:
            A tuple (grad_x, grad_y) with the image gradient of grad
            in x direction and in y direction.
        """
        g_x, g_y = ProjectionGradient.get_sobel_kernels(sobel_size_graddim,
                                                        sobel_size_orthdim)
        tile_shape = [1, 1, tf.shape(grad)[1]]
        g_x = tf.expand_dims(tf.tile(tf.expand_dims(g_x, -1), tile_shape), -1)
        g_y = tf.expand_dims(tf.tile(tf.expand_dims(g_y, -1), tile_shape), -1)
        a = {'padding': 'SAME', 'data_format': 'NCHW'}
        self.grad_x = tf.nn.depthwise_conv2d(grad, g_x, strides=[1] * 4, **a)
        self.grad_y = tf.nn.depthwise_conv2d(grad, g_y, strides=[1] * 4, **a)
        return self.grad_x, self.grad_y

    def get_insertion_grad_index(self, grad_axis, mapping_for_grad, object_id):
        """Computes an axis index mapping from grad_axis to depth axis."""
        axis_mappings = (U0(), U1(), U2())
        world_axis = mapping_for_grad.img_to_world_axis_name[grad_axis]
        world_axis = world_axis.replace("-", "")
        # get the index of the mapping where world axis is depth
        scene_index = [i for i, m in enumerate(axis_mappings)
                       if 'd' in m.world_to_img_axis_name[world_axis]][0]
        return scene_index * 2 + object_id

    def prepare_axis_gradient(self, projection, grad, axis_name,
                              mapping_for_grad):
        """Prepare grad (a grad w.r.t. an axis) for insertion into overall grad.

        This consists of three steps:
        1) Sum up the gradients over the axis held constant.
        2) Invert the values, if the axes point into the opposite
           direction.
        3) Prepare for broadcasting, i.e. set the shape to (img_size, 1) to
           broadcast over x and to (1, img_size) to broadcast over y.
        """
        # 1) Compute the sum on the gradient axis for each scan line.
        # All black pixels have no gradient.
        # Add small epsilon to be sure to cap all gradients.
        has_no_grad = tf.less_equal(projection, 0 + 0.5)
        # set the gradient to 0 for empty pixels
        grad = tf.where(has_no_grad, tf.zeros_like(grad), grad)
        gradient_axis_idx = 1 if axis_name == 'y' else 2
        grad_vector = tf.reduce_sum(grad, axis=gradient_axis_idx)
        # 2) Compute inversions
        axis_mappings = (U0(), U1(), U2())
        world_axis_name = mapping_for_grad.img_to_world_axis_name[axis_name]
        world_axis_stripped = world_axis_name.replace("-", "")
        insertion_mapping = \
            [m for m in axis_mappings
             if 'd' in m.world_to_img_axis_name[world_axis_stripped]][0]
        depth_axis_name = \
            insertion_mapping.world_to_img_axis_name[world_axis_stripped]
        # invert gradient axis if only one of the axes needs inversion.
        if ('-' in world_axis_name) ^ ('-' in depth_axis_name):  # XOR
            grad_vector = -grad_vector
        # reverse the axis over which we iterate if necessary
        orthogonal_axis = 'x' if axis_name == 'y' else 'y'
        world_axis_name = \
            mapping_for_grad.img_to_world_axis_name[orthogonal_axis]
        world_axis_stripped = world_axis_name.replace("-", "")
        orthogonal_axis_name = \
            insertion_mapping.world_to_img_axis_name[world_axis_stripped]
        if ('-' in world_axis_name) ^ ('-' in orthogonal_axis_name):  # XOR
            grad_vector = tf.reverse(grad_vector, axis=[-1])
        # 3) prepare for broadcasting over the original depth axis
        world_axis_name = mapping_for_grad.img_to_world_axis_name['d']
        world_axis_stripped = world_axis_name.replace("-", "")
        broadcast_axis_name = \
            insertion_mapping.world_to_img_axis_name[world_axis_stripped]
        broadcast_axis = 1 if 'y' in broadcast_axis_name else 2
        grad_vector = tf.expand_dims(grad_vector, broadcast_axis)
        return grad_vector

    def add_image_axis_gradients(self, op, grad):
        """Compute gradients w.r.t. to image x and y axis and add them."""
        grad_x, grad_y = self.apply_sobel(
            grad, sobel_size_graddim=5, sobel_size_orthdim=5)
        # unstack in channel dimension, i.e. separate the projections
        unstacked_grad = tf.unstack(grad, axis=1)
        unstacked_projection = tf.unstack(op.inputs[0], axis=1)
        grad_x = tf.unstack(grad_x, axis=1)
        grad_y = tf.unstack(grad_y, axis=1)
        axis_mappings = (U0(), U1(), U2())
        for i, projection in enumerate(unstacked_projection):
            axis_mapping = axis_mappings[i // 2]
            object_id = i % 2
            for axis_name, grad in zip(['x', 'y'], (grad_x[i], grad_y[i])):
                axis_grad = self.prepare_axis_gradient(projection, grad,
                                                       axis_name, axis_mapping)
                insertion_index = self.get_insertion_grad_index(
                    axis_name, axis_mapping, object_id)
                unstacked_grad[insertion_index] += axis_grad
        return tf.stack(unstacked_grad, axis=1)


# _____________________________________________________________________________
#
class ProjectionLayer:
    """Project point clouds to three orthogonal depth images."""

    instance_id = 0

    def __init__(self, img_size):
        """Constructor, defining the height and width of the projection.

        Args:
            img_size: `int` in range [0, 254], the height and width of the
                projection.
        """
        self.img_size = float(img_size)
        self.num_objects_per_scene = 2
        self.values_shift = 100
        # constant to shift the segmentation ids to the correct batch position.
        # (This constant is multiplied by the object position in the batch.)
        self.segment_id_shift = tf.constant(np.square(img_size), tf.float32)
        # intermediate tensor, located before the identity function with which
        # we alter the backward pass, i.e., when we run
        # tf.gradients(loss, projection_after_grad_approx), we get the
        # approximated gradients. (For visualization purposes.)
        self.projection_after_grad_approx = None
        self.gradient_approx = ProjectionGradient()
        self.gradient_name = ("proj_approx{}"
                              "".format(ProjectionLayer.instance_id))
        tf.RegisterGradient(self.gradient_name)(
            self.gradient_approx.add_image_axis_gradients)
        ProjectionLayer.instance_id += 1

    @scope_wrapper
    def project(self, point_clouds, ids):
        """Project the given points to the three orthogonal planes.

        Args:
            point_clouds: `tf.Tensor` with shape (N, 3), the 3D points
                of the objects.
            ids: `tf.Tensor` with shape (N,), the ids of the points,
                i.e. point_clouds[i] is of object ids[i]. Must be in
                [0, num_objects].
        Returns:
           A `Tensor` with shape [num_objects, img_size, img_size]
           that represents the depth buffer
        """
        # aligns the points in the image cube, returns the axes unstacked
        point_clouds = tf.convert_to_tensor(point_clouds)
        ids = tf.convert_to_tensor(ids)
        point_clouds = self._scale_and_center_cloud(point_clouds, ids)
        clouds_unstacked = tf.unstack(point_clouds, num=3, axis=-1)
        projections = []
        for axis_mapping in (U0(), U1(), U2()):
            map_name = type(axis_mapping).__name__
            with tf.name_scope("project_to_{}".format(map_name)):
                p = self._project_to_plane(axis_mapping, clouds_unstacked, ids)
                projections.append(p)
        projections = tf.concat(projections, 1)
        # the projection after the gradient approximation (after is
        # relative to the backward pass)
        self.projection_after_grad_approx = projections
        # Adapt backward pass:
        # Compute the gradient for the points here. As the meaningful
        # gradients are not on the depth axes, but on the image axes,
        # we compute the image gradient w.r.t to x and w.r.t. y
        # using a Sobel Kernel. And combine this to pass as the gradient
        return ProjectionGradient.backward_pass_mod(projections,
                                                    self.gradient_name)

    @scope_wrapper
    def _scale_and_center_cloud(self, points, ids):
        num_scenes = tf.to_int32((tf.reduce_max(ids) //
                                  self.num_objects_per_scene) + 1)
        scene_ids = tf.to_int32(ids // self.num_objects_per_scene)
        # convert to int, as unsorted_segment_{min, max} is faster for int
        points_int = tf.to_int32(points * 10000.)
        axis_min = tf.to_float(tf.unsorted_segment_min(points_int, scene_ids,
                                                       num_scenes)) / 10000.
        axis_max = tf.to_float(tf.unsorted_segment_max(points_int, scene_ids,
                                                       num_scenes)) / 10000.
        axis_min.set_shape(axis_min.shape.with_rank(1))
        axis_max.set_shape(axis_max.shape.with_rank(1))
        axis_span = axis_max - axis_min
        with tf.name_scope("scale_to_image_cube"):
            max_span = tf.reduce_max(axis_span, axis=-1, keepdims=True)
            sfactor = tf.divide((self.img_size - 2.0), max_span,
                                name="scale_factor")
            points = points * tf.gather(sfactor, scene_ids)
            # update min/max/span to save computation
        with tf.name_scope("center_objects"):
            axis_shift = (-axis_min * sfactor +
                          (((self.img_size - 2.0) - axis_span * sfactor) / 2))
            gathered_shift = tf.gather(axis_shift, scene_ids)
            points = points + gathered_shift
            return points

    @scope_wrapper
    def invert(self, values):
        """Invert 'values', i.e., move origin to the other side of the cube."""
        return self.img_size - 1 - values

    @scope_wrapper
    def _compute_segment_ids(self, points, axis_mapping, object_ids):
        """Compute the segment ids.

        The ids enumerate the image pixels
        starting with 0 in the topleft corner with ids increasing by 1 per
        column and increasing by img_size per row.
        The second channel is shifted by img_size * img_size
        For example an image with 2x2 pixels and 2 channels has the ids
        Channel 1 [0][1]   Channel 2 [4][5]
                  [2][3]             [6][7]
        """
        world_axis_of_img_y = axis_mapping.img_to_world_axis_name['y']
        world_axis_of_img_x = axis_mapping.img_to_world_axis_name['x']
        with tf.name_scope("img_y_coordinates"):
            img_y = points[axis_index(world_axis_of_img_y)]
            img_y = self.invert(img_y) if "-" in world_axis_of_img_y else img_y
        with tf.name_scope("img_x_coordinates"):
            img_x = points[axis_index(world_axis_of_img_x)]
            img_x = self.invert(img_x) if "-" in world_axis_of_img_x else img_x
        segment_ids = (tf.round(img_y) * self.img_size) + tf.round(img_x)
        segment_ids += self.segment_id_shift * object_ids  # shift ids
        return tf.to_int32(segment_ids)

    def _project_to_plane(self, axis_mapping, points,
                          object_ids):
        segment_ids = self._compute_segment_ids(points, axis_mapping,
                                                object_ids)
        with tf.name_scope("depth_values"):
            world_depth_axis = axis_mapping.img_to_world_axis_name['d']
            values = points[axis_index(world_depth_axis)]
            if "-" in world_depth_axis:
                values = self.invert(values)
            values += self.values_shift
        num_objects = tf.reduce_max(object_ids) + 1
        num_segments = tf.to_int32(np.square(self.img_size) * num_objects)
        segment_ids.set_shape(segment_ids.shape.with_rank(1))
        values.set_shape(values.shape.with_rank(1))
        segmented_min = tf.unsorted_segment_min(values, segment_ids,
                                                num_segments=num_segments)
        segmented_min.set_shape(segmented_min.shape.with_rank(1))
        # set empty pixels to 0
        segmented_min = tf.where(segmented_min >= 255.0,
                                 tf.zeros_like(segmented_min),
                                 segmented_min)
        shape = (-1, 2, int(self.img_size), int(self.img_size))
        return tf.reshape(segmented_min, shape)

    @scope_wrapper
    def project_to_rgb(self, points, ids):
        """Visualize the projections to each side.

        Projects the points to the three planes and adds a black channel
        to each plane so that the image can easily be saved as an
        RGB image.

        Returns:
            A batch of three `tf.Tensor`s, shape [N, 3, img_size, img_size, 3]
        """
        points = tf.convert_to_tensor(points)
        projections = self.project(points, ids)
        return tf.cast(self.batch_to_rgb(projections), tf.uint8)

    @scope_wrapper
    def batch_to_rgb(self, projections):
        """Convert the projections to images by adding an all-zero blue channel.

        Args:
            projections: A `tf.Tensor` with shape
                (num_objects, 2,  img_size, img_size).

            Returns:
                A batch of three `tf.Tensor`s with
                shape (N, 3, img_size, img_size, 3).
        """
        batch_size = tf.shape(projections)[0]
        shape = [batch_size, 3, 1, int(self.img_size), int(self.img_size)]
        empty_channel = tf.zeros(tf.convert_to_tensor(shape), dtype=tf.float32)
        # group the projections to the same side into separate channel (dim 2)
        shape[2] = 2
        projections = tf.reshape(projections, shape)
        # add a channel of zeros
        projections = tf.concat((projections, empty_channel), axis=2)
        # change CHW to HWC and cast to uint8
        return tf.transpose(projections, [0, 1, 3, 4, 2])
