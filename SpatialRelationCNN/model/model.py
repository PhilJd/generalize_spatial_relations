"""The model for learning and generalizing spatial relations.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from SpatialRelationCNN.model.projectionlayer import ProjectionLayer
from SpatialRelationCNN.model.utility import scope_wrapper
from SpatialRelationCNN.model.weight_decay_optimizers import MomentumWOptimizer

import tensorflow as tf
import numpy as np

DATA_MEAN = 0.0138672649823


@scope_wrapper
class SpatialRelationModel:
    """Model to learn a metric for spatial relations."""

    def __init__(self, cloud_tensor=None, id_tensor=None):
        """Model to learn a metric for spatial relations.

        Args:
            cloud_tensor: A `tf.Tensor` with shape (N, 3), the concatenated
                point clouds of the objects.
            id_tensor: A `tf.Tensor` with shape (N,), the ids assigning each
                point in object_cloud_tensor to the object the point originated
                from.)
        """
        self.cloud_tensor = cloud_tensor
        self.id_tensor = id_tensor
        self.projections = None
        self.projection_height = 100
        self.projection_layer = ProjectionLayer(self.projection_height)
        self.dropout_prob = tf.placeholder(dtype=tf.float32, shape=(),
                                           name="dropout_prob")
        self.embedding_size = 64
        self.weight_decay = 1e-5
        self.embedding = self.inference(self.cloud_tensor, self.id_tensor)
        self.embedding_x_xplus_xminus = self.embedding_x_xplus_xminus_fn()
        self.moving_mean_dist = tf.Variable(
            1, dtype=tf.float32, name='mean_distance', trainable=False)
        self.mean_dist_update_op = None
        self.global_step = tf.train.get_or_create_global_step()
        # vars contain the subgraphs once the respective function is called
        self._loss = None
        self._train_op = None
        self._scene_distance = None
        self._generalization_loss = None
        self._generalization_op = None
        self.saver = None
        self.recreate_saver()

    def recreate_saver(self):
        """Create a new saver to include recently added variables."""
        varlist = [v for v in tf.global_variables()
                   if "translation" not in v.name and "rotation" not in v.name]
        self.saver = tf.train.Saver(var_list=varlist, max_to_keep=3)

    @staticmethod
    def initializer(shape, dtype, partition_info=None):
        """Custom initializer, similar to xavier."""
        n_in = np.prod(shape[:-1])
        # random samples from N(mu, sigma^2): sigma * np.random.randn(...) + mu
        variance = 2.1 / n_in
        init_val = np.sqrt(variance) * np.random.randn(*shape)
        return init_val.astype(dtype.as_numpy_dtype)

    @scope_wrapper
    def inference(self, object_cloud_tensor, ids):
        """The mapping 'Gamma' from point cloud space to the metric space.

        (Gamma without the translation + rotation of the point cloud.)
        To deal with input point clouds with differing number of points,
        we concatenate the clouds of all objects into one big point cloud, with
        the ids mapping the individual points to the desired object. All points
        of one id will be projected into their own channels in the projection
        layer. Each pair of two clouds is regarded as one scene and fed
        through the view network.

        Args:
            object_cloud_tensor: A `tf.Tensor` with shape (N, 3), the
                concatenated point clouds of the objects.
            ids: A `tf.Tensor` with shape (N,), the ids assigning each point in
                object_cloud_tensor to the object the point originated from.
        """
        # the projection (rho in the paper)
        projection = self.projection_layer.project(object_cloud_tensor, ids)
        self._rgb_image_summary(projection)
        batch_shape = (-1, 6, self.projection_height, self.projection_height)
        self.projections = tf.reshape(projection, batch_shape)
        normalized_projection = self.projections / 255. - DATA_MEAN
        # unstack the channels and pass each view through the single view model
        u = tf.unstack(normalized_projection, axis=1)
        views = []
        for i in range(3):  # for each projection
            projection = tf.stack([u[2 * i], u[(2 * i) + 1]], axis=1)
            views += [self._single_view_convnet(projection)]
        # construct the embedding
        x = tf.layers.flatten(tf.concat(views, axis=1))
        x = tf.layers.dense(x, self.embedding_size, reuse=tf.AUTO_REUSE,
                            name="fc_1", activation=tf.nn.elu,
                            kernel_initializer=self.initializer)
        return tf.nn.dropout(x, 1.0 - self.dropout_prob)

    @scope_wrapper
    def _rgb_image_summary(self, projection):
        # add third channel
        rgb_projection = self.projection_layer.batch_to_rgb(projection)
        self.summary_rgbs = tf.cast(rgb_projection, tf.uint8)
        for i in range(3):  # visualize the first triplet in the batch
            tf.summary.image("projection_{}".format(i), self.summary_rgbs[i])

    @scope_wrapper
    def embedding_x_xplus_xminus_fn(self, embedding=None):
        """The embedding for triplet input tensors grouped by label.

        This function restructures the tensor of a batch of embeddings ordered
        as triplets such that they are grouped by label, i.e.
        reference embedding, similar embedding, dissimilar embedding.
        That is, we restructure
        [[emb_x0, emb_x0plus, emb_x0minus], [emb_x1, ...], ...]
        to
        [[emb_x0, emb_x1, ...], [emb_x0plus, ...], [emb_x0minus, ...]]
        """
        embedding = embedding if embedding is not None else self.embedding
        grouped_by_tower = tf.reshape(self.embedding, name="grouped_by_tower",
                                      shape=(-1, 3, self.embedding_size))
        embeddings_t = tf.transpose(grouped_by_tower, [1, 0, 2])
        return embeddings_t

    @scope_wrapper
    def _single_view_convnet(self, view_tensor):
        """Create the network for a single view of a sibling."""
        def conv_block(inputs, filters, kernel_size, name, num, pool=True):
            """Wrapper that handles convolution, pooling and dropout."""
            with tf.variable_scope(name):
                x = tf.layers.conv2d(inputs, filters, kernel_size,
                                     padding='SAME',
                                     reuse=tf.AUTO_REUSE,
                                     data_format='channels_first',
                                     activation=tf.nn.elu,
                                     kernel_initializer=self.initializer)
                if pool:
                    x = tf.layers.max_pooling2d(x, 2, 2, padding='same',
                                                data_format='channels_first')
                return x

        view_tensor = tf.convert_to_tensor(view_tensor)
        x = conv_block(view_tensor, 32, 10, "conv_1", 0)
        x = conv_block(x, 42, 8, "conv_2", 1)
        x = conv_block(x, 64, 6, "conv_3", 2)
        x = conv_block(x, 64, 4, "conv_4", 3)
        x = conv_block(x, 128, 4, "conv_5", 4)
        x = conv_block(x, 128, 4, "conv_6", 5, pool=False)
        return conv_block(x, 128, 2, "conv_7", 6, pool=False)

    @staticmethod
    @scope_wrapper
    def distance_fn(x, y):
        """Compute the euclidean distance of x and y."""
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), -1))

    @scope_wrapper
    def loss(self, margin=1.):
        """The triplet loss."""
        if self._loss is not None:
            return self._loss
        x_ref, x_sim, x_dissim = tf.unstack(self.embedding_x_xplus_xminus,
                                            axis=0)
        dist_sim = self.distance_fn(x_ref, x_sim)
        dist_dissim = self.distance_fn(x_ref, x_dissim)
        # This is loss slightly differs from the paper as we found this to be
        # more robust with varying initializations and to slightly improve
        # performance. It's the triplet loss as in Hoffer et al.: Deep metric
        # learning using Triplet network. (https://arxiv.org/abs/1412.6622)
        dplus = tf.exp(dist_sim) / (tf.exp(dist_sim) + tf.exp(dist_dissim))
        dminus = tf.exp(dist_dissim) / (tf.exp(dist_sim) + tf.exp(dist_dissim))
        # mean of the squared euclidean distances between d+ and d- in batch
        self._loss = tf.reduce_mean(tf.square(dplus - dminus + 1.0))
        # moving mean
        mean = 0.5 * (tf.reduce_mean(dist_sim) + tf.reduce_mean(dist_dissim))
        n = tf.to_float(self.global_step)
        new_mean_dist = (mean + (n * self.moving_mean_dist)) / (n + 1)
        self.mean_dist_update_op = self.moving_mean_dist.assign(new_mean_dist)
        tf.summary.scalar('mean_dist_similar', tf.reduce_mean(dist_sim))
        tf.summary.scalar('mean_dist_dissimilar', tf.reduce_mean(dist_dissim))
        tf.summary.scalar('moving_mean_dist', self.moving_mean_dist)
        tf.summary.scalar('train_loss', self._loss)
        return self._loss

    @property
    @scope_wrapper
    def train_op(self):
        """The training operation."""
        # update batch norm ops
        if self._train_op is not None:
            return self._train_op
        lr = tf.train.cosine_decay_restarts(1e-3, self.global_step,
                                            first_decay_steps=1500, t_mul=2.0,
                                            m_mul=1.0, alpha=0.0)
        # retain a minimum fraction of weight decay
        decay = tf.maximum(self.weight_decay * lr * 1e3,
                           self.weight_decay / 100.)
        tf.summary.scalar('learning_rate', lr)
        update_ops = [self.global_step.assign_add(1), self.mean_dist_update_op]
        with tf.control_dependencies(update_ops):
            optimizer = MomentumWOptimizer(decay, lr, 0.9)
            self._train_op = optimizer.minimize(self.loss())
        return self._train_op

    @property
    @scope_wrapper
    def generalization_loss(self):
        """The loss function for the generalization, i.e., the distance."""
        return self.scene_distance

    @property
    @scope_wrapper
    def scene_distance(self):
        """The distance between the pairs of consecutive scenes."""
        if self._scene_distance is not None:
            return self._scene_distance
        grouped_by_tower = tf.reshape(self.embedding, name="grouped_by_tower",
                                      shape=(-1, 2, self.embedding_size))
        embeddings_t = tf.transpose(grouped_by_tower, [1, 0, 2])
        dist = self.distance_fn(embeddings_t[0], embeddings_t[1])
        self._scene_distance = dist / self.moving_mean_dist
        return self._scene_distance

    def reset_generalization_optimizer(self, sess):
        """Reinitialize the variables of the optimizer."""
        optimizer = self.generalization_optimizer
        sess.run([v.initializer for v in optimizer.variables()])

    @scope_wrapper
    def generalization_op(self, var_list, regularization=None, sess=None):
        """The generalization operation."""
        if self._generalization_op is not None:
            return self._generalization_op
        loss = self.generalization_loss
        lr = 1e-3
        self.generalization_optimizer = tf.train.AdamOptimizer(lr)
        if regularization is not None:
            loss += regularization
        optimizer = self.generalization_optimizer
        self._generalization_op = optimizer.minimize(loss, var_list=var_list)
        if sess:
            sess.run([v.initializer for v in optimizer.variables()])
        return self._generalization_op
