"""Test for SpatialRelationCNN.model.evaluation_metrics.

Author: Philipp Jund, 2017
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from SpatialRelationCNN.model import evaluation_metrics as metrics


class EvaluationMetricsTest(tf.test.TestCase):

    def test_distance_matrix(self):
        embeddings = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        result = metrics.distance_matrix(embeddings)
        self.assertAllEqual(np.diag(result), np.ones((3,)) * np.nan)
        self.assertEqual(result[0][2],
                         np.linalg.norm(embeddings[0] - embeddings[2]))
        self.assertEqual(result[2][0],
                         np.linalg.norm(embeddings[0] - embeddings[2]))
        self.assertEqual(result[0][1], 0)

    def test_similarity_matrix(self):
        labels = np.array([0, 1, 0])
        result = metrics.similarity_matrix(labels)
        gt = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])
        self.assertAllEqual(result, gt)

    def test_knn_accuracy(self):
        # we want a distance matrix of 6 elements, i.e. shape is 6x6
        embeddings = np.arange(18).reshape((6, 3))
        embeddings[4:] += 10
        d_mat = metrics.distance_matrix(embeddings)
        # first four and last two are similar. Note that self-similarity is not
        # considered in k nearest neighbor computation, so 3of5 is max here.
        labels = (np.arange(6) / 4).astype(np.int32)
        similarity_mat = metrics.similarity_matrix(labels)
        acc_3of5 = metrics.knn_accuracy(d_mat, similarity_mat, 5, 3)
        self.assertEqual(acc_3of5, 2 / 3)
        acc_5of5 = metrics.knn_accuracy(d_mat, similarity_mat, 5, 5)
        self.assertEqual(acc_5of5, 0)
        acc_0of5 = metrics.knn_accuracy(d_mat, similarity_mat, 5, 0)
        self.assertEqual(acc_0of5, 1)
        acc_2of5 = metrics.knn_accuracy(d_mat, similarity_mat, 5, 1)
        self.assertEqual(acc_2of5, 1)


    def test_mean_distances(self):
        dist_mat = np.array([[np.nan, 2, 3],
                             [2, np.nan, 5],
                             [3, 5, np.nan]])
        similarity_mat = np.array([[True, False, True],
                                   [False, True, False],
                                   [True, False, True]])
        mean_sim, mean_dissim = metrics.mean_distances(dist_mat,
                                                       similarity_mat)
        self.assertEqual(mean_sim, 3)
        self.assertEqual(mean_dissim, 3.5)



if __name__ == "__main__":
    tf.test.main()
