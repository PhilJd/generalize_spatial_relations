"""Various evaluation metrics.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def distance_matrix(embeddings):
    """Compute the distances between all combinations of embeddings.

    Args:
        embeddings: `np.array` with shape (N, embedding_size)

    Returns:
        An `np.array` with shape (N, N), containing the euclidean distances
        between the embeddings and np.nan on the diagonal.

    """
    assert isinstance(embeddings, np.ndarray), "Embeddings must be a np.array."
    e = embeddings
    distance_matrix = np.linalg.norm(e[np.newaxis] - e[:, np.newaxis], axis=-1)
    # set distance between self to nan, makes it easier to ignore them later on
    np.fill_diagonal(distance_matrix, np.nan)
    return distance_matrix


def similarity_matrix(labels):
    """Compute the similarity matrix between combinations of labels.

    Args:
        labels: `np.array` containing the relation label of each
            embedding. Shape is (N,).

    Returns:
        An `np.array` with shape (N, N) and type np.bool, where entry (i,j)
            is True if labels[i] == labels[j].
    """
    assert isinstance(labels, np.ndarray), "Labels must be a np.array."
    return labels[np.newaxis] == labels[:, np.newaxis]


def get_sorted_index(distance_matrix):
    """Compute a row-wise sorted index.

    What this code does:
        >>> a
        array([[2, 5, 1],
               [5, 2, 6],
               [1, 6, 2]], dtype=uint8)
        >>> index = list(np.ix_(*[np.arange(i) for i in a.shape]))
        >>> index[-1] = a.argsort(-1)
        >>> index
        [array([[0],
           [1],
           [2]]), array([[2, 0, 1],
           [1, 0, 2],
           [0, 2, 1]])]
    """
    index = list(np.ix_(*[np.arange(i) for i in distance_matrix.shape]))
    index[-1] = distance_matrix.argsort(-1)
    return index


def knn_accuracy(distance_matrix, similarity_matrix, k, x):
    """Compute the x out of k-nearest neighbors performance.

    This function computes the k-nearest neighbors for each embedding based on
    the distance matrix, sums the number of similar examples and computes the
    percentage of entries where at least x samples are similar.

    Args:
        distance_matrix: The distances between embeddings with shape (N, N).
        similarity_matrix: The similarity matrix where the entry at (i,j) is
        True if the scenes corresponding to i and j are similar.
    Returns: A Scalar, the percentage of samples where at least x nearest
        neighbors where labeled as similar.
    """
    index = get_sorted_index(distance_matrix)
    # the first k entries of the distance matrix sorted ascending by distance
    knn_labels = similarity_matrix[index][:, :k]
    return np.mean(np.nanmean(knn_labels, -1) >= (float(x) / float(k)))


def mean_distances(distance_matrix, similarity_matrix):
    """Compute the mean distance between similar and dissimilar pairs."""
    # only use lower triangle, as mat is symmetric
    # the entries between different projections are nan
    tril_indices = np.tril_indices(distance_matrix.shape[0], -1)
    distance_tril = distance_matrix[tril_indices]
    similarity_tril = similarity_matrix[tril_indices]
    mean_dissim = np.ma.array(distance_tril, mask=similarity_tril).mean()
    mean_sim = np.ma.array(distance_tril, mask=(1 - similarity_tril)).mean()
    return mean_sim, mean_dissim
