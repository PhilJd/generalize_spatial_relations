"""
Train and/or evaluate a spatial relation model on one or multiple splits.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from SpatialRelationCNN.data_io.relation_dataset import RelationDataset
from SpatialRelationCNN.model.model import SpatialRelationModel
from SpatialRelationCNN.model.input_layer import InputLayer
from SpatialRelationCNN.model import evaluation_metrics as metrics
import SpatialRelationCNN.model.utility as util

import numpy as np

import tensorflow as tf
import tensorboard.plugins.projector as tensorboard_projector


# store the accuracies of all fifteen splits to compute mean + standard dev
accuracies = {p: {"3of5_accuracy": [],
                  "3of3_accuracy": [],
                  "5of5_accuracy": []}
              for p in ["test", "validation"]}
# code to store embeddings in tensorboard
embedding_var = None  # variable for tensorboard
assignment_op = None
config = tensorboard_projector.ProjectorConfig()
embedding_config = config.embeddings.add()


def export_embedding_to_tensorboard(embedding, model,
                                    summary_writer, sess):
    global embedding_var, assignment_op
    if embedding_var is None:
        embedding_var = tf.Variable(embedding, "tb_embeddings")
        sess.run(embedding_var.initializer)
        summary_writer.add_graph(sess.graph)
        embedding_config.tensor_name = embedding_var.name
        embedding_config.metadata_path = os.path.join(FLAGS.logdir,
                                                      "labels.tsv")
        model.recreate_saver()
        assignment_op = tf.assign(embedding_var, embedding)
    sess.run(assignment_op)
    tensorboard_projector.visualize_embeddings(summary_writer, config)


def train(model, sess, input_layer, labels, split, logdir,
          validate=False):
    """Train `model` on `split` of `dataset`."""
    global_step = tf.train.get_or_create_global_step()
    loss, train_op = model.loss(), model.train_op
    fd = {model.dropout_prob: 0.5}
    sess.run(tf.global_variables_initializer())
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir,
                                           sess.graph)
    print(summary_op)
    input_layer.switch_input("train", sess)
    run_summary = [train_op, loss, summary_op, global_step]
    run_ops = [train_op, loss]
    for i in range(FLAGS.num_iterations):
        if i % 1000 == 0:
            _, loss_value, summary, step = sess.run(run_summary, feed_dict=fd)
            print("step: {}, loss: {}".format(step, loss_value))
            summary_writer.add_summary(summary, i)
            summary_writer.flush()
        else:
            _, loss_value = sess.run(run_ops, feed_dict=fd)
        if validate and i % FLAGS.evaluate_every_n_steps == 0:
            evaluate(model, sess, input_layer, labels, split, "validation",
                     i, logdir, summary_writer)
        if i % FLAGS.snapshot_iterations == 0:
            util.save(i, model, sess, logdir)
    util.save(FLAGS.num_iterations, model, sess, logdir)


def evaluate(model, sess, input_layer, labels_gt, split, phase, step, logdir,
             summary_writer):
    """Evaluate the model on the test_set."""
    print("evaluating model")
    input_layer.switch_input(phase, sess)
    fd = {model.dropout_prob: 0.}
    embeddings = []
    labels = []
    try:
        run_ops = [model.embedding, labels_gt]
        while True:
            emb, y = sess.run(run_ops, feed_dict=fd)
            embeddings += [emb[0]]
            labels += [y]
    except tf.errors.OutOfRangeError:
        pass
    embeddings, labels = np.array(embeddings), np.squeeze(np.array(labels))
    with open(os.path.join(logdir, "labels.tsv"), 'w') as f:
        f.writelines([str(i) + "\n" for i in labels])
    export_embedding_to_tensorboard(embeddings, model, summary_writer, sess)
    dist_mat = metrics.distance_matrix(embeddings)
    similarity_mat = metrics.similarity_matrix(labels)
    mean_sim, mean_dissim = metrics.mean_distances(dist_mat, similarity_mat)
    args = {"step": step, "summary_writer": summary_writer}
    # log distances...
    util.log_np_summary(phase + "_mean_dist_similar", mean_sim, **args)
    util.log_np_summary(phase + "_mean_dist_dissimilar",
                        mean_dissim, **args)
    # ... and nearest neighbor performance
    for x_of_k, k in ((3, 5), (3, 3), (5, 5)):
        metric_name = "{}of{}_accuracy".format(x_of_k, k)
        acc = metrics.knn_accuracy(dist_mat, similarity_mat, k, x_of_k)
        accuracies[phase][metric_name].append(acc)
        util.log_np_summary(phase + metric_name, acc, **args)
    summary_writer.flush()
    input_layer.switch_input("train", sess)


def main(_):
    """Train and/or evaluate the fifteen splits."""
    global embedding_var
    tf.logging.set_verbosity(tf.logging.INFO)
    dataset = RelationDataset(FLAGS.data_dir,
                              validation_ratio=0.0)
    if FLAGS.train_on_all_data:
        dataset.splits[0]["train"] += dataset.splits[0]["test"]
    for split_index in FLAGS.splits:
        print("Training split {}.".format(split_index))
        logdir = os.path.join(FLAGS.logdir, str(split_index))
        with tf.Session() as sess:
            input_layer = InputLayer(dataset, FLAGS.more_augmentation)
            points, segment_ids, labels, is_clone_augmented = \
                input_layer.dataset_input_fn(FLAGS.batch_size, split_index)
            model = SpatialRelationModel(cloud_tensor=points,
                                         id_tensor=segment_ids)
            # preconstruct loss as we have augmentation information here
            with tf.name_scope("Loss"):
                ones = tf.ones_like(is_clone_augmented, dtype=tf.float32)
                margin = tf.where(is_clone_augmented, ones * 0.2, ones * 1)
                model.loss(margin=margin)
            sess.run(tf.global_variables_initializer())
            if os.path.exists(logdir):
                util.load_variables(model, sess, logdir)
            if not FLAGS.evaluate_only:
                validate = bool(dataset.splits[split_index]["validation"])
                train(model, sess, input_layer, labels, split_index, logdir,
                      validate=validate)
            tf_summary_writer = tf.summary.FileWriter(logdir + "/test",
                                                      sess.graph)
            evaluate(model, sess, input_layer, labels, split_index, "test",
                     0, logdir, tf_summary_writer)
            embedding_var = None
        tf.reset_default_graph()
        if FLAGS.train_on_all_data:
            break
    summary_writer = tf.summary.FileWriter(FLAGS.logdir + "/mean_summary")
    args = {"step": FLAGS.num_iterations, "summary_writer": summary_writer}
    #  generate final summary
    for name, values in accuracies['test'].items():
        util.log_np_summary("mean_" + name, np.mean(values), **args)
        util.log_np_summary("stddev_" + name, np.std(values), **args)


if __name__ == "__main__":
    # Using the Winograd non-fused algorithms provides a small performance
    # boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=list(range(15)),
                        help="The splits to train.", nargs='+')
    parser.add_argument("--data_dir", type=str, default=".",
                        help="The directory containing the training data.")
    parser.add_argument("--logdir", type=str, default=".",
                        help="The directory where the weights are saved during"
                             " training and tensorboard files are stored. If"
                             " the directory contains a checkpoint, the model"
                             " is restored from the latest checkpoint.")
    parser.add_argument("--evaluate_only", type=bool, default=False,
                        help="If true, the script only evaluates the given "
                             "test splits.")
    parser.add_argument("--train_on_all_data", type=bool, default=False,
                        help="If true, all data is used for training.")
    parser.add_argument("--evaluate_every_n_steps", type=int, default=1000,
                        help="The splits to train.")
    parser.add_argument("--more_augmentation", type=bool, default=False,
                        help="If true, we do additional augmentation, i.e.,"
                             "cloning scenes and random transforms.")

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.batch_size = 100
    # restarts: 1500 -> 4500 -> 10000 ->
    # duration: 1500 -> 3000 -> 6000 -> 12000
    FLAGS.num_iterations = 14000
    FLAGS.snapshot_iterations = 1000

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
