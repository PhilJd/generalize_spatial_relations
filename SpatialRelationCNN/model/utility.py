"""Utility functions for tensorflow.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


def scope_wrapper(func, *args, **kwargs):
    """Decorator to wrap a function in a tf.name_scope of the function's name.

    E.g., the following would create a name scope called "my_tf_function".

    @scope_wrapper
    def my_tf_function():
        tf.pass
    """
    def scoped_func(*args, **kwargs):
        scope_name = (func.__name__ if not func.__name__.startswith("_")
                      else "p" + func.__name__)
        with tf.name_scope("{}".format(scope_name)):
            return func(*args, **kwargs)
    return scoped_func


def log_np_summary(summary_name, value, step, summary_writer):
    """Log python variables during training."""
    summary = tf.Summary(value=[tf.Summary.Value(tag=summary_name,
                                                 simple_value=value)])
    summary_writer.add_summary(summary, step)
    print(summary_name, value)


def save(step, model, session, logdir):
    """Save the models current variables."""
    cwd = os.getcwd()
    # instead of using full path, specify the dir as relative path to
    # enable copying variables to different machines
    os.chdir(logdir)
    model.saver.save(session, "./model.ckpt",
                     global_step=step)
    os.chdir(cwd)


def load_variables(model, session, logdir):
    """Restore the variables present in logdir."""
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=logdir)
    if ckpt:
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        raise ValueError("No checkpoint found in '{}'.".format(logdir))


@scope_wrapper
def repeat(a, repeats, num_repeats=None):
    """Repeat each element a[i] repeats[i] times.

    The shape of repeats must be known at compile time. If tensorflow is
    not able to infer the shape of repeats but it is known, num_repeats
    can be passed.
    E.g. a = [4, 5], repeats = [1, 3] --> [4, 5, 5, 5]
    """
    a = tf.convert_to_tensor(a)
    repeats = tf.convert_to_tensor(repeats)
    # tf.tile doesn't allow scalar values. As we do slicing, we need to
    # reshape 1D-tensors to 2D-tensors
    if a.get_shape().ndims == 1:
        a = tf.expand_dims(a, axis=1)
    if repeats.get_shape().ndims == 1:
        repeats = tf.expand_dims(repeats, axis=1)
    if repeats.get_shape().as_list()[0] is not None:
        num_repeats = repeats.get_shape().as_list()[0]
    if num_repeats is None:
        raise ValueError("num_repeats could not be inferred, if possible"
                         "specify it manually")
    repeated = [tf.tile(a[i], repeats[i]) for i in range(num_repeats)]
    return tf.concat(repeated, axis=0)
