"""
Load a spatial relation model and generalize relations to new scenes.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import gc

from SpatialRelationCNN.data_io.relation_dataset import RelationDataset
from SpatialRelationCNN.data_io.relation_dataset import SceneDescription
from SpatialRelationCNN.model.model import SpatialRelationModel
from SpatialRelationCNN.model.input_layer import InputLayer
from SpatialRelationCNN.data_io.data_loaders import pcd_as_nparray
import SpatialRelationCNN.visualization.visualization_ops as vis
import SpatialRelationCNN.model.utility as util

import numpy as np
import tensorflow as tf


FLAGS = None

freiburg_scenes = ["coffeecup_plate_8335", "shampoo_sweetener_881",  # 0
                   "bowl_sweetener_3349", "boxbase_milk_6983",  # 1
                   "thuna_salt_5333", "teagreen_sugar_8027",  # 2
                   "boxbase_bowl_4380", "rice_teagreen_3271",  # 3
                   "rice_thuna_745", "muesli_bowl_5885",  # 4
                   "pot_salt_6424",  # 5
                   "pot_oil_3767", "boxbase_oil_222"]  # 6

model_names = ["armadillo.pcd", "torus.pcd", "bunny.pcd", "teapot.pcd",
               "cornellbox.pcd"]


def create_artificial_scenes(relation_dataset):
    """Initialize the scenes consisting of the Stanford objects."""
    clouds = {n: pcd_as_nparray(os.path.join(FLAGS.artificial_obj_dir, n))
              for n in model_names}
    # scale objects into unit cube, such that all objects have the same size
    clouds = {n: c / np.ptp(c, axis=0).max() for n, c in clouds.items()}
    relation_dataset.clouds.update(clouds)
    num_clouds_before_update = len(relation_dataset.cloud_names)
    relation_dataset.cloud_names += model_names
    scene_names = []
    for i, n in enumerate(model_names):
        relation_dataset.cloud_num_points[n] = len(clouds[n])
        i += num_clouds_before_update
        for j, m in enumerate(model_names):
            j += num_clouds_before_update
            if i == j:
                continue
            scene = SceneDescription("_".join([n[:-4], m[:-4], str(i), str(j)]))
            scene.cloud_ids = (i, j)
            delta_x = np.abs(clouds[n][:, 0].max()) + np.abs(clouds[m][:, 0].min())
            scene.translations = np.array(((0, 0, 0), (delta_x, 0, 0)))
            relation_dataset.scenes[scene.name] = scene
            scene_names.append(scene.name)
            tmp_concat = np.concatenate([clouds[n], clouds[m]])
            scene.span = np.ptp(tmp_concat, axis=0).max()
    return scene_names


def is_generalizable(scene1, scene2, relation_dataset):
    """Return true if scene1 can be generalized to scene2."""
    # Don't use the same scene as reference and initial scene, loss is always 0
    if scene1.name == scene2.name:
        return False
    # Check that at least one object can contain the other
    is_inside = relation_dataset.scene_to_label[scene1.name] in (1, 5, 6)
    inside_containers = ('bowl, boxbase', 'muesli', 'pot', 'cornellbox')
    if is_inside and not any([o in scene2.name for o in inside_containers]):
        return False
    return True


def get_visualization_ops(model):
    """Create various tensors for visualization, e.g. gradients."""
    loss = model.generalization_loss
    # tuples: (name, tf.Tensor)
    vis_projection_tensors = [
        ("projections",
         tf.cast(model.projection_layer.batch_to_rgb(model.projections),
                 tf.uint8))
    ]
    vis_gradient_tensors = [
        ("grad_with_approx_tensor", tf.gradients(
            loss, model.projection_layer.projection_after_grad_approx)[0]),
        ("grad_without_approx_tensor", tf.gradients(loss, model.projections)[0]),
        # grad of channels w.r.t. image x
        ("grad_wrt_x", model.projection_layer.gradient_approx.grad_x),
        # grad of channels w.r.t. image y
        ("grad_wrt_y", model.projection_layer.gradient_approx.grad_y)
    ]
    # add third empty channel
    vis_gradient_tensors = [(name, model.projection_layer.batch_to_rgb(t))
                            for name, t in vis_gradient_tensors]
    # normalize gradient visualizations such that the scaling is the same for
    # all gradients of one scene
    all_grads = tf.stack([t[1] for _, t in vis_gradient_tensors])
    # reduce_max over all dims except the batch dim
    max_grad_magnitude = tf.reduce_max(tf.abs(all_grads), axis=[0, 2, 3, 4],
                                       keepdims=True)
    vis_gradient_tensors = [(name, t / (2 * max_grad_magnitude) * 255 + 128)
                            for name, t in vis_gradient_tensors]
    vis_gradient_tensors = [(name, tf.cast(t, tf.uint8))
                            for name, t in vis_gradient_tensors]
    return vis_projection_tensors + vis_gradient_tensors


def save_projections(visualization_op_results, filename):
    """Export projections and corresponding gradients as time line and gif."""
    # stack the front, side and top visualizations
    stacked_sides = []  # stack all visualizations, including gradient
    # the shape of visualization_op_results is :
    # (#steps, #projections + #gradient_vis, batch_size=2, num_sides, H, W, 3)
    for vis_step in visualization_op_results:
        stacked_sides.append(
            [vis.stack_images_on_axis(v[1], 'y', border=2)
             for v in vis_step])
    steps_stacked = [
        vis.stack_images_on_axis(v, 'y', border=5)
        for v in stacked_sides]
    timeline = vis.create_timeline(steps_stacked, num_intermediate=10)
    ref_projs = [v for v in visualization_op_results[0][0][0]]
    ref_scene = vis.stack_images_on_axis(ref_projs, 'y', border=2)
    timeline = vis.stack_images_on_axis([ref_scene, timeline], 'x', border=15)
    timeline.save(filename + "projection.png")
    create_gif(visualization_op_results, filename)
    print("saved" + filename)


def create_gif(visualization_op_results, filename):
    """Create a gif of the projections in each step."""
    ref_projs = vis.stack_images_on_axis(
        visualization_op_results[0][0][0], 'y', border=2)
    stacked_projections = []
    for vis_step in visualization_op_results:
        opt_projs = vis.stack_images_on_axis(vis_step[0][1], 'y', border=2)
        stacked_projections.append(
            vis.stack_images_on_axis((ref_projs, opt_projs), 'x', border=7))
    vis.save_gif(stacked_projections, filename + "projection.gif")


def create_3d_gif(transformed_points, distances, filename):
    """Render all steps in 3D and create a gif."""
    distance_plots = vis.plot_distance_progress_to_nparray(distances, 800, 300)
    print(len(transformed_points))
    plots3d = [vis.create_3d_plots(i[0], i[1], [1])[0]
               for i in transformed_points]
    plot3d_ref = vis.create_3d_plots(transformed_points[0][0],
                                     transformed_points[0][1], [0])[0]
    plots = [vis.stack_images_on_axis([plot3d_ref, p], 'x', border=2)
             for p in plots3d]
    stacked = []
    for step_images in zip(plots, distance_plots):
        stacked.append(vis.stack_images_on_axis(step_images, 'y', border=2))
    vis.save_gif(stacked, filename + "projection3d.webp")


def generalize_scene(dataset, sess, model, input_layer, ref_scene,
                     initial_scene, run_ops):
    """Generalize the spatial relation of the ref_scene to initial_scene."""
    filename = ("ref:" + ref_scene.name + "__optimized:" + initial_scene.name)
    fd = {input_layer.translations_pl: (ref_scene.translations,
                                        initial_scene.translations),
          input_layer.obj_ids_pl: (ref_scene.cloud_ids,
                                   initial_scene.cloud_ids),
          input_layer.rotations_pl: (ref_scene.rotations,
                                     initial_scene.rotations),
          model.dropout_prob: 0.}
    input_layer.reset_transform_vars(sess)
    model.reset_generalization_optimizer(sess)
    all_visualizations = []  # store viz in memory and only save after the run
    all_transformed_points = []
    distances = []
    for step in range(FLAGS.num_iterations):
        _, distance, visualizations, transformed_points = \
            sess.run(run_ops, feed_dict=fd)
        distances.append(distance)
        print(step, distance)
        all_visualizations.append(visualizations)
        all_transformed_points.append(transformed_points)
    base_save_path = os.path.join(FLAGS.visualization_dir, filename)
    save_projections(all_visualizations, base_save_path)
    if FLAGS.render_3D:
        create_3d_gif(all_transformed_points, distances, base_save_path)
    # manually start the garbage collector as we allocate quite some memory for
    # the transformed point clouds
    gc.collect()


def generalize_freiburg_scenes(dataset, sess, model, input_layer, run_ops):
    """Generalize scenes from the Spatial Relations dataset into each other."""
    for ref_scene_name in freiburg_scenes:
        ref_scene = dataset.scenes[ref_scene_name]
        for initial_scene_name in freiburg_scenes:
            initial_scene = dataset.scenes[initial_scene_name]
            if is_generalizable(ref_scene, initial_scene, dataset):
                generalize_scene(dataset, sess, model, input_layer,
                                 ref_scene, initial_scene, run_ops)


def generalize_artificial_scenes(dataset, sess, model, input_layer,
                                 artificial_scenes, run_ops):
    """Generalize spatial relations to objects not seen during training."""
    for ref_scene_name in freiburg_scenes:
        ref_scene = dataset.scenes[ref_scene_name]
        for initial_scene_name in artificial_scenes:
            initial_scene = dataset.scenes[initial_scene_name]
            if is_generalizable(ref_scene, initial_scene, dataset):
                generalize_scene(dataset, sess, model, input_layer,
                                 ref_scene, initial_scene, run_ops)


def main(_):
    """Generalize scenes."""
    tf.logging.set_verbosity(tf.logging.INFO)
    dataset = RelationDataset(FLAGS.data_dir)
    artificial_scenes = create_artificial_scenes(dataset)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        input_layer = InputLayer(dataset)
        # keep the first scene fixed and generalize the second scene
        disable_rotation = [True, True, FLAGS.disable_rotation, False]
        points, segment_ids = \
            input_layer.generalize_input_fn([False, False, True, True],
                                            disable_rotation)
        model = SpatialRelationModel(cloud_tensor=points,
                                     id_tensor=segment_ids)
        sess.run(tf.global_variables_initializer())
        util.load_variables(model, sess, FLAGS.logdir)
        # Create the operations here already, as otherwise new tensors get
        # constructed all the time and fill up the RAM
        visualization_ops = [i[1] for i in get_visualization_ops(model)]
        transformed_points_ops = [model.cloud_tensor, model.id_tensor]
        transform_vars = input_layer.get_transform_vars()
        generalization_op = model.generalization_op(transform_vars, sess=sess)
        run_ops = [generalization_op, model.generalization_loss,
                   visualization_ops, transformed_points_ops]
        generalize_freiburg_scenes(dataset, sess, model, input_layer, run_ops)
        generalize_artificial_scenes(dataset, sess, model, input_layer,
                                     artificial_scenes, run_ops)


if __name__ == "__main__":
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".",
                        help="The directory containing the training data.")
    parser.add_argument("--logdir", type=str, default=".",
                        help="The directory where the weights are saved during"
                             " training and tensorboard files are stored. If"
                             " the directory contains a checkpoint, the model"
                             " is restored from the latest checkpoint.")
    parser.add_argument("--artificial_obj_dir", type=str, default=".",
                        help=("The path to the point clouds of the artificial"
                              "objects."))
    parser.add_argument("--visualization_dir", type=str, default=".",
                        help=("The path where the visualizations of the "
                              "generalizations will be stored."))
    parser.add_argument("--num_iterations", type=int, default="600",
                        help=("The number of gradient descent steps to run."))
    parser.add_argument("--rotate_only_one_object", type=bool, default=False,
                        help="If true, the first object won't have a trainable"
                             "rotation.")
    parser.add_argument("--render_3D", type=bool, default=False,
                        help="If true, we render 3D animations. Default is"
                             "False as this is extremely slow.")

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
