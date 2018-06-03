"""Visualization ops to export projections and gradients.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
try:
    from mayavi import mlab
except:
    print("3D plotting is unavailable as mayavi is not installed.")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if 'mlab' in globals():
    mlab.options.offscreen = True
    mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))


def plot_distances_to_nparray(distances, x_axislimits=None,
                              y_axislimits=None):
    """Line-plot distances and return the rendered result as np.array.

    Args:
        distances: `list of floats`, containing the distances between query
            scene and continuous check scenes.
        x_axislimits: tuple, (min, max)
        y_axislimits: tuple, (min, max)
    """
    fig, ax = plt.subplots()
    x = list(range(len(distances)))
    ax.plot(x, distances)
    # ax.legend()
    ax.set_xlim([0, x[-1]] if x_axislimits is None else x_axislimits)
    ax.set_ylim([0, max(distances)] if y_axislimits is None else y_axislimits)
    ax.grid(True)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    result = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return result


def plot_distance_progress_to_nparray(distances, width, height):
    """Plot an animated distance curve.

    Args:
        distances: List of floats.
        width: The desired width of the output images in pixels.
        height: The desired height of the output images in pixels.
    """
    x_axislimits = (0, len(distances) - 1)
    y_axislimits = (0, 1)
    frames = []
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams["figure.figsize"] = [width / 100, height / 100]  # in inch
    for i in range(1, len(distances) + 1):
        frames.append(plot_distances_to_nparray(distances[:i],
                                                x_axislimits, y_axislimits))
    return frames


def convert_to_pil_images(images):
    """Convert a list of images to PIL Images, if necessary."""
    return [i if isinstance(i, Image.Image) else Image.fromarray(i)
            for i in images]


def stack_images_on_axis(images, axis='y', border=5, shape=None):
    """Stack images on top of each other and return one image.

    Args:
        images: A list of `np.array`s and/or `PIL.Image`s.
        axis: A `string` from ('y', 'x'). The image axis on which the images
            will be stacked.
        border: `int`, the border width in pixels, defaults to 5.
        shape: optional shape the images should be resized to before stacking.

    Returns:
        A `PIL.Image`, with all image stacked on top of each other.
    """
    imgs = convert_to_pil_images(images)
    if shape is not None:
        for img in imgs:
            img.thumbnail(shape)
    heights = [0] + [i.height + border * int(axis == 'y') for i in imgs]
    heights[-1] = imgs[-1].height  # replace the last entry to have no border
    widths = [0] + [i.width + border * int(axis == 'x') for i in imgs]
    widths[-1] = imgs[-1].width
    if axis == 'y':
        h, w = np.sum(heights), max(widths)
    elif axis == 'x':
        h, w = max(heights), np.sum(widths)
    else:
        raise ValueError("axis must be one of {'x', 'y'}.")
    # PIL uses width, height to specify the shape
    stacked_img = Image.new('RGB', (w, h), color=(255, 255, 255))
    for img, y, x in zip(imgs, np.cumsum(heights), np.cumsum(widths)):
        stacked_img.paste(img, (0, y) if axis == 'y' else (x, 0))
    return stacked_img


def create_timeline(images, num_intermediate):
    """Create a timeline with start, end and num_intermediate samples."""
    num_intermediate = min(num_intermediate, len(images) - 2)
    selection_ids = [int(round(i / num_intermediate * len(images)))
                     for i in range(1, num_intermediate)]
    selection_ids = [0] + selection_ids + [len(images) - 1]
    selected_images = [images[i] for i in selection_ids]
    return stack_images_on_axis(selected_images, axis='x')


def save_gif(images, save_path):
    """Save an animated gif of images."""
    images = convert_to_pil_images(images)
    images[0].save(save_path, save_all=True, append_images=images[1:],
                   duration=15, lossless=True)


# def create_3d_plots(points, ids, scene_ids_to_render=None):
#     """Plot all scenes as 3D.

#     Args:
#         points: `np.array` with shape (N, 3), the points of all scenes.
#         ids: `np.array` with shape (N,), the ids for each point, indicating
#             to which object a point belongs.
#         scene_ids_to_render: optional list of scene_ids. If not None, only the
#             scenes with id in scene_ids will be rendered.
#     """
#     renderings = []
#     scene_ids = (ids // 2).astype(np.int32)
#     for i in range(max(scene_ids) + 1):
#         if scene_ids_to_render is not None and i not in scene_ids_to_render:
#             continue
#         scene_points = points[scene_ids == i].copy()
#         # scale points as a hacky way to not move the camera
#         scene_points *= 5
#         # center points close to origin
#         scene_points -= np.min(scene_points, axis=0) - 0.1
#         scenes_obj_ids = ids[scene_ids == i]
#         object_ids = np.expand_dims(scenes_obj_ids - min(scenes_obj_ids), 1)
#         color1 = np.array([(154, 0, 0)]) / 255  # AIS red
#         color2 = np.array([(51, 102, 153)]) / 255  # AIS blue
#         # use broadcasting to create an array of colors
#         object_colors = color1 * object_ids + color2 * (1 - object_ids)
#         canvas = vispy.scene.SceneCanvas(keys='interactive', show=True,
#                                          size=(500, 500), bgcolor=(1, 1, 1))
#         view = canvas.central_widget.add_view()
#         view.camera = 'turntable'
#         #ranges = [(max(scene_points[..., i]), min(scene_points[..., i]))
#         #          for i in range(3)]
#         #view.camera.set_range(ranges[0], ranges[1], ranges[2])
#         #view.camera.elevation = 50
#         #center = list(view.camera.center)
#         #center[2] += 0.2
#         #view.camera.center = center
#         print(view.camera.center)
#         view.camera.center = [0.15, 0.15, 0.1]

#         scatter = visuals.Markers()
#         scatter.set_data(scene_points, face_color=object_colors, size=2,
#                          edge_width=0, edge_color=None, symbol='o')
#         view.add(scatter)
#         # view.camera.azimuth = 50
#         # add a colored 3D axis for orientation
#         visuals.XYZAxis(parent=view.scene)
#         #view.camera = 'turntable'
#         #vispy.app.run()
#         print(view.camera.center)
#         renderings.append(canvas.render(size=(500,500)))
#     return renderings


# def create_3d_plots(points, ids, scene_ids_to_render=None):
#     """Plot all scenes as 3D.

#     Args:
#         points: `np.array` with shape (N, 3), the points of all scenes.
#         ids: `np.array` with shape (N,), the ids for each point, indicating
#             to which object a point belongs.
#         scene_ids_to_render: optional list of scene_ids. If not None, only the
#             scenes with id in scene_ids will be rendered.
#     """
#     renderings = []
#     scene_ids = (ids // 2).astype(np.int32)
#     for i in range(max(scene_ids) + 1):
#         if scene_ids_to_render is not None and i not in scene_ids_to_render:
#             continue
#         scene_points = points[scene_ids == i].copy()
#         # center points close to origin
#         scene_points -= np.min(scene_points, axis=0) - 0.1
#         scenes_obj_ids = ids[scene_ids == i]

#         object_ids = np.expand_dims(scenes_obj_ids - min(scenes_obj_ids), 1)
#         color1 = np.array([(154, 0, 0)]) / 255  # AIS red
#         color2 = np.array([(51, 102, 153)]) / 255  # AIS blue
#         # use broadcasting to create an array of colors
#         object_colors = color1 * object_ids + color2 * (1 - object_ids)
#         Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
#         canvas = scene.SceneCanvas(keys='interactive', show=True)
#         view = canvas.central_widget.add_view()
#         #view.camera.fov = 45
#         #view.camera.distance = 500
#         p1 = Scatter3D(parent=view.scene)
#         p1.set_gl_state(blend=True, depth_test=True)
#         p1.set_data(scene_points, face_color=object_colors, edge_width=0, scaling=False)
#         p1.symbol = visuals.marker_types[10]
#         p1.update()
#         view.camera = 'turntable'
#         canvas._update_transforms()
#         canvas.update()

#         # run
#         #vispy.app.run()
#         #exit()
#         renderings.append(canvas.render(size=(500,500), bgcolor=(1, 1, 1)))
#     return renderings

def create_3d_plots(points, ids, scene_ids_to_render=None):
    """Plot all scenes as 3D.

    Args:
        points: `np.array` with shape (N, 3), the points of all scenes.
        ids: `np.array` with shape (N,), the ids for each point, indicating
            to which object a point belongs.
        scene_ids_to_render: optional list of scene_ids. If not None, only the
            scenes with id in scene_ids will be rendered.
    """
    renderings = []
    scene_ids = (ids // 2).astype(np.int32)
    color1 = np.array((154, 0, 0, 255), np.uint8)  # AIS red
    color2 = np.array((51, 102, 153, 255), np.uint8)  # AIS blue
    color_map = np.array((color1, color2))
    for i in range(max(scene_ids) + 1):
        mlab.clf()
        if scene_ids_to_render is not None and i not in scene_ids_to_render:
            continue
        scene_points = points[scene_ids == i]
        scene_points = scene_points[np.arange(0, len(scene_points), 10)]
        # center points close to origin
        scene_points -= np.min(scene_points, axis=0) - 0.1
        scenes_obj_ids = ids[scene_ids == i]
        object_ids = scenes_obj_ids - min(scenes_obj_ids) + 1
        object_ids = object_ids[np.arange(0, len(object_ids), 10)]
        p = mlab.points3d(scene_points[..., 0], scene_points[..., 1], scene_points[..., 2], object_ids,
                          scale_factor=0.001, mode='2dcircle')
        p.glyph.color_mode = 'color_by_scalar'
        p.module_manager.scalar_lut_manager.lut.table = color_map
        #mlab.axes()
        mlab.orientation_axes(line_width=4)
        renderings.append(mlab.screenshot(antialiased=True))
        p.scene.disable_render = True
    return renderings

#
# def create_3d_plots(points, ids, scene_ids_to_render=None):
#     """Plot all scenes as 3D.
#
#     Args:
#         points: `np.array` with shape (N, 3), the points of all scenes.
#         ids: `np.array` with shape (N,), the ids for each point, indicating
#             to which object a point belongs.
#         scene_ids_to_render: optional list of scene_ids. If not None, only the
#             scenes with id in scene_ids will be rendered.
#     """
#     renderings = []
#     scene_ids = (ids // 2).astype(np.int32)
#     color1 = np.array((154, 0, 0, 255), np.uint8)  # AIS red
#     color2 = np.array((51, 102, 153, 255), np.uint8)  # AIS blue
#     for i in range(max(scene_ids) + 1):
#         if scene_ids_to_render is not None and i not in scene_ids_to_render:
#             continue
#         scene_points = points[scene_ids == i]
#         # center points close to origin
#         scene_points -= np.min(scene_points, axis=0) - 0.1
#         scenes_obj_ids = ids[scene_ids == i]
#         object_ids = np.expand_dims(scenes_obj_ids - min(scenes_obj_ids), 1)
#         color1 = np.array([(154, 0, 0)]) / 255  # AIS red
#         color2 = np.array([(51, 102, 153)]) / 255  # AIS blue
#         # use broadcasting to create an array of colors
#         object_colors = color1 * object_ids + color2 * (1 - object_ids)
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(scene_points[..., 0], scene_points[..., 1], scene_points[..., 2], c=object_colors, marker='o')
#         plt.show()
#         exit()
#     return renderings
