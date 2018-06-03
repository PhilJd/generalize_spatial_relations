"""
Data loading functions for point clouds and projection images.

Author: Philipp Jund, 2018
"""
from PIL import Image

import numpy as np

try:
    # if pandas is installed, we use this to load the point clouds
    import pandas
    use_pandas = True
except ImportError:
    use_pandas = False


def pcd_as_nparray(filepath):
    # this is faster than np.loadtxt
    if use_pandas:
        df = pandas.read_csv(filepath, sep=' ', skiprows=11, dtype=np.float32)
        return df.as_matrix()
    with open(filepath) as f:
        point_list = [np.fromstring(s, sep=' ') for s in f.readlines()[11:]]
        return np.array(point_list, dtype=np.float32)


def png_to_nparray(filepaths, order="CHW"):
    """ Loads one or multiple projections into a numpy array.
        Resulting array has shape (num_filepaths, C, H, W) if order="CHW"
        or (num_filepaths, H, W, C) if order="HWC".
    """
    if type(filepaths) is str:
        filepaths = [filepaths]
    projections = []
    for filepath in filepaths:
        with open(filepath, "rb") as f:
            img = Image.open(f).convert('RGB')
            np_img = (np.array(img) / 255.0).astype(np.float32)
            assert(np.sum(np_img[:, :, 2]) == 0.0), "last channel not empty"
            np_img = np_img[:, :, :2]  # remove the empty channel
            if order == "CHW":
                projections += [np.transpose(np_img, (2, 0, 1))]
            elif order == "HWC":
                projections += [np_img]
            raise Exception("Order must be either CHW or HWC")
    return np.concatenate(projections, axis=0)
