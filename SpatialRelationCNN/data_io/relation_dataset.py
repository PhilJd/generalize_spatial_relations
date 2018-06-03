"""Functions to read the spatial relations dataset.

(http://spatialrelations.cs.uni-freiburg.de/#dataset).

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import SpatialRelationCNN.data_io.data_loaders as load

import numpy as np


def _string_to_nparray(s, delimiter=","):
    """Convert a comma-separated string of numbers to a numpy array."""
    return np.array([float(i) for i in s.split(delimiter)])


# _____________________________________________________________________________
#
class RelationDataset(object):
    """Load the Freiburg spatial relations dataset into memory and query it."""

    def __init__(self, dataset_dir, validation_ratio=0.0):
        """Constructor, defining which parts of the dataset should be loaded.

        Args:
            dataset_dir: The directory of the spatial relation dataset.
                Must contain the folders 'objects', 'cross_validation_splits'
                and the files 'labels.tsv' and 'scenes.txt'.
            validation_ratio: scalar, if larger than zero this ratio of the
                training set is kept back as validation set.
        """
        self.validation_ratio = validation_ratio
        self.data_dir = os.path.expanduser(dataset_dir)
        self.scene_to_label = {}
        self.label_to_scene = {}
        self.splits = [{"train": [], "validation": [], "test": []}
                       for i in range(15)]
        self.scenes = {}  # self.scenes[scene_name] = SceneDescription
        # point cloud of the individual objects
        self.clouds = {}  # self.clouds[object_name] = np.array
        self.cloud_names = []  # name position in list serves as id
        self.cloud_num_points = {}  # the number of points each cloud contains
        # load the data
        self._load_clouds()
        self._read_labels()  # fills scene_to_label and label_to_scene
        self._read_scenes()  # fills self.scenes
        self._read_splits()  # fill train/test in self.splits with scene names

    def assert_phase_is_valid(self, phase):
        """Check if phase is present in data."""
        allowed_phases = ["train", "test", "validation"]
        phase_error_msg = "Phase must be one of {}".format(allowed_phases)
        assert phase in allowed_phases, phase_error_msg
        if len(self.splits[0]["validation"]) == 0 and phase == "validation":
            raise ValueError("Validation set contains no entries.")

    def _read_labels(self):
        """Load the scenes (class!) labels.

        Reads the labels.tsv file into self.scene_to_label and
        self.label_to_scene, i.e. two dictionaries where scene
        name and relation are key/value and value/key, respectively.
        """
        with open(os.path.join(self.data_dir, "labels.tsv"), 'r') as f:
            scene_label_pair = [(l.strip().split("\t")) for l in f.readlines()]
        # convert the label to integer (and remove the title line)
        scene_label_pair = [(s[0], int(s[1])) for s in scene_label_pair[1:]]
        # prepare the scene to label mapping
        self.scene_to_label = dict(scene_label_pair)
        # prepare the label to scene mapping
        max_key = max(self.scene_to_label.values())
        self.label_to_scene = dict([(i, []) for i in range(max_key + 1)])
        for k, v in self.scene_to_label.items():
            self.label_to_scene[v].append(k)

    def _read_splits(self):
        """Load the 15 cross validation splits.

        Reads the train and test split files into self.splits as a list of
        dictionaries with keys 'train' and 'test' and a list of scene
        names as values.
        """
        split_dir = os.path.join(self.data_dir, "cross_validation_splits")
        train_file = os.path.join(split_dir, "split{}_train.txt")
        test_file = os.path.join(split_dir, "split{}_test.txt")
        for i in range(15):
            with open(train_file.format(i + 1), 'r') as f:
                self.splits[i]["train"] = [l.strip() for l in f.readlines()]
            random.shuffle(self.splits[i]["train"])
            if self.validation_ratio > 0:
                len_val = int(self.validation_ratio * len(self.splits[i]["train"]))
                self.splits[i]["validation"] = self.splits[i]["train"][-len_val:]
                self.splits[i]["train"] = self.splits[i]["train"][:-len_val]
            with open(test_file.format(i + 1), 'r') as f:
                self.splits[i]["test"] = [l.strip() for l in f.readlines()]

    def _read_scenes(self):
        """Load the scenes, i.e. 2 objects and their translation and rotation.

        Reads the scenes into self.scenes as a dictionary with a scene
        name as key and the SceneDescription as value.
        """
        with open(os.path.join(self.data_dir, "scenes.txt"), 'r') as f:
            for row in [l.strip().split(";") for l in f.readlines()[1:]]:
                scene = SceneDescription(name=row[0])
                scene.translations[1] = _string_to_nparray(row[1])  # object 2
                scene.rotations[1] = _string_to_nparray(row[2])  # object 2
                if row[3] != '':
                    scene.rotations[0] = _string_to_nparray(row[3])  # object 1
                if self.clouds:
                    obj_names = scene.name.split("_")
                    scene.cloud_ids[0] = self.cloud_names.index(obj_names[0])
                    scene.cloud_ids[1] = self.cloud_names.index(obj_names[1])
                    # concatenate to determine the scene's span
                    clouds = np.concatenate([self.clouds[scene.obj1],
                                             self.clouds[scene.obj2]])
                    scene.span = np.ptp(clouds, axis=0).max()
                self.scenes[row[0]] = scene

    def _load_clouds(self):
        """Load the point clouds of the individual objects."""
        print("Loading point clouds.")
        for dirpath, dirnames, file_names in os.walk(self.data_dir):
            for f in [x for x in file_names if x.endswith(".pcd")]:
                object_name = f[:f.find("_uniform_leafzise_0.001.pcd")]
                cloud_path = os.path.join(dirpath, f)
                cloud = load.pcd_as_nparray(cloud_path)
                self.clouds[object_name] = cloud
                self.cloud_num_points[object_name] = len(cloud)
        self.cloud_names = sorted(self.clouds.keys())
        if not self.cloud_names:
            raise ValueError("No point clouds found at {}. Please check that "
                             "dataset_dir points to the root directory of the "
                             "spatial relations dataset and that the clouds "
                             "are generated (see object-models/scripts/"
                             "create_uniform_pcd.sh).".format(self.data_dir))
        print("Done.")


# _____________________________________________________________________________
#
class SceneDescription(object):
    """Struct to describe a scene, i.e. objects and their transformations."""

    def __init__(self, name, translations=(None, None), rotations=(None, None),
                 cloud_ids=(None, None), span=None):
        """A struct for the scenes.

        Contains the two objects (names and cloud ids) and their respective
        translations and rotations. All properties are a list with length two
        where the first entry corresponds to object 1 and the second to
        object 2. Entries of single objects are preferred over entries for
        both, i.e. SceneDescription(translations=(a,b), translation_obj1=c)
        will result in SceneDescription.translation == (c, b).

        Args:
            name: The name of the scene, must match the ones in the splits.
            translation_obj2: np.array with shape (3,), the translation
                (x, y, z)of object 2 relative to the origin of object 1.
                Defaults to (0, 0, 0).
            rotations: np.array with shape (2, 4), the two quaternions
                (w, x, y, z) describing the rotations of obj1 and obj2.
                Defaults to (1, 0, 0, 0).
            cloud_ids: np.array with shape (2,), the ids of the point clouds.
            span: Optional `int`, the span of the scene.
        """
        self.name = name
        self.obj1 = name.split("_")[0]
        self.obj2 = name.split("_")[1]
        self.translations = translations
        self.rotations = rotations
        self.cloud_ids = cloud_ids
        self.span = span

    @property
    def translations(self):
        """2-Tuple of 3D translation vectors of object 1 and object 2."""
        return self.__translations

    @translations.setter
    def translations(self, ts):
        self.__translations = np.zeros((len(ts), 3), dtype=np.float32)
        for i, t in enumerate(ts):
            if t is not None:
                self.__translations[i] = t

    @property
    def rotations(self):
        """2-Tuple of 4D rotation quaternions of object 1 and object 2."""
        return self.__rotations

    @rotations.setter
    def rotations(self, rs):
        self.__rotations = np.array(len(rs) * [[1, 0, 0, 0]], dtype=np.float32)
        for i, r in enumerate(rs):
            if r is not None:
                self.__rotations[i] = r

    @property
    def cloud_ids(self):
        """The indices of the point clouds in the RelationDataset."""
        return self.__cloud_ids

    @cloud_ids.setter
    def cloud_ids(self, cs):
        self.__cloud_ids = np.array(cs)
        for i, c in enumerate(cs):
            if c is None:
                self.__cloud_ids[i] = np.nan

    def __repr__(self):
        """String representation of SceneDescription."""
        return ("<SceneDescription: name: {}, translations: {}, rotations: {}>"
                "".format(self.name, self.translations, self.rotations))
