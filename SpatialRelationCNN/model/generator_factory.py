"""
Python generator to create triplets of scenes.

Author: Philipp Jund, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import SpatialRelationCNN.model.augmentation as augment

import random
from math import pi
import numpy as np


class GeneratorFactory:
    """Base class for creating generator functions for input tuples."""

    def __init__(self, dataset, more_augmentation):
        """Create generator functions that generate training triplets.

        E.g., generators exist for scene names, projections and
        scene descriptions.

        Args:
            dataset: A `RelationDatset` instance.
        """
        self.dataset = dataset
        self.more_augmentation = more_augmentation

    def scene_desc_generator(self, split, phase):
        """Create a generator function that yields scene tuple triplets.

        The scene tuple triplets (s, s+, s-) are uniformly distributed over
        class labels. One tuple s is composed as
        (cloud ids, translations, rotations)

        Args:
            split: `int` in the interval [1, 15], the index of the split.
            phase: either 'train', "validation", or 'test'

        Returns:
            A `generator function` that yields a 3-tuple
            (cloud ids, translations, rotations) of 3-tuples (train)
            or 1-tuples (validation, test).
        """
        self.dataset.assert_phase_is_valid(phase)

        def generator_fn():
            if phase == "train":
                name_generator = self.triplet_scene_name_generator(split)()
            else:
                name_generator = self.scene_name_generator(split, phase)()
            while True:
                # either one scene or a a triplet
                scenes = [self.dataset.scenes[n] for n in next(name_generator)]
                is_clone_augmented = False
                if phase == "train":
                    scenes, is_clone_augmented = \
                        augment.scene_description_triplet(scenes, self.dataset,
                                                          self.more_augmentation)
                yield ([s.cloud_ids for s in scenes],
                       [s.translations for s in scenes],
                       [s.rotations for s in scenes],
                       [self.dataset.scene_to_label[s.name] for s in scenes],
                       is_clone_augmented)

        return generator_fn

    def scene_name_generator(self, split, phase="validation"):
        """Create a generator function that yields all scene names once.

        Args:
            split: `int` in the interval [1, 15], the index of the split.
            phase: either 'train', "validation", or 'test'

        Returns:
            A `generator function` that yields each scene name once.
        """
        def generator_fn():
            for k in self.dataset.splits[split][phase]:
                yield [k]

        return generator_fn

    def triplet_scene_name_generator(self, split):
        """Create a generator that infinitely yields scene name triplets.

        The scene name triplets (s, s+, s-) are uniformly distributed over
        labels of s.

        Args:
            split: `int` in the interval [1, 15], the index of the split.

        Returns:
            A `generator function` that yields a 3-tuple of scene names.
        """
        train_scenes = self.dataset.splits[split]["train"]
        all_labels_and_scenes = self.dataset.label_to_scene.items()
        label_to_scene = {label: [n for n in names if n in train_scenes]
                          for label, names in all_labels_and_scenes}
        # remove a label if it contains less than 2 examples
        empty_ys = [y for y in label_to_scene if len(label_to_scene[y]) < 2]
        for label in empty_ys:
            label_to_scene.pop(label)

        def generator_fn():
            while True:
                y, y_dissim = random.sample(label_to_scene.keys(), 2)
                s, s_plus = random.sample(label_to_scene[y], 2)
                s_minus = random.choice(label_to_scene[y_dissim])
                yield (s, s_plus, s_minus)

        return generator_fn
