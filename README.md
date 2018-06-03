# Code for: Optimization Beyond the Convolution: Generalizing Spatial Relations with End-to-End Metric Learning

This repository contains the code for our paper on generalizing spatial
relations with end-to end metric learning, published at ICRA2018.

## News

- 24.06.2018 **We won the ICRA2018 Best Paper Award in Robot Vision!**

## Setup:

Install this repository (in development mode) using pip:
```
git clone https://github.com/PhilJd/generalize_spatial_relations
cd generalize_spatial_relations
pip install -e .
```
If you'd like to extend the code, I highly recommend to
additionally install pandas as it reads in the point clouds in 3 seconds,
while numpy takes over 30 seconds.

- Download the [dataset](http://spatialrelations.cs.uni-freiburg.de/#dataset)
- Navigate to the relations dataset and create the point clouds from the .obj files (needs pcl):
  ```cd scripts; ./create_uniform_pcd.sh```


## Training
To train all 15 splits run `train.py`:
```
CUDA_VISIBLE_DEVICES=0 python train.py --logdir=$STORE_WEIGHTS_HERE --data_dir=$OBJECT_MODELS_ARE_HERE --more_augmentation=True
```
Adding the flag --more_augmentation applies stronger augmentation, i.e. 
clones a scene three times and applies stronger augmentation to the third clone. This leads
to a better metric performance but might lead to less realistic generalizations.

To train a model on all the data add the flag `--train_on_all_data`.


## Experiments (generalize relations)

To generalize relations from one scene to another, take a look at `generalize.py`.
Here we picked a random subset of the scenes and use each scene as a reference to
generalize the relation to all other scenes of this subset. The 3d visualization
requires Mayavi to be installed and running it is extremely slow (~12 hours to generate
all scenes.)

## Integrate the model into your code
For a simple example of how you could use this model in your code, see
`SpatialRelationCNN/inference_example.py`.
Please note that the code currently runs on GPU only.
