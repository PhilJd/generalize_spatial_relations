"""Setup script.

Author: Philipp Jund (jundp@informatik.uni-freiburg.de)
"""
from setuptools import setup, find_packages
import os


dirname = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dirname, 'README.md')) as f:
    long_description = f.read()


requirements = ['numpy', 'tfquaternion', 'pillow']
try:
    import tensorflow  # noqa
except ImportError:
    print("WARNING: Installing tensorflow. Make sure you have CUDA"
          "and CUDNN installed!")
    requirements += ['tensorflow-gpu']


setup(
    name='SpatialRelationCNN',
    version='0.1',
    description="Generalize Spatial Relations using a WeightSharing CNN.",
    long_description=long_description,
    url='',

    author='Philipp Jund',
    author_email='jundp@cs.uni-freiburg.de',

    keywords='Spatial Relations Optimization Beyond Convolution',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={'fast point cloud loading': ['pandas']},

    license='Apache 2.0',

    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
