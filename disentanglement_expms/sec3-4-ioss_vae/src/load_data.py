# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract class for data sets that are two-step generative models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import PIL
from PIL import Image
import scipy.io as sio
from sklearn.utils import extmath
import numpy.random as npr

os.environ['DISENTANGLEMENT_LIB_DATA'] = '/proj/sml/usr/yixinwang/representation-causal/src/disentanglement_expms/data/'


class GroundTruthData(object):
  """Abstract class for data sets that are two-step generative models."""

  @property
  def num_factors(self):
    raise NotImplementedError()

  @property
  def factors_num_values(self):
    raise NotImplementedError()

  @property
  def observation_shape(self):
    raise NotImplementedError()

  def sample_factors(self, num, random_state=npr.RandomState(0)):
    """Sample a batch of factors Y."""
    raise NotImplementedError()

  def sample_observations_from_factors(self, factors, random_state=npr.RandomState(0)):
    """Sample a batch of observations X given a batch of factors Y."""
    raise NotImplementedError()

  def sample(self, num, random_state=npr.RandomState(0)):
    """Sample a batch of factors Y and observations X."""
    factors = self.sample_factors(num, random_state)
    return factors, self.sample_observations_from_factors(factors, random_state)

  def sample_observations(self, num, random_state=npr.RandomState(0)):
    """Sample a batch of observations X."""
    return self.sample(num, random_state)[1]


class SplitDiscreteStateSpace(object):
    """State space with factors split between latent variable and observations."""

    def __init__(self, factor_sizes, latent_factor_indices):
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [
                i for i in range(self.num_factors)
                if i not in self.latent_factor_indices
        ]

    @property
    def num_latent_factors(self):
        return len(self.latent_factor_indices)

    def sample_latent_factors(self, num, random_state=npr.RandomState(0)):
        """Sample a batch of the latent factors."""
        factors = np.zeros(
                shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = self._sample_factor(i, num, random_state)
        return factors

    def sample_all_factors(self, latent_factors, random_state=npr.RandomState(0)):
        """Samples the remaining factors based on the latent factors."""
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(
                shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self.latent_factor_indices] = latent_factors
        # Complete all the other factors
        for i in self.observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, num_samples, random_state)
        return all_factors

    def _sample_factor(self, i, num, random_state=npr.RandomState(0)):
        return random_state.randint(self.factor_sizes[i], size=num)

    def pos_to_idx(self, positions) -> np.ndarray:
        """
        Convert a position to an index (or convert a list of positions to a list of indices)
        - positions are lists of integers, with each element < their corresponding factor size
        - indices are integers < size
        """
        positions = np.moveaxis(positions, source=-1, destination=0)
        return np.ravel_multi_index(positions, self.factor_sizes)

    def idx_to_pos(self, indices) -> np.ndarray:
        """
        Convert an index to a position (or convert a list of indices to a list of positions)
        - indices are integers < size
        - positions are lists of integers, with each element < their corresponding factor size
        """
        positions = np.array(np.unravel_index(indices, self.factor_sizes))
        return np.moveaxis(positions, source=0, destination=-1)

class DSprites(GroundTruthData):
    """DSprites dataset.
    The data set was originally introduced in "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework" and can be downloaded from
    https://github.com/deepmind/dsprites-dataset.
    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """

    def __init__(self, latent_factor_indices=None):
        # By default, all factors (including shape) are considered ground truth
        # factors.
        if latent_factor_indices is None:
            latent_factor_indices = list(range(6))
        self.latent_factor_indices = latent_factor_indices
        self.data_shape = [64, 64, 1]
        # Load the data so that we can sample from it.
        # with gfile.Open(DSPRITES_PATH, "rb") as data_file:
            # Data was saved originally using python2, so we need to set the encoding.
        data = np.load(os.path.join(os.environ['DISENTANGLEMENT_LIB_DATA'],'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), encoding="latin1", allow_pickle=True)
        self.images = np.array(data["imgs"], dtype=np.float32)
        self.factor_sizes = np.array(
            data["metadata"][()]["latents_sizes"], dtype=np.int64)
        self.full_factor_sizes = [1, 3, 6, 40, 32, 32]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
                self.factor_sizes)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)
        self.latents_values = data['latents_values']
        self.latents_classes = data['latents_classes']

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        return self.data_shape


    def sample_factors(self, num, random_state=npr.RandomState(0)):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state=npr.RandomState(0)):
        return self.sample_observations_from_factors_no_color(factors, random_state)

    def sample_observations_from_factors_no_color(self, factors, random_state=npr.RandomState(0)):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return np.expand_dims(self.images[indices].astype(np.float32), axis=3)

    def _sample_factor(self, i, num, random_state=npr.RandomState(0)):
        return random_state.randint(self.factor_sizes[i], size=num)



CARS3D_PATH = os.path.join(
        os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "cars")


class Cars3D(GroundTruthData):
    """Cars3D data set.
    The data set was first used in the paper "Deep Visual Analogy-Making"
    (https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
    downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.
    The ground-truth factors of variation are:
    0 - elevation (4 different values)
    1 - azimuth (24 different values)
    2 - object type (183 different values)
    """

    def __init__(self):
        self.factor_sizes = [4, 24, 183]
        features = extmath.cartesian(
                [np.array(list(range(i))) for i in self.factor_sizes])
        self.latent_factor_indices = [0, 1, 2]
        self.num_total_factors = features.shape[1]
        self.index = StateSpaceAtomIndex(self.factor_sizes, features)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)
        self.data_shape = [64, 64, 3]
        self.images = self._load_data()

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes


    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state=npr.RandomState(0)):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state=npr.RandomState(0)):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = self.index.features_to_index(all_factors)
        return self.images[indices].astype(np.float32)

    def _load_data(self):
        dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
        all_files = [x for x in os.listdir(CARS3D_PATH) if ".mat" in x]
        for i, filename in enumerate(all_files):
            data_mesh = _load_mesh(filename)
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                    np.tile(factor1, len(factor2)),
                    np.repeat(factor2, len(factor1)),
                    np.tile(i,len(factor1) * len(factor2))
            ])
            indexes = self.index.features_to_index(all_factors)
            dataset[indexes] = data_mesh
        return dataset


def _load_mesh(filename):
    """Parses a single source file and rescales contained images."""
    with open(os.path.join(CARS3D_PATH, filename), "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
    for i in range(flattened_mesh.shape[0]):
        pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
        pic.thumbnail(size=(64, 64))
        # pic.thumbnail(size=(64, 64, 3), PIL.Image.ANTIALIAS)
        rescaled_mesh[i, :, :, :] = np.array(pic)
    return rescaled_mesh * 1. / 255



class StateSpaceAtomIndex(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes, features):
        """Creates the StateSpaceAtomIndex.
        Args:
            factor_sizes: List of integers with the number of distinct values for each
                of the factors.
            features: Numpy matrix where each row contains a different factor
                configuration. The matrix needs to cover the whole state space.
        """
        self.factor_sizes = factor_sizes
        num_total_atoms = np.prod(self.factor_sizes)
        self.factor_bases = num_total_atoms / np.cumprod(self.factor_sizes)
        feature_state_space_index = self._features_to_state_space_index(features)
        if np.unique(feature_state_space_index).size != num_total_atoms:
            raise ValueError("Features matrix does not cover the whole state space.")
        lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
        lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
        self.state_space_to_save_space_index = lookup_table

    def features_to_index(self, features):
        """Returns the indices in the input space for given factor configurations.
        Args:
            features: Numpy matrix where each row contains a different factor
                configuration for which the indices in the input space should be
                returned.
        """
        state_space_index = self._features_to_state_space_index(features)
        return self.state_space_to_save_space_index[state_space_index]

    def _features_to_state_space_index(self, features):
        """Returns the indices in the atom space for given factor configurations.
        Args:
            features: Numpy matrix where each row contains a different factor
                configuration for which the indices in the atom space should be
                returned.
        """
        if (np.any(features > np.expand_dims(self.factor_sizes, 0)) or
                np.any(features < 0)):
            raise ValueError("Feature indices have to be within [0, factor_size-1]!")
        return np.array(np.dot(features, self.factor_bases), dtype=np.int64)



class MPI3D(GroundTruthData):
    """MPI3D dataset.
    MPI3D datasets have been introduced as a part of NEURIPS 2019 Disentanglement
    Competition.(http://www.disentanglement-challenge.com).
    There are three different datasets:
    1. Simplistic rendered images (mpi3d_toy).
    2. Realistic rendered images (mpi3d_realistic).
    3. Real world images (mpi3d_real).
    Currently only mpi3d_toy is publicly available. More details about this
    dataset can be found in "On the Transfer of Inductive Bias from Simulation to
    the Real World: a New Disentanglement Dataset"
    (https://arxiv.org/abs/1906.03292).
    The ground-truth factors of variation in the dataset are:
    0 - Object color (4 different values for the simulated datasets and 6 for the
        real one)
    1 - Object shape (4 different values for the simulated datasets and 6 for the
        real one)
    2 - Object size (2 different values)
    3 - Camera height (3 different values)
    4 - Background colors (3 different values)
    5 - First DOF (40 different values)
    6 - Second DOF (40 different values)
    """

    def __init__(self, mode="mpi3d_toy"):
        if mode == "mpi3d_toy":
            mpi3d_path = os.path.join(
                    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "mpi3d_toy",
                    "mpi3d_toy.npz")
            # if not tf.io.gfile.exists(mpi3d_path):
            if not os.path.exists(mpi3d_path):
                raise ValueError(
                        "Dataset '{}' not found. Make sure the dataset is publicly available and downloaded correctly."
                        .format(mode))
            else:
                # with tf.io.gfile.GFile(mpi3d_path, "rb") as f:
                # with open(mpi3d_path, "rb") as f:
                    # data = np.load(f)
                data = np.load(mpi3d_path)
            self.factor_sizes = [4, 4, 2, 3, 3, 40, 40]
        elif mode == "mpi3d_realistic":
            mpi3d_path = os.path.join(
                    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "mpi3d_realistic",
                    "mpi3d_realistic.npz")
            # if not tf.io.gfile.exists(mpi3d_path):
            if not os.path.exists(mpi3d_path):
                raise ValueError(
                        "Dataset '{}' not found. Make sure the dataset is publicly available and downloaded correctly."
                        .format(mode))
            else:
                # with tf.io.gfile.GFile(mpi3d_path, "rb") as f:
                # with open(mpi3d_path, "rb") as f:
                    # data = np.load(f)
                data = np.load(mpi3d_path)
            self.factor_sizes = [4, 4, 2, 3, 3, 40, 40]
        elif mode == "mpi3d_real":
            mpi3d_path = os.path.join(
                    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "mpi3d_real",
                    "mpi3d_real.npz")
            # if not tf.io.gfile.exists(mpi3d_path):
            if not os.path.exists(mpi3d_path):
                raise ValueError(
                        "Dataset '{}' not found. Make sure the dataset is publicly available and downloaded correctly."
                        .format(mode))
            else:
                # with tf.io.gfile.GFile(mpi3d_path, "rb") as f:
                # with open(mpi3d_path, "rb") as f:
                    # data = np.load(f)
                data = np.load(mpi3d_path)
            self.factor_sizes = [6, 6, 2, 3, 3, 40, 40]
        else:
            raise ValueError("Unknown mode provided.")

        self.images = data["images"]
        self.latent_factor_indices = [0, 1, 2, 3, 4, 5, 6]
        self.num_total_factors = 7
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
                self.factor_sizes)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [64, 64, 3]


    def sample_factors(self, num, random_state=npr.RandomState(0)):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state=npr.RandomState(0)):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.images[indices] / 255.




SMALLNORB_TEMPLATE = os.path.join(
        os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "small_norb",
        "smallnorb-{}-{}.mat")

SMALLNORB_CHUNKS = [
        "5x46789x9x18x6x2x96x96-training",
        "5x01235x9x18x6x2x96x96-testing",
]


class SmallNORB(GroundTruthData):
    """SmallNORB dataset.
    The data set can be downloaded from
    https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/. Images are resized to 64x64.
    The ground-truth factors of variation are:
    0 - category (5 different values)
    1 - elevation (9 different values)
    2 - azimuth (18 different values)
    3 - lighting condition (6 different values)
    The instance in each category is randomly sampled when generating the images.
    """

    def __init__(self):
        self.images, features = _load_small_norb_chunks(SMALLNORB_TEMPLATE, SMALLNORB_CHUNKS)
        self.factor_sizes = [5, 10, 9, 18, 6]
        # Instances are not part of the latent space.
        self.latent_factor_indices = [0, 2, 3, 4]
        self.num_total_factors = features.shape[1]
        self.index = StateSpaceAtomIndex(self.factor_sizes, features)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        return [64, 64, 1]


    def sample_factors(self, num, random_state=npr.RandomState(0)):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state=npr.RandomState(0)):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = self.index.features_to_index(all_factors)
        return np.expand_dims(self.images[indices].astype(np.float32), axis=3)


def _load_small_norb_chunks(path_template, chunk_names):
    """Loads several chunks of the small norb data set for final use."""
    list_of_images, list_of_features = _load_chunks(path_template, chunk_names)
    features = np.concatenate(list_of_features, axis=0)
    features[:, 3] = features[:, 3] / 2  # azimuth values are 0, 2, 4, ..., 24
    return np.concatenate(list_of_images, axis=0), features




def _load_chunks(path_template, chunk_names):
    """Loads several chunks of the small norb data set into lists."""
    list_of_images = []
    list_of_features = []
    for chunk_name in chunk_names:
        norb = _read_binary_matrix(path_template.format(chunk_name, "dat"))
        list_of_images.append(_resize_images(norb[:, 0]))
        norb_class = _read_binary_matrix(path_template.format(chunk_name, "cat"))
        norb_info = _read_binary_matrix(path_template.format(chunk_name, "info"))
        list_of_features.append(np.column_stack((norb_class, norb_info)))
    return list_of_images, list_of_features


def _read_binary_matrix(filename):
    """Reads and returns binary formatted matrix stored in filename."""
    # with tf.gfile.GFile(filename, "rb") as f:
    f = open(filename, "rb")
    s = f.read()
    magic = int(np.frombuffer(s, "int32", 1))
    ndim = int(np.frombuffer(s, "int32", 1, 4))
    eff_dim = max(3, ndim)
    raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
    dims = []
    for i in range(0, ndim):
        dims.append(raw_dims[i])

    dtype_map = {
            507333717: "int8",
            507333716: "int32",
            507333713: "float",
            507333715: "double"
    }
    data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data


def _resize_images(integer_images):
    resized_images = np.zeros((integer_images.shape[0], 64, 64))
    for i in range(integer_images.shape[0]):
        image = PIL.Image.fromarray(integer_images[i, :, :])
        image = image.resize((64, 64), PIL.Image.ANTIALIAS)
        resized_images[i, :, :] = image
    return resized_images / 255.