"""Datasets for testing augmented signatures method."""
from collections.abc import Sequence
import os
import warnings

import numpy as np
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe


class DataPathWarning(UserWarning):
    """Warnings regarding the data path."""


DEFAULT_DATA_DIR = os.path.expanduser('~/.data/UEA_UCR')
UEA_UCR_DATA_DIR = os.environ.get('UEA_UCR_DATA_DIR', DEFAULT_DATA_DIR)
if UEA_UCR_DATA_DIR == DEFAULT_DATA_DIR:
    warnings.warn(
        'Using default data path "{}". You can change this by setting '
        'the environment variable "UEA_UCR_DATA_DIR".'
        .format(DEFAULT_DATA_DIR),
        DataPathWarning
    )


if not os.path.exists(UEA_UCR_DATA_DIR):
    warnings.warn(
        'The data path "{}" does not exist. All funcitons in this module will '
        'likely fail'.format(UEA_UCR_DATA_DIR)
    )


def _build_UEA_UCR_data_path(name, train):
    """Build path to UEA_UCR dataset.

    Args:
        name: Name of dataset
        train: If True returns path to train data, otherwise to test data

    Returns:
        Path to UEA_UCR dataset
    """
    path = os.path.join(UEA_UCR_DATA_DIR, name, name)
    return path + ('_TRAIN.ts' if train else '_TEST.ts')


def list_datasets():
    """Get list of available datasets."""
    return [
        d for d in os.listdir(UEA_UCR_DATA_DIR)
        if os.path.isdir(os.path.join(UEA_UCR_DATA_DIR, d))
    ]


class Dataset(Sequence):
    """Datasets from the UCR time series archiv."""

    def __init__(self, name, train=True):
        """Datasets from the UCR time series archiv.

        Args:
            name: Name of the dataset.
            train: Return train split when True, test split when False.

        """
        data_path = _build_UEA_UCR_data_path(name, train)

        self.data_x, self.data_y = load_from_tsfile_to_dataframe(data_path)
        # We do not support time series with time stamps yet. It seems as if
        # timestamps are stored in the index of the individual series. Thus
        # this check would fail if we don't have a regularly sampled time
        # series without time stamps.
        assert isinstance(self.data_x.iloc[0, 0].index, pd.RangeIndex)

        self.class_mapping = self.__build_class_mapping(name)
        self._n_classes = len(self.class_mapping.keys())

    @staticmethod
    def __build_class_mapping(name):
        """Build a class mapping mapping from class labels to ids of int type.

        Args:
            name: Dataset name

        Return:
            dict with dict[class_label] = class_id

        """
        train_path = _build_UEA_UCR_data_path(name, True)
        # test_path = build_UEA_UCR_data_path(name, False)
        _, train_y = load_from_tsfile_to_dataframe(train_path)
        # _, test_y = load_from_tsfile_to_dataframe(test_path)
        # all_labels = np.concatenate([train_y, test_y], axis=0)
        unique_labels = np.unique(train_y)
        return dict(zip(unique_labels, range(len(unique_labels))))

    def __len__(self):
        """Get number of instances in the dataset."""
        return len(self.data_y)

    def __getitem__(self, index):
        """Get dataset instance."""
        instance = self.data_x.iloc[index, :]
        # Combine into a single dataframe and then into a numpy array
        instance_x = pd.concat(list(instance), axis=1).values
        instance_y = self.class_mapping[self.data_y[index]]

        return instance_x.astype(np.float32), instance_y

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_channels(self):
        return self.data_x.shape[-1]
