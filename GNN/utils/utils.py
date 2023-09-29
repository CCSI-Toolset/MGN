import random
import time
import torch
from torch.utils.data import Subset


def batchnorm_inverse(bn, normalized_data):
    """
    Inverse a batchnorm layer
    :param bn: batchnorm layer
    :param normalized_data: normalized data (by this bn layer) to be unnormalized
    :return: unnormalized data
    """
    assert not bn.training  # batchnorm layer must be in eval() mode
    return ((normalized_data - bn.bias) / bn.weight) * torch.sqrt(
        bn.running_var + bn.eps
    ) + bn.running_mean


def get_all_val_attrs(obj):
    def _good(attr):
        return (not attr.startswith("__")) and (not callable(getattr(obj, attr)))

    attrs = [attr for attr in dir(obj) if _good(attr)]

    return attrs


def get_subset(dataset, ratio, attributes=None):
    length = len(dataset)
    subset = Subset(dataset, indices=random.sample(range(length), int(ratio * length)))

    if attributes:
        for attr in attributes:
            assert hasattr(dataset, attr)
            setattr(subset, attr, getattr(dataset, attr))

    return subset


def get_delta_minute(start_time):
    # start_time in seconds
    return (time.time() - start_time) / 60.0
