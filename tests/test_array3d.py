# import foo needing to be fixed
import sys
import os.path
path_to_parent = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              os.path.pardir))
sys.path.append(path_to_parent)
import array3d as a3d
sys.path.remove(path_to_parent)
# end import foo

import numpy as np
import pytest


@pytest.fixture
def labels():
    data = np.genfromtxt('tests/mock_data.csv', delimiter=',')
    return data[:, -1]


@pytest.fixture
def features():
    data = np.genfromtxt('tests/mock_data.csv', delimiter=',')
    return data


def test_simple():
    '''At least this will work!'''
    assert True


def test_2d_non_rolling(labels):
    time_step = 10
    two_dim = a3d.create_2d_target_array(labels, time_step, False)
    assert two_dim[0] == labels[time_step-1]


def test_2d_odd_non_rolling(labels):
    time_step = 8
    data = np.copy(labels)
    data = a3d.check_array_time_step_divisible(data, time_step)
    two_dim = a3d.create_2d_target_array(labels, time_step, False)
    assert two_dim[0] == data.reshape(-1, time_step, 1)[0][-1]


def test_2d_rolling(labels):
    time_step = 10
    two_dim = a3d.create_2d_target_array(labels, time_step, True)
    assert two_dim[0] == labels[time_step-1]


def test_3d_non_rolling(features, labels):
    time_step = 10
    three_dim = a3d.convert_3d_array(features, time_step, features.shape[1], False)
    assert three_dim[0, -1, -1] == labels[time_step-1]


def test_3d_odd_non_rolling(features, labels):
    time_step = 8
    labels = a3d.check_array_time_step_divisible(labels, time_step)
    three_dim = a3d.convert_3d_array(features, time_step, features.shape[1], False)
    assert three_dim[0, -1, -1] == labels.reshape(-1, time_step, 1)[0][-1]


def test_3d_rolling(features, labels):
    time_step = 10
    three_dim = a3d.convert_3d_array(features, time_step, features.shape[1], True)
    assert three_dim[0, -1, -1] == labels[time_step-1]
