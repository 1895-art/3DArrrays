import array3d as a3d
import numpy as np
import pytest


@pytest.fixture
def labels():
    data = np.genfromtxt('mock_data.csv', delimiter=',')
    return data[:, -1]


@pytest.fixture
def features():
    data = np.genfromtxt('mock_data.csv', delimiter=',')
    return data


def test_simple():
    '''At least this will work!'''
    assert True


def test_2d_non_rolling(labels):
    time_step = 10
    two_dim = a3d.create_2d_target_array(labels, time_step, True)
    assert two_dim[0] == labels[time_step-1]


def test_2d_odd_non_rolling(labels):
    time_step = 8
    data = np.copy(labels)
    two_dim = a3d.create_2d_target_array(labels, time_step, True)
    # labels = a3d.check_array_time_step_divisible(labels, time_step)
    assert two_dim[0] == data[time_step-1]


def test_2d_rolling(labels):
    time_step = 10
    two_dim = a3d.create_2d_target_array(labels, time_step, False)
    assert two_dim[0] == labels[time_step-1]


def test_3d_rolling(features, labels):
    time_step = 10
    three_dim = a3d.convert_3d_array(features, time_step, features.shape[1], False)
    assert three_dim[0, -1, -1] == labels[time_step-1]
