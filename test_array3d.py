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
    assert True
    return


def test_2d_non_rolling(labels):
    two_dim = a3d.create_2d_target_array(labels, 10, True)
    assert two_dim[0] == labels[9]
