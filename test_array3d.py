import array3d as a3d
import numpy as np


def test_simple():
    assert True
    return


def test_2d_non_rolling():
    data = np.genfromtxt('mock_data.csv', delimiter=',')
    two_dim = a3d.create_2d_target_array(data[:, -1], 10, True)
    assert two_dim[0] == data[:, -1][9]



