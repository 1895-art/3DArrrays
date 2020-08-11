import numpy as np
from numpy.core.numeric import roll
from sklearn.model_selection import train_test_split
# from numpy.core.numeric import roll


def check_array_time_step_divisible(arr, time_step):
    """Checks the array if evenly divisible by time step, if not compute next
    closest number and return back array with those elements

    Args:
        arr (numpy array): a numpy 2 dimensional array
        time_step (int): number time steps or look back periods

    Returns:
        arr: numpy array with only the latest elements for length divisible by
        time step
    """
    if len(arr) % time_step == 0:
        return arr
    else:
        quotient = int(len(arr)/time_step)
        n_length = quotient * time_step
        return arr[-n_length:]


def shift_array(arr, num, rolling=False, fill_value=-777):
    #  num = (np.abs(num) - 1) * np.sign(num)
    if rolling is True:
        result = np.zeros_like(arr)
        num = (np.abs(num) - 1) * np.sign(num)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
        result = result[result != fill_value]
        return result[:-1]
    else:
        num = np.abs(num)
        arr = check_array_time_step_divisible(arr, num)

        result = arr.reshape(-1, num, 1)
        nonroll_target = []
        for i in range(result.shape[0]):
            nonroll_target.append(result[i][-1])

        return np.array(nonroll_target)


def convert_3d_array(data, n_time_steps, features, rolling=False):
    """
    ---
    Arguments:
        data:
        n_time_steps:
        features:
    Returns:
        NumPy 3D array formatted for LSTM analysis.
    """
    if rolling is False:
        data = check_array_time_step_divisible(data, n_time_steps)
        return data.reshape((-1, n_time_steps, features))
    else:
        rolling3d = []
        for i in range(data.shape[0]-n_time_steps):
            rolling3d.append(data[i:i+n_time_steps].reshape((-1, n_time_steps, features)))

        rolling3d = np.array(rolling3d)
        rolling3d = rolling3d.reshape((rolling3d.shape[0], n_time_steps, features))

        return rolling3d


def create_2d_target_array(arr, time_steps, rolling=False, features=1):
    if np.abs(time_steps) <= 1:
        return arr
    else:
        target_arr = shift_array(arr, -time_steps, rolling)

    return target_arr.reshape((-1, features))


def create_3d_test_train(features, labels, timesteps, rolling=False, test_size=0.2, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        shuffle=shuffle,
                                                        test_size=0.2)
    X3d_train = convert_3d_array(X_train, timesteps, X_train.shape[1],
                                 rolling=rolling)
    X3d_test = convert_3d_array(X_test, timesteps, X_test.shape[1],
                                rolling=rolling)

    y2d_train = create_2d_target_array(y_train, timesteps, rolling=rolling,
                                       features=1)

    y2d_test = create_2d_target_array(y_test, timesteps, rolling=rolling,
                                      features=1)

    return X3d_train, X3d_test, y2d_train, y2d_test