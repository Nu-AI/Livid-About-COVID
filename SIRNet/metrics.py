from functools import wraps

import numpy as np

from . import util

__all__ = ['mean_squared_error_elementwise',
           'mean_squared_error_samplewise']


def metric(func):
    @wraps(func)
    def wrapped(y_pred, y_true, *args, **kwargs):
        assert len(y_pred) == len(y_true), \
            'Cannot compute metric with inputs of different length'
        return func(y_pred, y_true, *args, **kwargs)

    return wrapped


@metric
def mean_squared_error_samplewise(y_pred, y_true, agg_func=np.mean):
    """Compute MSE between predicted and ground truth sample-wise, i.e.
    the mean of all MSEs of samples.
    """
    mses = []
    for yp, yt in zip(y_pred, y_true):
        mses.append(
            np.mean((util.to_numpy(yp, False, False) -
                     util.to_numpy(yt, False, False)) ** 2)
        )
    return agg_func(mses)


@metric
def mean_squared_error_elementwise(y_pred, y_true):
    """Compute MSE between predicted and ground truth element-wise, i.e.
    the mean of all SEs for all time steps of each sample.
    """
    mses = []
    for yp, yt in zip(y_pred, y_true):
        mses.append(
            (util.to_numpy(yp, False, False) -
             util.to_numpy(yt, False, False)) ** 2
        )
    return np.mean(np.concatenate(mses))


def root_mean_squared_error_samplewise(y_pred, y_true, agg_func=np.mean):
    """Compute RMSE between predicted and ground truth sample-wise, i.e.
    the sqaure root of the mean of all MSEs of samples.
    """
    return np.sqrt(mean_squared_error_samplewise(y_pred, y_true, agg_func))


def root_mean_squared_error_elementwise(y_pred, y_true):
    """Compute RMSE between predicted and ground truth element-wise, i.e.
    the square root of the mean of all SEs for all time steps of each sample.
    """
    return np.sqrt(root_mean_squared_error_elementwise(y_pred, y_true))


@metric
def mean_absolute_percentage_error_samplewise(y_pred, y_true, agg_func=np.mean):
    """Compute MSE between predicted and ground truth sample-wise, i.e.
    the mean of all MSEs of samples.
    """
    mapes = []
    for yp, yt in zip(y_pred, y_true):
        mapes.append(
            np.mean(
                np.abs((util.to_numpy(yp, False, False) -
                        util.to_numpy(yt, False, False))
                       / util.to_numpy(yt, False, False))
            )
        )
    return agg_func(mapes)
