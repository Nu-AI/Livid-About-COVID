import numpy as np

from . import util


def mean_squared_error_samplewise(y_pred, y_true, agg_func=np.mean):
    """Compute MSE between predicted and ground truth sample-wise, i.e.
    the mean of all MSEs of samples.
    """
    assert len(y_pred) == len(y_true), \
        'Cannot compute metric with inputs of different length'

    mses = []
    for yp, yt in zip(y_pred, y_true):
        mses.append(
            np.mean((util.to_numpy(yp, False, False) -
                     util.to_numpy(yt, False, False)) ** 2)
        )
    return agg_func(mses)


def mean_squared_error_elementwise(y_pred, y_true):
    """Compute MSE between predicted and ground truth element-wise, i.e.
    the mean of all SEs for all time steps of each sample.
    """
    assert len(y_pred) == len(y_true), \
        'Cannot compute metric with inputs of different length'

    mses = []
    for yp, yt in zip(y_pred, y_true):
        mses.append(
            (util.to_numpy(yp, False, False) -
             util.to_numpy(yt, False, False)) ** 2
        )
    return np.mean(np.concatenate(mses))
