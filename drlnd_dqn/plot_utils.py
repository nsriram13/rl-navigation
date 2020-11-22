# -*- coding: utf-8 -*-

# Plotting utilities from OpenAI baselines package
# Source: https://github.com/openai/baselines/blob/master/baselines/common/plot_util.py

import numpy as np


class _TFColor(object):
    """Enum of colors used in TF docs."""

    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'

    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]


TFColor = _TFColor()
COLORS = [
    TFColor.red,
    TFColor.orange,
    TFColor.green,
    TFColor.blue,
    TFColor.pink,
    TFColor.brown,
    TFColor.purple,
    TFColor.yellow,
    TFColor.gray,
]


# Plotting utility to smooth out a noisy series
def smooth(y, radius, mode='two_sided', valid_only=False):
    """
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    """
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / np.convolve(
            np.ones_like(y), convkernel, mode='same'
        )
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / np.convolve(
            np.ones_like(y), convkernel, mode='full'
        )
        out = out[: -radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def one_sided_ema(
    xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=1e-8
):
    """
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]
    Arguments:
    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds
    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]
    n: int                - number of points in new x grid
    decay_steps: float    - EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN
    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid
    """

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert (
        xolds[0] <= low
    ), 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert (
        xolds[-1] >= high
    ), 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(
        high, xolds[-1]
    )
    assert len(xolds) == len(
        yolds
    ), 'length of xolds ({}) and yolds ({}) do not match!'.format(
        len(xolds), len(yolds)
    )

    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0  # last unused old index
    sum_y = 0.0
    count_y = 0.0
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(-1.0 / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(-(xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys


def symmetric_ema(
    xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=1e-8
):
    """
    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]
    Arguments:
    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds
    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]
    n: int                - number of points in new x grid
    decay_steps: float    - EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN
    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid
    """
    xs, ys1, count_ys1 = one_sided_ema(
        xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0
    )
    _, ys2, count_ys2 = one_sided_ema(
        -xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0
    )
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys
