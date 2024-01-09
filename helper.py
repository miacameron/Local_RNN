# Copied over from PredRec by M.C. 2023-12-27

import numpy as np
from scipy.special import softmax


def gaussian_activation(t, event_t, interval, sigma, peak=0.1):
    return (
        peak
        * np.exp(
            -(((t - event_t + interval / 2) % interval - interval / 2) ** 2)
            / (2 * sigma**2)
        )
        / (np.sqrt(2 * np.pi) * sigma)
    )


def create_inputs(N, T, interval, sigma=0.5):
    ts = np.arange(0, T).astype("float32")
    peak = sigma * 2

    # Events are distributed "diagonally"
    events = np.arange(1, interval + 1, interval / N)
    inputs = np.zeros((N, T))

    for i in range(0, N):
        inputs[i] = gaussian_activation(ts, events[i], interval, sigma, peak)

    # normalize ec_inputs across neurons at each timestep, i.e. make they sum to 1, so that they can be represented using a softmax function
    for t in range(T):
        # inputs[:, t] = inputs[:, t] / (np.max(inputs) - np.min(inputs))
        # inputs[:, t] = inputs[:, t] / np.linalg.norm(inputs[:, t])
        inputs[:, t] = softmax(inputs[:, t])

    return inputs
