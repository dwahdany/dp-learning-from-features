from typing import Optional

import numpy as np

from .utils import clip_features, dp_covariance


def noisy_sum(
    X_clip,
    clipping_norm,
    noise_multiplier,
    rng,
    k_classes=1,
):
    d = X_clip.shape[1]
    b = np.sum(X_clip, axis=0)
    b += rng.normal(
        scale=clipping_norm * noise_multiplier * np.sqrt(k_classes), size=(d)
    )
    return b


def dp_least_squares(
    A,
    y,
    weight_alpha,
    reg_lambda,
    clipping_norm,
    noise_multiplier,
    seed=42,
    k_classes: Optional[int] = None,
):
    """Build and solve the differentially private least squares problem.
    Algorithm attempts to follow the description (Algorithm 3) in:

    Mehta, H., Krichene, W., Thakurta, A., Kurakin, A., & Cutkosky, A. (2022).
    Differentially private image classification from features.
    arXiv preprint arXiv:2211.13403.

    Args:
        A: (n, d) matrix of features
        y: (n,) vector of labels
        weight_alpha: weight of the global covariance matrix
        reg_lambda: regularization parameter
        clipping_norm: L2 norm to clip to
        noise_multiplier: noise multiplier for DP-SGD
        k_classes: maximum number of positive classes per sample
    Returns:
        x: (d,) vector of weights
    """
    n, d = A.shape
    assert y.shape == (n,)
    assert clipping_norm > 0
    assert noise_multiplier > 0
    assert reg_lambda >= 0
    assert weight_alpha >= 0

    rng = np.random.default_rng(seed)

    A_clip = clip_features(A, clipping_norm)

    G = dp_covariance(
        A_clip,
        (noise_multiplier * clipping_norm**2),
        rng,
    )  # k_classes is always 1 for global G
    targets = np.unique(y)
    if k_classes is None:
        k_classes = 1
    else:
        assert (
            k_classes < targets
        ), "K_classes cannot be larger than the number of unique classes"
        assert (
            k_classes >= 1
        ), "There must be at least one sample with at least one positive class (k_classes > 1)"
    thetas = []
    for target in targets:
        x_class = A_clip[np.where(y == target)[0]]
        A_class = dp_covariance(
            x_class,
            (noise_multiplier * np.sqrt(k_classes) * clipping_norm**2),
            rng,
        )
        b_class = noisy_sum(x_class, clipping_norm, noise_multiplier, rng, k_classes)
        theta_class = np.linalg.solve(
            A_class + weight_alpha * G + reg_lambda * np.eye(d), b_class
        )
        thetas.append(theta_class)

    return np.asarray(thetas)


def least_squares_classification(
    observations: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Returns the predictions of the least squares classifier.
    `n` is the number of observations, `d` is the dimension of the observations, and `k` is the number of classes.
    Args:
        observations (np.ndarray): (n, d)-array containing the observations.
        theta (np.ndarray): (k, d)-array containing the least squares solution.

    Returns:
        np.ndarray: (n, )-array containing the predictions.
    """
    return np.argmax(observations @ theta.T, axis=1)
