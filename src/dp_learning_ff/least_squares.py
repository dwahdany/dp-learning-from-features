import numpy as np


def clip_features(x, max_norm):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    clip_coef = max_norm / x_norm
    clip_coef = np.minimum(clip_coef, 1.0)
    x_clip = clip_coef * x
    return x_clip


def dp_covariance(
    X_clip,
    clipping_norm,
    noise_multiplier,
    rng,
    k_classes=1,
):
    """Compute the differentially private covariance matrix.
    Args:
        X_clip: (n,d), matrix of clipped samples
        clipping_norm: L2 norm to clip to
        noise_multiplier: noise multiplier
        seed: random seed
    Returns:
        cov: (d, d) covariance matrix
    """
    d = X_clip.shape[1]
    assert clipping_norm > 0
    assert noise_multiplier > 0

    # Compute the covariance matrix
    cov = X_clip.T @ X_clip
    # Add Gaussian noise to the matrix
    cov += rng.normal(
        scale=clipping_norm**2 * noise_multiplier * np.sqrt(k_classes), size=(d, d)
    )
    return cov


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
    k_classes=None,
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
        A_clip, clipping_norm, noise_multiplier, rng, k_classes=1
    )  # k_classes is always 1 for global G
    targets = np.unique(y)
    if k_classes is None:
        k_classes = np.ones(n, dtype=int)
    else:
        assert len(k_classes) == n  # each sample has a number of positive classes
        assert np.max(k_classes) <= len(
            targets
        )  # no sample is assigned to more than all classes
        assert np.min(k_classes) >= 1  # no sample is assigned to less than 1 class
    thetas = []
    for i, target in enumerate(targets):
        x_class = A[np.where(y == target)[0]]
        A_class = dp_covariance(
            x_class, clipping_norm, noise_multiplier, rng, k_classes[i]
        )
        b_class = noisy_sum(x_class, clipping_norm, noise_multiplier, rng, k_classes[i])
        theta_class = (
            np.linalg.inv(A_class + weight_alpha * G + reg_lambda * np.eye(d)) @ b_class
        )
        thetas.append(theta_class)

    return np.asarray(thetas)
