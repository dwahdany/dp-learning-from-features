import numpy as np


def clip_features(x, max_norm):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    clip_coef = max_norm / x_norm
    clip_coef = np.minimum(clip_coef, 1.0)
    x_clip = clip_coef * x
    return x_clip


def dp_covariance(
    X_clip,
    noise_std,
    rng,
):
    """Compute the differentially private covariance matrix.
    Args:
        X_clip: (n,d), matrix of clipped samples
        noise_std: standard deviation of the noise added
        seed: random seed
    Returns:
        cov: (d, d) covariance matrix
    """
    d = X_clip.shape[1]
    assert noise_std > 0

    # Compute the covariance matrix
    cov = X_clip.T @ X_clip
    # Add Gaussian noise to the matrix
    cov += rng.normal(scale=noise_std, size=(d, d))
    return cov
