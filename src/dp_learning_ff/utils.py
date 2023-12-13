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