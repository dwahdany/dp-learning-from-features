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


def binary_optimize(
    f,
    target: float,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
    min_param: float = 0,
    min_open: bool = True,
    max_param: float = np.inf,
    max_open: bool = True,
    verbose: bool = False,
    initial: float = 1,
    initial_step: float = 0.5,
    steps: int = 1000,
):
    x = initial
    step = initial_step
    over = f(x) > target

    def max_comp(x, max_param, max_open):
        if max_open:
            return x > max_param
        else:
            return x >= max_param

    def min_comp(x, min_param, min_open):
        if min_open:
            return x < min_param
        else:
            return x <= min_param

    it = 0
    while True:
        if verbose:
            print(f"obj({x}) = {f(x)}")
        curr_obj = f(x)
        if (
            curr_obj < target
            and target - curr_obj < abs_tol
            and (target - curr_obj) / target < rel_tol
        ):
            break
        if curr_obj > target:
            if not over:
                step /= 2
                over = True
            while min_comp(x - step, min_param, min_open):
                step /= 2
            x -= step
        else:
            if over:
                step /= 2
                over = False
            while max_comp(x + step, max_param, max_open):
                step /= 2
            x += step
        it += 1
        if it > steps:
            raise RuntimeError("binary_optimize did not converge")
    return x
