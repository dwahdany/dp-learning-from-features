import numpy as np


def true_max(scores: np.ndarray, size: int = 1) -> np.ndarray:
    max_idx = scores.argmax()
    max_idx = max_idx.repeat(size)
    return max_idx


def exponential(
    scores: np.ndarray,
    sensitivity: float,
    epsilon: float,
    size: int = 1,
    max_fix: bool = True,
) -> np.ndarray:
    if np.isposinf(epsilon):
        return true_max(scores, size)

    # Substract maximum exponent to avoid overflow
    if max_fix:
        max_exponent = epsilon * scores.max() / (2 * sensitivity)
    else:
        max_exponent = 0
    # Calculate the probability for each element, based on its score
    probabilities = np.exp(epsilon * scores / (2 * sensitivity) - max_exponent)

    # Normalize the probabilties so they sum to 1
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

    # Choose an element from R based on the probabilities
    return np.random.choice(len(scores), size, p=probabilities, replace=True)


def report_noisy_max(
    scores: np.ndarray,
    sensitivity: float,
    epsilon: float,
    size: int = 1,
) -> np.ndarray:
    if np.isposinf(epsilon):
        return true_max(scores, size)

    # Add size-dim noise to each score
    noisy_scores = scores[:, np.newaxis] + np.random.laplace(
        loc=0, scale=sensitivity / epsilon, size=(len(scores), size)
    )

    # Find the index of the maximum score
    max_idx = noisy_scores.argmax(axis=0)

    return max_idx
