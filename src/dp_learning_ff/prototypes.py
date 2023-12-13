from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dp_learning_ff.coinpress import algos


def give_non_private_prototypes(
    train_preds, train_targets: np.ndarray, subsampling, seed
):
    targets = np.unique(train_targets)
    train_preds_sorted = np.stack(
        [train_preds[train_targets == target] for target in targets]
    ).copy()
    if subsampling < 1.0:
        rng = np.random.default_rng(seed)
        rng.shuffle(train_preds_sorted, axis=1)
        train_preds_sorted = train_preds_sorted[
            :, : int(subsampling * train_preds_sorted.shape[1])
        ]
    protos = np.asarray(
        [train_preds_sorted[i].mean(axis=0) for i, target in enumerate(targets)]
    )
    return protos


def give_private_prototypes(
    train_preds: np.ndarray,
    train_targets: np.ndarray,
    Ps: np.ndarray[float],
    seed: int = 42,
    subsampling: float = 1.0,
):
    """Returns a private prototype for each class.

    Args:
        train_preds (np.ndarray): (n, d)-array containing the predictions of the training set.
        train_targets (np.ndarray): (n, )-array containing the labels of the training set.
        Ps (np.ndarray[float]): Array of privacy budget per step in (0,rho)-cZDP. To total privacy cost is the sum of this array. The algorithm will perform len(Ps) steps.
        seed (int): RNG seed
        subsampling (float): Ratio in (0, 1] of samples to use



    Returns:
        np.ndarray: (k, d)-array containing the private prototypes for each class.
    """
    targets = np.unique(train_targets)
    train_preds_sorted = [
        train_preds[train_targets == target].copy() for target in targets
    ]
    if subsampling < 1.0:
        rng = np.random.default_rng(seed)
        subsampled = []
        for M_x in train_preds_sorted:
            rng.shuffle(M_x, axis=0)
            subsampled.append(M_x[: int(subsampling * M_x.shape[0])])
        train_preds_sorted = subsampled
    protos = np.asarray(
        [private_mean(train_preds_sorted[i], Ps) for i, target in enumerate(targets)]
    )
    return protos


def private_mean(X, Ps, r=None, c=None):
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array, but received shape: {}".format(X.shape))
    d = X.shape[1]
    if r is None:
        r = np.sqrt(d) * 3
    if c is None:
        c = np.zeros(d)
    t = len(Ps)
    mean = algos.multivariate_mean_iterative(X, c=c, r=r, t=t, Ps=Ps)
    return mean


def roh_of_epsilon_delta(epsilon, delta, c=0):
    roh = (np.sqrt(epsilon - c + np.log(1 / delta)) - np.sqrt(np.log(1 / delta))) ** 2
    return roh


@dataclass
class ClassificationScheme(ABC):
    @abstractmethod
    def classify(self, v_pred, m_protos):
        assert (
            len(v_pred.shape) == 1
        ), f"Expected 1-D sample vector, got shape {v_pred.shape}"
        assert (
            len(m_protos.shape) == 2
        ), f"Expected 2-D matrix of prototypes, got shape {m_protos.shape}"
        assert (
            v_pred.shape[0] == m_protos.shape[1]
        ), f"Expected same dimensionality of sample and each class prototype, but got shapes {v_pred.shape} and {m_protos.shape}"


class CosineClassification(ClassificationScheme):
    name: str = "cosine"

    def classify(self, v_pred, m_protos):
        super().classify(v_pred, m_protos)
        return np.argmax(cosine_similarity(v_pred.reshape(1, -1), m_protos))


class EuclideanClassification(ClassificationScheme):
    name: str = "euclidean"

    def classify(self, v_pred, m_protos):
        super().classify(v_pred, m_protos)
        return np.argmin(np.linalg.norm(v_pred - m_protos, ord=2, axis=1))
