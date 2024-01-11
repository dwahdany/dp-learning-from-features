from typing import Optional

import numpy as np


def give_non_private_prototypes(
    train_preds,
    train_targets: np.ndarray,
    subsampling,
    seed,
    poisson_sampling: bool = True,
):
    targets = np.unique(train_targets)
    train_preds_sorted = [
        train_preds[train_targets == target].copy() for target in targets
    ]
    if subsampling < 1.0:
        rng = np.random.default_rng(seed)
        subsampled = []
        for M_x in train_preds_sorted:
            if poisson_sampling:
                occurences = rng.poisson(lam=subsampling, size=M_x.shape[0])
                subsampled_indices = np.arange(M_x.shape[0]).repeat(occurences)
                subsampled.append(M_x[subsampled_indices])
            else:
                rng.shuffle(M_x, axis=0)
                subsampled.append(M_x[: int(subsampling * M_x.shape[0])])
        train_preds_sorted = subsampled
    protos = np.asarray(
        [train_preds_sorted[i].mean(axis=0) for i, target in enumerate(targets)]
    )
    return protos


class NonPrivatePrototyping:
    def __init__(
        self,
        p_sampling: float = 1,
        sample_each_step: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.p_sampling = p_sampling
        self.sample_each_step = sample_each_step
        self.seed = seed
        self.verbose = verbose

    def prototypes(
        self, train_preds, train_targets, overwrite_seed: Optional[int] = None
    ):
        seed = self.seed if overwrite_seed is None else overwrite_seed
        return give_non_private_prototypes(
            train_preds,
            train_targets,
            seed=seed,
            subsampling=self.p_sampling,
        )

    @property
    def epsilon(self):
        return np.inf

    @epsilon.setter
    def epsilon(self, value):
        raise RuntimeError("Cannot set epsilon for non-private prototyping")

    @property
    def delta(self):
        return 1.0

    @delta.setter
    def delta(self, value):
        raise RuntimeError("Cannot set delta for non-private prototyping")

    @property
    def p_sampling(self):
        return self._p_sampling

    @p_sampling.setter
    def p_sampling(self, value):
        assert value is None or (
            value > 0 and value <= 1
        ), "p_sampling must be in (0, 1]"
        self._p_sampling = value
