from typing import Any, Callable, Literal

import numpy as np
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
)


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


class OptimizationError(ValueError):
    pass


class OptimizationNotFeasibleError(OptimizationError):
    pass


class OptimizationNotConvergedError(OptimizationError):
    pass


def binary_optimize(
    f,
    target: float,
    strictly: Literal["none", "lower", "upper", "leq", "geq"] = "none",
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
    min_param: float = 0,
    min_open: bool = True,
    max_param: float = np.inf,
    max_open: bool = True,
    verbose: bool = False,
    initial: float = 1,
    initial_step: float = 1e10,
    steps: int = 1000,
):
    if verbose:
        print(f"target = {target}, strictly = {strictly}")
        print(f"min_param = {min_param}, min_open = {min_open}")
        print(f"max_param = {max_param}, max_open = {max_open}")
        print(f"initial = {initial}")
        print(f"initial_step = {initial_step}")
        print(f"steps = {steps}")
        print(f"abs_tol = {abs_tol}")
        print(f"rel_tol = {rel_tol}")

    def max_violated(x, max_param, max_open):
        """Return true if x is greater than max_param (or equal if max_open is True)

        Args:
            x (_type_): _description_
            max_param (_type_): _description_
            max_open (_type_): _description_

        Returns:
            _type_: _description_
        """
        if max_open:
            return x >= max_param
        else:
            return x > max_param

    def min_violated(x, min_param, min_open):
        """Return true if x is smaller than min_param (or equal if min_open is True)"""
        if min_open:
            return x <= min_param
        else:
            return x < min_param

    def monotonicity_comp(
        x: float, target: float, f_monotonicity: Literal["positive", "negative"]
    ) -> bool:
        """Return true if we need to decrease x to get closer to target.


        Args:
            x (float): Current value of x
            target (float): Target value of f(x)
            f_monotonicity (Literal["positive", "negative"]): Whether f is increasing or decreasing over x

        Returns:
            bool: True if we need to decrease x to get closer to target, False if we need to increase x to get closer to target
        """
        if f_monotonicity == "positive":
            return f(x) > target
        else:
            return f(x) < target

    def strictness_ok(
        curr_obj: float,
        target: float,
        strictly: Literal["none", "lower", "upper", "leq", "geq"],
    ) -> bool:
        if strictly == "lower":
            return curr_obj < target
        elif strictly == "upper":
            return curr_obj > target
        elif strictly == "leq":
            return curr_obj <= target
        elif strictly == "geq":
            return curr_obj >= target
        else:
            return True

    def check_monotonicity(f, max_param, min_param):
        if min_param == -np.inf:
            if max_param == np.inf:
                a = -5
            else:
                a = max_param - max(0.5 * abs(max_param), 5)
        else:
            a = min_param
        if max_param == np.inf:
            if min_param == -np.inf:
                b = 5
            else:
                b = min_param + max(0.5 * abs(min_param), 5)
        else:
            b = max_param
        points = np.linspace(a, b, 4)
        if f(points[1]) < f(points[2]):
            return "positive"
        else:
            return "negative"

    def check_feasibility(
        f: Callable[[float], float],
        max_param: float,
        max_open: bool,
        min_param: float,
        min_open: bool,
        strictly: Literal["none", "lower", "upper", "leq", "geq"],
        abs_tol: float,
        rel_tol: float,
        monotonicity: str,
    ):
        sign_f = {"positive": 1, "negative": -1}[monotonicity]
        highest_param = max_param - 1e-8 if max_open else max_param
        lowest_param = min_param + 1e-8 if min_open else min_param
        if verbose:
            print(f"highest_param = {highest_param}, lowest_param = {lowest_param}")
            print(
                f"f(highest_param) = {f(highest_param)}, f(lowest_param) = {f(lowest_param)}"
            )
        if (
            (
                strictness_ok(f(highest_param), target, strictly)
                or strictness_ok(f(lowest_param), target, strictly)
            )
            and (
                (f(highest_param) > target - abs_tol)
                if (sign_f == 1)
                else (f(highest_param) < target + abs_tol)
            )
            and (
                (f(lowest_param) < target + abs_tol)
                if (sign_f == 1)
                else (f(lowest_param) > target - abs_tol)
            )
        ):
            return True
        return False

    x = initial
    step = initial_step
    over = f(x) > target
    assert not min_violated(
        x, min_param, min_open
    ), f"Initial {x} is smaller than min_param {min_param} (or equal if min_open is True)"
    assert not max_violated(
        x, max_param, max_open
    ), f"Initial {x} is greater than max_param {max_param} (or equal if max_open is True)"
    f_monotonicity = check_monotonicity(f, max_param, min_param)

    if not check_feasibility(
        f,
        max_param,
        max_open,
        min_param,
        min_open,
        strictly,
        abs_tol,
        rel_tol,
        f_monotonicity,
    ):
        raise OptimizationNotFeasibleError("Optimization is not feasible")

    it = 0
    while True:
        if verbose:
            print(f"obj({x}) =", end=" ")
        curr_obj = f(x)
        if verbose:
            print(curr_obj)
        if (
            strictness_ok(curr_obj, target, strictly)
            and abs(target - curr_obj) < abs_tol
            and abs((target - curr_obj) / target) < rel_tol
        ):
            break
        if monotonicity_comp(x, target, f_monotonicity):
            if not over:
                step /= 2
                over = True
            while min_violated(x - step, min_param, min_open):
                step /= 2
            x -= step
        else:
            if over:
                step /= 2
                over = False
            while max_violated(x + step, max_param, max_open):
                step /= 2
            x += step
        it += 1
        if it > steps:
            raise OptimizationNotConvergedError("binary_optimize did not converge")
    return x


class ImmediateEarlyStop(EarlyStopping):
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        try:
            self._run_early_stopping_check(trainer)
        except Exception as e:
            print(e)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
