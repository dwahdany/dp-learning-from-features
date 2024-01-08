import math
from typing import Iterable, Literal, Optional

import numpy as np
from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism, zCDP_Mechanism
from autodp.transformer_zoo import AmplificationBySampling, ComposeGaussian, Composition

from .utils import binary_optimize


class CoinpressGM(Mechanism):
    def __init__(
        self,
        Ps: list,
        p_sampling: float = 1,
        sample_each_step: bool = False,
        name: str = "CoinpressGM",
    ):
        """
        Initialize the CoinpressGM object.

        Args:
            Ps (list): List of privacy costs per step in (0,rho)-zCDP.
            p_sampling (float, optional): Probability of sampling. Defaults to 1.
            name (str, optional): Name of the object. Defaults to "CoinpressGM".
        """
        assert p_sampling <= 1, "p_sampling must be <= 1"
        assert p_sampling > 0, "p_sampling must be positive"
        assert all(p > 0 for p in Ps), "Ps must be positive"
        self.params = {"Ps": Ps}
        self.name = name
        mechanisms = [GaussianMechanism(1 / math.sqrt(2 * p)) for p in Ps]
        compose = ComposeGaussian()
        mech = compose(mechanisms, [1 for _ in mechanisms])
        if p_sampling < 1:
            preprocessing = AmplificationBySampling(PoissonSampling=True)
            if sample_each_step:
                compose = Composition()
                mechanisms = [
                    preprocessing.amplify(mech, p_sampling, improved_bound_flag=True)
                    for mech in mechanisms
                ]
                mech = compose(mechanisms, [1 for _ in mechanisms])
            else:
                mech = preprocessing.amplify(mech, p_sampling)
        self.set_all_representation(mech)


class ScaledCoinpressGM(CoinpressGM):
    def __init__(
        self,
        scale: float,
        steps: int = 10,
        dist: Literal["lin", "exp", "log", "eq"] = "exp",
        ord: float = 1,
        p_sampling: float = 1,
        sample_each_step: bool = False,
        name="ScaledCoinpressGM",
        Ps: Optional[Iterable[float]] = None,
    ):
        """
        Initialize the ScaledCoinpressGM mechanism.

        Args:
            scale (float): The scaling factor.
            steps (int): The number of steps. Ignored if Ps is set. Defaults to 10.
            dist (Literal["lin", "exp", "log", "eq"]): The distribution type. Ignored if Ps is set. Defaults to "exp".
            ord (float, optional): The order of the distribution. Ignored if Ps is set. Defaults to 1.
            name (str, optional): The name of the mechanism. Defaults to "ScaledCoinpressGM".
            p_sampling (float, optional): The probability of sampling. Defaults to 1.
            Ps (Optional[Iterable[float]], optional): The privacy costs per step. Overwrites steps, dist, ord. Defaults to None.
        """
        assert scale > 0, "scale must be positive"
        assert steps > 0, "steps must be positive"

        self.scale = scale
        self.sample_each_step = sample_each_step
        if Ps is not None:
            Ps = [scale * p for p in Ps]
        elif dist == "lin":
            Ps = [math.pow(scale * (t + 1), ord) for t in range(steps)]
        elif dist == "exp":
            Ps = [math.pow(scale * math.exp(t / steps), ord) for t in range(steps)]
        elif dist == "log":
            Ps = [math.pow(scale * math.log(t + 1), ord) for t in range(steps)]
        elif dist == "eq":
            Ps = [scale] * steps
        super().__init__(
            name=name, p_sampling=p_sampling, sample_each_step=sample_each_step, Ps=Ps
        )


class LeastSquaresCDPM(Mechanism):
    def __init__(self, noise_multiplier, p_sampling: float = 1, name="LeastSquares"):
        assert noise_multiplier > 0, "noise_multiplier must be positive"
        assert p_sampling <= 1, "p_sampling must be <= 1"
        assert p_sampling > 0, "p_sampling must be positive"
        self.params = {"noise_multiplier": noise_multiplier}
        mechanism = zCDP_Mechanism(rho=3 / (2 * noise_multiplier**2), xi=0)
        if p_sampling < 1:
            preprocessing = AmplificationBySampling()
            mechanism = preprocessing.amplify(mechanism, p_sampling)
        self.set_all_representation(mechanism)


def calibrate_single_param(mechanism_class, epsilon, delta, verbose: bool = False):
    def obj(x):
        return mechanism_class(x).get_approxDP(delta)

    scale = binary_optimize(obj, epsilon, verbose=verbose, min_open=True)
    return mechanism_class(scale)


def approxzCDP_to_approxDP(
    rho: float,
    xi: float,
    delta_zcdp: float,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    return_tuple: bool = False,
):
    if epsilon is None and delta is None:
        raise ValueError("Either epsilon or delta must be set")
    if epsilon is not None and delta is not None:
        raise ValueError("Only one of epsilon and delta must be set")
    if delta is None:
        assert epsilon >= xi + rho, "epsilon must be >= xi + rho"
        excess_epsilon = epsilon - xi - rho
        scaled_excess_epsilon = excess_epsilon / (2 * rho)
        delta_excess = np.exp(-((excess_epsilon) ** 2) / (4 * rho)) * min(
            1,
            np.sqrt(np.pi * rho),
            1 / (1.0 + scaled_excess_epsilon),
            2
            / (
                1.0
                + scaled_excess_epsilon
                + np.sqrt((1.0 + scaled_excess_epsilon) ** 2 + 4 / (np.pi * rho))
            ),
        )
        delta = delta_zcdp + (1 - delta_zcdp) * delta_excess
        if not return_tuple:
            return delta
    else:
        assert delta <= 1, "delta must be <= 1"
        assert delta > 0, "delta must be > 0"

        def obj(x):
            return approxzCDP_to_approxDP(
                rho, xi, delta_zcdp, epsilon=x, return_tuple=False
            )

        epsilon = binary_optimize(obj, delta)
        if not return_tuple:
            return epsilon
    return epsilon, delta
