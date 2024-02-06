import math
from typing import Iterable, Literal, Optional

import numpy as np
from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism, zCDP_Mechanism
from autodp.transformer_zoo import AmplificationBySampling, ComposeGaussian, Composition
from opacus.accountants.analysis import gdp

from .utils import binary_optimize


class OpacusSGD(Mechanism):
    def __init__(
        self,
        steps: int,
        sampling_rate: float,
        noise_multiplier: Optional[float] = None,
        target_mu: Optional[float] = None,
        name: str = "OpacusSGD",
    ):
        super().__init__()
        self.name = name
        if noise_multiplier and target_mu:
            raise ValueError("Only one of noise_multiplier and target_mu must be set")
        if noise_multiplier is None and target_mu is None:
            raise ValueError("Either noise_multiplier or target_mu must be set")
        if noise_multiplier is None:

            def sigma_from_mu(mu, steps, sampling_rate):
                return 1 / (
                    np.sqrt(
                        np.log(mu**2 + steps * sampling_rate**2)
                        - np.log(steps * sampling_rate**2)
                    )
                )

            noise_multiplier = sigma_from_mu(target_mu, steps, sampling_rate)

        self.params = {
            "noise_multiplier": noise_multiplier,
            "steps": steps,
            "sampling_rate": sampling_rate,
        }
        mu = gdp.compute_mu_poisson(
            steps=steps,
            noise_multiplier=noise_multiplier,
            sample_rate=sampling_rate,
        )
        mech = GaussianMechanism(1 / mu)
        self.set_all_representation(mech)


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
        self.params = {"Ps": Ps, "p_sampling": p_sampling}
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


class PrototypicalNetwork(Mechanism):
    def __init__(
        self,
        share_projection: float,
        mu_total: float,
        proj_sampling_rate: float,
        proj_steps: int,
        est_sampling_rate: float,
        name: str = "PrototypicalNetwork",
    ):
        assert share_projection > 0, "share_projection must be > 0"
        assert share_projection < 1, "share_projection must be < 1"
        self.name = name

        def mu_projection(share, mu_total):
            return mu_total * share / (np.sqrt(share**2 + (1 - share) ** 2))

        def sigma_projection(share, mu_total):
            return 1 / mu_projection(share, mu_total)

        def mu_estimation(share, mu_total):
            return mu_total * (1 - share) / (np.sqrt(share**2 + (1 - share) ** 2))

        def sigma_estimation(share, mu_total):
            return 1 / mu_estimation(share, mu_total)

        self.mech_projection = OpacusSGD(
            target_mu=mu_projection(share_projection, mu_total),
            steps=proj_steps,
            sampling_rate=proj_sampling_rate,
        )
        rho = mu_estimation(share_projection, mu_total) ** 2 / 2  # convert GDP to zCDP
        self.mech_estimation = CoinpressGM(
            [5 / 64 * rho, 7 / 64 * rho, 52 / 64 * rho], p_sampling=est_sampling_rate
        )
        mech = Composition()([self.mech_projection, self.mech_estimation], [1, 1])
        self.set_all_representation(mech)

        self.params = {
            "share_projection": share_projection,
            "mu_total": mu_total,
            "proj_sampling_rate": proj_sampling_rate,
            "proj_steps": proj_steps,
            "est_sampling_rate": est_sampling_rate,
        }

    @classmethod
    def from_eps_delta(
        cls,
        epsilon: float,
        delta: float,
        share_projection: float,
        proj_sampling_rate: float,
        proj_steps: int,
        est_sampling_rate: float,
        verbose: bool = False,
    ):
        def obj(mu_total):
            return cls(
                share_projection=share_projection,
                mu_total=mu_total,
                proj_sampling_rate=proj_sampling_rate,
                proj_steps=proj_steps,
                est_sampling_rate=est_sampling_rate,
            ).get_approxDP(delta)

        mu_total = binary_optimize(
            obj, epsilon, min_open=True, strictly="leq", verbose=verbose
        )
        return cls(
            share_projection=share_projection,
            mu_total=mu_total,
            proj_sampling_rate=proj_sampling_rate,
            proj_steps=proj_steps,
            est_sampling_rate=est_sampling_rate,
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


def approx_zCDP_to_approxDP(
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
            return approx_zCDP_to_approxDP(
                rho, xi, delta_zcdp, epsilon=x, return_tuple=False
            )

        epsilon = binary_optimize(obj, delta)
        if not return_tuple:
            return epsilon
    return epsilon, delta
