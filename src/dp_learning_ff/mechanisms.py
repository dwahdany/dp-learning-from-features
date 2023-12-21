from typing import Literal

from autodp.autodp_core import Mechanism
from autodp.calibrator_zoo import eps_delta_calibrator
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import ComposeGaussian
import math


class CoinpressGM(Mechanism):
    def __init__(self, Ps: list, name: str = "CoinpressGM"):
        """
        Initialize the CoinpressGM object.

        Args:
            Ps (list): List of privacy costs per step in (0,rho)-zCDP.
            name (str, optional): Name of the object. Defaults to "CoinpressGM".
        """
        self.params = {"Ps": Ps}
        self.name = name
        mechanisms = [GaussianMechanism(1 / math.sqrt(2 * p)) for p in Ps]
        compose = ComposeGaussian()
        mech = compose(mechanisms, [1 for _ in mechanisms])
        self.set_all_representation(mech)


class ScaledCoinpressGM(CoinpressGM):
    def __init__(
        self,
        scale: float,
        steps: int = 10,
        dist: Literal["lin", "exp", "log", "eq"] = "exp",
        ord: float = 1,
        name="ScaledCoinpressGM",
    ):
        """
        Initialize the ScaledCoinpressGM mechanism.

        Args:
            scale (float): The scaling factor.
            steps (int): The number of steps.
            dist (Literal["lin", "exp", "log", "eq"]): The distribution type.
            ord (float, optional): The order of the distribution. Defaults to 1.
            name (str, optional): The name of the mechanism. Defaults to "ScaledCoinpressGM".
        """
        assert scale > 0, "scale must be positive"
        assert steps > 0, "steps must be positive"
        self.scale = scale
        if dist == "lin":
            Ps = [math.pow(scale * (t + 1), ord) for t in range(steps)]
        elif dist == "exp":
            Ps = [math.pow(scale * math.exp(t / steps), ord) for t in range(steps)]
        elif dist == "log":
            Ps = [math.pow(scale * math.log(t + 1), ord) for t in range(steps)]
        elif dist == "eq":
            Ps = [scale] * steps
        super().__init__(name=name, Ps=Ps)


def calibrate_single_param(mechanism_class, epsilon, delta, p_min=0, p_max=1000):
    calibrator = eps_delta_calibrator()
    return calibrator(mechanism_class, epsilon, delta, [p_min, p_max])
