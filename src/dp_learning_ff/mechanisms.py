from typing import Literal

from autodp.autodp_core import Mechanism
from autodp.calibrator_zoo import eps_delta_calibrator
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import ComposeGaussian
import math


class CoinpressGM(Mechanism):
    def __init__(self, Ps, name="CoinpressGM"):
        self.params = {"Ps": Ps}
        self.name = name
        mechanisms = [GaussianMechanism(1 / math.sqrt(2 * p)) for p in Ps]
        compose = ComposeGaussian()
        mech = compose(mechanisms, [1 for _ in mechanisms])
        self.set_all_representation(mech)


class ScaledCoinpressGM(CoinpressGM):
    def __init__(
        self,
        scale,
        steps,
        dist: Literal["lin", "exp", "log", "eq"],
        name="ScaledCoinpressGM",
    ):
        self.scale = scale
        if dist == "lin":
            Ps = [scale * (t + 1) for t in range(steps)]
        elif dist == "exp":
            Ps = [scale * math.exp(t / steps) - 1 for t in range(steps)]
        elif dist == "log":
            Ps = [scale * math.log(t + 1) for t in range(steps)]
        elif dist == "eq":
            Ps = [scale] * steps
        super().__init__(name=name, Ps=Ps)


def calibrate(mechanism_class, epsilon, delta, p_min=0, p_max=1000):
    calibrator = eps_delta_calibrator()
    return calibrator(mechanism_class, epsilon, delta, p_min, p_max)
