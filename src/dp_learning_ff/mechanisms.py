from typing import Iterable, Literal, Optional

from autodp.autodp_core import Mechanism
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
            Ps (Optional[Iterable[float]], optional): The privacy costs per step. Overwrites steps, dist, ord. Defaults to None.
        """
        assert scale > 0, "scale must be positive"
        assert steps > 0, "steps must be positive"

        self.scale = scale
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
        super().__init__(name=name, Ps=Ps)


def calibrate_single_param(mechanism_class, epsilon, delta, verbose: bool = False):
    def obj(x):
        return mechanism_class(x).get_approxDP(delta)

    x = 1
    step = 0.5
    over = obj(x) > epsilon
    while True:
        if verbose:
            print(f"obj({x}) = {obj(x)}")
        curr_obj = obj(x)
        if (
            curr_obj < epsilon
            and epsilon - curr_obj < 1e-5
            and (epsilon - curr_obj) / epsilon < 1e-2
        ):
            break
        if curr_obj > epsilon:
            if not over:
                step /= 2
                over = True
            while x - step <= 0:
                step /= 2
            x -= step
        else:
            if over:
                step /= 2
                over = False
            x += step
    return mechanism_class(x)
