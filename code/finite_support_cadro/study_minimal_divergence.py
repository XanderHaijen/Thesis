
from typing import Callable
from enum import Enum
import textwrap
import json

from dataclasses import dataclass

import numpy as np
from scipy.stats import dirichlet
from scipy.special import betainc, binom
from scipy.optimize import brentq


def scenario_bound(*, degree: int = 1, level: float = 0.05, confidence: float = None, nb_scenarios: int = None):
    if confidence is None and nb_scenarios is None:
        raise ValueError('Either prior or nscenario should differ from None or both.')
    
    if confidence < 0:
        if nb_scenarios is None:
            raise ValueError('Total scenario bound with zero dropout is not implemented.') 
        else:
            return int(np.floor(level*nb_scenarios)) - 1 

    if nb_scenarios is None:
        f = lambda n: betainc(n-degree+1, degree, 1-level) - confidence
        n1, n2 = degree, np.ceil(2/level*(degree-1+np.log(1/confidence)))
        return int(np.ceil(brentq(f, n1, n2, full_output=False, xtol=0.1)))
    elif confidence is None:
        return betainc(nb_scenarios-degree+1, degree, 1-level)
    else:
        if betainc(nb_scenarios-degree+1, degree, 1-level) > confidence:
            raise ValueError(f'Insufficient samples for requested guarantee required: {scenario_bound(degree, level, confidence)}.')
        f = lambda k: binom(k+degree-1, k) * betainc(nb_scenarios-degree-k+1, degree+k, 1-level) - confidence
        k1, k2 = 0, nb_scenarios-degree
        return int(np.floor(brentq(f, k1, k2, full_output=False, xtol=0.1)))



class Divergence(str, Enum):
    KL = "kl"
    BURG = "burg"
    HELLINGER = "hellinger"
    CHISQ = "chi-sq"
    TV = "tv" 


divergences = {
    Divergence.KL: (
        lambda t: t * np.log(t) - t + 1,  # phi function
        lambda t: np.exp(t) - 1,  # convex conjugate of phi
        lambda t: np.log(t),  # subgradient of phi
    ),
    Divergence.BURG: (
        lambda t: t - np.log(t) - 1,
        lambda t: -np.log(1 - t),
        lambda t: 1 - 1 / t,
    ),
    Divergence.HELLINGER: (
        lambda t: (np.sqrt(t) - 1) ** 2,
        lambda t: t / (1 - t),
        lambda t: 1 - 1 / np.sqrt(t),
    ),
    Divergence.CHISQ: (
        lambda t: (t - 1) ** 2 / t,
        lambda t: 2 - 2 * np.sqrt(1 - t),
        lambda t: 1 - t**(-2),
    ),
    Divergence.TV: (
        lambda t: np.abs(t - 1),
        lambda t: np.maximum(-1, t),
        lambda t: -1 * (t <= 1) + 1 * (t > 1),
    ),
}

# should implement y -> argmax_x y*x - f(x)
EXTREMUM = Callable[[float], float]


class Block:
    """Blocks of linked list used for pool adjacent violators."""
    __slots__ = 'value', 'size', 'optimizer', 'previous', 'next'  # provides more efficient memory usage
    def __init__(self, value: float, optimizer: float, size: int = 1, previous: 'Block' = None, next: 'Block' = None):
        self.value = value
        self.optimizer = optimizer
        self.size = size

        self.previous = previous
        self.next = next

    def __repr__(self):
        return f'block({self.value}, {self.size}, {self.optimizer})'

    @property
    def minimizer(self):
        return np.full(self.size, self.optimizer)
    
    def merge(self):
        """Merge with the next block."""
        if self.next is None:
            raise ValueError('No next block to merge with.')
        result = self.__class__(self.value + self.next.value, None, self.size + self.next.size, self.previous, self.next.next)
        if result.next is not None:
            result.next.previous = result
        if result.previous is not None:
            result.previous.next = result
        return result


def assign_links(lst: list[Block]):
    """Setup linked list based on list."""
    lst.insert(0, None)
    lst.append(None)
    for p, c, n in zip(lst[:-2], lst[1:-1], lst[2:]):
        c.previous = p
        c.next = n
    lst.pop(0), lst.pop()
    return lst


def ordered_conjugate(argmax: EXTREMUM, value: np.ndarray, *, return_blocks: bool = False):
    """Compute the maximizer of the ordered conjugate at the provided value using the pool adjacent violators algorithm.

    Source: Best, M. J., Chakravarti, N., & Ubhaya, V. A. (2000). 
                Minimizing Separable Convex Functions Subject to Simple 
                Chain Constraints. SIAM Journal on Optimization, 10(3), 658–672. 
                https://doi.org/10.1137/S1052623497314970

    The ordered conjugate of a separable function (x1, ... xn) |-> f(x1) + ... + f(xn) is defined as.
        minimize_x    <x, value> - (f(x1) + f(x2) + ... + f(xn))/n
        subject to    x1 <= x2 <= ... <= xn
    The algorithm requires the subgradient of the convex conjugate is required, 
    which is argmax_x {y*x - f(x)}. 

    Args:
        argmax (EXTREMUM): takes in y and returns argmax y*x - f(x)
        value (np.ndarray): the values at which to evaluated the ordered conjugate.
        return_blocks (bool, optional): return the linked list used to generate the solution. Defaults to False
    
    Returns:
        float: Maximizer of the ordered conjugate at the provided value. 
    """
    blocks = [Block(v, argmax(len(value)*v)) for v in value]
    blocks = assign_links(blocks)

    this = blocks[0]
    while this.next is not None:
        if this.optimizer > this.next.optimizer:
            this = this.merge()
            this.optimizer = argmax(len(value)*this.value/this.size)
            while this.previous is not None and this.previous.optimizer > this.optimizer:
                this = this.previous.merge()
                this.optimizer = argmax(len(value)*this.value/this.size)
        else:
            this = this.next

    result = []
    blocks = []
    while this is not None:
        blocks.append(this)
        result.append(this.minimizer)
        this = this.previous

    result = np.flip(np.concatenate(result))
    if return_blocks:
        return result, blocks[::-1]
    return result


@dataclass
class Calibration:
    label: str
    radius: float
    size: int 
    mode: Divergence
    info: dict = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.label}, mode={self.mode})'

    def __str__(self):
        content = f"radius: {self.radius}\n"
        content += f"mode: {self.mode}"
        content += "\ninfo: "
        content += json.dumps(self.info, indent=2)
        return (
            self.__class__.__name__
            + f' "{self.label}" :\n'
            + textwrap.indent(content, prefix=" " * 2)
        )


def calibrate(
    mode: Divergence,
    nb_samples: int,
    level: float = 0.05,
    *,
    nb_scenarios: int = 1000,
    confidence: float = -1,
    full_output: bool = True,
    **kwargs,
):
    """Calibrate divergence ambiguity set using ORM framework.

    Args:
        mode (Divergence): Type of divergence.
        nb_samples (int): Number of data points used in ORM (including robust term).
        level (float, optional): Confidence level of mean bound. Defaults to 0.05.
        nb_scenarios (int, optional): Number of samples from ordered conjugate used. Defaults to 1000.
        confidence (float, optional): Confidence of the data-driven estimate of optimal radius. Defaults to -1.
        full_output (bool, optional): Return meta data. Defaults to True.

    Returns:
        float: calibrated radius (and metadata).  
    """
    _, phi_conj, subgradient = divergences[mode]
    nb_dropout = scenario_bound(level=level, nb_scenarios=nb_scenarios, confidence=confidence)

    scenarios = dirichlet.rvs(
        np.ones(nb_samples), size=nb_scenarios, random_state=kwargs.get("random_state", None)
    )

    radii = np.empty(nb_scenarios)
    for i, scenario in enumerate(scenarios):
        optimum = ordered_conjugate(subgradient, scenario)  # get maximizer of ordered conjugate
        radii[i] = np.inner(optimum, scenario) - np.average(phi_conj(optimum))  # evaluate ordered conjugate at the maximizer
    radius = -np.partition(-radii, nb_dropout)[nb_dropout]  # remove nb_dropout largest values.

    if full_output:
        return Calibration(
            label=mode,
            radius=radius,
            size=nb_samples,
            mode=mode,
            info={
                'level': level,   
                'confidence': confidence,  
                'nb_scenarios': nb_scenarios,
                'nb_dropout': nb_dropout,       
            }
        )
    return radius



if __name__ == '__main__':
    nb_samples = 100

    # evaluate radius
    calibration = calibrate(Divergence.TV, nb_samples, 0.1, nb_scenarios=1000, confidence=0.01, full_output=True)
    print('calibration tv .........................................................')
    print(calibration)
    print('radius:', calibration.radius)

    # evaluate radius
    calibration = calibrate(Divergence.KL, nb_samples, 0.1, nb_scenarios=1000, confidence=0.01, full_output=True)
    print('calibration kl .........................................................')
    print(calibration)
    print('radius:', calibration.radius)