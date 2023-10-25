import textwrap
import json
from math import floor, sqrt

from typing import Literal
import copy

from dataclasses import dataclass

import numpy as np
import numpy.random as npr

from scipy.optimize import brentq
from scipy.stats import dirichlet
from scipy import special
from scipy.special import betainc, binom


# ------------------------------------------------------------------------------
# Calibration of Risk
# ------------------------------------------------------------------------------

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


@dataclass
class Calibration:
    label: str
    weights: np.ndarray
    mode: str
    info: dict = None

    def __post_init__(self):
        self.weights = np.array(self.weights)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.label}, length={len(self.weights)}, mode={self.mode})'

    def __str__(self):
        content = "weights:\n" + textwrap.indent(str(self.weights), prefix=" " * 2)
        content += f"\nmode: {self.mode}"
        content += "\ninfo: "
        content += json.dumps(self.info, indent=2)
        return (
            self.__class__.__name__
            + f' "{self.label}" :\n'
            + textwrap.indent(content, prefix=" " * 2)
        )

    @property
    def size(self):
        return len(self.weights)
    

def _poisson(k: np.ndarray, loc: np.ndarray):
    squeeze = [False, False]
    if isinstance(k, (int, float)):
        k = np.array([k])
        squeeze[0] = True
    if isinstance(loc, (int, float)):
        loc = np.array([loc])
        squeeze[1] = True
    if loc.ndim == 0:
        loc = np.array([loc])
        squeeze[1] = True
    res = [
        np.exp(special.xlogy(k, np.abs(ell)) - special.gammaln(k + 1) - np.abs(ell))
        * np.sign(ell) ** k
        for ell in loc
    ]
    res = np.array(res)
    if squeeze[0] and squeeze[1]:
        return np.squeeze(res)
    if squeeze[0] or squeeze[1]:
        return np.reshape(res, (-1))
    return res


def crossprob(w: np.ndarray, k: int = None):
    nsample = len(w) - 1
    if k is None:
        k = floor(sqrt(nsample)) - 1

    w = np.flip(w)
    q = np.concatenate([np.array([0.0, 1.0]), np.zeros(nsample)])
    for i in range(0, nsample + 1, k + 1):
        # store part of state vector
        r = np.copy(q[: k + 1])

        # perform leap
        q = np.convolve(
            q[1:],
            _poisson(np.arange(nsample + 1 - i), nsample * np.sum(w[i : i + k + 1])),
        )[k : nsample + 1 - i]

        # correct for failures
        for j in range(0, k):
            r = np.convolve(
                r[1:], _poisson(np.arange(k - j + 1), nsample * w[i + j])
            )[: k - j]
            if i + j == nsample:
                return r[0] / _poisson(nsample, nsample)
            q = q - (
                r[0]
                * _poisson(
                    np.arange(k - j, nsample + 1 - (i + j)),
                    nsample * sum(w[i + j + 1 : i + k + 1]),
                )
            )
        if i + k == nsample:
            return q[0] / _poisson(nsample, nsample)



def crossprob_direct(w: np.ndarray):
    """Equivalent to ``crossprob(np.flip(w))``."""
    w = np.flip(w)
    nsample = len(w) - 1
    q = np.concatenate([np.array([0.0, 1.0]), np.zeros(nsample)])
    for i in range(nsample):
        q = np.convolve(q[1:], _poisson(np.arange(nsample+1-i), nsample * w[i]))[:nsample+1-i]  # only compute necessary values
        # q = np.convolve(q, _poisson(np.arange(nsample+1), nsample * w[i]))[1:]  # compute all values ...
    return q[1] * _poisson(0, nsample * w[nsample]) / _poisson(nsample, nsample)


def odirichlet(alpha: np.ndarray, size: np.ndarray, *, rg: npr.Generator = None):
    res = dirichlet(alpha).rvs(size=size, random_state=rg)
    return np.cumsum(res[..., :-1], axis=-1)


def cov2bnd(w: np.ndarray):
    return np.cumsum(w)[:-1]


def bnd2cov(b: np.ndarray):
    return np.diff(np.concatenate([np.zeros(1), b, np.ones(1)]))


def radius2cvar(length: int, gamma: float, *, normalized: bool = False):
    return bnd2cov(np.maximum(np.arange(1, length) / (length if normalized else length - 1) - gamma, 0.0))


Method = Literal["brentq", "scenario", "asymptotic"]


def calibrate(
    length: int,
    level: float = 0.05,
    *,
    method: Method = "brentq",
    full_output: bool = True,
    normalized: bool = False,
    prob = crossprob,
    **kwargs,
):
    if normalized and method != "brentq":
        raise NotImplementedError(f'Normalized cvar not implemented for method: "{method}".')
    if method == "brentq":
        f = lambda gamma: prob(radius2cvar(length, gamma, normalized=normalized)) - (1 - level)
        gamma, info = brentq(
            f,
            -1,
            1,
            **{k: kwargs[k] for k in ["xtol", "rtol", "maxiter"] if k in kwargs},
            full_output=True,
        )

        vertex = radius2cvar(length, gamma, normalized=normalized)
        if full_output:
            return Calibration(
                "cvar.n" if normalized else "cvar",
                vertex,
                "cvar",
                {
                    "mode": "cvar",
                    "level": level,
                    "monotone": False,
                    "method": method,
                    "radius": gamma,
                    "normalized": normalized
                }
                | info.__dict__,
            )
        return vertex
    elif method == "scenario":
        # get options
        nb_scenarios = kwargs.get("nb_scenarios", 10_000)
        confidence = kwargs.get("confidence", -1)
        rg = kwargs.get("rg", npr.default_rng())

        # configure constraint removal
        # dropout = int(np.floor(level * nb_scenarios)) - 1
        dropout = scenario_bound(level=level, confidence=confidence, nb_scenarios=nb_scenarios)
        if dropout <= 0:
            res = np.zeros(length)
            res[-1] = 1.0
            return res

        # sample from ordered dirichlet
        scenario = odirichlet(np.ones(length), nb_scenarios, rg=rg)

        # compute radius parameter for all scenarios
        center = np.arange(1, length) / (length - 1)
        radius = np.max(center - scenario, axis=-1)

        # dropout the largest ones
        radius = -np.partition(-radius, dropout)[dropout]

        # compute vertex
        offset = np.maximum(center - radius, np.zeros((length - 1,)))
        vertex = np.append(offset, np.ones((1,)))
        vertex[1:, ...] = vertex[1:, ...] - vertex[:-1, ...].copy()

        info = {
            "mode": "cvar",
            "level": level,
            "monotone": False,
            "method": method,
            "nb_scenarios": nb_scenarios,
            "confidence": confidence,
            "dropout": dropout,
            "radius": radius,
            "normalized": False
        }

        if full_output:
            return Calibration("cvar", vertex, "cvar", info)
        return vertex
    elif method == "asymptotic":
        gamma = np.sqrt(np.log(1 / level) / (2 * (length - 1)))
        vertex = radius2cvar(length, gamma, normalized=normalized)
        if not full_output:
            return vertex
        return Calibration(
            "cvar",
            vertex,
            "cvar",
            {
                "level": level,
                "monotone": False,
                "method": method,
                "radius": float(gamma),
                "normalized": False
            },
        )
    else:
        raise ValueError(f"Invalid method {method}. Expected 'brentq', 'scenario' or 'asymptotic'.")
    

# ------------------------------------------------------------------------------
# Evaluation of Risk
# ------------------------------------------------------------------------------

def risk(samples: np.ndarray, calibration: np.ndarray | Calibration, *, bound: float = None):
    rs = False
    if samples.ndim == 1:
        samples = samples.reshape((1, -1))
        rs = True
    if bound is not None:
        samples = np.hstack([samples, np.full((samples.shape[0], 1), bound)])
    
    samples = np.sort(samples, axis=-1)

    if isinstance(calibration, Calibration):
        calibration = calibration.weights

    res = np.einsum('ij, j->i', samples, calibration)
    return res[0] if rs else res
    

# ------------------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------------------
    

if __name__ == '__main__':
    # evaluate radius exactly
    calibration = calibrate(10, 0.1, method="brentq", full_output=False)
    print('calibration exact (brentq)')
    print(calibration)

    # evaluate radius using asymptotic bound
    calibration = calibrate(100, 0.1, method="asymptotic", full_output=True)
    print('calibration asymptotic')
    print(calibration)

    # evaluate radius using data-driven bound
    calibration = calibrate(10, 0.1, method="scenario", nb_scenarios=100_000, confidence=0.0001)
    print('calibration data-driven')
    print(calibration)

    # evaluate risk using given calibration
    print(risk(np.arange(9), calibration, bound=20))