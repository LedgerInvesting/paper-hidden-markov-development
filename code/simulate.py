from typing import Tuple, Sequence, Union
import numpy as np
import scipy.stats as stat
from functools import partial


def ilogit(x: Union[np.ndarray, float]) -> np.ndarray:
    return np.exp(x) / (1 + np.exp(x))


def lognormal_from_mean_variance(
    mean: float, variance: float, rng: np.random.Generator
) -> float:
    mu = np.log(mean**2 / np.sqrt(mean**2 + variance))
    sigma = np.sqrt(np.log(1 + variance / mean**2))
    return stat.lognorm(scale=np.exp(mu), s=sigma).rvs(random_state=rng)


def lognormal_lpdf_from_mean_variance(
    y: float, mean: float, variance: float, rng: np.random.Generator
) -> float:
    mu = np.log(mean**2 / np.sqrt(mean**2 + variance))
    sigma = np.sqrt(np.log(1 + variance / mean**2))
    return stat.lognorm(scale=np.exp(mu), s=sigma).logpdf(y)


def gamma_from_mean_variance(
    mean: float, variance: float, rng: np.random.Generator
) -> float:
    rate = mean / variance
    shape = mean**2 / variance
    return stat.gamma(a=shape, scale=rate**-1).rvs(random_state=rng)


def gamma_lpdf_from_mean_variance(
    y: float, mean: float, variance: float, rng: np.random.Generator
) -> float:
    rate = mean / variance
    shape = mean**2 / variance
    return stat.gamma(a=shape, scale=rate**-1).logpdf(y)


def hmm(
    N: int,
    M: int,
    theta: np.ndarray,
    alpha: Sequence[float],
    omega: float,
    beta: float,
    gamma: Tuple[float, float],
    init: Tuple[float, float] = (1, 1),
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)

    def generate_z(theta: float) -> int:
        return rng.binomial(n=1, p=theta)

    years = list(range(N))
    lags = list(range(M))
    years_lags = [(year, lag) for year in years for lag in lags]

    y = np.empty(shape=(N, M))
    mu = np.empty(shape=(N, M, 2))
    sigma2 = np.empty(shape=(N, M))
    z = np.empty(shape=(N, M), dtype=int)
    ll = np.empty(shape=(N, M))

    for i, j in years_lags:
        lag = j + 1
        if not j:
            z[i, j] = generate_z(theta[j][0, 1])
            y[i, j] = lognormal_from_mean_variance(*init, rng)
            ll[i, j] = 0.0
            mu[i, j] = [0, 0]
            sigma2[i, j] = 0
        else:
            z[i, j] = generate_z(theta[j][z[i, j - 1], 1])
            lagged_y = y[i, j - 1]
            mu[i, j] = [lagged_y * alpha[j - 1], lagged_y * omega**beta**lag]
            sigma2[i, j] = np.exp(gamma[0] + gamma[1] * lag + np.log(lagged_y))
            try:
                distr = stat.lognorm(scale=mu[i, j, z[i, j]], s=np.sqrt(sigma2[i, j]))
                y[i, j] = distr.rvs(random_state=rng)
                ll[i, j] = distr.logpdf(y[i, j])
            except ValueError:
                breakpoint()
                y[i, j] = np.nan
                ll[i, j] = np.nan

    return y, z, ll
