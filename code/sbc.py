"""
Simulation-based calibration of the HMM model.
"""
from typing import Dict, Union
import cmdstanpy as csp
import scipy.stats as stat
import numpy as np
import json
from tqdm import tqdm
import logging

from simulate import hmm, ilogit

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.setLevel(logging.ERROR)
if not cmdstanpy_logger.handlers:
    cmdstanpy_logger.addHandler(logging.StreamHandler())
for handler in cmdstanpy_logger.handlers:
    handler.setLevel(logging.ERROR)

SEED = 1234

RNG = np.random.default_rng(SEED)

RESULTS = "results"

HMM = csp.CmdStanModel(stan_file="stan/hmm-sbc.stan")

P = 1000
SAMPLES = 1000
STAN_CONFIG = {
    "iter_sampling": SAMPLES,
    "iter_warmup": SAMPLES,
    "parallel_chains": 4,
    "inits": 0,
    "seed": SEED,
    "show_progress": False,
}

def stan_data(y: np.ndarray) -> Dict[str, Union[int, np.ndarray]]:
    N, M = y.shape
    index = np.array([[i * M + j for j, _ in enumerate(yy)] for i, yy in enumerate(y)])
    train_i, test_i = (
        np.concatenate([i[:n] for i, n in zip(index, range(N, 0, -1))]),
        np.concatenate([i[-n:] if n else np.array([], dtype=int) for i, n in zip(index, range(0, N))]),
    )
    train_y = y.flatten()[train_i]
    test_y = y.flatten()[test_i]
    ii, jj = (
        np.array([[i + 1] * len(yy) for i, yy in enumerate(y)]),
        np.array([list(range(1, len(yy) + 1)) for yy in y]),
    )
    return { 
        "T": len(train_i),
        "T_prime": len(test_i),
        "N": N,
        "M": M,
        "K": 2,
        "ii": np.concatenate([ii.flatten()[train_i], ii.flatten()[test_i]]),
        "jj": np.concatenate([jj.flatten()[train_i], jj.flatten()[test_i]]),
        "B": np.concatenate([index.flatten()[train_i], index.flatten()[test_i]]),
        "y": y.flatten(),
        "learn": 1,
    }


def generate_data(P: int = P):
    N = 10
    M = 10
    alpha_star_rng = stat.norm(
        [0] * (M - 1),
        [1 / (j + 1) for j in range(M - 1)],
    )
    omega_star_rng = stat.halfnorm(0, 1)
    beta_star_rng = stat.norm(0, 1)
    gamma_rng = stat.norm((-3, -1), (0.25, 0.1))
    pi_star_rng = stat.norm(0, 1)

    data = []
    while len(data) < P:
        pars = dict(
            alpha=np.exp(alpha_star_rng.rvs(random_state=RNG)),
            omega=np.exp(omega_star_rng.rvs(random_state=RNG)),
            beta=ilogit(beta_star_rng.rvs(random_state=RNG)),
            gamma=gamma_rng.rvs(random_state=RNG),
        )
        pi = [1] + [ilogit(pi_star_rng.rvs(random_state=RNG))] * (M - 1)
        pars |= {
            "theta": np.array([
                [ [p, 1 - p], [0, 1] ]
                for p
                in pi
            ])
        }
        y, z, ll = hmm(
            N=N,
            M=M,
            **pars,
            init=(1, 0.01),
            seed=None,
        )
        if np.isnan(y).any():
            continue
        else:
            data.append(((y, z, ll), pars))

    return data

def fit_hmm(data):
    fits = []
    for (y, z, ll), pars in tqdm(data):
        fits.append(
            (
                HMM.sample(
                    data=stan_data(y),
                    **STAN_CONFIG,
                ),
                y,
                z, 
                ll,
                pars,
            )
        )
    return fits


def _log_epsilon(ranks: np.ndarray, L: int) -> np.ndarray:
    K = L + 1
    R = np.zeros(K)
    log_epsilon = np.repeat(np.log(2), len(ranks))
    z = np.arange(1, K) / K

    for i in range(len(ranks)):
        R[ranks[i] + 1] = R[ranks[i] + 1] + 1
        ecdf = np.cumsum(R[:L])
        log_epsilon[i] += np.min(
            [
                stat.binom(i, z).logcdf(ecdf),
                np.log1p(stat.binom(i, z).cdf(ecdf - 1)),
            ]
        )
    return log_epsilon

def rank(fits):
    ranks = {}
    N, M = 10, 10
    index = np.array([
        i for i in range(N * M)
    ]).reshape((N, M))
    train_index = np.array([idx for i, n in zip(index, range(10, 0, -1)) for idx in i[:n]])
    thin = 10
    for fit, y, z, ll, pars in fits:
        draws_raw = fit.stan_variables()
        draws = {
            k: v[::thin]
            for k, v
            in draws_raw.items()
        }
        pars |= {
            "pi_": pars["theta"][-1, 0, 0]
        }
        keys = (k for k in draws if k in pars)
        for par in keys:
            if par in ranks:
                ranks[par].append(sum(pars[par] > draws[par]))
            else:
                ranks[par] = [sum(pars[par] > draws[par])]
        z_star = (draws["z_star"][:, train_index] - 1).mean(axis=0).round()
        if "z" in ranks:
            ranks["z"].append((z_star == z.flatten()[train_index]).mean())
        else:
            ranks["z"] = [(z_star == z.flatten()[train_index]).mean()]
        if "log likelihood" in ranks:
            ranks["log likelihood"].append(sum(ll.flatten()[train_index].sum() > draws["log_lik"][:,train_index].sum(axis=1)))
        else:
            ranks["log likelihood"] = [sum(ll.flatten().sum() > draws["log_lik"][:,train_index].sum(axis=1))]
        tilde_y = "tilde{y}[{1,10}]"
        if tilde_y in ranks:
            ranks[tilde_y].append(sum(y[0][-1] > draws["y_tilde"][:,9]))
        else:
            ranks[tilde_y] = [sum(y[0][-1] > draws["y_tilde"][:,9])]

    ranks["L"] = max([np.max(rank) for rank in ranks.values()])
    json.dump({k: np.asarray(v).tolist() for k, v in ranks.items()}, open(RESULTS + "/ranks.json", "w"))

def main():
    data = generate_data()
    fits = fit_hmm(data)
    rank(fits)

if __name__ == "__main__":
    main()
