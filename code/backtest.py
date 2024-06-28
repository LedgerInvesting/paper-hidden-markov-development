from typing import List, Dict, Union, Optional, Tuple
import cmdstanpy as csp
import json
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
import logging

from summarize import Score
from __init__ import logger

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.setLevel(logging.ERROR)
if not cmdstanpy_logger.handlers:
    cmdstanpy_logger.addHandler(logging.StreamHandler())
for handler in cmdstanpy_logger.handlers:
    handler.setLevel(logging.ERROR)

SEED = 1234
SCALER = 1e4
DATA = {
    "PP": "data/pp.json",
    "WC": "data/wc.json",
    "CA": "data/ca.json",
    "OO": "data/oo.json",
}
HMM = csp.CmdStanModel(stan_file="stan/hmm.stan")
HMM_NU = csp.CmdStanModel(stan_file="stan/hmm-nu.stan")
HMM_LAG = csp.CmdStanModel(stan_file="stan/hmm-lag.stan")
TRADITIONAL = csp.CmdStanModel(stan_file="stan/traditional.stan")

RESULTS = "results"

CURSOR = "*************"

SAMPLES = 2500

STAN_CONFIG = {
    "iter_sampling": SAMPLES,
    "iter_warmup": SAMPLES,
    "parallel_chains": 4,
    "inits": 0,
    "seed": SEED,
    "show_progress": False,
}


def load_data(lob: str) -> List[Dict[str, List[Union[float, int]]]]:
    return json.load(open(DATA[lob], "r"))


def stan_data(d: List) -> Dict[str, int | float | np.ndarray]:
    d = np.array(d)[..., 0]
    N, M = d.shape
    index = np.array([[i * M + j for j, _ in enumerate(yy)] for i, yy in enumerate(d)])
    train_i, test_i = (
        np.concatenate([i[:n] for i, n in zip(index, range(N, 0, -1))]),
        np.concatenate(
            [
                i[-n:] if n else np.array([], dtype=int)
                for i, n in zip(index, range(0, N))
            ]
        ),
    )

    ii, jj = (
        np.array([[i + 1] * len(yy) for i, yy in enumerate(d)]),
        np.array([list(range(1, len(yy) + 1)) for yy in d]),
    )

    MAX_PRED = max(d.flatten() / SCALER) * 100

    return {
        "T": len(train_i),
        "T_prime": len(test_i),
        "N": N,
        "M": M,
        "K": 2,
        "tau": 6,
        "rho": [4, 10],
        "ii": np.concatenate([ii.flatten()[train_i], ii.flatten()[test_i]]),
        "jj": np.concatenate([jj.flatten()[train_i], jj.flatten()[test_i]]),
        "B": np.concatenate([index.flatten()[train_i], index.flatten()[test_i]]),
        "y": d.flatten() / SCALER,
        "learn": 1,
        "MAX_PRED": MAX_PRED,
    }


def fit_hmm(data):
    base = []
    nu = []
    lag = []
    for i, (key, d) in enumerate(tqdm(data.items())):
        base.append(
            HMM.sample(
                data=stan_data(d),
                **STAN_CONFIG,
            )
        )
        nu.append(
            HMM_NU.sample(
                data=stan_data(d),
                **STAN_CONFIG,
            )
        )
        lag.append(
            HMM_LAG.sample(
                data=stan_data(d),
                **STAN_CONFIG,
            )
        )
    return base, nu, lag


def fit_traditional(data):
    fits = []
    for i, (key, d) in enumerate(tqdm(data.items())):
        fits.append(
            TRADITIONAL.sample(
                data=stan_data(d),
                **STAN_CONFIG,
            )
        )
    return fits


def elpd(d, *fits):
    indices = stan_data(d)
    test = indices["B"][55:]
    elpds = [
        logsumexp(f.log_lik.T[test], axis=1) - np.log(f.log_lik.shape[0]) for f in fits
    ]
    return elpds


def squared_error(d, *fits):
    indices = stan_data(d)
    test = indices["B"][55:]
    y = indices["y"][test]
    se = [(f.y_tilde.T[test].mean(axis=1) - y) ** 2 for f in fits]
    return se


def percentile(d, *fits):
    indices = stan_data(d)
    test = indices["B"][55:]
    y = indices["y"][test]
    p = [np.mean(f.y_tilde[:, test] <= y, axis=0) for f in fits]
    return p


def score(data, models, lob) -> Tuple[Score, Score, np.ndarray, np.ndarray]:
    indices = stan_data(data[next(iter(data))])
    test = indices["B"][55:]
    ii = indices["ii"][-len(test) :]
    jj = indices["jj"][-len(test) :]
    raw_elpds = np.array(
        [elpd(d, *fits) for d, fits in zip(data.values(), zip(*models.values()))]
    )
    ses = np.array(
        [
            squared_error(d, *fits)
            for d, fits in zip(data.values(), zip(*models.values()))
        ]
    )
    percentiles = np.array(
        [percentile(d, *fits) for d, fits in zip(data.values(), zip(*models.values()))]
    )
    z_stars = np.array(
        [
            [
                (f.z_star - 1).mean(axis=0).reshape((10, 10))
                for f in fits
                if hasattr(f, "z_star")
            ]
            for fits in zip(*models.values())
        ]
    )
    elpds = Score(raw_elpds, np.sum, ii, jj)

    def sqrt_mean(x, axis):
        return np.sqrt(np.mean(x, axis=axis))

    rmse = Score(ses, sqrt_mean, ii, jj)

    elpds.write(RESULTS + f"/elpd-{lob}.json")
    elpds.write(RESULTS + f"/elpd-{lob}-filter-1e4.json", lambda x: x > -1e4)
    elpds.write(RESULTS + f"/elpd-{lob}-filter-1e3.json", lambda x: x > -1e3)
    elpds.write(RESULTS + f"/elpd-{lob}-filter-1e2.json", lambda x: x > -1e2)
    elpds.write(RESULTS + f"/elpd-{lob}-filter-0.json", lambda x: x > 0)
    rmse.write(RESULTS + f"/rmse-{lob}.json")

    with open(RESULTS + f"/zstar-{lob}.json", "w") as f:
        json.dump(z_stars.tolist(), f)

    with open(RESULTS + f"/percentiles-{lob}.json", "w") as f:
        json.dump(percentiles.tolist(), f)

    return elpds, rmse, percentiles, z_stars


def main() -> None:
    for lob in DATA:
        logger.info(f"Backtesting line of business {lob}")
        data = load_data(lob)
        hmm, hmm_nu, hmm_lag = fit_hmm(data)
        traditional = fit_traditional(data)
        score(
            data,
            {
                "hmm": hmm,
                "hmm_nu": hmm_nu,
                "hmm_lag": hmm_lag,
                "traditional": traditional,
            },
            lob,
        )


if __name__ == "__main__":
    main()
