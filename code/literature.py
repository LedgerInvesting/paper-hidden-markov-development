from typing import List
import cmdstanpy as csp
import numpy as np
import json
from scipy.special import logsumexp
import re
import logging

from summarize import Score

logger = logging.getLogger(__name__)

RESULTS = "results"
SCALER = 1e5

SEED = 1234

TAU_RHOS = {
    "Balona & Richman (2022), long-tailed liability": (4, [4, 16]),
    "Balona & Richman (2022), short-tailed property": (4, [3, 20]),
    "Gisler (2015)": (4, [3, 21]),
    "Merz & Wuthrich (2015)": (4, [3, 16]),
    "Verrall & Wuthrich (2015)": (11, [10, 21]),
}


csp.set_cmdstan_path(".cmdstan/cmdstan-2.36.0")
HMM = csp.CmdStanModel(stan_file="stan/hmm.stan")
HMM_NU = csp.CmdStanModel(stan_file="stan/hmm-nu.stan")
HMM_LAG = csp.CmdStanModel(stan_file="stan/hmm-lag.stan")
CHANGEPOINT = csp.CmdStanModel(stan_file="stan/changepoint.stan")
TRADITIONAL = csp.CmdStanModel(stan_file="stan/traditional.stan")

SAMPLES = 2500
STAN_CONFIG = {
    "iter_sampling": SAMPLES,
    "iter_warmup": SAMPLES,
    "parallel_chains": 4,
    "inits": 0,
    "seed": SEED,
}

TRIANGLES = {
    "Balona & Richman (2022), long-tailed liability": "data/balona-richman-2022-long-tailed-liability.json",
    "Balona & Richman (2022), short-tailed property": "data/balona-richman-2022-short-tailed-property.json",
    "Gisler (2015)": "data/gisler-2015.json",
    "Merz & Wuthrich (2015)": "data/merz-wuthrich-2015.json",
    "Verrall & Wuthrich (2015)": "data/verrall-wuthrich-2015.json",
}


def flatten(x: List[List]) -> np.ndarray:
    return np.array([v for i in x for v in i])


def stan_data(triangle: List[List[float]]) -> np.ndarray:
    N = M = len(triangle)
    index_raw = [
        (i, j)
        for i, j in enumerate([j for j, period in enumerate(triangle) for v in period])
    ]
    index = np.array(
        [
            [i for i, j in index_raw if j == period] + [-9999] * (M - len(v))
            for period, v in enumerate(triangle)
        ]
    )
    mask = index == -9999
    train_i, test_i = (
        np.concatenate(
            [i[~m][:-1] for i, m, n in zip(index, mask, range(N - 1, 0, -1))]
        ),
        np.concatenate(
            [
                i[~m][-1:] if len(i[~m]) > 1 else np.array([], dtype=int)
                for i, m, n in zip(index, mask, range(N))
            ]
        ),
    )
    ii, jj = (
        [[i + 1] * len(yy[~m]) for i, (m, yy) in enumerate(zip(mask, index))],
        [list(range(1, len(yy[~m]) + 1)) for m, yy in zip(mask, index)],
    )
    keep = np.array([i for i in index[~mask] if i in train_i or i in test_i])
    y = np.array([y for values in triangle for y in values])[keep]
    MAX_PRED = np.max(y / SCALER) * 100
    return {
        "T": len(train_i),
        "T_prime": len(test_i),
        "N": N,
        "M": M,
        "K": 2,
        "tau": None,
        "rho": None,
        "ii": np.concatenate([flatten(ii)[train_i], flatten(ii)[test_i]]),
        "jj": np.concatenate([flatten(jj)[train_i], flatten(jj)[test_i]]),
        "B": np.concatenate([index[~mask][train_i], index[~mask][test_i]]),
        "y": y / SCALER,
        "learn": 1,
        "MAX_PRED": MAX_PRED,
        "train": train_i,
        "test": test_i,
    }


def fit_hmms(triangle: List[List[float]], paper: str) -> csp.CmdStanMCMC:
    d = stan_data(triangle) | {
        "tau": TAU_RHOS[paper][0],
        "rho": TAU_RHOS[paper][1],
    }
    base = HMM.sample(
        data=d,
        **STAN_CONFIG,
    )
    nu = HMM_NU.sample(
        data=d,
        **STAN_CONFIG,
    )
    lag = HMM_LAG.sample(
        data=d,
        **STAN_CONFIG,
    )
    return base, nu, lag


def fit_changepoint(triangle: List[List[float]], paper: str) -> csp.CmdStanMCMC:
    d = stan_data(triangle) | {
        "tau": TAU_RHOS[paper][0],
        "rho": TAU_RHOS[paper][1],
    }
    changepoint = CHANGEPOINT.sample(
        data=d,
        **STAN_CONFIG,
    )
    return changepoint


def fit_traditional(triangle: List[List[float]], paper: str) -> csp.CmdStanMCMC:
    d = stan_data(triangle) | {
        "tau": TAU_RHOS[paper][0],
        "rho": TAU_RHOS[paper][1],
    }
    traditional = TRADITIONAL.sample(
        data=d,
        **STAN_CONFIG,
    )
    return traditional


def elpd(triangle, models):
    indices = stan_data(triangle)
    test = indices["test"]
    S = models["hmm"].log_lik.shape[0]
    return np.array(
        [logsumexp(f.log_lik.T[test], axis=1) - np.log(S) for f in models.values()]
    )


def squared_error(triangle, models):
    indices = stan_data(triangle)
    test = indices["test"]
    y = indices["y"][test]
    return np.array(
        [(f.y_tilde[:, test].mean(axis=0) - y) ** 2 for f in models.values()]
    )


def percentile(d, models):
    indices = stan_data(d)
    test = indices["test"]
    y = indices["y"][test]
    p = [np.mean(f.y_tilde[:, test] <= y, axis=0) for f in models.values()]
    return p


def score(models, triangle, paper):
    paper = (
        re.sub(r"\(|\)| |&|,", "-", paper.lower())
        .replace("---", "-")
        .replace("--", "-")
    )
    indices = stan_data(triangle)
    test = indices["test"]
    ii = indices["ii"][-len(test) :]
    jj = indices["jj"][-len(test) :]
    raw_elpds = np.expand_dims(elpd(triangle, models), axis=0)
    ses = np.expand_dims(squared_error(triangle, models), axis=0)
    percentiles = np.expand_dims(percentile(triangle, models), axis=0)
    z_stars = np.array(
        [(f.z_star - 1).mean(axis=0) for f in models.values() if hasattr(f, "z_star")]
    )
    tau_stars = models["changepoint"].tau_star

    elpds = Score(raw_elpds, np.sum, ii, jj)

    def sqrt_mean(x, axis):
        return np.sqrt(np.mean(x, axis=axis))

    rmse = Score(ses, sqrt_mean, ii, jj)

    elpds.write(RESULTS + f"/elpd-{paper}.json")
    rmse.write(RESULTS + f"/rmse-{paper}.json")
    json.dump(percentiles.tolist(), open(RESULTS + f"/percentiles-{paper}.json", "w"))
    json.dump(z_stars.tolist(), open(RESULTS + f"/zstar-{paper}.json", "w"))

    with open(RESULTS + f"/taustar-{paper}.json", "w") as f:
        json.dump(tau_stars.tolist(), f)

def main():
    for paper, file in TRIANGLES.items():
        logger.info(f"Fitting models to {paper}")
        triangle = list(json.load(open(file)).values())
        logger.info("Fitting the HMM models")
        hmm, hmm_nu, hmm_lag = fit_hmms(triangle, paper)
        logger.info("Fitting the changepoint model")
        changepoint = fit_changepoint(triangle, paper)
        logger.info("Fitting the traditional model")
        traditional = fit_traditional(triangle, paper)
        score(
            {
                "hmm": hmm,
                "hmm_nu": hmm_nu,
                "hmm_lag": hmm_lag,
                "changepoint": changepoint,
                "traditional": traditional,
            },
            triangle,
            paper,
        )


if __name__ == "__main__":
    main()
