import numpy as np
import cmdstanpy as csp
import json
from scipy.special import logsumexp

import simulate

HMM = csp.CmdStanModel(stan_file="stan/hmm.stan")
TRADITIONAL = csp.CmdStanModel(stan_file="stan/traditional.stan")

SEED = 1234

N = 10
M = 10

TAU = 6
RHO = [6, 10]

def simulate_data(alpha, pi, nu):
    omega = 2
    beta = 0.2
    gamma = (-4, -1)
    theta = np.array([
        [ [p, 1 - p], [n, 1 - n] ]
        for p, n
        in zip(pi, nu)
    ])
    init = (1, 0.01)

    y, z, ll = simulate.hmm(
        N=N, 
        M=M, 
        alpha=alpha,
        omega=omega,
        beta=beta,
        gamma=gamma,
        theta=theta, 
        init=init,
        seed=SEED,
    )

    return y, z

def stan_data(y, z):
    index = np.array([[i * M + j for j, _ in enumerate(yy)] for i, yy in enumerate(y)])

    train_i, test_i = (
        np.concatenate([i[:n] for i, n in zip(index, range(N, 0, -1))]),
        np.concatenate([i[-n:] if n else np.array([], dtype=int) for i, n in zip(index, range(0, N))]),
    )

    train_y, train_z = y.flatten()[train_i], z.flatten()[train_i]
    test_y, test_z = y.flatten()[test_i], z.flatten()[test_i]

    ii, jj = (
        np.array([[i + 1] * len(yy) for i, yy in enumerate(y)]),
        np.array([list(range(1, len(yy) + 1)) for yy in y]),
    )

    B = np.concatenate([index.flatten()[train_i], index.flatten()[test_i]])

    MAX_PRED = np.max(y) * 100

    return {
        "T": len(train_i),
        "T_prime": len(test_i),
        "N": N,
        "M": M,
        "K": 2,
        "ii": np.concatenate([ii.flatten()[train_i], ii.flatten()[test_i]]),
        "jj": np.concatenate([jj.flatten()[train_i], jj.flatten()[test_i]]),
        "B": B,
        "y": y.flatten(),
        "learn": 1,
        "MAX_PRED": MAX_PRED,
        "tau": TAU,
        "rho": RHO,
    }

def fit(data, model):
    fit = model.sample(
        data=data,
        iter_warmup=2500,
        iter_sampling=2500,
        inits=0,
        seed=SEED,
    )

    y_tilde = fit.y_tilde.reshape((fit.y_tilde.shape[0], N, M))
    if "z_star" in fit.stan_variables():
        z_star = fit.z_star.reshape((fit.z_star.shape[0], N, M))
    else:
        z_star = []
    ll = fit.log_lik.reshape((fit.log_lik.shape[0], N, M))
    return y_tilde, z_star, ll

def simple():
    alpha = [3.5, 2.2, 1.6, 1.3, 1.2, 1.1, 1.05, 1.0, 1.0]
    pi = [1] + [0.7] * (M - 1)
    nu = [0] * M
    y, z = simulate_data(alpha, pi, nu)
    data = stan_data(y, z)
    y_tilde, z_star, ll = fit(data, model=HMM)
    y_tilde_trad, _, ll_trad = fit(data, model=TRADITIONAL)
    elpds = [
        np.sum(
            logsumexp(
                ll.reshape((ll.shape[0], N, M))[:,i, (M - i):], axis=0
            ) - np.log(ll.shape[0])
        )
        for i
        in range(N)
    ][1:]
    elpds_trad = [
        np.sum(
            logsumexp(
                ll_trad.reshape((ll.shape[0], N, M))[:,i, (M - i):], axis=0
            ) - np.log(ll_trad.shape[0])
        )
        for i
        in range(N)
    ][1:]
    z_star_trad = np.array([
        [
            [1] * TAU + [2] * (M - TAU)
            for n in range(N)
        ]
        for s in range(y_tilde.shape[0])
    ])
    json.dump(
        {
            "y": y.tolist(),
            "z": z.tolist(),
            "y_tilde": y_tilde.tolist(), 
            "z_star": z_star.tolist(),
            "y_tilde_trad": y_tilde_trad.tolist(), 
            "z_star_trad": z_star_trad.tolist(), 
            "elpds": list(zip(elpds, elpds_trad)),
        }, 
        open("results/simple_sim.json", "w")
    )

def main():
    simple()

if __name__ == "__main__":
    main()
