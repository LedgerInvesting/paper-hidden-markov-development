import numpy as np
import cmdstanpy as csp

import simulate
from plot import plot_paths

HMM = csp.CmdStanModel(stan_file="stan/hmm.stan")
TRADITIONAL = csp.CmdStanModel(stan_file="stan/traditional.stan")

N = 10
M = 10
alpha = [3.5, 2.2, 1.6, 1.3, 1.2, 1.1, 1.05, 1.0, 1.0]
omega = 2
beta = 0.2
pi = [1] + [0.7] * 9
gamma = (-2, -1)
theta = np.array([
    [ [p, 1 - p], [0, 1] ]
    for p
    in pi
])
init = (1, 0.01)
seed = 1234

y, z, ll = simulate.hmm(
    N=N, 
    M=M, 
    alpha=alpha,
    omega=omega,
    beta=beta,
    gamma=gamma,
    theta=theta, 
    init=init,
    seed=None,
)

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

fit = HMM.sample(
    data={
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
    },
    iter_warmup=1000,
    iter_sampling=1000,
    seed=seed,
)

y_tilde = fit.y_tilde.reshape((fit.y_tilde.shape[0], N, M))
z_star = fit.z_star.reshape((fit.z_star.shape[0], N, M))

plot_paths((y, z), (y_tilde, z_star), "simulated.png")

fit2 = TRADITIONAL.sample(
    data={
        "T": len(train_i),
        "T_prime": len(test_i),
        "N": N,
        "M": M,
        "tau": 6,
        "rho": [4, 10],
        "ii": np.concatenate([ii.flatten()[train_i], ii.flatten()[test_i]]),
        "jj": np.concatenate([jj.flatten()[train_i], jj.flatten()[test_i]]),
        "B": np.concatenate([index.flatten()[train_i], index.flatten()[test_i]]),
        "y": y.flatten(),
        "learn": 1,
    },
    iter_warmup=1000,
    iter_sampling=1000,
    seed=seed,
)

y_tilde = fit2.y_tilde.reshape((fit2.y_tilde.shape[0], N, M))
z_star = np.zeros_like(y_tilde)
plot_paths((y, z), (y_tilde, z_star), "simulated-clb.png")
