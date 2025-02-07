from typing import List, Tuple, Dict
import numpy as np
import scipy.stats as stat
from arviz import hdi
import re
import matplotlib.pyplot as plt
from matplotlib import colors

from summarize import Score
from backtest import SCALER as BACKTEST_SCALER
from literature import SCALER as LITERATURE_SCALER
from literature import TAU_RHOS
from __init__ import logger

plt.style.use("publication.mplstyle")

BODY = "#5b84b1"
TAIL = "#fc766a"

LETTERS = list(map(chr, range(65, 91)))

MODEL_NAMES = [
    "HMM",
    r"HMM-$\nu$",
    "HMM-lag",
    "Change-point",
    "Two-step",
]

LOB_COLORS = {
    "PP": "#332288",
    "WC": "#44aa99",
    "CA": "#88ccee",
    "OO": "#ddcc77",
}

PROGRAM_COLORS = {
    "band": "#fde49e",
    "z": "#6dc5d1",
}

PATH = "figures/"

DataType = List[List[int]]


def plot_numerical(numerical: Dict[str, Dict[str, List]]):
    R = 10
    C = 3
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C, figsize=(12, 12), sharex=True)
    fig.supxlabel("Development period", fontsize=16)
    fig.supylabel("Losses", fontsize=16)
    rng = np.random.default_rng(1234)

    name = "simple"
    y, z, y_tilde, z_star, y_tilde_changepoint, z_star_changepoint, y_tilde_trad, z_star_trad, elpds = numerical[name].values()
    preds = zip(np.array([y_tilde, y_tilde_changepoint, y_tilde_trad]), np.array([z_star, z_star_changepoint, z_star_trad]))
    N, M = y.shape

    for c, (y_tilde, z_star) in enumerate(preds):
        for r in range(R):
            ax[r, c].grid(zorder=-5)
            ax[r, c].set_yticklabels([])
            ax[r, c].set_xticks(range(0, M, 3))
            ax[r, c].set_xticklabels(range(1, M + 1, 3))
            body = y[r][z[r] == 0]
            tail = y[r][z[r] == 1]
            train_index = [i for i in range(M) if i < N - r]
            test_index = [i for i in range(M) if i >= N - r]
            body_index = [i for i in range(M) if not z[r][i]]
            tail_index = [i for i in range(M) if z[r][i]]
            train = y[r][train_index]
            test = y[r][test_index]
            samples = rng.choice(range(y_tilde.shape[0]), 200)
            for sample in samples:
                z_pred = z_star[sample][r] - 1
                y_pred = y_tilde[sample][r]
                body_pred = [i for i in range(M) if not z_pred[i]]
                tail_pred = [i for i in range(M) if z_pred[i]]
                for i in range(1, M):
                    ax[r, c].plot(
                        [i - 1, i],
                        [y_pred[i - 1], y_pred[i]],
                        color=BODY if i - 1 in body_pred else TAIL,
                        lw=0.3,
                        alpha=0.3,
                        zorder=-1,
                    )
            y_bar = y_tilde.mean(axis=0)[r]
            body_bar = [i for i in range(M) if (z_star - 1).mean(axis=0)[r][i] < 0.5]
            for i in range(1, M):
                if not c and not r and i == 1:
                    label = "Body sample"
                elif not c and not r and i == (M - 1):
                    label = "Tail sample"
                else:
                    label = ""
                ax[r, c].plot(
                    [i - 1, i],
                    [y_bar[i - 1], y_bar[i]],
                    color=BODY if i - 1 in body_bar else TAIL,
                    lw=3,
                    label=label,
                )
            for i in range(M):
                marker = "o" if i in train_index else "^"
                if not c and r == 1 and not i:
                    label = "Body (train)"
                elif not c and r == 5 and i == (M - 1):
                    label = "Body (test)"
                elif not c and not r and i == (M - 1):
                    label = "Tail (train)"
                elif not c and r == 1 and i == (M - 1):
                    label = "Tail (test)"
                else:
                    label = ""
                if i in body_index:
                    ax[r, c].scatter(
                        i,
                        y[r][i],
                        color=BODY,
                        edgecolor="black",
                        marker=marker,
                        s=40,
                        zorder=1,
                        label=label,
                    )
                else:
                    ax[r, c].scatter(
                        i,
                        y[r][i],
                        color=TAIL,
                        edgecolor="black",
                        marker=marker,
                        s=40,
                        zorder=1,
                        label=label,
                    )
            if not r:
                ax[r, c].text(
                    0,
                    1.1,
                    LETTERS[c],
                    transform=ax[r, c].transAxes,
                    fontsize=20,
                    fontweight="bold",
                )
            if r:
                ax[r, c].text(
                    0,
                    1.025,
                    f"ELPD: {round(elpds[r - 1][c], 2)}",
                    transform=ax[r, c].transAxes,
                    fontsize=12,
                )

    labels_handles = {
        label: handle
        for ax in fig.axes
        for handle, label in zip(*ax.get_legend_handles_labels())
    }
    labels_handles = {
        label: labels_handles[label]
        for label in (
            "Body (train)",
            "Body (test)",
            "Tail (train)",
            "Tail (test)",
            "Body sample",
            "Tail sample",
        )
    }
    fig.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        ncol=3,  # len(labels_handles),
        bbox_to_anchor=(0.5, 1.075),
    )
    plt.savefig(PATH + "numerical.png")
    plt.close()


def _extract_atas(cells: DataType) -> np.ndarray:
    raw = [
        [loss / prev for loss, prev in zip(period[1:], period[:-1])] for period in cells
    ]
    return np.array(list(zip(*raw)))


def plot_atas(data: List[List[DataType]]) -> None:
    R = len(data)
    C = 1
    fig, ax = plt.subplots(R, C, sharex=True)
    rc = [(r, c) for r in range(R) for c in range(C)]

    for idx, (r, c) in enumerate(rc):
        lob = list(data)[idx]
        atas = np.array(
            [_extract_atas(np.array(d)[..., 0]) for d in data[lob].values()]
        )
        means = atas.mean(axis=0).mean(axis=1)
        error = 1.96 * atas.mean(axis=0).std(axis=1)
        grid = np.arange(1, atas.shape[1] + 1)
        ax[r].axhline(y=1.0, ls=":", color="gray")
        ax[r].errorbar(
            grid, means, yerr=error, color=LOB_COLORS[lob], fmt="-o", label=lob.upper()
        )
        ax[r].set_ylabel("Link ratio")
        ax[r].text(
            -0.075,
            1.1,
            LETTERS[r],
            transform=ax[r].transAxes,
            fontsize=20,
            fontweight="bold",
        )

    ax[-1].set_xlabel("Development lag")
    plt.xticks(ticks=range(1, atas.shape[1]), labels=range(1, atas.shape[1]))
    fig.align_ylabels(ax)
    labels_handles = {
        label: handle
        for ax in fig.axes
        for handle, label in zip(*ax.get_legend_handles_labels())
    }
    fig.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        ncol=len(LOB_COLORS),
        bbox_to_anchor=(0.5, 1.075),
    )
    plt.savefig(PATH + "/atas.png")
    plt.close()


def plot_scores(
    scores: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]],
) -> None:
    R = len(scores)
    C = len(scores[next(iter(scores))])
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C)

    for i, (r, c) in enumerate(rc):
        lob = list(scores)[r]
        score_raw = np.array(scores[lob][c]["scores"])
        score = np.sqrt(score_raw) * BACKTEST_SCALER / 1e3 if c else score_raw
        if c:
            ordered = sorted(
                (s, i) for i, s in enumerate(score.mean(axis=2).mean(axis=0))
            )
            diffs = np.array(
                [
                    score[:, min(ordered)[1], :] - score[:, m, :]
                    for m in [i for _, i in ordered]
                ]
            )
            diffs_mu = diffs.mean(axis=2).mean(axis=1)
            diffs_se = (np.sqrt(diffs.var(axis=2, ddof=1) / score.shape[-1])).mean(
                axis=1
            )
        else:
            ordered = sorted(
                [(s, i) for i, s in enumerate(score.sum(axis=2).mean(axis=0))],
                reverse=True,
            )
            diffs = np.array(
                [
                    score[:, max(ordered)[1], :] - score[:, m, :]
                    for m in [i for _, i in ordered]
                ]
            )
            diffs_mu = diffs.sum(axis=2).mean(axis=1)
            diffs_se = (np.sqrt(diffs.var(axis=2, ddof=1) * score.shape[-1])).mean(
                axis=1
            )
        names = [MODEL_NAMES[i] for _, i in ordered]
        lowers, uppers = diffs_mu - diffs_se * 2, diffs_mu + diffs_se * 2
        marker = "^" if c else "o"
        errors = [
            (abs(low - mu), abs(mu - high))
            for mu, low, high in zip(diffs_mu, lowers, uppers)
        ]
        for i, (name, mu, error) in enumerate(zip(names, diffs_mu, errors)):
            ax[r, c].errorbar(
                mu,
                i,
                xerr=error[0],
                color=LOB_COLORS[lob],
                ecolor="black",
                elinewidth=2,
                fmt=marker,
                alpha=0.3 if mu == 0.0 else 1,
                label=lob.upper() if i == 1 else None,
            )
            if not i:
                ax[r, c].text(
                    0,
                    -0.3,
                    ordered[0][0].round(1),
                    fontsize=12,
                    ha="center",
                    va="center",
                )
        ax[r, c].set_yticks(range(score.shape[1]))
        ax[r, c].set_yticklabels(names)
        ax[r, c].axvline(0, ls=":", color="gray")
        ax[r, c].set_ylim(ax[r, c].get_ylim()[::-1])
        if r == (R - 1):
            ax[r, c].set_xlabel("ELPD difference" * (1 - c) + c * "RMSE difference")
        if not c:
            ax[r, c].text(
                -0.1,
                1.1,
                LETTERS[r],
                transform=ax[r, c].transAxes,
                fontsize=20,
                fontweight="bold",
            )

    labels_handles = {
        label + str(i % 2): handle
        for i, ax in enumerate(fig.axes)
        for handle, label in zip(*ax.get_legend_handles_labels())
    }
    fig.legend(
        [h for l, h in labels_handles.items() if l[-1] == "0"],
        [l[:-1] for l, h in labels_handles.items() if l[-1] == "0"],
        title="ELPD difference +/- 2 SE",
        loc="upper center",
        ncol=len(LOB_COLORS),
        bbox_to_anchor=(0.32, 1.1),
    )
    fig.legend(
        [h for l, h in labels_handles.items() if l[-1] == "1"],
        [l[:-1] for l, h in labels_handles.items() if l[-1] == "1"],
        title="RMSE +/- 2 SE difference ($1000s)",
        loc="upper center",
        ncol=len(LOB_COLORS),
        bbox_to_anchor=(0.775, 1.1),
    )
    plt.savefig(PATH + "/scores.png")
    plt.close()


def plot_percentiles(percentiles: Dict[str, np.ndarray]):
    R = len(percentiles)
    C = percentiles[next(iter(percentiles))].shape[1]
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C, sharex=True, sharey="row")

    L = 100
    M = percentiles[next(iter(percentiles))][:, 0, :].size
    bins = int(min(L + 1, max(np.floor(M / 10), 5)))
    lower = stat.binom(M, 1 / bins).ppf(0.005)
    upper = stat.binom(M, 1 / bins).ppf(0.995)
    mean = stat.binom(M, 1 / bins).ppf(0.5)
    nudge = L * 0.1
    uniform = [
        (-nudge, lower),
        (0, mean),
        (-nudge, upper),
        (L + nudge, upper),
        (L, mean),
        (L + nudge, lower),
    ]
    hist_pars = dict(alpha=0.6, bins=bins)
    for i, (r, c) in enumerate(rc):
        lob = list(percentiles)[r]
        p = percentiles[lob][:, c, :] * 100
        ax[r, c].fill(
            *zip(*uniform), color="lightgray", edgecolor="skyblue", lw=2, alpha=0.5
        )
        ax[r, c].hist(
            p.flatten(), color=LOB_COLORS[lob], **hist_pars, label=lob.upper()
        )
        if not c:
            ax[r, c].set_ylabel("Frequency")
            ax[r, c].text(
                -0.125,
                1.1,
                LETTERS[r],
                transform=ax[r, c].transAxes,
                fontsize=20,
                fontweight="bold",
            )
        if r == (R - 1):
            ax[r, c].set_xlabel("Percentile")
        if not r:
            ax[r, c].set_title(MODEL_NAMES[c])

    labels_handles = {
        label: handle
        for ax in fig.axes
        for handle, label in zip(*ax.get_legend_handles_labels())
    }
    fig.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        ncol=len(LOB_COLORS),
        bbox_to_anchor=(0.5, 1.075),
    )
    plt.savefig(PATH + "/calibration.png")
    plt.close()


def plot_zstars(z_stars: Dict[str, np.ndarray]):
    R = len(z_stars)
    C = z_stars[next(iter(z_stars))].shape[1]
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C)
    thresholds_raw = [0] + [i * 0.1 for i in range(1, 10)] + [1]
    thresholds = {round(k, 1): i for i, k in enumerate(thresholds_raw)}
    alphas = ["ff", "cc", "99", "66", "4d"]
    colors_list = [BODY + alpha for alpha in alphas] + [
        TAIL + alpha for alpha in alphas[::-1]
    ]
    cmap = colors.ListedColormap(colors_list)
    norm = plt.Normalize(0, 10)

    images = []
    for i, (r, c) in enumerate(rc):
        lob = list(z_stars)[r]
        p_z = z_stars[lob].mean(axis=0)[c]
        p_z_alpha = [[thresholds[z] for z in zz] for zz in p_z.round(1)]
        ax[r, c].grid(False)
        im = ax[r, c].imshow(p_z_alpha, cmap=cmap, norm=norm, label="p(tail)")
        ax[r, c].set_xticks(range(0, 10, 2), labels=np.arange(1, 11, 2), size=14)
        ax[r, c].set_yticks(range(0, 10, 2), labels=np.arange(1, 11, 2), size=14)

        if not c:
            ax[r, c].set_ylabel("Accident period", size=14)
            ax[r, c].text(-6.5, 5, lob.upper(), size=20)
            ax[r, c].text(
                -0.4,
                1.1,
                LETTERS[r],
                transform=ax[r, c].transAxes,
                fontsize=20,
                fontweight="bold",
            )

        if r == (R - 1):
            ax[r, c].set_xlabel("Development period", size=14)

        if not r:
            ax[r, c].text(0, -2, MODEL_NAMES[c], size=20)

        images.append(im)
        ax[r, c].label_outer()

    cbar = fig.colorbar(images[0], ax=ax, orientation="vertical", fraction=0.1)
    cbar.set_ticks([0, 2.5, 5, 7.5, 10])
    cbar.set_ticklabels(["0", "(body)", "0.5", "(tail)", "1"])
    cbar.set_label("P(tail)", rotation=270)

    plt.savefig(PATH + "/z_stars.png")
    plt.close()


def _flatten_ranks(rank: np.ndarray):
    if isinstance(rank, (list, np.ndarray)):
        return {i: _flatten_ranks(r) for i, r in enumerate(rank)}
    else:
        return rank


def flat_ranks(ranks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    flattened_ranks = {}
    for par, rank in ranks.items():
        if par in ["z", "L"]:
            continue
        flat = _flatten_ranks(rank)
        for _, f in flat.items():
            if isinstance(f, dict):
                for idx, r in f.items():
                    name = f"{par}[{idx}]"
                    if name in flattened_ranks:
                        flattened_ranks[name] = np.append(flattened_ranks[name], r)
                    else:
                        flattened_ranks[name] = np.array([r])
            else:
                if par in flattened_ranks:
                    flattened_ranks[par] = np.append(flattened_ranks[par], f)
                else:
                    flattened_ranks[par] = np.array([f])
    keep = ranks["z"] > 0
    logger.info(f"Dropping {len(ranks['z']) - sum(keep)} ranks due to bad accuracy.")
    return {k: v[keep] for k, v in flattened_ranks.items()}


def plot_ranks(ranks: Dict[str, np.ndarray]) -> None:
    flattened_ranks = flat_ranks(ranks)
    R = 4
    C = 4
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C, sharex=True, sharey=True)

    keys = list(flattened_ranks)
    L = int(ranks["L"])
    M = len(flattened_ranks[keys[0]])
    bins = int(min(L + 1, max(np.floor(M / 10), 5)))
    lower = stat.binom(M, np.floor((L + 1) / bins) / L).ppf(0.005)
    upper = stat.binom(M, np.ceil((L + 1) / bins) / L).ppf(0.995)
    mean = stat.binom(M, 1 / bins).ppf(0.5)
    hist_pars = {"bins": bins, "color": "white", "edgecolor": "black"}
    greeks = ["alpha", "gamma", "omega", "beta", "pi", "tilde"]
    nudge = int(L * 0.1)
    uniform = [
        (-nudge, lower),
        (0, mean),
        (-nudge, upper),
        (L + nudge, upper),
        (L, mean),
        (L + nudge, lower),
    ]
    for i, (r, c) in enumerate(rc):
        if i >= len(flattened_ranks):
            break
        key = keys[i]
        clean_key = key.replace("_", "").replace("[", "_").replace("]", "")
        if r < 3 and c < 3:
            clean_key = re.sub(r"[0-9]", lambda i: str(int(i.group(0)) + 1), clean_key)
        rank = flattened_ranks[key]
        ax[r, c].fill(
            *zip(*uniform), color="lightgray", edgecolor="skyblue", lw=2, alpha=0.5
        )
        ax[r, c].hist(rank, **hist_pars)
        if any(greek in key for greek in greeks):
            ax[r, c].set_title(rf"$\{clean_key}$")
        else:
            ax[r, c].set_title(clean_key)
        if r == (R - 1):
            ax[r, c].set_xlabel("Rank statistics")
        if not c:
            ax[r, c].set_ylabel("Frequency")

    plt.savefig(PATH + "/ranks.png")
    plt.close()


def plot_z_accuracy(z_star: np.ndarray):
    fig, ax = plt.subplots(1, 1)
    z_star = z_star[z_star > 0]
    mu = z_star.mean()
    lower, upper = hdi(z_star, 0.95)
    y_grid = 10
    ax.hist(z_star, bins=40, color="white", edgecolor="black")
    ax.scatter(mu, y_grid, color="black", label="Mean +/- 95% HDI")
    ax.plot([lower, upper], [y_grid] * 2, color="black")
    ax.set_ylabel("Frequency")
    ax.set_xlabel(f"Average $z$ classification accuracy")
    ax.legend()
    plt.savefig(PATH + "/z_star_accuracy.png")
    plt.close()


def plot_literature_results(literature, literature_scores, literature_zstars, literature_taustars):
    R, C = 5, 4
    rc = [(r, c) for r in range(R) for c in range(C)]
    fig, ax = plt.subplots(R, C, figsize=(14, 12))

    keys = list(literature)
    for r, c in rc:
        if not c:
            key = keys[r]
            triangle, scores = list(literature[key].values()), literature_scores[key]
            atas = np.array(
                [
                    [loss / prev for loss, prev in zip(period[1:], period[:-1])]
                    + [-9999] * (len(triangle[0]) - len(period))
                    for period in triangle
                ]
            )
            tau, rho = TAU_RHOS[key]
            masked = np.ma.masked_array(atas, atas == -9999)
            means = masked.mean(axis=0).data
            error = 1.96 * masked.std(axis=0).data
            grid = np.arange(1, masked.shape[1] + 1)
            ax[r, c].axhline(y=1.0, ls=":", color="gray")
            ax[r, c].axvline(tau - 1, color="black", lw=2, label=r"$\tau$")
            ax[r, c].axvline(rho[0] - 1, color="blue", lw=2, label=r"$\rho_{1,2}$")
            ax[r, c].axvline(rho[1] - 1, color="blue", lw=2)
            ax[r, c].errorbar(grid, means, yerr=error, fmt="o", ms=6, color="black")
            ax[r, c].set_title(key.replace(", ", "\n"), fontsize=12)
            ax[r, c].set_ylabel("Link Ratio")
            ax[r, c].legend(loc="upper right")
            if r == (R - 1):
                ax[r, c].set_xlabel(r"Development lag")
            ax[r, c].text(
                -0.125,
                1.1,
                LETTERS[r],
                transform=ax[r, c].transAxes,
                fontsize=20,
                fontweight="bold",
            )
        if c == 1:
            zstars = literature_zstars[key][0]
            period_lengths = [
                int(v)
                for period, values in literature[key].items()
                for v in [period] * len(values)
            ][: len(zstars)]
            zstar_zip = list(zip(period_lengths, zstars))
            tails = [
                sum(not round(z) for i, z in zstar_zip if i == p) + 1
                for p in sorted(set(period_lengths))
            ]
            tails_counts = np.unique(tails, return_counts=True)
            zstar_max = sorted(zip(*tails_counts), key=lambda i: i[1], reverse=True)[0][0]
            taustar = np.unique(literature_taustars[key], return_counts=True)
            taustar_max = sorted(zip(*taustar), key=lambda i: i[1], reverse=True)[0][0]
            ax[r, c].bar(taustar[0], taustar[1] / 10e3, facecolor="black", edgecolor="black")
            ax[r, c].set_xticks(np.arange(2, len(grid) + 1, 3))
            ax[r, c].set_ylabel("Proportion")
            if r == (R - 1):
                ax[r, c].set_xlabel(r"$\tau$")
        if c in (2, 3):
            score_raw = np.array(scores[c - 2]["scores"])
            score = np.sqrt(score_raw) * LITERATURE_SCALER if c == 3 else score_raw
            if c == 2:
                ordered = sorted(
                    [(s, i) for i, s in enumerate(score.sum(axis=2).mean(axis=0))],
                    reverse=True,
                )
                diffs = np.array(
                    [
                        score[:, max(ordered)[1], :] - score[:, m, :]
                        for m in [i for _, i in ordered]
                    ]
                )
                diffs_mu = diffs.sum(axis=2).mean(axis=1)
                diffs_se = (np.sqrt(diffs.var(axis=2, ddof=1) * score.shape[-1])).mean(
                    axis=1
                )
            else:
                ordered = sorted(
                    (s, i) for i, s in enumerate(score.mean(axis=2).mean(axis=0))
                )
                diffs = np.array(
                    [
                        score[:, min(ordered)[1], :] - score[:, m, :]
                        for m in [i for _, i in ordered]
                    ]
                )
                diffs_mu = diffs.mean(axis=2).mean(axis=1)
                diffs_se = (np.sqrt(diffs.var(axis=2, ddof=1) / score.shape[-1])).mean(
                    axis=1
                )
            names = [MODEL_NAMES[i] for _, i in ordered]
            lowers, uppers = diffs_mu - diffs_se * 2, diffs_mu + diffs_se * 2
            marker = "^" if c == 3 else "o"
            errors = [
                (abs(low - mu), abs(mu - high))
                for mu, low, high in zip(diffs_mu, lowers, uppers)
            ]
            for i, (name, mu, error) in enumerate(zip(names, diffs_mu, errors)):
                ax[r, c].errorbar(
                    mu,
                    i,
                    xerr=error[0],
                    color="black",
                    elinewidth=2,
                    fmt=marker,
                    alpha=0.3 if mu == 0.0 else 1,
                )
                if not i:
                    ax[r, c].text(
                        0,
                        -0.5,
                        ordered[0][0].round(1),
                        fontsize=12,
                        ha="center",
                        va="center",
                    )
            ax[r, c].set_yticks(range(score.shape[1]))
            ax[r, c].set_yticklabels(names)
            ax[r, c].axvline(0, ls=":", color="gray")
            ax[r, c].set_ylim(ax[r, c].get_ylim()[::-1])
            if r == (R - 1):
                ax[r, c].set_xlabel("ELPD difference" if c == 2 else "RMSE difference")

    fig.align_ylabels()
    plt.savefig(PATH + "/literature.png")
    plt.close()
