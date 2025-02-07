from typing import List, Dict, Union, Any
import json
import numpy as np
import logging
import matplotlib.pyplot as plt

from plot import (
    plot_atas,
    plot_scores,
    plot_percentiles,
    plot_zstars,
    plot_ranks,
    plot_z_accuracy,
    plot_literature_results,
    plot_numerical,
)
from __init__ import logger

NUMERICAL = {
    "simple": "results/simple_sim.json",
}

BACKTEST = {
    "PP": "data/pp.json",
    "WC": "data/wc.json",
    "CA": "data/ca.json",
    "OO": "data/oo.json",
}

SCORES = {
    "PP": (
        "results/elpd-PP-filter-1e2.json",
        "results/rmse-PP.json",
    ),
    "WC": (
        "results/elpd-WC-filter-1e2.json",
        "results/rmse-WC.json",
    ),
    "CA": (
        "results/elpd-CA-filter-1e2.json",
        "results/rmse-CA.json",
    ),
    "OO": (
        "results/elpd-OO-filter-1e2.json",
        "results/rmse-OO.json",
    ),
}

PERCENTILES = {
    "PP": "results/percentiles-PP.json",
    "WC": "results/percentiles-WC.json",
    "CA": "results/percentiles-CA.json",
    "OO": "results/percentiles-OO.json",
}

ZSTARS = {
    "PP": "results/zstar-PP.json",
    "WC": "results/zstar-WC.json",
    "CA": "results/zstar-CA.json",
    "OO": "results/zstar-OO.json",
}

RANKS = "results/ranks.json"

LITERATURE = {
    "Balona & Richman (2022), long-tailed liability": "data/balona-richman-2022-long-tailed-liability.json",
    "Balona & Richman (2022), short-tailed property": "data/balona-richman-2022-short-tailed-property.json",
    "Gisler (2015)": "data/gisler-2015.json",
    "Merz & Wuthrich (2015)": "data/merz-wuthrich-2015.json",
    "Verrall & Wuthrich (2015)": "data/verrall-wuthrich-2015.json",
}

LITERATURE_SCORES = {
    "Balona & Richman (2022), long-tailed liability": (
        "results/elpd-balona-richman-2022-long-tailed-liability.json",
        "results/rmse-balona-richman-2022-long-tailed-liability.json",
    ),
    "Balona & Richman (2022), short-tailed property": (
        "results/elpd-balona-richman-2022-short-tailed-property.json",
        "results/rmse-balona-richman-2022-short-tailed-property.json",
    ),
    "Gisler (2015)": (
        "results/elpd-gisler-2015-.json",
        "results/rmse-gisler-2015-.json",
    ),
    "Merz & Wuthrich (2015)": (
        "results/elpd-merz-wuthrich-2015-.json",
        "results/rmse-merz-wuthrich-2015-.json",
    ),
    "Verrall & Wuthrich (2015)": (
        "results/elpd-verrall-wuthrich-2015-.json",
        "results/rmse-verrall-wuthrich-2015-.json",
    ),
}

LITERATURE_ZSTARS = {
    "Balona & Richman (2022), long-tailed liability": "results/zstar-balona-richman-2022-long-tailed-liability.json",
    "Balona & Richman (2022), short-tailed property": "results/zstar-balona-richman-2022-short-tailed-property.json",
    "Gisler (2015)": "results/zstar-gisler-2015-.json",
    "Merz & Wuthrich (2015)": "results/zstar-merz-wuthrich-2015-.json",
    "Verrall & Wuthrich (2015)": "results/zstar-verrall-wuthrich-2015-.json",
}

LITERATURE_TAUSTARS = {
    "Balona & Richman (2022), long-tailed liability": "results/taustar-balona-richman-2022-long-tailed-liability.json",
    "Balona & Richman (2022), short-tailed property": "results/taustar-balona-richman-2022-short-tailed-property.json",
    "Gisler (2015)": "results/taustar-gisler-2015-.json",
    "Merz & Wuthrich (2015)": "results/taustar-merz-wuthrich-2015-.json",
    "Verrall & Wuthrich (2015)": "results/taustar-verrall-wuthrich-2015-.json",
}

def load(file: str) -> Dict[str, Any]:
    return json.load(open(file, "r"))


def _dict_ndarray(d: Dict[str, List]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in d.items()}


def main() -> None:
    numerical = {name: _dict_ndarray(load(file)) for name, file in NUMERICAL.items()}
    backtest = {lob: load(file) for lob, file in BACKTEST.items()}
    scores = {
        lob: (_dict_ndarray(load(elpd)), _dict_ndarray(load(rmse)))
        for lob, (elpd, rmse) in SCORES.items()
    }
    percentiles = _dict_ndarray(
        {lob: load(props) for lob, props in PERCENTILES.items()}
    )
    zstars = _dict_ndarray({lob: load(zstar) for lob, zstar in ZSTARS.items()})
    ranks = _dict_ndarray(load(RANKS))
    literature = {name: load(file) for name, file in LITERATURE.items()}
    literature_scores = {
        name: (load(file[0]), load(file[1])) for name, file in LITERATURE_SCORES.items()
    }
    literature_zstars = {name: load(file) for name, file in LITERATURE_ZSTARS.items()}
    literature_taustars = {name: load(file) for name, file in LITERATURE_TAUSTARS.items()}
    plot_numerical(numerical)
    plot_atas(backtest)
    plot_scores(scores)
    plot_percentiles(percentiles)
    plot_zstars(zstars)
    plot_ranks(ranks)
    plot_z_accuracy(ranks["z"])
    plot_literature_results(literature, literature_scores, literature_zstars, literature_taustars)


if __name__ == "__main__":
    logger.info("Generating figures.")
    main()
