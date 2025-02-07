import cmdstanpy as csp
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not os.path.exists(".cmdstan"):
    raise ValueError(
        "CmdStan installation not found in .cmdstan. Set the path in `__init__.py` manually."
    )
csp.set_cmdstan_path(".cmdstan/cmdstan-2.36.0")
