Joint estimation of insurance loss development factors using Bayesian hidden Markov models
--------------------------------------------------------------------------------------------

This repository holds code and the workflow to build the paper:

    Goold, C. 2025. (preprint). *Joint estimation of insurance 
    loss development factors using Bayesian hidden Markov models*

A preprint is available on Arxiv `here <https://arxiv.org/abs/2406.19903>`_.

The code and data are available in the :code:`code/` and
:code:`code/data/` directories, respectively.

To run the code,
you will first need to install the requirements,
ideally within a virtual environment. Using Python 3.10:

.. code-block:: bash

   python3.10 -m venv .env
   source .env/bin/activate
   python3.10 -m pip install -r requirements.txt

You will also need a working version of CmdStan,
which can be installed after the above requiremnets using
(on Linux/MacOSX):

.. code-block:: bash

   mkdir -p code/.cmdstan
   install_cmdstan -d code/.cmdstan -v 2.36.0

See `CmdStanPy's installation page <https://mc-stan.org/cmdstanpy/installation.html#cmdstan-installation>`_
for more information.

Each of the Python files in the :code:`code/` directory can
be run as standalone modules, using:

.. code-block:: bash

   python3.10 [file].py

replacing :code:`[file]` with a particular filename.
Results will automatically be saved out in the
:code:`code/results/` directory.
