# Coupled technology adoption and energy system optimisation model

The goal of this project is to couple an [agent-based technology adoption model](https://github.com/david-huck/abetam) with an energy system model [COPPER](https://gitlab.com/McPherson/copper.git) to investigate feedback dynamics between these models.

## Quickstart

To get the coupled model running, open a terminal, navigate to a destination where you want to store the project, then follow these steps:

* `git clone --recursive https://github.com/david-huck/canadopt`
* `mamba env create -f env.yml`
* `conda activate cando`
* `python main.py`

Note, that this project relies on `marginals` i.e. dual values in COPPER, which are only available if `hydro_development=false` in config entry. This is necessary, because the presence of binary/integer variables inhibits the export of duals.