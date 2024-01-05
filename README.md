# Coupled technology adaption and energy system optimisation model

## Quickstart

To get the coupled model running, follow these steps:

* `mamba env create -f env.yml`
* `conda activate cando`
* `python main.py`

Note, that this repository relies on `marginals` in COPPER, which are only available if the respective config contains the `hydro_development=false` entry.