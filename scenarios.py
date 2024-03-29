import numpy as np
import pandas as pd
import toml


def price_reduction(x, lr, p0=1):
    b = -np.log((1 - lr / 100)) / np.log(2)
    return p0 * x**-b


def lr_based_cost_factors(x, lr, p0=1):
    # sets first (2020) data point to 1
    price_factors = price_reduction(x, lr, p0)
    return price_factors / list(price_factors)[0]


def iea_built_cap_projection():
    # in GW, according to IEA
    annual_builds = {
        "solar": [134, 630, 630],
        "wind": [114, 390, 350],
    }  # 2020, 2030, 2050
    all_years = range(2019, 2051)
    ann_build_df = pd.DataFrame(annual_builds, index=[2021, 2030, 2050])
    missing_years = set(all_years).difference(ann_build_df.index)
    empty_frame = pd.DataFrame(index=list(missing_years), columns=ann_build_df.columns)
    ann_build_df = pd.concat([ann_build_df, empty_frame]).sort_index()
    ann_build_df = ann_build_df.astype(float).interpolate()
    # https://iea-pvps.org/snapshot-reports/snapshot-2020/
    # https://www.iea.org/energy-system/renewables/wind section technology deployment
    ann_build_df.loc[2020, ["solar", "wind"]] = [627, 900]
    total_build_df = ann_build_df.cumsum()
    total_build_df = total_build_df.dropna()
    return total_build_df


PV_CAPACITY_PROJECTION = iea_built_cap_projection()["solar"]
WIND_CAPACITY_PROJECTION = iea_built_cap_projection()["wind"]

# results of fitting lr_based_cost_factors to technology_evolution.csv
COPPER_LR_PV = 18.268
COPPER_LR_WIND = 20.395

FAST_TRANSITION_LR_PV = COPPER_LR_PV + 5
FAST_TRANSITION_LR_WIND = COPPER_LR_WIND + 5
SLOW_TRANSITION_LR_PV = COPPER_LR_PV - 5
SLOW_TRANSITION_LR_WIND = COPPER_LR_WIND - 5


def update_tech_evo(
    scenario, pv_lr=COPPER_LR_PV, wind_lr=COPPER_LR_WIND, write_csv=False
):
    evol_fn = f"copper/scenarios/{scenario}/technology_evolution.csv"
    tech_evo = pd.read_csv(evol_fn, index_col=0)
    pds = [int(col) for col in tech_evo.columns]
    tech_evo.loc["solar.base", :] = (
        lr_based_cost_factors(PV_CAPACITY_PROJECTION, pv_lr).loc[pds].values
    )
    tech_evo.loc["wind_ons.base", :] = (
        lr_based_cost_factors(WIND_CAPACITY_PROJECTION, wind_lr).loc[pds].values
    )
    if write_csv:
        tech_evo.to_csv(evol_fn)
    return tech_evo


def modify_carbon_tax(config, flat_factor):
    carbon_tax = pd.Series(config["Carbon"]["ctax"])
    modified_carbon_tax = (carbon_tax * flat_factor).to_dict()
    config["Carbon"]["ctax"] = modified_carbon_tax
    return config

