import subprocess as sp
import pandas as pd
import toml
from pathlib import Path
import git
import sys
from datetime import datetime
from functools import partial
import importlib

root_dir = git.Repo().working_dir
abetam_dir = Path(root_dir) / "abetam"
copper_dir = Path(root_dir) / "copper"
sys.path.append(abetam_dir.as_posix())
# ruff: noqa: E402
from e_prices import global_adjustment
from abetam.scenarios import (
    generate_scenario_attitudes,
    MODES_2020,
    FAST_TRANSITION_MODES_AND_YEARS,
    SLOW_TRANSITION_MODES_AND_YEARS,
    update_price_w_new_CT,
    CT,
)
from abetam.data.canada.timeseries import demand_projection
from abetam.data.canada import end_use_prices
from abetam.batch import BatchResult
from scenarios import modify_carbon_tax


def spread_model_demand(model_demand_df):
    if len(model_demand_df) > 8500:
        return model_demand_df
    series = []
    for col in model_demand_df.columns:
        province = col.split(".")[0]
        c_series = spread_non_hourly_demand(model_demand_df[col], province)
        c_series.name = col
        series.append(c_series)
    return pd.concat(series, axis=1)


def spread_non_hourly_demand(demand_ts, province):
    if len(demand_ts) > 8500:
        return demand_ts
    demand_shape = pd.read_pickle(
        "abetam/data/canada/timeseries/canada_agg_demand_shapes.pkl"
    )[province]
    demand = demand_shape * demand_ts.sum() / demand_shape.sum()
    return demand


def add_abm_demand_to_projection(model_demand: pd.DataFrame, scenario="BAU_scenario"):
    model_demand = model_demand.reset_index()
    # colnames are integer years represented as string
    model_demand["COPPER_colnames"] = (
        model_demand["province"] + "." + model_demand["year"].astype(int).astype(str)
    )
    model_demand = model_demand[["hour", "Electricity", "COPPER_colnames"]].pivot(
        index="hour", columns=["COPPER_colnames"], values="Electricity"
    )
    model_demand /= 1000  # kWh -> MWh

    # in the implementation of 15.03. the abm works on a weekly basis,
    # this adds back the hourly resolution
    model_demand = spread_model_demand(model_demand).reset_index(drop=True)

    projection_df = demand_projection.query(
        "Scenario=='Global Net-zero' and Variable=='Electricity' and Sector!='Total End-Use'"
    )
    valid_years = (projection_df["Year"] % 5 == 0) & (projection_df["Year"] > 2019) | (
        projection_df["Year"] == 2021
    )
    # print(f"valid_years = {projection_df['Year'][valid_years]}")

    projection_df = projection_df.loc[valid_years, :]
    projection_df.loc[:, "COPPER_colnames"] = (
        projection_df["Region"] + "." + projection_df["Year"].astype(str)
    )
    # print(f"{projection_df.sum()=}")

    def agg_sectors(sector):
        if sector == "Residential":
            return sector
        else:
            return "Comm_Trans_Industr"

    projection_df["agg_sector"] = projection_df["Sector"].apply(agg_sectors)
    annual_sector_PJ_prov_demand = (
        projection_df.groupby(["agg_sector", "COPPER_colnames"])["Value"]
        .sum()
        .reset_index()
    )
    copper_demand = pd.read_csv(
        f"copper/scenarios/{scenario}/demand/demand.csv", index_col=0
    )

    copper_normalized_profiles = copper_demand / copper_demand.sum()
    test = {}
    for i, row in annual_sector_PJ_prov_demand.query(
        "agg_sector!='Residential'"
    ).iterrows():
        province = row["COPPER_colnames"].split(".")[0]
        if province not in copper_normalized_profiles.columns:
            continue
        rest_demand = row["Value"] * copper_normalized_profiles[province]
        test[row["COPPER_colnames"]] = rest_demand

    rest_demand_df = pd.DataFrame(test)
    #                   J->Wh, P -> M
    rest_demand_df *= (1 / 3600) * 1e9

    # determine common columns
    common_cols = sorted(set(rest_demand_df.columns).intersection(model_demand.columns))

    # add demands with common (int) index
    copper_input = rest_demand_df.loc[:, common_cols] + model_demand.reset_index(
        drop=True
    )
    assert copper_input.isna().sum().sum() == 0, AssertionError(
        f"Calculated copper input contains nans:\n{copper_input[copper_input.isna()]}"
    )
    return copper_input


def multi_index_copper_demand(copper_demand):
    """transforms the copper demand dataframe from a wide to a long format

    Args:
        copper_demand (pd.DataFrame): copper demand in wide format where column
    names are `PROVINCE.YEAR`

    Returns:
        copper_midx_demand: copper demand in long format with `hour`, `year`
        and `province` as indices
    """

    copper_demand_bidx = copper_demand.melt(ignore_index=False)
    copper_demand_bidx.loc[:, ["province", "year"]] = (
        copper_demand_bidx["variable"].str.split(".", expand=True).values
    )
    copper_demand_bidx["year"] = copper_demand_bidx["year"].astype(int)
    copper_demand_bidx = (
        copper_demand_bidx.reset_index(names=["hour"])
        .drop("variable", axis=1)
        .set_index(["year", "hour", "province"])
    )
    return copper_demand_bidx


def add_province(df):
    if "province" in df.columns:
        return df
    else:
        df.reset_index(inplace=True)
        df["province"] = df["node"].str.split(".", expand=True)[0]
        return df


def set_batch_params_to_copper_config(batch_parameters, config):
    config["Simulation_Settings"]["ap"] = batch_parameters["province"]
    config["Simulation_Settings"]["aba"] = [
        ba
        for ba in config["Simulation_Settings"]["aba"]
        if any(prov in ba for prov in config["Simulation_Settings"]["ap"])
    ]
    toml.dump(config, open(config_path, "w"))


hp_subsidies = {
    "BAU": 0.0,
    "CER": 0.15,
    "CER_plus": 0.15,
    "Rapid": 0.30,
    "Rapid_plus": 0.30,
}

refurbishment_rate = {
    "BAU": 0.01,
    "CER": 0.02,
    "CER_plus": 0.02,
    "Rapid": 0.03,
    "Rapid_plus": 0.03,
}

carbon_tax_mod = {"BAU": 1, "CER": 1, "CER_plus": 1, "Rapid": 2, "Rapid_plus": 2}

emission_limit = {
    "BAU": False,
    "CER": False,
    "CER_plus": False,
    "Rapid": True,
    "Rapid_plus": True,
}

start_atts = {
    "Electric furnace": 0.283833,
    "Gas furnace": 0.653435,
    "Heat pump": 0.050000,
    "Oil furnace": 0.728319,
    "Biomass furnace": 0.514116,
}

DEFAULT_MODES_AND_YEARS = {
    "Electric furnace": {"end_att": 0.383833, "at_year": 2030},
    "Gas furnace": {"end_att": 0.45, "at_year": 2030},
    "Heat pump": {"end_att": 0.25, "at_year": 2030},
    "Oil furnace": {"end_att": 0.728319, "at_year": 2030},
    "Biomass furnace": {"end_att": 0.514116, "at_year": 2030},
}
PLUS_TRANSITION_MODES_AND_YEARS = {
    "Electric furnace": {"end_att": 0.383833, "at_year": 2030},
    "Gas furnace": {"end_att": 0.45, "at_year": 2030},
    "Heat pump": {"end_att": 0.35, "at_year": 2030},
    "Oil furnace": {"end_att": 0.728319, "at_year": 2030},
    "Biomass furnace": {"end_att": 0.514116, "at_year": 2030},
}


att_modes = {
    "BAU": DEFAULT_MODES_AND_YEARS, #SLOW_TRANSITION_MODES_AND_YEARS,
    "CER": DEFAULT_MODES_AND_YEARS, #SLOW_TRANSITION_MODES_AND_YEARS,
    "CER_plus": PLUS_TRANSITION_MODES_AND_YEARS, #SLOW_TRANSITION_MODES_AND_YEARS,
    "Rapid": DEFAULT_MODES_AND_YEARS, #MODERATE_MODES_AND_YEARS,
    "Rapid_plus": PLUS_TRANSITION_MODES_AND_YEARS, #MODERATE_MODES_AND_YEARS,
}

fossil_ban_years = {
    "BAU": None,
    "CER": None,
    "CER_plus": 2030,
    "Rapid": None,
    "Rapid_plus": 2026,
}

if __name__ == "__main__":
    print("Mode distributions:",att_modes)
    if len(sys.argv) > 1:
        scen_name = sys.argv[1]
    else:
        scen_name = "BAU"  # "BAU", "CER", "Rapid"
    print(f"=== Scenario: {scen_name} ===")
    results_dir = f"./results/{scen_name}_" + datetime.now().strftime(r"%Y%m%d_%H%M")
    # which model to run first?
    scenario = (
        f"{scen_name}_scenario"
        if "Rapid" not in scen_name and scen_name != "CER_plus"
        else "CER_scenario"
    )
    config_path = f"copper/scenarios/{scenario}/config.toml"
    config = toml.load(config_path)

    # SCENARIO parameters for COPPER
    config["Simulation_Settings"]["test"] = False
    config["Simulation_Settings"]["user_specified_demand"] = True

    config["Carbon"]["national_emission_limit"] = emission_limit[scen_name]
    config = modify_carbon_tax(config, carbon_tax_mod[scen_name])

    # SCENARIO parameters for ABETAM

    tech_attitude_scenario = generate_scenario_attitudes(
        MODES_2020, att_modes[scen_name]
    )
    p_mode = 0.65  # result of fit
    province = "Ontario"
    batch_parameters = {
        "N": [500],
        "province": [province],
        "random_seed": list(range(42, 48)),
        "n_segregation_steps": [40],
        "tech_att_mode_table": [tech_attitude_scenario],
        "price_weight_mode": [p_mode],
        "ts_step_length": ["w"],
        "start_year": 2020,
        "refurbishment_rate": refurbishment_rate[scen_name],
        "hp_subsidy": hp_subsidies[scen_name],
        "fossil_ban_year": fossil_ban_years[scen_name],
    }

    if carbon_tax_mod[scen_name] != 1:
        new_CT = CT * carbon_tax_mod[scen_name]
        update_prices = partial(update_price_w_new_CT, new_CT=new_CT)
        end_use_prices["Price (ct/kWh)"] = end_use_prices[
            ["Year", "Price (ct/kWh)", "Type of fuel", "GEO"]
        ].apply(update_prices, axis=1)
        end_use_prices.to_csv(
            "abetam/data/canada/residential_GNZ_end-use-prices-2023_ct_per_kWh.csv",
            index=False,
        )

    # ensure electricity prices are reset before execution
    el_price_path = "abetam/data/canada/ca_electricity_prices.csv"
    el_prices_df = pd.read_csv(el_price_path).set_index("REF_DATE")
    el_prices_df = el_prices_df.loc[:2022, :]

    for i in range(2):
        if i:
            # remove projected prices after first iteration, to only use the COPPER-determined prices
            prices = pd.read_csv(
                "abetam/data/canada/residential_GNZ_end-use-prices-2023_ct_per_kWh.csv"
            )
            drop_rows = (
                (prices["GEO"] == province)
                & (prices["Year"] > 2020)
                & (prices["Type of fuel"] == "Electricity")
            )
            prices.loc[~drop_rows, :].to_csv(
                "abetam/data/canada/residential_GNZ_end-use-prices-2023_ct_per_kWh.csv",
                index=False,
            )
        # reload data module, to ensure price updates are reloaded
        importlib.import_module("data.canada")

        print(f"Iteration {i}, running ABM...")
        batch_result = BatchResult.from_parameters(
            batch_parameters, max_steps=(2050 - 2020) * 4, force_rerun=True
        )
        batch_result.save(custom_path=results_dir + f"_{i}")

        # retrieve abm time series results...
        demand_df = batch_result.mean_carrier_demand_df
        copper_demand = add_abm_demand_to_projection(demand_df, scenario=scenario)

        # ... and set them as copper demands
        copper_demand.to_csv(
            f"copper/scenarios/{scenario}/demand/user_input_demand.csv"
        )

        # adapt copper config to the current run
        set_batch_params_to_copper_config(batch_parameters, config)

        # run copper model for the selected provinces
        print(f"Iteration {i}, running COPPER with demands:\n{copper_demand.sum()}")
        result = sp.run(
            ["python", "COPPER.py", "-s", scenario, "-sl", "gurobi", "-o", results_dir],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            cwd=copper_dir.as_posix(),
        )
        stdout = result.stdout.decode("utf-8")
        stderr = result.stderr.decode("utf-8")

        print(stdout[:50])

        if "Traceback" in stderr or "Error" in stderr:
            print(f"{stderr=}")
            print(f"{stdout=}")
            raise ChildProcessError(
                "An error has occured during the execution of COPPER."
            )
        # retrieve copper results...
        last_run_dir = sorted(
            Path("copper/results").glob("*/LastRun"),
            key=lambda p: p.lstat().st_ctime,
            reverse=True,
        )[0]
        prices = (
            pd.read_csv(f"{last_run_dir}/annual_avg_prices.csv", index_col=0)
            .rename({"pds": "year"}, axis=1)
            .set_index(["year", "province"])
        )
        mean_electricity_prices = prices["price(CAD/MWh)"] / 10  # $/MWh -> ct/kWh
        mean_electricity_prices.name = "ct/kWh"

        ga = global_adjustment(mean_electricity_prices)
        effective_el_prices = mean_electricity_prices + ga
        el_price_copper = effective_el_prices.reset_index().pivot(
            index="year", columns="province", values="ct/kWh"
        )

        ga.name = "GA"
        mean_electricity_prices.name = "HOEP"
        effective_el_prices.name = "final price"
        price_info = pd.concat(
            [mean_electricity_prices, ga, effective_el_prices], axis=1
        )
        price_info["scenario"] = scen_name
        price_info["iteration"] = i
        print(price_info)

        # ... and merge them with the abm inputs
        for year in el_price_copper.index:
            for prov in el_price_copper.columns:
                el_prices_df.at[year, prov] = el_price_copper.loc[year, prov]
        el_prices_df.to_csv(el_price_path)
    pass
