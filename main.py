import subprocess as sp
import pandas as pd
import toml
from pathlib import Path
import os
import git
import sys

root_dir = git.Repo().working_dir
abetam_dir = Path(root_dir) / "abetam"
copper_dir = Path(root_dir) / "copper"
sys.path.append(abetam_dir.as_posix())

from abetam.data.canada import Provinces
from abetam.data.canada.timeseries import demand_projection
from abetam.batch import BatchResult
from abetam.components.probability import beta_with_mode_at


# best peaks for tech attitude distribution
best_tech_modes = {
    "Electric furnace": 0.788965,
    "Gas furnace": 0.388231,
    "Heat pump": 0.346424,
    "Oil furnace": 0.460226,
    "Wood or wood pellets furnace": 0.516056,
}


def add_abm_demand_to_projection(model_demand: pd.DataFrame):
    model_demand = model_demand.reset_index()
    # colnames are integer years represented as string
    model_demand["COPPER_colnames"] = (
        model_demand["province"] + "." + model_demand["year"].astype(int).astype(str)
    )
    model_demand = model_demand[["hour", "Electricity", "COPPER_colnames"]].pivot(
        index="hour", columns=["COPPER_colnames"], values="Electricity"
    )
    model_demand /= 1000  # kWh -> MWh

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
        "copper/scenarios/BAU_scenario/demand/demand.csv", index_col=0
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

    # print(f"{model_demand=}")
    common_cols = set(rest_demand_df.columns).intersection(model_demand.columns)
    # set(rest_demand_df.columns).difference(model_demand.columns)
    copper_input = rest_demand_df.loc[:, list(common_cols)] + model_demand
    # print(f"{copper_input=}")
    return copper_input


def get_copper_duals(copper_result_dir="copper/results/LastRun"):
    duals = pd.read_csv(f"{copper_result_dir}/duals.csv").set_index(
        ["year", "hour", "node"]
    )
    duals = add_province(duals)
    return duals


def get_copper_el_prices(config, copper_result_dir="copper/results/LastRun"):
    duals = get_copper_duals(copper_result_dir)

    run_day_key = "run_days"
    is_test_run = config["Simulation_Settings"]["test"]
    if is_test_run:
        run_day_key += "_test"
    n_days = len(config["Simulation_Settings"][run_day_key])
    print(
        f"n_days: {n_days}, using {run_day_key}: {config['Simulation_Settings'][run_day_key]}"
    )
    prices = duals.set_index(["year","hour","province"])[["dual_price"]]* -n_days / 365

    prices /= 10  # $/MWh -> c/kWh
    return prices


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


if __name__ == "__main__":
    # which model to run first?
    batch_parameters = {
        "N": [79],
        "province": ["Ontario",],
        "random_seed": list(range(42, 48)),
        "tech_attitude_dist_func": beta_with_mode_at,
        "tech_attitude_dist_params": [best_tech_modes],
        "start_year": 2020,
        "price_weight_mode": 0.875,
    }

    config_path = "copper/scenarios/BAU_scenario/config.toml"
    config = toml.load(config_path)

    for i in range(2):
        print(f"Iteration {i}, running ABM...")
        batch_result = BatchResult.from_parameters(
            batch_parameters, max_steps=(2050 - 2020) * 4, force_rerun=True
        )
        batch_result.save()

        # retrieve abm time series results...
        demand_df = batch_result.mean_carrier_demand_df
        copper_demand = add_abm_demand_to_projection(demand_df)

        # ... and set them as copper demands
        copper_demand.to_csv(
            "copper/scenarios/BAU_scenario/demand/user_input_demand.csv"
        )

        # adapt copper config to the current run
        set_batch_params_to_copper_config(batch_parameters, config)

        # run copper model for the selected provinces
        print(f"Iteration {i}, running COPPER...")
        result = sp.run(
            ["python", "COPPER.py"],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            cwd=copper_dir.as_posix(),
        )
        stdout = result.stdout.decode("utf-8")
        stderr = result.stderr.decode("utf-8")
        # print(stdout, stderr)
        if "Traceback" in stderr or "Error" in stderr:
            print(stderr)
            raise ChildProcessError("An error has occured during the execution of COPPER.")
        # retrieve copper results...
        electricity_prices = get_copper_el_prices(config)
        mean_electricity_prices = electricity_prices.groupby(["year", "province"]).mean()

        el_price_copper = (
            mean_electricity_prices
            .reset_index()
            .pivot(index="year", columns="province", values="dual_price")
        )
        print(f"Copper electricity prices:\n{el_price_copper}")
        # ... and merge them with the abm inputs
        el_price_path = "abetam/data/canada/ca_electricity_prices.csv"
        el_prices_df = pd.read_csv(el_price_path).set_index("REF_DATE")

        for year in el_price_copper.index:
            for prov in el_price_copper.columns:
                el_prices_df.at[year, prov] = el_price_copper.loc[year, prov]
        el_prices_df.to_csv(el_price_path)
    pass
