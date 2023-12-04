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
    model_demand["COPPER_colnames"] = (
        model_demand["province"] + "." + model_demand["year"].astype(int).astype(str)
    )
    model_demand = model_demand[["hour", "Electricity", "COPPER_colnames"]].pivot(
        index="hour", columns=["COPPER_colnames"], values="Electricity"
    )
    model_demand /= 1000  # kWh -> MWh

    projection_df = pd.read_csv(
        "data/canada/timeseries/end-use-demand-2023.csv", index_col=0
    ).query(
        "Scenario=='Global Net-zero' and Variable=='Electricity'  and Sector!='Total End-Use'"
    )
    valid_years = (projection_df["Year"] % 5 == 0) & (projection_df["Year"] > 2019) | (
        projection_df["Year"] == 2021
    )

    projection_df = projection_df.loc[valid_years, :]
    projection_df.loc[:, "COPPER_colnames"] = (
        projection_df["Region"] + "." + projection_df["Year"].astype("str")
    )

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

    common_cols = set(rest_demand_df.columns).intersection(model_demand.columns)
    # set(rest_demand_df.columns).difference(model_demand.columns)
    copper_input = rest_demand_df.loc[:, list(common_cols)] + model_demand
    return copper_input




# which model to run first?

if __name__ == "__main__":
    
    os.chdir(abetam_dir)
    #
    batch_parameters = {
        "N": [150],
        "province": [Provinces.ON],
        "random_seed": list(range(42, 43)),
        "tech_attitude_dist_func": [beta_with_mode_at],
        "tech_attitude_dist_params": [best_tech_modes],
        "start_year": 2020,
        "price_weight_mode": 0.875,
    }

    batch_result = BatchResult.from_parameters(
        batch_parameters, max_steps=(2050 - 2020) * 4
    )

    # retrieve abm time series results...
    demand_df = batch_result.mean_carrier_demand
    copper_demand = add_abm_demand_to_projection(demand_df)

    # ... and set them as copper demands
    copper_demand.to_csv("copper/scenarios/BAU_scenario/demand/user_input_demand.csv")

    # adapt copper config to the current run
    config_path = "copper/scenarios/BAU_scenario/config.toml"
    config = toml.load(config_path)
    config["Simulation_Settings"]["ap"] = batch_parameters["province"]
    config["Simulation_Settings"]["aba"] = [
        ba
        for ba in config["Simulation_Settings"]["aba"]
        if any(prov in ba for prov in config["Simulation_Settings"]["ap"])
    ]
    toml.dump(config, open(config_path, "w"))

    os.chdir(copper_dir)
    # run copper model for the selected provinces
    result = sp.run(
        ["python", "COPPER.py"], stdout=sp.PIPE, stderr=sp.PIPE, cwd="copper"
    )

    # TODO retrieve copper results and put them in the ABM folder

    pass
