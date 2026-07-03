import logging

import pandas as pd
import yaml
import os

logger = logging.getLogger(__name__)

matching_id_EIC = pd.read_csv("inputs/generation_data/Matching_idBNA_EIC.csv")

entsoe_gen_types = pd.read_csv(
    "inputs/generation_data/gen_types_and_emission_factors.csv"
)
entsoe_gen_types = entsoe_gen_types["entsoe"].unique()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BUSES = config["states"].values()


def prepare_gen_per_unit(per_unit_data):
    gen_per_unit = pd.DataFrame()

    temp_df = pd.DataFrame(columns=["technology", "EIC", "Generation [MW]"])

    for keys, values in per_unit_data.items():
        temp = temp_df.copy()
        keys = dict(keys[1])
        eic = keys["EIC"]
        technology = keys["technology"]

        temp["Generation [MW]"] = values["Generation [MW]"]
        temp.loc[:, "technology"] = technology
        temp.loc[:, "EIC"] = eic
        gen_per_unit = pd.concat([gen_per_unit, temp], axis=0)

    gen_per_unit = gen_per_unit.sort_index()
    gen_per_unit.index.name = "DateTime"
    gen_per_unit = gen_per_unit.fillna(0)

    return gen_per_unit


def preprocess_gen_per_unit(per_unit_data, start, end):
    global matching_id_EIC

    gen_per_unit_df = pd.DataFrame()
    gen_per_unit_info = {}
    # bna_used = []

    # a dataframe where columns are technology and index is states filled with 0
    capacities_used_in_per_unit = pd.DataFrame(
        index=BUSES,
        columns=entsoe_gen_types,
        data=0.0,
    )

    index = pd.date_range(start=start, end=end, freq="H")

    for keys, values in per_unit_data.items():
        keys = dict(keys[1])
        eic = keys["EIC"]
        technology = keys["technology"]
        state = matching_id_EIC.loc[matching_id_EIC["eic_code_block"] == eic, "state"]

        if len(state) == 0:
            logger.warning(f"Missing EIC: {eic}! Please update the matching_id_EIC.csv")

            # Ensure the file exists before opening it
            missing_eic_path = "inputs/generation_data/missing_eic.txt"
            if not os.path.exists(missing_eic_path):
                with open(missing_eic_path, "w") as f:
                    pass  # Create the file if it doesn't exist

            # Open missing_eic.txt and append EIC if not already in there
            with open(missing_eic_path, "r+") as f:
                if eic not in f.read():
                    f.write(f"{eic}\n")

            continue

        state = state.iloc[0]

        if state == "missing":
            continue

        temp = pd.DataFrame(columns=[eic])
        temp[eic] = values["Generation [MW]"]
        if len(temp) == 1:
            continue

        if len(temp) == len(index):
            capacity = matching_id_EIC.loc[
                matching_id_EIC["eic_code_block"] == eic, "capacity"
            ].values[0]
            capacities_used_in_per_unit.loc[state, technology] += capacity

        gen_per_unit_df = pd.concat([gen_per_unit_df, temp], axis=1)
        gen_per_unit_info[eic] = {"bus": state, "carrier": technology}

    # sum columns with same name in gen_per_unit_df
    gen_per_unit_df = gen_per_unit_df.groupby(gen_per_unit_df.columns, axis=1).sum()
    gen_per_unit_df = gen_per_unit_df.fillna(0)

    gen_per_unit_info = pd.DataFrame(gen_per_unit_info).T
    gen_per_unit_dict = {"gen": gen_per_unit_df, "info": gen_per_unit_info}

    # remove all columns with only 0 from capacities_used_in_per_unit
    capacities_used_in_per_unit = capacities_used_in_per_unit.loc[
        :, (capacities_used_in_per_unit != 0).any(axis=0)
    ]

    return gen_per_unit_dict, capacities_used_in_per_unit
