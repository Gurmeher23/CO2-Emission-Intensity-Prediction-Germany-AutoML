# logging level correctly
import logging

import pandas as pd
import yaml

from cosema.per_unit_scripts import prepare_gen_per_unit

logger = logging.getLogger(__name__)

gen_types_df = pd.read_csv("inputs/generation_data/gen_types_and_emission_factors.csv")
ALLOW_NEGATIVE_GENERATION = list(
    gen_types_df[gen_types_df["is_storage"]]["entsoe"].unique()
)
GEN_TYPES = gen_types_df["entsoe"].unique()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BUSES = config["states"].values()


def query_per_type_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_client,
    countries: list = None,
):
    gen_per_type = {}

    if countries is None:
        countries = config["countries"]

    for country in countries:
        try:
            country_gen = db_client.query_per_type_gen(
                start=start, end=end, technologies=GEN_TYPES, country=country
            )
        except Exception:
            logger.warning(
                f"No data for {country} for period {start} - {end}. Returning 0.0..."
            )
            country_gen = pd.DataFrame(
                index=pd.date_range(start, end, freq="1h"),
                columns=["Other"],
                data=0.0,
            )

        gen_per_type[country] = country_gen

    gen_per_type = pd.concat(gen_per_type, axis=1)

    return gen_per_type


def query_reg_per_type_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_client,
    mode="with_per_unit" or "only_per_type",
    column_name="Generation [MW]",
    balanced=False,
):
    reg_gen_per_type = {}

    for state in BUSES:
        try:
            state_gen = db_client.query_reg_gen_data(
                start=start,
                end=end,
                technologies=GEN_TYPES,
                state=state,
                mode=mode,
                column_name=column_name,
                balanced=balanced,
            )
            # fill nan values with 0
            state_gen = state_gen.fillna(0)
            # drop columns with all 0 values
            state_gen = state_gen.loc[:, (state_gen != 0).any(axis=0)]
        except Exception:
            logger.warning(f"No data for {state} for period {start} - {end}.")
            continue

        reg_gen_per_type[f"DE_{state}"] = state_gen

    reg_gen_per_type = pd.concat(reg_gen_per_type, axis=1)

    return reg_gen_per_type


def calc_leftover_gen_per_type(raw_per_unit_data, gen_per_type):
    gen_per_unit = prepare_gen_per_unit(raw_per_unit_data)

    gen_per_unit_grouped = gen_per_unit.groupby(
        ["DateTime", "technology"], as_index=True
    ).sum(numeric_only=True)

    gen_per_unit_grouped = gen_per_unit_grouped.reset_index()
    gen_per_unit_grouped = gen_per_unit_grouped.set_index("DateTime", drop=True)
    gen_per_unit_grouped = gen_per_unit_grouped.fillna(0)

    gen_types_per_unit = gen_per_unit_grouped["technology"].unique()

    leftover_gen_per_type = gen_per_type.copy()

    for gen_type, generation in leftover_gen_per_type.items():
        if gen_type in gen_types_per_unit:
            per_unit = gen_per_unit_grouped.loc[
                gen_per_unit_grouped["technology"] == gen_type, "Generation [MW]"
            ]

            if len(per_unit) != len(generation):
                # add missing hours to per unit data and fill with 0.0
                missing_hours = generation.index.difference(per_unit.index)
                per_unit = per_unit.reindex(generation.index)
                per_unit[missing_hours] = 0.0

            dif = generation - per_unit

            if gen_type not in ALLOW_NEGATIVE_GENERATION:
                dif[dif < 0] = 0.0

            leftover_gen_per_type[gen_type] = dif

    return leftover_gen_per_type


def collect_vre(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_client,
    mode: str = "historical" or "forecast",
    value: str = "Generation [MW]" or "reg_factor",
):
    index = pd.date_range(start=start, end=end, freq="1h")
    solar = pd.DataFrame(index=index, columns=BUSES, data=0.0)
    wind_onshore = pd.DataFrame(index=index, columns=BUSES, data=0.0)
    wind_offshore = pd.DataFrame(index=index, columns=BUSES, data=0.0)

    for state in BUSES:
        temp_df = db_client.query_vre(
            start=start,
            end=end,
            technology="solar",
            state=state,
            value=value,
            mode=mode,
        )
        solar[state] = temp_df.values

        temp_df = db_client.query_vre(
            start=start,
            end=end,
            technology="wind_onshore",
            state=state,
            value=value,
            mode=mode,
        )
        wind_onshore[state] = temp_df.values

        if state not in ["MV", "NI", "SH"]:
            continue

        temp_df = db_client.query_vre(
            start=start,
            end=end,
            technology="wind_offshore",
            state=state,
            value=value,
            mode=mode,
        )
        wind_offshore[state] = temp_df.values

    return solar, wind_onshore, wind_offshore


def calc_reg_gen_by_type(
    gen_per_type, reg_cap_factors, solar_cf, onshore_cf, offshore_cf
):
    index = gen_per_type.index
    reg_gen_by_type = pd.DataFrame()

    # create  list with all original energy types from gen_per_type.columns
    # and conv_cap_factors.index
    energy_types = gen_per_type.columns.tolist()
    energy_types.extend(reg_cap_factors.columns.tolist())
    energy_types = list(set(energy_types))

    for energy_type in energy_types:
        if energy_type in gen_per_type.columns:
            temp = gen_per_type[energy_type]

            if energy_type in reg_cap_factors.columns and energy_type not in [
                "Solar",
                "Wind Onshore",
                "Wind Offshore",
            ]:
                cap_factor = pd.DataFrame(
                    index=index, columns=reg_cap_factors.index, data=0.0
                )
                cap_factor += reg_cap_factors[energy_type]
                temp = cap_factor.mul(temp, axis=0)

            elif energy_type == "Solar":
                temp = solar_cf.mul(temp, axis=0)

            elif energy_type == "Wind Onshore":
                temp = onshore_cf.mul(temp, axis=0)

            elif energy_type == "Wind Offshore":
                temp = offshore_cf.mul(temp, axis=0)

            else:
                temp = pd.DataFrame(index=index, columns=BUSES, data=0.0)

        else:
            temp = pd.DataFrame(index=index, columns=BUSES, data=0.0)

        temp = temp.add_suffix(f"_{energy_type}")

        reg_gen_by_type = pd.concat([reg_gen_by_type, temp], axis=1)

    return reg_gen_by_type


def collect_intensities(start: pd.Timestamp, end: pd.Timestamp, db_client):
    index = pd.date_range(start=start, end=end, freq="1h")
    cons_intensity = pd.DataFrame(index=index, columns=BUSES, data=0.0)
    gen_intensity = pd.DataFrame(index=index, columns=BUSES, data=0.0)

    for state in BUSES:
        temp_df = db_client.query_intensities(
            start=start,
            end=end,
            state=state,
            emission_type="Consumption",
            mode="with_per_unit",
        )
        cons_intensity[state] = temp_df.values

        temp_df = db_client.query_intensities(
            start=start,
            end=end,
            state=state,
            emission_type="Generation",
            mode="with_per_unit",
        )
        gen_intensity[state] = temp_df.values

    return cons_intensity, gen_intensity
