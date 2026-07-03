import logging

import yaml

from cosema.demand_scripts import reg_demand_data_dynamic
from cosema.per_type_scripts import (
    calc_leftover_gen_per_type,
    calc_reg_gen_by_type,
    collect_vre,
    query_per_type_data,
)
from cosema.per_unit_scripts import preprocess_gen_per_unit

logger = logging.getLogger(__name__)

import traceback
import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BUSES = config["states"].values()

gen_types = pd.read_csv("inputs/generation_data/gen_types_and_emission_factors.csv")
gen_types = gen_types["entsoe"].unique()


def get_per_type_data(
    db_client,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    logger.debug("Quering per type generation data for DE...")
    try:
        gen_per_type = query_per_type_data(
            start=start,
            end=end,
            db_client=db_client,
            countries=["DE"],
        )

        # remove upper level column
        gen_per_type.columns = gen_per_type.columns.droplevel(0)

    except Exception as e:
        error = traceback.format_exc().splitlines()[-1]
        logger.error(
            f"Critical error {error}! No per type data for DE for period {start} - {end}. Using blank data..."
        )
        index = pd.date_range(start=start, end=end, freq="1h", tz=start.tz)
        gen_per_type = pd.DataFrame(index=index, columns=gen_types, data=0.0)

    return gen_per_type


def get_vre_data(
    db_client,
    start: pd.Timestamp,
    end: pd.Timestamp,
    mode: str = "historical" or "forecast",
):
    logger.debug("Quering VRE capacity factors...")
    try:
        solar_cf, onshore_cf, offshore_cf = collect_vre(
            start=start,
            end=end,
            db_client=db_client,
            mode=mode,
            value="reg_factor",
        )

    except Exception as e:
        error = traceback.format_exc().splitlines()[-1]
        logger.error(
            f"Critical error {error}! No VRE data for period {start} - {end}. Using blank data..."
        )
        index = pd.date_range(start=start, end=end, freq="1h", tz=start.tz)
        solar_cf = pd.DataFrame(index=index, columns=BUSES, data=0.0)
        onshore_cf = pd.DataFrame(index=index, columns=BUSES, data=0.0)
        offshore_cf = pd.DataFrame(index=index, columns=BUSES, data=0.0)

    return solar_cf, onshore_cf, offshore_cf


def get_demand_data(
    db_client,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    # Prepare demand data
    logger.debug("Preparing demand data...")

    try:
        demand_DE = db_client.query_demand_data(
            start=start,
            end=end,
            country="DE",
        )

    except Exception as e:
        error = traceback.format_exc().splitlines()[-1]
        logger.error(
            f"Critical error {error}! No demand data for period {start} - {end}. Using blank data..."
        )
        index = pd.date_range(start=start, end=end, freq="1h", tz=start.tz)
        demand_DE = pd.DataFrame(index=index, columns=BUSES, data=0.0)

    return demand_DE


def calculate_regionalized_gen_and_demand(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_client,
    mode: str = "only_per_type" or "with_per_unit",
    extra_tags: dict = None,
):
    if mode == "with_per_unit":
        available_gen_per_unit = db_client.check_per_unit_data(start=start, end=end)
        new_end = available_gen_per_unit[-1]
        if end != new_end:
            logger.warning(
                f"Per unit data is not available until {end}, using data until {new_end}"
            )
        end = new_end

    gen_per_unit_dict = None

    # Prepare power plants list and generation per unit
    logger.debug("Reading power plants list...")

    gen_per_type = get_per_type_data(db_client=db_client, start=start, end=end)

    # Prepare generation per unit data
    if mode == "with_per_unit":
        logger.debug("Preparing generation per unit data...")
        raw_per_unit_data = db_client.query_per_unit_gen(start=start, end=end)

        gen_per_unit_dict, capacities_used_in_per_unit = preprocess_gen_per_unit(
            raw_per_unit_data, start, end
        )

        gen_per_type = calc_leftover_gen_per_type(
            raw_per_unit_data=raw_per_unit_data, gen_per_type=gen_per_type
        )

    vre_mode = "historical" if mode == "with_per_unit" else "forecast"
    solar_cf, onshore_cf, offshore_cf = get_vre_data(
        db_client=db_client,
        start=start,
        end=end,
        mode=vre_mode,
    )

    logger.debug("Calculating regional capacity factors...")

    try:
        regional_capacities = pd.read_parquet(
            f"inputs/capacities/{start.strftime('%Y_%m')}/conv_capacities_{start.strftime('%Y_%m')}.parquet"
        )
    except FileNotFoundError:
        # if capacities are not available, use one month before
        month_before = start - pd.DateOffset(months=1)
        regional_capacities = pd.read_parquet(
            f"inputs/capacities/{month_before.strftime('%Y_%m')}/conv_capacities_{month_before.strftime('%Y_%m')}.parquet"
        )
        logger.warning(
            f"Capacities for {start.strftime('%Y_%m')} are not available, using capacities from {month_before.strftime('%Y_%m')}"
        )

    # remove columns from regional_capacities which are not in gen_types
    columns_to_remove = list(set(regional_capacities.columns) - set(gen_types))
    regional_capacities = regional_capacities.drop(columns=columns_to_remove)

    if mode == "with_per_unit":
        # substract capacities used in per unit from regional capacities
        regional_capacities = regional_capacities.sub(
            capacities_used_in_per_unit
        ).fillna(regional_capacities)
        # remove negative values
        regional_capacities[regional_capacities < 0] = 0

    # calculate regional capacity factors by normalizing per row
    reg_cap_factors = regional_capacities.div(regional_capacities.sum(axis=0), axis=1)
    # remove nan values
    reg_cap_factors = reg_cap_factors.fillna(0)

    logger.debug("Calculating regional per type generation...")
    reg_gen_by_type = calc_reg_gen_by_type(
        gen_per_type=gen_per_type,
        reg_cap_factors=reg_cap_factors,
        solar_cf=solar_cf,
        onshore_cf=onshore_cf,
        offshore_cf=offshore_cf,
    )

    demand_DE = get_demand_data(
        db_client=db_client,
        start=start,
        end=end,
    )
    reg_demand_DE = reg_demand_data_dynamic(demand_DE)

    # iterate over gen_per_unit data and assign to reg_gen_by_type by state and technology
    if mode == "with_per_unit":
        for _, row in gen_per_unit_dict["info"].iterrows():
            state = row["bus"]
            carrier = row["carrier"]
            reg_gen_by_type[f"{state}_{carrier}"] += gen_per_unit_dict["gen"][row.name]

    # check if total regionalized generation per technology is equal the initial total generation per technology
    # if not, raise a warning
    # NOTES:
    # 1: Hydro Water reservoir has a mismatch due to multiple nehative values in the initial data which are
    # set to 0 during processing
    # 2: Nuclear doesn't typically match because per unit data if not equal to the per type data
    # and all nuclear is in per_unit, which we trust more
    # 3. Hard coal has a mitmatch because per unit data is usually larger than per type data and we trust
    # per unit data more
    # 4. Fossil gas has a mismatch because coal-derived gas is not included in per type data
    for gen_type in gen_types:
        if gen_type in gen_per_type.columns:
            total_reg_gen = reg_gen_by_type.filter(regex=f"{gen_type}$").sum(axis=1)
            difference = abs(total_reg_gen - gen_per_type[gen_type])
            if (difference > 100).any():
                logger.warning(
                    f"Total regionalized generation for {gen_type} is not equal to initial total generation!"
                )

    # fill nan values with 0
    reg_gen_by_type = reg_gen_by_type.fillna(0)
    reg_demand_DE = reg_demand_DE.fillna(0)

    db_client.write_reg_demand_data(
        df=reg_demand_DE, balanced=False, extra_tags=extra_tags
    )
    db_client.write_reg_generation_data(
        df=reg_gen_by_type, mode=mode, balanced=False, extra_tags=extra_tags
    )

    logger.info(
        f"Regionalized generation and demand written to the database, period {start}-{end}."
    )
