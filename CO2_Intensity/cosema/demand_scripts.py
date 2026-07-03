import logging
import traceback

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BUSES = config["states"].values()


def query_demand_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_client,
    mode="historical",
    countries: list = None,
):
    if countries is None:
        countries = config["countries"]

    demand_df = pd.DataFrame(
        index=pd.date_range(start=start, end=end, freq="1h"),
        columns=countries,
        data=0.0,
    )

    for country in countries:
        try:
            demand = db_client.query_demand_data(
                start=start, end=end, country=country, mode=mode
            )
            demand_df[country] = demand.values
        except Exception as e:
            logger.warning(
                f"No data for {country} for period {start} - {end}. Returning 0.0..."
            )
            demand_df[country] = 0.0

    return demand_df


def query_DE_demand_data(
    start: pd.Timestamp, end: pd.Timestamp, db_client, country="DE", mode="historical"
):
    demand_DE = db_client.query_demand_data(
        start=start, end=end, country=country, mode=mode
    )

    return demand_DE


def query_reg_demand_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_client,
    column_name="Demand [MW]",
    balanced=False,
):
    demand_df = pd.DataFrame(
        index=pd.date_range(start=start, end=end, freq="1h"),
        columns=[f"DE_{bus}" for bus in BUSES],
        data=0.0,
    )

    for state in BUSES:
        try:
            demand = db_client.query_reg_demand_data(
                start=start,
                end=end,
                state=state,
                column_name=column_name,
                balanced=balanced,
            )
        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f'Error "{error}" during demand data query: state {state}, period {start}-{end}.'
            )
            continue

        demand_df[f"DE_{state}"] = demand.values

    return demand_df


def reg_demand_data_dynamic(demand_DE):
    index = demand_DE.index

    # Identify all years in the index
    years = sorted(set(index.year))
    
    # Initialize an empty DataFrame to hold demand_reg_factors from all years
    demand_reg_factors_all = pd.DataFrame()

    # Read and concatenate demand_reg_factors for each year
    for year in years:
        yearly_factors = pd.read_csv(
            f"inputs/demand_reg_data/demand_reg_factors_{year}.csv", index_col=0
        )
        yearly_factors.index = pd.to_datetime(yearly_factors.index, utc=True)
        demand_reg_factors_all = pd.concat([demand_reg_factors_all, yearly_factors])

    # Remove duplicates if present
    demand_reg_factors_all = demand_reg_factors_all[~demand_reg_factors_all.index.duplicated(keep='first')]

    # Filter for the relevant index range
    reg_factors = demand_reg_factors_all.loc[index]

    # Calculate the regionalized demand
    reg_demand_DE = demand_DE.values * reg_factors.values
    reg_demand_DE = pd.DataFrame(
        columns=demand_reg_factors_all.columns, index=demand_DE.index, data=reg_demand_DE
    )

    return reg_demand_DE
