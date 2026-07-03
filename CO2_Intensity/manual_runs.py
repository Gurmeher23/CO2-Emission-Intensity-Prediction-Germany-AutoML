# %%
import logging

import pandas as pd
import yaml
from entsoe import EntsoePandasClient

from cosema.calc_forecast import forecast_intensities
from cosema.calc_installed_capacities import calculate_total_capacities_for_cosema
# from cosema.calc_intensities import calculate_intensities
from cosema.calc_vre import run_vre_calculations
from cosema.db_client import DBClient
from cosema.download_scripts import (
    download_cross_border_flows,
    download_demand_data,
    download_demand_forecast_data,
    download_per_type_data,
    download_per_unit_data,
)
from cosema.loggers import get_handlers
# from cosema.regionalization import calculate_regionalized_gen_and_demand

handlers = get_handlers(log_path="logs/manual_runs.log")
logging.basicConfig(level=logging.INFO, handlers=handlers)

logger = logging.getLogger(__name__)

# read config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# load keys.yaml where the database and entsoe keys are stored
with open("keys.yaml", "r") as f:
    keys = yaml.safe_load(f)

db_client = DBClient(
    database_name="cosema",
    username=keys["influxdb"]["username"],
    password=keys["influxdb"]["password"],
)

entsoe_client = EntsoePandasClient(api_key=keys["entsoe-key"])

# Start and end date
start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
end = pd.Timestamp("2024-02-03 00:00", tz="UTC")

get_per_type_data = 0
get_demand_data = 1
get_demand_forecast_data = 1
get_generation_forecast = 1
get_wind_and_solar_forecast = 1
get_day_ahead_prices = 1
get_cross_border_flows = 0
get_per_unit_data = 0
get_vre_historical_data = 0
get_vre_forecast_data = 0
run_regionalization = 0
calc_intensities = 0

reg_mode = "only_per_type"  # "only_per_type" / "with_per_unit"


# %%
for month in pd.date_range(start=start, end=end, freq="MS"):
    if end - start > pd.Timedelta(days=30):
        temp_start = month
        temp_end = month + pd.DateOffset(months=1)
    else:
        temp_start = start
        temp_end = end

    logger.info(f"Running for {temp_start} to {temp_end}")

    # Download per type data
    if get_per_type_data:
        download_per_type_data(
            start=temp_start,
            end=temp_end,
            entsoe_client=entsoe_client,
            db_client=db_client,
        )

    # Download demand and demand forecast data
    if get_demand_data:
        download_demand_data(
            start=temp_start,
            end=temp_end,
            entsoe_client=entsoe_client,
            db_client=db_client,
        )

    if get_demand_forecast_data:
        download_demand_forecast_data(
            start=temp_start,
            end=temp_end,
            entsoe_client=entsoe_client,
            db_client=db_client,
        )

    # Download generation forecast
    if get_generation_forecast:
        download_generation_forecast(
            start=temp_start,
            end=temp_end,
            entsoe_client=entsoe_client,
            db_client=db_client,
        )

    # Download wind and solar forecast
    if get_wind_and_solar_forecast:
        download_wind_and_solar_forecast(
            start=temp_start,
            end=temp_end,
            entsoe_client=entsoe_client,
            db_client=db_client,
        )

    # Download day-ahead prices
    if get_day_ahead_prices:
        download_day_ahead_prices(
            start=temp_start,
            end=temp_end,
            entsoe_client=entsoe_client,
            db_client=db_client,
        )

    # Download cross border flows
    if get_cross_border_flows:
        download_cross_border_flows(
            start=temp_start,
            end=temp_end,
            entsoe_client=entsoe_client,
            db_client=db_client,
        )

    # Download per unit data
    if get_per_unit_data:
        download_per_unit_data(
            start=temp_start,
            end=temp_end,
            entsoe_client=entsoe_client,
            db_client=db_client,
        )

    # VRE historical calculations
    if get_vre_historical_data:
        run_vre_calculations(
            start=temp_start,
            end=temp_end,
            db_client=db_client,
            mode="historical",
            level="federal",
        )

    # VRE forecast calculations
    if get_vre_forecast_data:
        run_vre_calculations(
            start=temp_start,
            end=temp_end,
            db_client=db_client,
            mode="forecast",
            level="federal",
            overwrite=True,
        )

    # Calculate regionalized generation and demand
    if run_regionalization:
        calculate_regionalized_gen_and_demand(
            start=temp_start,
            end=temp_end,
            db_client=db_client,
            mode=reg_mode,
        )

    if calc_intensities:
        calculate_intensities(
            start=temp_start,
            end=temp_end,
            db_client=db_client,
            mode=reg_mode,
        )



# %%
# VRE capacity calculation
# update_mastr_db = False

# calculate_total_capacities_for_cosema(
#     start_date=start,
#     end_date=end,
#     update_mastr_db=update_mastr_db,
# )
