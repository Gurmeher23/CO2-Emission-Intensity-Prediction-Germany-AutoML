import logging
import time
import traceback

import entsoe
import pandas as pd
import requests
import yaml

from cosema.download_scripts_SMARD import download_DE_per_type_data
from cosema.download_scripts_UK import (
    download_GB_demand_data,
    download_GB_IE_flows,
    download_GB_per_type_data,
    download_IE_demand_data,
    download_IE_per_type_data,
)

logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

COUNTRIES_FOR_FORECAST = config["countries_for_forecast"]


def retry_function(func, *args, **kwargs):
    # try 3 times with 15 seconds delay
    counter = 0
    while counter < 3:
        time.sleep(10)
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error during {func.__name__}. Retrying.")

            counter += 1
            continue

    raise Exception(f"Failed to download data after 3 attempts.")


def download_per_unit_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    db_client,
    control_areas: dict = None,
):
    if control_areas is None:
        control_areas = config["control_areas"]

    for tso, zone_code in control_areas.items():
        try:
            # try 3 times with 15 seconds delay
            tso_gen = retry_function(
                entsoe_client.query_generation_per_plant,
                country_code=zone_code,
                start=start,
                end=end,
                include_eic=True,
                nett=True,
            )

        except entsoe.exceptions.NoMatchingDataError:
            logger.warning(
                f"No generation per unit data found for TSO {tso}, period {start}-{end}."
            )
            continue

        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f'Error "{error}" during per unit data download: {tso}, period {start}-{end}.'
            )
            continue

        tso_gen.index = pd.to_datetime(tso_gen.index).tz_convert("UTC")

        for info, gen in tso_gen.items():
            tempDF = pd.DataFrame(
                index=gen.index, columns=["Generation [MW]"], data=gen.values
            ).astype("float32")

            db_client.write_df(
                df=tempDF,
                measurement="per_unit_gen",
                tags={
                    "ID": info[0],
                    "technology": info[1],
                    "TSO": tso,
                    "EIC": info[3],
                },
            )

    logger.info(
        f"Per unit data downloaded and written to database, period {start}-{end}."
    )


def _download_per_type_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    country: str,
):
    if country == "GB":
        country_gen = download_GB_per_type_data(start=start, end=end)

    elif country == "IE":
        country_gen = download_IE_per_type_data(start=start, end=end)

    elif country == "DE":
        country_gen_1 = entsoe_client.query_generation(
            country_code=country, start=start, end=end, nett=True
        )
        country_gen_2 = download_DE_per_type_data(start=start, end=end, nett=True)

        # Combine data, filling missing data in country_gen_1 with data from country_gen_2
        country_gen = country_gen_1.combine_first(country_gen_2)

    else:
        country_gen = entsoe_client.query_generation(
            country_code=country, start=start, end=end, nett=True
        )

    # make sure index timezone is UTC
    country_gen.index = pd.to_datetime(country_gen.index).tz_convert("UTC")

    return country_gen


def download_per_type_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    db_client,
    countries: list = None,
):
    if countries is None:
        countries = config["countries"]

    for country in countries:
        try:
            # try 3 times with 15 seconds delay
            country_gen = retry_function(
                _download_per_type_data,
                start=start,
                end=end,
                entsoe_client=entsoe_client,
                country=country,
            )

        except entsoe.exceptions.NoMatchingDataError:
            logger.warning(
                f"No generation per type data found for country {country}, period {start}-{end}."
            )
            continue

        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f'Error "{error}" during per type data download: country {country}, period {start}-{end}.'
            )
            continue

        for technology, gen in country_gen.items():
            tempDF = pd.DataFrame(
                index=gen.index, columns=["Generation [MW]"], data=gen.values
            ).astype("float32")

            db_client.write_df(
                df=tempDF,
                measurement="per_type_gen",
                tags={"country": country, "technology": technology},
            )

    logger.info(
        f"Per type data downloaded and written to database, period {start}-{end}."
    )


def _download_demand_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    country: str,
):
    if country == "GB":
        demand = download_GB_demand_data(start=start, end=end)

    elif country == "IE":
        demand = download_IE_demand_data(start=start, end=end)

    else:
        demand = entsoe_client.query_load(country_code=country, start=start, end=end)

    demand.index = pd.to_datetime(demand.index).tz_convert("UTC")
    demand = demand.astype("float32")
    demand = demand.rename(columns={"Actual Load": "Demand [MW]"})

    return demand


def download_demand_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    db_client,
    countries: list = None,
):
    if countries is None:
        countries = config["countries"]

    for country in countries:
        try:
            # try 3 times with 15 seconds delay
            demand = retry_function(
                _download_demand_data,
                start=start,
                end=end,
                entsoe_client=entsoe_client,
                country=country,
            )
            time.sleep(0.5)

        except entsoe.exceptions.NoMatchingDataError:
            logger.warning(
                f"No demand data found for country {country}, period {start}-{end}."
            )
            continue

        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f' Error "{error}" during demand data download: country {country}, period {start}-{end}.'
            )
            continue

        db_client.write_df(df=demand, measurement="demand", tags={"country": country})

    logger.info(
        f"Demand data downloaded and written to database, period {start}-{end}."
    )


def download_demand_forecast_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    db_client,
):
    for country_code in COUNTRIES_FOR_FORECAST:
        try:
            # try 3 times with 15 seconds delay
            demand_forecast = retry_function(
                entsoe_client.query_load_forecast,
                country_code,
                start=start,
                end=end,
            )

        except entsoe.exceptions.NoMatchingDataError:
            logger.warning(
                f"No demand forecast data found for country {country_code}, period {start}-{end}."
            )
            continue

        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f' Error "{error}" during demand forecast data download: country {country_code}, period {start}-{end}.'
            )
            continue

        demand_forecast.index = pd.to_datetime(demand_forecast.index).tz_convert("UTC")

        demand_forecast = demand_forecast.rename(
            columns={"Forecasted Load": "Demand [MW]"}
        )
        demand_forecast = demand_forecast.astype("float32")

        db_client.write_df(
            df=demand_forecast,
            measurement="demand_forecast",
            tags={"country": country_code},
        )

    logger.info(
        f"Demand forecast data downloaded and written to database, period {start}-{end}."
    )


def download_vre_forecast_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    db_client,
):
    for country_code in COUNTRIES_FOR_FORECAST:
        try:
            # try 3 times with 15 seconds delay
            vre_forecast = retry_function(
                entsoe_client.query_wind_and_solar_forecast,
                country_code,
                start=start,
                end=end,
            )

        except entsoe.exceptions.NoMatchingDataError:
            logger.warning(
                f"No VRE forecast data found for country {country_code}, period {start}-{end}."
            )
            continue

        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f' Error "{error}" during VRE forecast data download: country {country_code}, period {start}-{end}.'
            )
            continue

        # make sure index timezone is UTC
        vre_forecast.index = pd.to_datetime(vre_forecast.index).tz_convert("UTC")

        for technology, gen in vre_forecast.items():
            tempDF = pd.DataFrame(
                index=gen.index, columns=["Generation [MW]"], data=gen.values
            ).astype("float32")

            db_client.write_df(
                df=tempDF,
                measurement="per_type_gen_forecast",
                tags={"country": country_code, "technology": technology},
            )

    logger.info(
        f"VRE forecast data downloaded and written to database, period {start}-{end}."
    )


def _download_cross_border_flows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    border: str,
):
    country_code_from, country_code_to = border.split("-")

    if border == "GB-IE":
        flows = download_GB_IE_flows(start=start, end=end)
        flow_to = flows["GB > IE"]
        flow_from = flows["IE > GB"]

    else:
        flow_to = entsoe_client.query_crossborder_flows(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
        )
        flow_from = entsoe_client.query_crossborder_flows(
            country_code_to,
            country_code_from,
            start=start,
            end=end,
        )

    flow_to.index = pd.to_datetime(flow_to.index).tz_convert("UTC")
    flow_from.index = pd.to_datetime(flow_from.index).tz_convert("UTC")

    flow_to = pd.DataFrame(
        index=flow_to.index,
        columns=["Flow [MW]"],
        data=flow_to.values,
    ).astype("float32")

    flow_from = pd.DataFrame(
        index=flow_from.index,
        columns=["Flow [MW]"],
        data=flow_from.values,
    ).astype("float32")

    return flow_to, flow_from


def download_cross_border_flows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    db_client,
    country_borders: list = None,
):
    if country_borders is None:
        country_borders = config["country_borders"]

    for border in country_borders:
        try:
            # try 3 times with 15 seconds delay
            flow_to, flow_from = retry_function(
                _download_cross_border_flows,
                start=start,
                end=end,
                entsoe_client=entsoe_client,
                border=border,
            )

        except entsoe.exceptions.NoMatchingDataError:
            logger.warning(
                f"No cross border flow data found for line {border}, period {start}-{end}."
            )
            continue

        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f' Error "{error}" during cross border flow data download: line {border}, period {start}-{end}.'
            )
            continue

        country_code_from, country_code_to = border.split("-")

        db_client.write_df(
            df=flow_to,
            measurement="cross_border_flow",
            tags={"from": country_code_from, "to": country_code_to},
        )

        db_client.write_df(
            df=flow_from,
            measurement="cross_border_flow",
            tags={"from": country_code_to, "to": country_code_from},
        )

    logger.info(
        f"Cross border flow data downloaded and written to database, period {start}-{end}."
    )

def download_generation_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    db_client,
    countries: list = None,
):
    if countries is None:
        countries = config["countries"]

    for country_code in countries:
        try:
            # Retry 3 times with delay
            generation_forecast = retry_function(
                entsoe_client.query_generation_forecast,
                country_code,
                start=start,
                end=end,
            )
        except entsoe.exceptions.NoMatchingDataError:
            logger.warning(
                f"No generation forecast data found for country {country_code}, period {start}-{end}."
            )
            continue
        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f'Error "{error}" during generation forecast download: country {country_code}, period {start}-{end}.'
            )
            continue

        # Convert to UTC
        generation_forecast.index = pd.to_datetime(generation_forecast.index).tz_convert("UTC")

        # Save data to the database
        for technology, gen in generation_forecast.items():
            tempDF = pd.DataFrame(
                index=gen.index, columns=["Generation [MW]"], data=gen.values
            ).astype("float32")

            db_client.write_df
                df=tempDF,
                measurement="generation_forecast",
                tags={"country": country_code, "technology": technology},
            )

    logger.info(
        f"Generation forecast data downloaded and written to database, period {start}-{end}."
    )


def download_wind_and_solar_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    db_client,
    countries: list = None,
):
    if countries is None:
        countries = config["countries"]

    for country_code in countries:
        try:
            wind_solar_forecast = retry_function(
                entsoe_client.query_wind_and_solar_forecast,
                country_code,
                start=start,
                end=end,
            )
        except entsoe.exceptions.NoMatchingDataError:
            logger.warning(
                f"No wind and solar forecast data found for country {country_code}, period {start}-{end}."
            )
            continue
        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f'Error "{error}" during wind and solar forecast download: country {country_code}, period {start}-{end}.'
            )
            continue

        # Convert to UTC
        wind_solar_forecast.index = pd.to_datetime(wind_solar_forecast.index).tz_convert("UTC")

        # Save data to the database
        for psr_type, gen in wind_solar_forecast.items():
            tempDF = pd.DataFrame(
                index=gen.index, columns=["Generation [MW]"], data=gen.values
            ).astype("float32")

            db_client.write_df(
                df=tempDF,
                measurement="wind_solar_forecast",
                tags={"country": country_code, "psr_type": psr_type},
            )

    logger.info(
        f"Wind and solar forecast data downloaded and written to database, period {start}-{end}."
    )

def download_day_ahead_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    entsoe_client,
    db_client,
    countries: list = None,
):
    if countries is None:
        countries = config["countries"]

    for country_code in countries:
        try:
            day_ahead_prices = retry_function(
                entsoe_client.query_day_ahead_prices,
                country_code,
                start=start,
                end=end,
            )
        except entsoe.exceptions.NoMatchingDataError:
            logger.warning(
                f"No day-ahead prices found for country {country_code}, period {start}-{end}."
            )
            continue
        except Exception as e:
            error = traceback.format_exc().splitlines()[-1]
            logger.warning(
                f'Error "{error}" during day-ahead prices download: country {country_code}, period {start}-{end}.'
            )
            continue

        # Convert to UTC
        day_ahead_prices.index = pd.to_datetime(day_ahead_prices.index).tz_convert("UTC")
        tempDF = pd.DataFrame(
            index=day_ahead_prices.index, columns=["Price [EUR/MWh]"], data=day_ahead_prices.values
        ).astype("float32")

        db_client.write_df(
            df=tempDF,
            measurement="day_ahead_prices",
            tags={"country": country_code},
        )

    logger.info(
        f"Day-ahead prices downloaded and written to database, period {start}-{end}."
    )

