import logging

import yaml

from cosema.demand_scripts import reg_demand_data_dynamic
from cosema.forecast_scripts import forecast_bus_intensity, prepare_inputs
from cosema.per_type_scripts import collect_vre

logger = logging.getLogger(__name__)

import pandas as pd

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BUSES = config["states"].values()


def get_demand_data(
    db_client,
    index: pd.date_range,
):
    start = index[0]
    end = index[-1]

    # Prepare demand data
    logger.debug("Calculating regionalized forecasted demand...")
    demand_DE = db_client.query_demand_data(
        start=start, end=end, country="DE", mode="forecast"
    )

    demand_DE = demand_DE.dropna()
    demand_DE = demand_DE.interpolate(method="linear", axis=0, limit_area="inside")

    demand_DE = demand_DE.fillna(0)

    reg_demand_DE = reg_demand_data_dynamic(demand_DE)

    return reg_demand_DE


def get_vre_data(
    db_client,
    index: pd.date_range,
):
    start = index[0]
    end = index[-1]

    logger.debug("Quering forecasted VRE generation...")
    reg_solar, reg_onshore, reg_offshore = collect_vre(
        start=start,
        end=end,
        db_client=db_client,
        mode="forecast",
        value="Generation [MW]",
    )

    reg_solar = reg_solar.interpolate(method="linear", axis=0, limit_area="inside")
    reg_onshore = reg_onshore.interpolate(method="linear", axis=0, limit_area="inside")
    reg_offshore = reg_offshore.interpolate(
        method="linear", axis=0, limit_area="inside"
    )

    reg_solar = reg_solar.fillna(0)
    reg_onshore = reg_onshore.fillna(0)
    reg_offshore = reg_offshore.fillna(0)

    return reg_solar, reg_onshore, reg_offshore


def get_forecasts(
    db_client,
    index: pd.date_range,
    reg_solar,
    reg_onshore,
    reg_demand_DE,
):
    logger.debug("Calculating emission intensitiy forecasts...")
    start = index[0]
    end = index[-1]

    gen_intensity = pd.DataFrame(columns=BUSES, index=index, data=0.0)

    cons_intensity = pd.DataFrame(columns=BUSES, index=index, data=0.0)

    for state in BUSES:
        features = pd.concat(
            [reg_solar[state], reg_onshore[state], reg_demand_DE[state]], axis=1
        )
        features.columns = ["Solar", "Onshore", "demand"]
        features = prepare_inputs(features_state=features, state=state)

        gen_intensity[state] = forecast_bus_intensity(
            state=state, emission_type="gen", input=features
        )
        cons_intensity[state] = forecast_bus_intensity(
            state=state, emission_type="cons", input=features
        )

    gen_intensity = gen_intensity / 1000
    cons_intensity = cons_intensity / 1000

    logger.debug("Writing emission forecast to the db...")
    db_client.write_intensities(
        intensity=gen_intensity, type="Generation", mode="forecast"
    )

    db_client.write_intensities(
        intensity=cons_intensity, type="Consumption", mode="forecast"
    )

    logger.info(
        f"Intensities forecasted and written to database, period {start}-{end}."
    )


def get_temp_forecast(
    db_client,
    index: pd.date_range,
):
    try_start = index[0] - pd.Timedelta(days=7)
    try_end = index[-1] - pd.Timedelta(days=7)
    try_index = pd.date_range(start=try_start, end=try_end, freq="1h")

    gen_intensity = pd.DataFrame(columns=BUSES, index=index, data=0.0)

    cons_intensity = pd.DataFrame(columns=BUSES, index=try_index, data=0.0)

    for state in BUSES:
        gen_intensity[state] = db_client.query_intensities(
            start=try_start,
            end=try_end,
            state=state,
            emission_type="Generation",
            mode="historic",
        )

        cons_intensity[state] = db_client.query_intensities(
            start=try_start,
            end=try_end,
            state=state,
            emission_type="Consumption",
            mode="historic",
        )

    gen_intensity.index = index
    cons_intensity.index = index

    logger.debug("Writing temp emission forecast to the db...")
    db_client.write_intensities(
        intensity=gen_intensity, type="Generation", mode="forecast"
    )

    db_client.write_intensities(
        intensity=cons_intensity, type="Consumption", mode="forecast"
    )

    logger.warning(
        f"Temp forecasted intensities written to database, period {index[0]}-{index[-1]}."
    )


def forecast_intensities(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_client,
):
    index = pd.date_range(start=start, end=end, freq="1h")

    try:
        reg_demand_DE = get_demand_data(db_client=db_client, index=index)
        index = reg_demand_DE.index

        reg_solar, reg_onshore = get_vre_data(db_client=db_client, index=index)
        sucess = True
    except Exception as e:
        logger.error(f"Error {e}. No forecasted values for period {start} - {end}.")
        sucess = False

    if sucess:
        get_forecasts(
            db_client=db_client,
            index=index,
            reg_solar=reg_solar,
            reg_onshore=reg_onshore,
            reg_demand_DE=reg_demand_DE,
        )
    else:
        get_temp_forecast(db_client=db_client, index=index)
