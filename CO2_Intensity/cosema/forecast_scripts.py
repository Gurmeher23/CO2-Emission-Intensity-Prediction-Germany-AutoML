import logging

import yaml

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    logger.warning("LightGBM not installed. Forecasting will not work.")
    lgb = None

import numpy as np
import pandas as pd

from cosema.demand_scripts import query_reg_demand_data
from cosema.per_type_scripts import collect_intensities, collect_vre

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BUSES = config["states"].values()


def prepare_inputs(features_state, state):
    features_norm = norm_data(features_state, state)

    sin_hour = pd.DataFrame(
        np.array(np.sin(features_state.index.hour)),
        index=features_state.index,
        columns=["sin_hour"],
    )
    cos_hour = pd.DataFrame(
        np.array(np.cos(features_state.index.hour)),
        index=features_state.index,
        columns=["cos_hour"],
    )
    sin_day = pd.DataFrame(
        np.array(np.sin(features_state.index.dayofweek)),
        index=features_state.index,
        columns=["sin_day"],
    )
    cos_day = pd.DataFrame(
        np.array(np.cos(features_state.index.dayofweek)),
        index=features_state.index,
        columns=["cos_day"],
    )

    X = pd.concat([features_norm, sin_hour, cos_hour, sin_day, cos_day], axis=1)

    return X


def prepare_outputs(CO2_state, state):
    y = norm_data(CO2_state, state)
    y = y.fillna(value=y.mean(axis=0))
    return y


def norm_data(data, state):
    # norm the data!
    # use the normation values used for training! Needs to be actualized for each new model
    maxima = pd.read_csv("./inputs/forecast_models/norm_max.csv", index_col=0)
    minima = pd.read_csv("./inputs/forecast_models/norm_min.csv", index_col=0)

    if isinstance(data, pd.DataFrame):
        maxima = maxima.loc[data.columns, state]
        minima = minima.loc[data.columns, state]

    else:
        maxima = maxima.loc[data.name, state]
        minima = minima.loc[data.name, state]

    data = (data - minima) / (maxima - minima)

    return data


def denorm_data(data, state):
    # scale back to original scale

    maxima = pd.read_csv("./inputs/forecast_models/norm_max.csv", index_col=0)
    minima = pd.read_csv("./inputs/forecast_models/norm_min.csv", index_col=0)

    if isinstance(data, pd.DataFrame):
        maxima = maxima.loc[data.columns, state]
        minima = minima.loc[data.columns, state]

    else:
        maxima = maxima.loc[data.name, state]
        minima = minima.loc[data.name, state]

    data = data * (maxima - minima) + minima

    return data


def load_forecast_model(state=None, emission_type="cons"):
    # gets the state as a string and returns the recent model for that state

    file_path = "inputs/forecast_models"
    model = lgb.Booster(model_file=f"{file_path}/{state}_model_{emission_type}.txt")

    return model


def forecast_bus_intensity(state, emission_type, input):
    # gets the preprocessed features X, the state and emission_type and returns a forecast
    model = load_forecast_model(state, emission_type)

    CO2_forecast = model.predict(input)
    CO2_forecast = pd.DataFrame(
        denorm_data(pd.Series(CO2_forecast, index=input.index, name="CO2"), state)
    )

    return CO2_forecast


def update_model(model, inputs, output, state, emission_type="cons"):
    # gets a model and preprocessed input and ouput data for small time intervall
    # and saves the updated model
    file_path = "inputs/forecast_models"

    params = pd.read_csv(f"{file_path}/train_params.csv", index_col=0)
    params = params[state]

    train_data = lgb.Dataset(inputs, label=output, free_raw_data=False)

    model_updated = lgb.train(
        params={"force_col_wise": True, "min_data_in_leaf": 1},
        train_set=train_data,
        num_boost_round=1,
        init_model=model,
        keep_training_booster=True,
    )

    model_updated.save_model(
        f"{file_path}/{state}_model_{emission_type}.txt",
        num_iteration=model_updated.best_iteration,
    )

    return model_updated


def update_forecast_model(db_client):
    start = pd.Timestamp("now") - pd.Timedelta("37d")
    end = pd.Timestamp("now") - pd.Timedelta("7d")

    logger.debug("Quering regionalized generation ...")
    reg_solar, reg_onshore, reg_offshore = collect_vre(
        start=start,
        end=end,
        db_client=db_client,
        mode="historical",
        values="Generation [MW]",
    )

    # Prepare demand data
    logger.debug("Quering regionalized demand...")
    reg_demand_DE = query_reg_demand_data(start=start, end=end, db_client=db_client)

    logger.debug("Preparing past emission intensities...")
    cons_intensity, gen_intensity = collect_intensities(
        start=start, end=end, db_client=db_client
    )

    cons_intensity = cons_intensity * 1000
    gen_intensity = gen_intensity * 1000

    logger.debug("Updating intensitiy forecast models...")

    for state in BUSES:
        features = pd.concat(
            [reg_solar[state], reg_onshore[state], reg_demand_DE[state]], axis=1
        )
        features.columns = ["Solar", "Onshore", "demand"]
        features = prepare_inputs(features_state=features, state=state)

        cons_output = cons_intensity[state]
        cons_output.name = "CO2"
        gen_output = gen_intensity[state]
        gen_output.name = "CO2"

        cons_CO2 = prepare_outputs(CO2_state=cons_output, state=state)
        gen_CO2 = prepare_outputs(CO2_state=gen_output, state=state)

        cons_model = load_forecast_model(state=state, emission_type="cons")
        cons_model = update_model(
            cons_model,
            inputs=features,
            output=cons_CO2,
            state=state,
            emission_type="cons",
        )

        gen_model = load_forecast_model(state=state, emission_type="gen")
        gen_model = update_model(
            gen_model, inputs=features, output=gen_CO2, state=state, emission_type="gen"
        )

    logger.info("Forecast models updated.")
