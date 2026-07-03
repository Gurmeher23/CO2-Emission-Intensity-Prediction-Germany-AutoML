# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:45:26 2022

@author: Tim Fürmann

Version: 1.0

Description: This is the main file for calculating the generation of the renewable
powerplants (Solar,Onshore,Offshore) in Germany for the year 2021 in the spatial
resolution of the federal states.


"""
import logging

import yaml

logger = logging.getLogger(__name__)

import os

import geopandas as gpd
import pandas as pd
import shapely

from cosema.vre_scripts import create_cutout, get_borders_federal, renewable_generation

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

regions = gpd.read_file(f"{config['Shapefiles']['states']['path']}")

x1, y1 = regions.bounds[["minx", "miny"]].min(axis=0).tolist()
x2, y2 = regions.bounds[["maxx", "maxy"]].max(axis=0).tolist()

grid = {
    "dx": 0.25,
    "dy": 0.25,
    "x": [x1, x2],
    "y": [y1, y2],
}


def run_vre_calculations(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_client,
    mode: str = "forecast" or "historical",
    level: str = "federal" or "nuts3",
    overwrite: bool = False,
):
    if pd.Timestamp("now", tz="UTC") - start < pd.Timedelta("7d"):
        era_start_date = (start - pd.Timedelta("7d")).strftime("%Y-%m-%d %H:%M")
    else:
        era_start_date = start.strftime("%Y-%m-%d %H:%M")

    era_end_date = end.strftime("%Y-%m-%d %H:%M")

    time = {"start": era_start_date, "end": era_end_date}

    module = "meteo_hist" if mode == "historical" else "meteo"
    features = "all"
    month = start.strftime("%Y_%m")
    files_daterange = f"{start.strftime('%Y_%m_%d')}_{end.strftime('%m_%d')}"

    cutout_path = f"./inputs/cutouts/{month}"
    cutout_file_path = f"{cutout_path}/cutout_{files_daterange}_{mode}.nc"
    if not os.path.exists(cutout_path):
        os.makedirs(cutout_path)

    shapes = get_borders_federal(path=config["Shapefiles"]["states"]["path"])

    # check if cutout already exists and delete if overwrite is set to True
    if os.path.isfile(cutout_file_path) and overwrite:
        os.remove(cutout_file_path)

    logger.info(f"Creating cutout for {start}-{end} using {module} module")
    cutout_GER = create_cutout(
        path=cutout_file_path,
        grid=grid,
        time=time,
        module=module,
        features=features,
    )
    logger.info(f"Sucessfully created cutout for {start}-{end}")

    capacity_path = f"./inputs/capacities/{month}"
    check_path = f"{capacity_path}/solar_capacities_BW_{month}.parquet"
    if not os.path.isfile(check_path):
        logger.warning(
            f"VRE capacities for {month} not found. Using values from last months."
        )
        month = (start - pd.Timedelta("30d")).strftime("%Y_%m")
        capacity_path = f"./inputs/capacities/{month}"

    for technology in ["solar", "wind_onshore", "wind_offshore"]:
        generation_df = pd.DataFrame()
        for idx, _ in shapes.iterrows():
            identifier = (
                shapes.loc[idx].name if level == "federal" else shapes.loc[idx].NUTS_ID
            )
            if technology == "wind_offshore" and identifier not in ["MV", "NI", "SH"]:
                continue

            local_capacity_path = (
                f"{capacity_path}/{technology}_capacities_{identifier}_{month}.parquet"
            )

            local_capacity = gpd.read_parquet(local_capacity_path)
            local_capacity["geometry"] = shapely.make_valid(local_capacity["geometry"])

            local_shape = shapes.loc[idx]["geometry"]

            generation = renewable_generation(
                technology=technology,
                local_capacity=local_capacity,
                shape_file=local_shape,
                time=time,
                cutout_path=cutout_file_path,
                cutout=cutout_GER,
            )

            generation.index = generation.index - pd.Timedelta("1h")
            # rename column specific generation to identifier
            generation.rename(columns={generation.columns[0]: identifier}, inplace=True)
            generation_df = pd.concat([generation_df, generation], axis=1)

            if db_client is not None:
                db_client.write_vre_data(
                    df=generation,
                    technology=technology,
                    state=identifier,
                    mode=mode,
                    column_name="Generation [MW]",
                )

        total_generation = generation_df.sum(axis=1)
        for idx, _ in shapes.iterrows():
            identifier = (
                shapes.loc[idx].name if level == "federal" else shapes.loc[idx].NUTS_ID
            )
            if technology == "wind_offshore" and identifier not in ["MV", "NI", "SH"]:
                continue

            capacity_factors = generation_df[identifier] / total_generation
            capacity_factors = capacity_factors.fillna(0)
            
            if db_client is not None:
                db_client.write_vre_data(
                    df=capacity_factors,
                    technology=technology,
                    state=identifier,
                    mode=mode,
                    column_name="reg_factor",
                )

    logger.info(f"Sucessfully calculated VRE generation and CF for {generation_df.index[0]}-{generation_df.index[-1]}")
