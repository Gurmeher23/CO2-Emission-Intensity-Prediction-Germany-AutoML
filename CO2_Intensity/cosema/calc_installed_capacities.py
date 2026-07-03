# -*- coding: utf-8 -*-
"""
Created on Friday January 12 2024

@author: Tim Fürmann

Version: 1.0

Description: In this file a power plant list for Germany based on the MaStr
is created. Data is cleaned and filters for capacity, comissioning and decomissioning
can be applied. Furthermore regional identifiers can be added using different region
shapefiles. The resulting data is returned seperated in conventional and vre 
power plants and the per type capacities.
"""

import logging
import os

# Package for Parallelization
import dask

# Packages for Geometric Datamanipulation
import geopandas as gpd

# Packages for Datamanipulation
import numpy as np
import pandas as pd
import yaml
from dask import delayed
from dask.diagnostics import ProgressBar

# Package for Datadownload from the MaStr
from open_mastr import Mastr

logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "config.yaml"


# Load configurations
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config(CONFIG_PATH)


# %% Internal Helper Functions used in the script
def _clean_names(string):
    """
    Removes umlauts from strings and replaces them with the letter+e convention
    :param string: string to remove umlauts from
    :return: unumlauted string
    """
    u = "ü".encode()
    U = "Ü".encode()
    a = "ä".encode()
    A = "Ä".encode()
    o = "ö".encode()
    O = "Ö".encode()
    ss = "ß".encode()

    if ~isinstance(string, str):
        return string

    string = string.encode()
    string = string.replace(u, b"ue")
    string = string.replace(U, b"Ue")
    string = string.replace(a, b"ae")
    string = string.replace(A, b"Ae")
    string = string.replace(o, b"oe")
    string = string.replace(O, b"Oe")
    string = string.replace(ss, b"ss")
    string = string.toUpperCase()

    string = string.decode("utf-8")
    return string


def _get_postcodes(path, crs):
    """
    Loads a list of postcodes for Germany given as input data
    :path string: location where to find the postcode list
    :crs string: coordinate system to use
    :return: postcodes
    """
    # Open postcode list manipulate it and return it
    postcodes = gpd.read_file(path)
    postcodes = postcodes.to_crs(crs)
    postcodes = postcodes[["plz", "geometry"]].rename(columns={"plz": "Postleitzahl"})
    postcodes = postcodes.set_index("Postleitzahl")

    return postcodes


def _clean_default_data(pp_data, gen_types, decommissioning_date):
    """
    Function to clean the default data that is the same for all energy carrier
    fields in the MaStr. This includes cleaning location, types, capacity,
    commissioning and decommissioning data
    :pp_data dataframe: data to clean
    :gen_types dataframe: specific mapping of carrier names
    :return: postcodes
    """

    # Use unique energy carriers for the pp data list, spcified in gen_types
    # Create new column 'Type' with this energy carrier, default is 'Other'
    pp_data["Type"] = "Other"

    for carrier in pp_data["Energietraeger"].unique():
        # Since not all energy carriers are defined in the field Energietraeger, use
        # a specific mapping routine stored in a input .csv file (gen_types)
        if carrier in gen_types[gen_types["field"].notna()].index.unique():
            for resource, row in gen_types.loc[carrier].iterrows():
                pp_data.loc[
                    (pp_data["Energietraeger"].isin([carrier]))
                    & (pp_data[row["field"]].isin([row["value"]])),
                    "Type",
                ] = row["converted"]
        else:
            pp_data.loc[
                (pp_data["Energietraeger"].isin([carrier])), "Type"
            ] = gen_types.loc[carrier, "converted"]

    # Define unique new 'capacity' columns in which the brutto capacity is stored, and if not accesible the netto capacity
    # drop data with missing capacity entries and convert from kW to MW
    pp_data.loc[
        pp_data["Nettonennleistung"].isnull(), "Nettonennleistung"
    ] = pp_data.loc[pp_data["Nettonennleistung"].isnull(), "Bruttoleistung"]
    pp_data["Nettonennleistung"] = pp_data["Nettonennleistung"] / 1000

    pp_data = pp_data.drop(pp_data[pp_data["Nettonennleistung"].isnull()].index, axis=0)

    pp_data.loc[pp_data["Bruttoleistung"].isnull(), "Bruttoleistung"] = pp_data.loc[
        pp_data["Bruttoleistung"].isnull(), "Nettonennleistung"
    ]
    pp_data["Bruttoleistung"] = pp_data["Bruttoleistung"] / 1000

    # Additional column where the difference between Brutto and Netto is stored
    pp_data["Capacity Difference"] = (
        pp_data["Bruttoleistung"] - pp_data["Nettonennleistung"]
    )

    # Get location of each generation unit based on longitude, latitude or plz data
    # drop from the total data set all data where no location information is given
    # use uniform labeling (x, y, postcode)

    missing_locations = pp_data[
        (pp_data["Laengengrad"].isnull()) | (pp_data["Breitengrad"].isnull())
    ]

    # Get country postcode and location mapping only valid for Germany
    postcodes = _get_postcodes(
        f"{config['MaStr']['postcodes']['path']}/{config['MaStr']['postcodes']['name']}",
        config["MaStr"]["crs"],
    )

    postcodes["Location"] = postcodes.representative_point()
    # x laengengrad longitude values
    postcodes["Laengengrad"] = postcodes["Location"].x
    # y breitengrad latitude values
    postcodes["Breitengrad"] = postcodes["Location"].y

    # Fill missing lat/ lon data using postal codes
    pp_data.loc[
        missing_locations["Laengengrad"].index, "Laengengrad"
    ] = missing_locations["Postleitzahl"].map(postcodes["Laengengrad"])
    pp_data.loc[
        missing_locations["Breitengrad"].index, "Breitengrad"
    ] = missing_locations["Postleitzahl"].map(postcodes["Breitengrad"])

    # Drop all without further locational information
    pp_data = pp_data.drop(
        missing_locations[missing_locations["Postleitzahl"].isnull()].index, axis=0
    )

    # Fill missing postal codes based on lat / lon entries
    missing_postcodes = pp_data.loc[pp_data["Postleitzahl"].isnull()]
    missing_postcodes = gpd.GeoDataFrame(
        missing_postcodes,
        geometry=gpd.points_from_xy(
            missing_postcodes["Laengengrad"], missing_postcodes["Breitengrad"]
        ),
        crs=config["MaStr"]["crs"],
    )
    missing_postcodes = missing_postcodes.to_crs(config["MaStr"]["crs"])
    missing_postcodes["Postleitzahl"] = gpd.sjoin(
        missing_postcodes, postcodes, how="inner", predicate="within"
    )["index_right"]
    pp_data.loc[pp_data["Postleitzahl"].isnull(), "Postleitzahl"] = missing_postcodes[
        "Postleitzahl"
    ]

    # Ignore all locations which have not be determined yet
    pp_data = pp_data[
        pp_data["Laengengrad"].notnull() & pp_data["Breitengrad"].notnull()
    ]

    # Clean power plant names by removing umlaute and replacing them with letter+e
    pp_data["NameStromerzeugungseinheit"] = pp_data["NameStromerzeugungseinheit"].apply(
        _clean_names
    )
    pp_data["WeicDisplayName"] = pp_data["WeicDisplayName"].apply(_clean_names)
    pp_data["Strasse"] = pp_data["Strasse"].apply(_clean_names)

    # Map the betriebsstatus of the unit to english names, only relevant for newest updates
    pp_data["EinheitBetriebsstatus"] = pp_data["EinheitBetriebsstatus"].map(
        {
            "In Betrieb": "on",
            "Endgültig stillgelegt": "off",
            "Vorübergehend stillgelegt": "temporarily off",
            "In Planung": "planned",
        }
    )

    # Map the betriebsstatus of the unit to english names, only relevant in case not activated plans would be considered
    pp_data["EinheitSystemstatus"] = pp_data["EinheitSystemstatus"].map(
        {
            "Aktiviert": "activated",
            "Deaktiviert": "deactivated",
            "Unvollständig": "incomplete",
        }
    )

    # Set the default decomissioning date based on different date fields in the database
    pp_data["Decommissioning"] = pp_data[
        pp_data.columns[
            pp_data.columns.isin(
                [
                    "DatumEndgueltigeStilllegung",
                    "DatumBeginnVoruebergehendeStilllegung",
                    "DatumBeginnVorlaeufigenOderEndgueltigenStilllegung",
                ]
            )
        ]
    ].min(skipna=True, axis=1)
    pp_data["Decommissioning"] = pp_data["Decommissioning"].fillna(
        decommissioning_date + pd.Timedelta(20 * 52, "W")
    )

    # Add unavailabilities and correct the decommissioning date for units that returned to the market
    pp_data_with_unavailabilities = pp_data[
        pp_data[
            ["DatumWiederaufnahmeBetrieb", "DatumBeendigungVorlaeufigenStilllegung"]
        ]
        .notna()
        .any(axis=1)
    ]

    pp_data["Unavailability Start"] = pd.NaT
    pp_data["Unavailability End"] = pd.NaT

    for idx, unit in pp_data_with_unavailabilities.iterrows():
        # out date = unit leaves market, return date = unit gets back into market
        out_date = unit["DatumBeginnVoruebergehendeStilllegung"]
        return_date = (
            unit[
                ["DatumWiederaufnahmeBetrieb", "DatumBeendigungVorlaeufigenStilllegung"]
            ]
            .astype("datetime64[ns]")
            .min(skipna=True)
        )

        # store unavailability period
        if out_date <= return_date:
            pp_data.loc[idx, "Unavailability Start"] = out_date
            pp_data.loc[idx, "Unavailability End"] = return_date
        else:
            pp_data.loc[idx, "Unavailability Start"] = out_date
            pp_data.loc[idx, "Unavailability End"] = out_date

        # correct decommissioning date based on return date
        if unit["Decommissioning"] <= return_date:
            decommissioning = (
                unit[
                    unit.index[
                        unit.index.isin(
                            [
                                "DatumEndgueltigeStilllegung",
                                "DatumBeginnVorlaeufigenOderEndgueltigenStilllegung",
                            ]
                        )
                    ]
                ]
                .astype("datetime64[ns]")
                .max(skipna=True)
            )
            if pd.isnull(decommissioning):
                pp_data.loc[idx, "Decommissioning"] = decommissioning_date
            else:
                pp_data.loc[idx, "Decommissioning"] = decommissioning

    # Set the status of a certain unit to on in case its is not decommissioned until
    # the decommissioning filter date
    pp_data.loc[
        pp_data["Decommissioning"] >= decommissioning_date, "EinheitBetriebsstatus"
    ] = "on"

    # Drop not needed columns
    pp_data = pp_data[
        pp_data.columns[
            ~pp_data.columns.isin(
                [
                    "Energietraeger",
                    "Hauptbrennstoff",
                    "Lage",
                    "Technologie",
                    "ArtDerWasserkraftanlage",
                    "Bruttoleistung",
                    "DatumEndgueltigeStilllegung",
                    "DatumBeginnVoruebergehendeStilllegung",
                    "DatumWiederaufnahmeBetrieb",
                    "DatumBeendigungVorlaeufigenStilllegung",
                    "DatumBeginnVorlaeufigenOderEndgueltigenStilllegung",
                ]
            )
        ]
    ]

    # Rename all columns of interest to unique english names, fitting smard data
    pp_data = pp_data.rename(
        columns={
            "Nettonennleistung": "Capacity",
            "Laengengrad": "Longitude",
            "Breitengrad": "Latitude",
            "Postleitzahl": "Postal Code",
            "EinheitBetriebsstatus": "Unit Status",
            "EinheitSystemstatus": "System Status",
            "Weic": "EIC",
            "Kraftwerksnummer": "BNA",
            "NameStromerzeugungseinheit": "Power Plant Name",
            "WeicDisplayName": "EIC Name",
            "Strasse": "Adress",
            "Land": "Country",
            "Inbetriebnahmedatum": "Commissioning",
        }
    )

    return pp_data


def _clean_specific_data(pp_data):
    """
    Function to clean the specific data that is the not the same for all energy carrier
    fields in the MaStr. This includes cleaning data for Solar and Wind data, in
    particular the solar orientation and hub height for wind turbines, as well as
    specific power plant names.
    :pp_data dataframe: data to clean
    :return: postcodes
    """
    # Clean all unit names and store them, if no unit names there set nan
    try:
        pp_data["NameKraftwerk"] = pp_data["NameKraftwerk"].apply(_clean_names)

        pp_data = pp_data.rename(columns={"NameKraftwerk": "Unit Name"})

    except KeyError:
        pp_data["Unit Name"] = np.nan

    # Clean all block names and store them, if no block names there set nan
    try:
        pp_data["NameKraftwerksblock"] = pp_data["NameKraftwerksblock"].apply(
            _clean_names
        )

        pp_data = pp_data.rename(columns={"NameKraftwerksblock": "Block Name"})
    except KeyError:
        pp_data["Block Name"] = np.nan

    # Clean the orientation of solar panels, azimuth and slope, this is required
    # to have a better calculation of the power generation later on, default 180° and 35°
    # Add a value of 180 and 35 degree to all power plants as default, but take for solar plants
    # the values given in the MaStr this will have no influence for the results
    try:
        slope = pp_data["HauptausrichtungNeigungswinkel"].str.findall(r"\d+")
        slope = (
            pd.DataFrame({"0": slope.str[0], "1": slope.str[1]}, index=slope.index)
            .astype(float)
            .ffill(axis=1)
            .fillna(35.0)
        )
        pp_data["HauptausrichtungNeigungswinkel"] = slope.mean(axis=1)

        pp_data["Hauptausrichtung"] = (
            pp_data["Hauptausrichtung"]
            .map(config["MaStr"]["technologies"]["solar"]["azimuth mapping"])
            .fillna(180.0)
        )

        pp_data = pp_data.rename(
            columns={
                "HauptausrichtungNeigungswinkel": "Slope",
                "Hauptausrichtung": "Azimuth",
            }
        )

    except KeyError:
        pp_data["Azimuth"] = 180.0
        pp_data["Slope"] = 35.0

    # Clean the hub height of the wind turbines, default is 100m
    # Add a value of 100m to all power plants as default hub height, but take for wind plants
    # the values given in the MaStr this will have no influence for the results
    try:
        pp_data["Nabenhoehe"] = pp_data["Nabenhoehe"].fillna(100.0)

        pp_data = pp_data.rename(columns={"Nabenhoehe": "Height"})

    except KeyError:
        pp_data["Height"] = 100.0

    return pp_data


def _get_region_identifier(pp_data, shapefiles, closest=False):
    """
    Function for parallization using a subset of the pp_data and matches its
    location with the regions given as shapefiles.
    :pp_data geodataframe: data to match
    :shapefiles geodataframe: region location information
    :closest bool: returns closest shape
    :return: region_ids
    """
    # Loop over each power plant in the subset and match its location to one
    # of the regions given in the shapefiles, default: np.nan (no match)
    # Nan values are later good for filtering as not in region
    region_ids = {}
    for idx, pp in pp_data.iterrows():
        location = pp["geometry"]
        try:
            if closest:
                region_ids[idx] = location.distance(shapefiles.geometry).idxmin()
            else:
                region_ids[idx] = shapefiles.loc[
                    location.within(shapefiles.geometry).values
                ].index[0]
        except IndexError:
            region_ids[idx] = np.nan

    region_ids = pd.Series(region_ids)

    return region_ids


def _add_region_information(pp_data, shapefiles, closest=False):
    """
    Function for matching the location of power plants to the regions of interest
    given as input. This allows later to sum up the capacities for each region
    and power plant type and for filtering.
    :pp_data geodataframe: data to match
    :shapefiles geodataframe: region location information
    :closest bool: allows to return the closest shape
    :return: region_ids
    """
    # Define pp data as geodataframe to add region identifier
    pp_data = gpd.GeoDataFrame(
        pp_data,
        geometry=gpd.points_from_xy(pp_data["Longitude"], pp_data["Latitude"]),
        crs=config["MaStr"]["crs"],
    )

    # Use dask parallelization to increase matchign speed and use lower memory levels
    # therefore a reasonable stepsize is required to exploit the parallelization
    region_ids = []
    for idx in range(0, pp_data.shape[0], config["MaStr"]["dask"]["steps"]):
        region_ids.append(
            delayed(_get_region_identifier)(
                pp_data.iloc[idx : idx + config["MaStr"]["dask"]["steps"]],
                shapefiles,
                closest,
            )
        )

    with ProgressBar():
        print("Adding regional identifier to power plant list")
        region_ids = dask.compute(
            *region_ids,
            scheduler="threads",
            num_workers=config["MaStr"]["dask"]["workers"],
            threads_per_worker=config["MaStr"]["dask"]["threads"],
        )

        try:
            region_ids = pd.concat(region_ids)
        except ValueError:
            region_ids = pd.DataFrame()

    return region_ids


def get_pp_MaStr(
    regions,
    default_decommissioning_date="2045-01-01",
    tech_groups=[
        "biomass",
        "combustion",
        "gsgk",
        "hydro",
        "nuclear",
        "solar",
        "storage",
        "wind",
    ],
    update_regions=False,
    closest=False,
):
    """
    Main function in which all the loading, updating and cleaning of the pp
    data is organized. Inputs are specified via a config file with all important
    settings, the shapes and tech_groups which allow to only load specific data.
    :shapefiles geodataframe: region location information
    :default_decommissioning_date str: default decommissiononing timepoint
    :tech_groups list: list of the technology groups in the MaStr which should used
    :update_regions bool: update region identifier
    :closest bool: method used to update regions
    :return: pp_data
    """

    # Get unique MaStr specific carrier mapping, that is required to have a unique identification
    # of the generation types
    gen_types = pd.read_csv(
        f"{config['MaStr']['technologies']['carrier mapping']['path']}/{config['MaStr']['technologies']['carrier mapping']['name']}"
    )
    gen_types = gen_types.set_index("resource")

    # Convert default decommissioning date to timestamp
    default_decommissioning_date = pd.Timestamp(default_decommissioning_date)

    # Initialize database of the MaStr and update it if desired
    db_MaStr = Mastr()

    # Load powerplant data from MaStr for specific technology groups
    # Use dask parallelization to increase data loading
    pp_data = []
    for tech in tech_groups:
        pp_data.append(
            delayed(pd.read_sql)(sql=f"{tech}_extended", con=db_MaStr.engine)
        )

    with ProgressBar():
        print("Loading data from MaStr")
        pp_data = dask.compute(
            *pp_data,
            scheduler="threads",
            num_workers=config["MaStr"]["dask"]["workers"],
            threads_per_worker=config["MaStr"]["dask"]["threads"],
        )

        pp_data = pd.concat(pp_data)

    # Reduce memory consumption by working only on columns of interest
    MaStr_columns = pp_data.columns[
        pp_data.columns.isin(
            [
                "Lage",
                "Land",
                "Energietraeger",
                "Technologie",
                "Hauptbrennstoff",
                "ArtDerWasserkraftanlage",
                "Weic",
                "WeicDisplayName",
                "Kraftwerksnummer",
                "EinheitMastrNummer",
                "NameKraftwerk",
                "NameKraftwerksblock",
                "EinheitBetriebsstatus",
                "NameStromerzeugungseinheit",
                "Inbetriebnahmedatum",
                "DatumEndgueltigeStilllegung",
                "Postleitzahl",
                "Strasse",
                "Laengengrad",
                "Breitengrad",
                "Bruttoleistung",
                "Nettonennleistung",
                "Nabenhoehe",
                "Hauptausrichtung",
                "HauptausrichtungNeigungswinkel",
                "DatumBeginnVoruebergehendeStilllegung",
                "DatumBeginnVoruebergehendeStilllegung",
                "DatumBeendigungVorlaeufigenStilllegung",
                "DatumWiederaufnahmeBetrieb",
                "DatumBeginnVorlaeufigenOderEndgueltigenStilllegung",
                "EinheitSystemstatus",
            ]
        )
    ]

    pp_data = pp_data[MaStr_columns]
    pp_data = pp_data.reset_index(drop=True)

    # Use the unique SEE MastrNr as identifier
    pp_data = pp_data.rename(columns={"EinheitMastrNummer": "SEE"}).set_index("SEE")

    # Clean the default data that is for all energy carriers given
    pp_data = _clean_default_data(pp_data, gen_types, default_decommissioning_date)

    # Clean specific data that is not given for all energy carriers given
    pp_data = _clean_specific_data(pp_data)

    # Fill all none and na values with np.nan this allows filtering
    pp_data = pp_data.fillna(value=np.nan)

    # Filter for capacitiy entries larger than a certaion threshold (100 kw by default)
    pp_data = pp_data[pp_data["Capacity"] >= config["MaStr"]["capacity threshold"]]

    # Remove all untis without a commissioning date, these are mainly units which are In Planung
    pp_data = pp_data[pp_data["Commissioning"].notna()]

    # Filter for power plant unit status, use only power plants with an unit status 'on'
    # pp_data = pp_data[pp_data["Unit Status"].isin(["on"])]

    # Filter for power plant system status, use only power plants with an activated system status
    pp_data = pp_data[pp_data["System Status"].isin(["activated"])]

    # Add region identifier based on shapefiles if desired
    if update_regions:
        pp_data["Region Identifier"] = _add_region_information(
            pp_data=pp_data, shapefiles=regions, closest=closest
        )

    return pp_data


def calculate_total_capacities_for_cosema(
    start_date,
    end_date,
    freq="M",
    conv_tech_groups=[
        "biomass",
        "combustion",
        "gsgk",
        "hydro",
        "nuclear",
        "storage",
    ],
    vre_tech_groups=["solar", "wind"],
    update_mastr_db=False,
    overwrite=False,
):
    """
    Main function in which all the loading, updating and cleaning of the pp
    data is organized. Inputs are specified via a config file with all important
    settings, the shapes and tech_groups which allow to only load specific data.

    :start_date pd.Timestamp: date from which powerplants should be filtered
    :end_date pd.Timestamp: date until which powerplants should be filtered
    :freg str: frequency with which powerplants should be filtered
    :decomission_date pd.Timestamp: date at which the power plants should be decomissioned
    :update_mastr_db bool: update database (MaStr)

    :return: pp_data_conventional_per_type, pp_data_vre_per_type
    """

    default_decommissioning_date = config["MaStr"]["default"]["decommissioning date"]

    # remove tz info from start and end
    start_date = start_date.tz_localize(None)
    end_date = end_date.tz_localize(None)

    nuts2_to_state = config["nuts2_to_state"]
    nuts2_to_state = {k[3:]: v[3:] for k, v in nuts2_to_state.items()}

    # Load shapefiles from cosema
    regions = gpd.read_file(f"{config['Shapefiles']['states']['path']}")
    regions = regions.to_crs(config["MaStr"]["crs"])
    regions = regions.set_index(config["Shapefiles"]["states"]["identifier"])[
        "geometry"
    ]

    if update_mastr_db:
        logger.info("Updating MaStr database")
        db_MaStr = Mastr()
        db_MaStr.download()

    # Get all conventional generation from the MaStr and set region identifier
    # to closest location (all units are considered independent whether they are
    # located within Germany or not)
    powerplants_conv = get_pp_MaStr(
        regions=regions,
        default_decommissioning_date=default_decommissioning_date,
        tech_groups=conv_tech_groups,
        update_regions=True,
        closest=True,
    )

    powerplants_vre = get_pp_MaStr(
        regions=regions,
        default_decommissioning_date=default_decommissioning_date,
        tech_groups=vre_tech_groups,
        update_regions=True,
        closest=False,
    )

    # calculate conventional and VRE capacities per type
    logger.info("Calculating conventional capacities per type")
    for date in pd.date_range(start=start_date, end=end_date, freq=freq):
        path = f"inputs/capacities/{date.strftime('%Y_%m')}"

        if not os.path.exists(path):
            os.makedirs(path)

        if (
            os.path.exists(f"{path}/conv_capacities_{date.strftime('%Y_%m')}.parquet")
            and not overwrite
        ):
            logger.info(
                f"Capacities are already calculated for {date}, skipping calculation"
            )
            continue

        pp_data = powerplants_conv.copy()
        pp_data = pp_data[
            ~pp_data["Type"].isin(["Solar", "Wind Onshore", "Wind Offshore"])
        ]

        # Filter for a commisioning year larger equal than start date and decommissioning year smaller equal than end date
        pp_data = pp_data[pp_data["Commissioning"] <= date]
        pp_data = pp_data[pp_data["Decommissioning"] >= date + pd.DateOffset(months=1)]

        # Sum the capacities to per type generation, either with region identifiers or without
        pp_data_per_type = (
            pp_data[["Capacity", "Type", "Region Identifier"]]
            .groupby(["Type", "Region Identifier"])
            .sum()
        )

        pp_data_per_type["Region"] = pp_data_per_type.index.get_level_values(1).map(
            nuts2_to_state
        )

        # make a dataframe with states as index and columns as technologies and capacities as values
        pp_data_per_type = pp_data_per_type.pivot_table(
            index="Region", columns="Type", values="Capacity"
        )

        # fill nan values with 0
        pp_data_per_type = pp_data_per_type.fillna(0)
        # round to 3 significant digits
        pp_data_per_type = pp_data_per_type.round(3)

        # save to parquet
        pp_data_per_type.to_parquet(
            f"{path}/conv_capacities_{date.strftime('%Y_%m')}.parquet"
        )
        # %% calculate VRE capacities per type
        if (
            os.path.exists(f"{path}/solar_capacities_{date.strftime('%Y_%m')}.parquet")
            and not overwrite
        ):
            logger.info(
                f"Capacities are already calculated for {date}, skipping calculation"
            )
            continue

        pp_data_vre = powerplants_vre.copy()
        pp_data_vre = pp_data_vre[
            pp_data_vre["Type"].isin(["Solar", "Wind Onshore", "Wind Offshore"])
        ]

        # Filter for a commisioning year larger equal than start date and decommissioning year smaller equal than end date
        pp_data_vre = pp_data_vre[pp_data_vre["Commissioning"] <= date]
        pp_data_vre = pp_data_vre[
            pp_data_vre["Decommissioning"] >= date + pd.DateOffset(months=1)
        ]

        pp_data_vre["Region Identifier"] = pp_data_vre["Region Identifier"].map(
            nuts2_to_state
        )

        # Filter all powerplants for region and technology and store for each month
        # their capacity in seperate files. The dataframe should contain the following
        # columns: plz, longitude, latitude, capacity
        pp_data_vre = pp_data_vre[
            ["Type", "Longitude", "Latitude", "Capacity", "Region Identifier"]
        ]
        pp_data_vre = pp_data_vre.rename(columns={"Longitude": "x", "Latitude": "y"})
        pp_data_vre = pp_data_vre.set_index(["Region Identifier", "Type"], drop=False)
        pp_data_vre = gpd.GeoDataFrame(
            pp_data_vre,
            geometry=gpd.points_from_xy(
                pp_data_vre["x"], pp_data_vre["y"], crs=config["MaStr"]["crs"]
            ),
        )

        # Store data to file
        for idx in pp_data_vre.index.unique():
            region = idx[0]
            # continue if region is nan
            if pd.isna(region):
                continue
            carrier = idx[1].lower()

            if carrier == "wind offshore":
                carrier = "wind_offshore"
            elif carrier == "wind onshore":
                carrier = "wind_onshore"

            temp_df = pp_data_vre.loc[idx].reset_index(drop=True)
            temp_df = temp_df.drop(columns=["Type", "Region Identifier"])

            temp_df.to_parquet(
                f"{path}/{carrier}_capacities_{region}_{date.strftime('%Y_%m')}.parquet"
            )

        logger.info(f"Capacities for {date} are calculated and stored")
