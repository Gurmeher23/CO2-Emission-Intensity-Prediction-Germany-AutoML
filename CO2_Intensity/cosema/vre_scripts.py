import logging

import geopandas as gp
import numpy as np
import shapely
import yaml

import atlite as at

logging.getLogger("atlite").setLevel(logging.ERROR)
logging.getLogger("open_mastr").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

nuts2_to_state = config["nuts2_to_state"]
nuts2_to_state = {k[3:]: v[3:] for k, v in nuts2_to_state.items()}


def get_borders_federal(path):
    """Download simple german shape file with cartopy."""

    df = gp.read_file(path)  # only use the simplest polygon
    polygons = df[["NUTS_ID", "geometry"]].set_index("NUTS_ID")

    # match NUTS_ID to state abbreviation
    polygons.index = polygons.index.map(nuts2_to_state)

    polygons["geometry"] = shapely.make_valid(polygons["geometry"])

    return polygons


# cutouts
def create_cutout(path, grid, time, module="meteo", features="all"):
    cutout = at.Cutout(
        path=path,
        module=module,
        x=slice(grid["x"][0] - grid["dx"], grid["x"][1] + grid["dx"]),
        y=slice(grid["y"][0] - grid["dy"], grid["y"][1] + grid["dy"]),
        time=slice(time["start"], time["end"]),
    )

    cutout.prepare(features=features)

    if "expver" in cutout.data.coords:
        cutout.data = cutout.data.reduce(np.nansum, dim="expver", keep_attrs=True)

    return cutout


def select_cutout(cutout, path, grid, time):
    if abs(grid["x"][0] - grid["dx"] - (grid["x"][1] + grid["dx"])) < 0.5:
        a = (cutout.data.x.values < grid["x"][0]) * cutout.data.x.values
        grid["x"][0] = np.max(a[np.nonzero(a)])
        a = (cutout.data.x.values > grid["x"][1]) * cutout.data.x.values
        grid["x"][1] = np.min(a[np.nonzero(a)])

    if abs(grid["y"][0] - grid["dy"] - (grid["y"][1] + grid["dy"])) < 0.5:
        a = (cutout.data.y.values < grid["y"][0]) * cutout.data.y.values
        grid["y"][0] = np.max(a[np.nonzero(a)])
        a = (cutout.data.y.values > grid["y"][1]) * cutout.data.y.values
        grid["y"][1] = np.min(a[np.nonzero(a)])

        cutout = cutout.sel(
            path=path,
            x=slice(grid["x"][0] - grid["dx"], grid["x"][1] + grid["dx"]),
            y=slice(grid["y"][0] - grid["dy"], grid["y"][1] + grid["dy"]),
            time=slice(time["start"], time["end"]),
        )
    return cutout


def renewable_generation(
    technology, local_capacity, shape_file, time, cutout_path, cutout
):
    x1, y1, x2, y2 = shape_file.bounds
    grid = {
        "dx": 0.25,
        "dy": 0.25,
        "x": [x1, x2],
        "y": [y1, y2],
    }

    cutout = select_cutout(cutout=cutout, path=cutout_path, grid=grid, time=time)

    # add identifier bus and state to the data based on the nearest bus of each powerplant
    layout = cutout.layout_from_capacity_list(local_capacity, col="Capacity")

    # calculate generation and capacity factors for 'Solar', 'Onshore' and 'Offshore'
    if technology == "solar":
        generation = cutout.pv(
            panel="CSi",
            orientation={"slope": 30.0, "azimuth": 180.0},
            shapes=[shape_file],
            layout=layout,
            show_progress=False,
        )
    if technology == "wind_onshore":
        generation = cutout.wind(
            turbine="Vestas_V112_3MW",
            shapes=[shape_file],
            layout=layout,
            show_progress=False,
        )
    if technology == "wind_offshore":
        generation = cutout.wind(
            turbine="NREL_ReferenceTurbine_5MW_offshore",
            layout=layout,
            show_progress=False,
        )

    generation = generation.to_dataframe()
    generation = generation.reset_index(level=[1])
    generation = generation.drop("dim_0", axis=1)

    return generation
