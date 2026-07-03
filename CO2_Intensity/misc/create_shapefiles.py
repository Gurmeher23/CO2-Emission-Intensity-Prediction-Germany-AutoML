#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:01:53 2024

@author: tim
"""

from itertools import combinations

import geopandas as gpd
import numpy as np
import shapely
from longsgis import voronoiDiagram4plg
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

# %% Function to create Island Free Polygons


def remove_islands(poly, crs, crs_dist):
    if isinstance(poly, shapely.Polygon):
        return poly

    polygons = gpd.GeoSeries(poly.geoms, crs=crs)
    polygons = polygons.to_crs(crs_dist)

    # now build geodataframe limited by biggest area polygons
    polygons = (
        gpd.GeoDataFrame(
            data={"area": polygons.area / 10**6}, geometry=polygons, crs=polygons.crs
        )
        .sort_values("area", ascending=0)
        .loc[lambda d: d["area"].gt(100)]
        # .head(3) # only one tasmania will be excluded
        # .loc[lambda d: d["area"].gt(5000)]  # filter by size of polygon
        .reset_index()
    )

    polygons = polygons.to_crs(crs)
    main_polygon = polygons["geometry"][0]

    return main_polygon


def remove_interiors(poly):
    """
    Close polygon holes by limitation to the exterior ring.

    Arguments
    ---------
    poly: shapely.geometry.Polygon
        Input shapely Polygon

    Returns
    ---------
    Polygon without any interior holes
    """
    if poly.interiors:
        return shapely.Polygon(list(poly.exterior.coords))
    else:
        return poly


def pop_largest(gs):
    """
    Pop the largest polygon off of a GeoSeries

    Arguments
    ---------
    gs: geopandas.GeoSeries
        Geoseries of Polygon or MultiPolygon objects

    Returns
    ---------
    Largest Polygon in a Geoseries
    """
    geoms = [g.area for g in gs]
    return gs.pop(geoms.index(max(geoms)))


def close_holes(geom):
    """
    Remove holes in a polygon geometry

    Arguments
    ---------
    gseries: geopandas.GeoSeries
        Geoseries of Polygon or MultiPolygon objects

    Returns
    ---------
    Largest Polygon in a Geoseries
    """
    if isinstance(geom, shapely.MultiPolygon):
        ser = gpd.GeoSeries([remove_interiors(g) for g in geom.geoms])
        big = pop_largest(ser)
        outers = ser.loc[~ser.within(big)].tolist()
        if outers:
            return shapely.MultiPolygon([big] + outers)
        return shapely.Polygon(big)
    if isinstance(geom, shapely.Polygon):
        return remove_interiors(geom)


def merge_polygons(geoms, crs):
    merged_geom = gpd.GeoSeries(shapely.unary_union(geoms), crs=crs)
    return merged_geom[0]


def remove_feature(poly, bounds, crs):
    if isinstance(poly, shapely.Polygon):
        return poly

    polygons = gpd.GeoSeries(poly.geoms, crs=crs)
    polygons = polygons[~(polygons.bounds <= bounds).all(axis=1)]

    main_polygon = merge_polygons(polygons, crs=crs)

    return main_polygon


def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the polygon
    `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py.

    Attributes
    ----------
    points : Nx2 - ndarray[dtype=float]
    outline : Polygon
    Returns
    -------
    polygons : N - ndarray[dtype=Polygon|MultiPolygon]
    """
    points = np.asarray(points)

    if len(points) == 1:
        polygons = [outline]
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(
            np.vstack(
                (
                    points,
                    [
                        [xmin - 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmin - 3.0 * xspan, ymax + 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymax + 3.0 * yspan],
                    ],
                )
            )
        )

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            with np.errstate(invalid="ignore"):
                poly = poly.intersection(outline)

            polygons.append(poly)

    return polygons


# %% Main Script

crs = "EPSG:4326"
crs_dist = "EPSG:3035"

nuts3 = gpd.read_file(
    "/home/tim/Documents/timfuermann/code/github/CO2_Intensity/inputs/vre/nuts3/NUTS_RG_10M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp"
)
eez = gpd.read_file(
    "/home/tim/Documents/timfuermann/code/github/CO2_Intensity/inputs/vre/eez/eez.shp"
)

nuts3 = nuts3.to_crs(crs)
eez = eez.to_crs(crs)

nuts1_DE = nuts3.loc[(nuts3["LEVL_CODE"] == 1) & (nuts3["CNTR_CODE"].isin(["DE"]))]
nuts1_DE = nuts1_DE.to_crs(crs)

nuts1_DE = nuts1_DE.set_index("NUTS_ID")
nuts1_DE = nuts1_DE[["NUTS_NAME", "geometry"]]

eez_DE = eez[["iso_ter1", "geometry"]]
eez_DE = eez_DE.rename(columns={"iso_ter1": "NUTS_ID"})
eez_DE = eez_DE.set_index("NUTS_ID")
eez_DE.loc["DEU", "geometry"] = close_holes(eez_DE.loc["DEU", "geometry"])


nuts1_DE.loc["DEF", "geometry"] = merge_polygons(
    nuts1_DE.loc[["DEF", "DE6"], "geometry"], crs=crs
)
nuts1_DE.loc["DE9", "geometry"] = merge_polygons(
    nuts1_DE.loc[["DE9", "DE5"], "geometry"], crs=crs
)
nuts1_DE.loc["DE4", "geometry"] = merge_polygons(
    nuts1_DE.loc[["DE4", "DE3"], "geometry"], crs=crs
)

nuts1_DE = nuts1_DE.drop(["DE3", "DE5", "DE6"])

# These are the shapefiles for the webpage and contain all island an no maritim areas
nuts1_DE_webpage = nuts1_DE.copy()

# Continue with adding maritim areas
onshore_states = gpd.GeoDataFrame(
    nuts1_DE.loc[["DEF", "DE8", "DE9"], "geometry"]
).sort_index()
onshore_states.geometry = onshore_states.geometry.apply(
    remove_islands, crs=crs, crs_dist=crs_dist
)
onshore_states.geometry = onshore_states.geometry.apply(lambda p: close_holes(p))

offshore_voronois = voronoiDiagram4plg(onshore_states, eez_DE)
offshore_voronois.index = ["DE9", "DE8", "DEF"]

for idx in onshore_states.index:
    geometries = [nuts1_DE.loc[idx].geometry, offshore_voronois.loc[idx].geometry]
    geometry = merge_polygons(geometries, crs=crs)
    geometry = remove_islands(geometry, crs=crs, crs_dist=crs_dist)
    geometry = close_holes(geometry)

    nuts1_DE.loc[idx, "geometry"] = geometry


nuts1_DE_webpage = nuts1_DE_webpage.sort_index()
nuts1_DE = nuts1_DE.sort_index()

nuts1_DE_webpage = nuts1_DE_webpage.to_crs(crs_dist)
nuts1_DE = nuts1_DE.to_crs(crs_dist)

# add a buffer of really small distance
nuts1_DE_webpage.geometry = nuts1_DE_webpage.geometry.buffer(0.001)
nuts1_DE.geometry = nuts1_DE.geometry.buffer(0.001)

# remove overlaps https://gis.stackexchange.com/questions/370521/erase-overlapping-qgis-polygon-using-geopandas-within-jupyter
poly = nuts1_DE_webpage.geometry
for p1_idx, p2_idx in combinations(poly.index, 2):
    if poly.loc[p1_idx].intersects(poly.loc[p2_idx]):
        # Store intermediary results back to poly
        poly.loc[p2_idx] -= poly.loc[p1_idx]

poly = nuts1_DE.geometry
for p1_idx, p2_idx in combinations(poly.index, 2):
    if poly.loc[p1_idx].intersects(poly.loc[p2_idx]):
        # Store intermediary results back to poly
        poly.loc[p2_idx] -= poly.loc[p1_idx]

nuts1_DE = nuts1_DE.to_crs(crs)

# Scale down the resolution of the map for the webpage to 100m accuracy
nuts1_DE_webpage.geometry = nuts1_DE_webpage.geometry.simplify(100)
nuts1_DE_webpage = nuts1_DE_webpage.to_crs(crs)

nuts1_DE.to_file("states_for_geomatching.geojson", driver="GeoJSON")
nuts1_DE_webpage.to_file("states_for_webpage.geojson", driver="GeoJSON")
