# %%
import itertools

import numpy as np
import pandas as pd
import pypsa
from pypsa.linopt import (
    _str_array,
    broadcasted_axes,
    define_constraints,
    get_var,
    linexpr,
    to_pandas,
    write_objective,
)


def get_regions(network, subset=None):
    if subset is None:
        subset = []
    # Get the country identfier and from there all unique countries
    countries = pd.Series(country.split("_")[0] for country in network.buses.country)
    countries = pd.Series(countries.unique())

    regions = network.buses.country[~network.buses.country.isin(countries)].unique()

    if len(subset) > 0:
        regions = regions[regions.isin(subset)]

    return (countries.to_list(), regions.tolist())


def get_crossborders(network, subset=None):
    if subset is None:
        subset = []
    # Get all crossborder (country) lines and links and from there all unique borders
    links = pd.DataFrame(
        [pd.Series(network.links.bus0), pd.Series(network.links.bus1)]
    ).T
    lines = pd.DataFrame(
        [pd.Series(network.lines.bus0), pd.Series(network.lines.bus1)]
    ).T

    connections = pd.concat([lines, links])

    mask = (
        connections["bus0"].str.split("_").str[0]
        != connections["bus1"].str.split("_").str[0]
    )

    borders = connections["bus0"] + "->" + connections["bus1"]

    if len(subset) > 0:
        subset = pd.Series(itertools.product(subset, subset))
        borders = borders[borders.isin(subset)]

    country_borders = (
        connections[mask]["bus0"].str.split("_").str[0]
        + "->"
        + connections[mask]["bus1"].str.split("_").str[0]
    )
    country_borders = country_borders.unique()

    region_borders = borders[mask].reset_index(drop=True)
    region_borders = region_borders[~region_borders.isin(country_borders)].unique()

    country_borders = pd.Series(
        tuple(sorted(country_border.split("->"))) for country_border in country_borders
    )
    country_borders = pd.Series(country_borders)

    region_borders = pd.Series(
        tuple(sorted(region_border.split("->"))) for region_border in region_borders
    )
    region_borders = pd.Series(region_borders)

    return (country_borders, region_borders)


def get_reference_network(network_path, level=None):
    return pypsa.Network(f"{network_path}_{level}.nc")


# Create new network based on the reference network and entsoe data
def prepare_network(generation, demand, flows, config, n_ref):
    # Copy original_data and work on that
    generation = generation.copy()
    demand = demand.copy()
    flows = flows.copy()

    # Create empty Network
    network = pypsa.Network()
    # Set snapshots based on config file and convert to UTC
    network.set_snapshots(
        pd.date_range(
            start=config["start"], end=config["end"], freq=config["frequency"]
        )
        .tz_convert("UTC")
        .tz_localize(None)
    )

    # Attach all buses from the reference network
    buses_ref = n_ref.buses[
        n_ref.buses.index.isin(config["regions"])
        | n_ref.buses.index.isin(config["countries"])
    ]
    buses_ref_country = [country.split("_")[0] for country in buses_ref.country]
    network.madd(
        "Bus",
        buses_ref.index,
        country=buses_ref_country,
        **buses_ref.drop("country", axis=1),
    )

    # Get lines and links from the reference network
    lines_ref = n_ref.lines[
        (n_ref.lines.bus0.isin(buses_ref.index))
        & (n_ref.lines.bus1.isin(buses_ref.index))
    ]
    lines_ref = lines_ref.rename(
        columns={
            "s_nom": "p_nom",
            "s_max_pu": "p_max_pu",
            "s_min_pu": "p_min_pu",
            "s_nom_max": "p_nom_max",
            "s_nom_min": "p_nom_min",
            "s_nom_extendable": "p_nom_extendable",
        }
    )
    lines_ref["under_construction"] = False
    lines_ref.loc[:, "x"] = (
        n_ref.lines.length
        * (n_ref.line_types.loc[n_ref.lines.type].x_per_length).values
    )

    links_ref = n_ref.links[
        (n_ref.links.bus0.isin(buses_ref.index))
        & (n_ref.links.bus1.isin(buses_ref.index))
    ]

    links_ref.loc[:, "num_parallel"] = 1
    links_ref.loc[:, "x"] = 1

    variables_of_interest = [
        "bus0",
        "bus1",
        "num_parallel",
        "length",
        "x",
        "p_nom",
        "capital_cost",
        "under_construction",
    ]

    connections_ref = pd.concat([lines_ref, links_ref])[variables_of_interest]
    connections_ref[["bus0", "bus1"]] = (
        connections_ref[["bus0", "bus1"]].apply(sorted, axis=1).to_list()
    )

    # Add identifier for crossborder links
    crossborder_connections = pd.DataFrame(
        [pd.Series(connections_ref.bus0), pd.Series(connections_ref.bus1)]
    ).T
    crossborder_connections = crossborder_connections[
        crossborder_connections["bus0"].str.split("_").str[0]
        != crossborder_connections["bus1"].str.split("_").str[0]
    ]

    connections_ref["crossborder_id"] = False
    connections_ref.loc[crossborder_connections.index, "crossborder_id"] = True

    # Add identifier for backup links
    connections_ref["backup_id"] = False

    # Add borders and country_borders
    borders = connections_ref["bus0"] + "->" + connections_ref["bus1"]
    country_borders = (
        connections_ref["bus0"].str.split("_").str[0]
        + "->"
        + connections_ref["bus1"].str.split("_").str[0]
    )

    connections_ref["index"] = borders
    connections_ref["border"] = borders
    connections_ref["country_border"] = country_borders

    # Add identifier for links with flows from entsoe
    connections_ref["entsoe_id"] = False
    connections_ref.loc[
        connections_ref["border"].isin(flows.columns)
        | connections_ref["country_border"].isin(flows.columns),
        "entsoe_id",
    ] = True

    # Add identifier for links with flows from entsoe, but missing in reference network
    connections_ref["missing_entsoe_id"] = False

    connections_ref = connections_ref.groupby("index").agg(
        {
            "border": "first",
            "country_border": "first",
            "bus0": "first",
            "bus1": "first",
            "x": "sum",
            "p_nom": "sum",
            "length": "sum",
            "num_parallel": "sum",
            "capital_cost": "sum",
            "under_construction": "prod",
            "backup_id": "first",
            "entsoe_id": "first",
            "missing_entsoe_id": "first",
            "crossborder_id": "first",
        }
    )

    # Sort flows and reference connections alphabetically
    flows = flows.sort_index(axis=1)
    connections_ref = connections_ref.sort_index(axis=0)

    # Add field with maxium balanced crossborder flow value for all entsoe lines within reference network
    connections_ref["max_cc_flow_entsoe"] = np.nan
    cols = flows.columns[
        (
            flows.columns.isin(connections_ref.index)
            | flows.columns.isin(connections_ref["country_border"])
        )
    ]
    for col in cols:
        bool_mask = connections_ref.index.isin([col]) | connections_ref[
            "country_border"
        ].isin([col])
        connections_ref.loc[bool_mask, "max_cc_flow_entsoe"] = abs(flows[col].max())

    # Attach reference connections to network
    network.madd(
        "Link",
        names=connections_ref.index,
        p_nom_extendable=False,
        p_min_pu=-1,
        p_max_pu=1,
        **connections_ref,
        axis=1,
    )

    # Get all flows from ENTSOE that have a network connection in the reference network and
    # add them with zero capacity and zero resistance
    # shortcut: line_flows_entsoe_without_reference (lF_E_w_ref)

    lF_E_w_ref = flows.columns[
        ~(
            flows.columns.isin(connections_ref.index)
            | flows.columns.isin(connections_ref["country_border"])
        )
    ]

    for flow in lF_E_w_ref:
        bus0 = flow.split("->")[0]
        bus1 = flow.split("->")[1]
        border = bus0 + "->" + bus1

        bool_mask = (
            (len(bus0.split("_")) > 1)
            & (len(bus1.split("_")) > 1)
            & (bus0.split("_")[0] == bus1.split("_")[0])
        )
        if bool_mask:
            crossborder_id = False
        else:
            crossborder_id = True

        network.add(
            "Link",
            name=border,
            bus0=bus0,
            bus1=bus1,
            p_max_pu=1,
            p_min_pu=-1,
            p_nom=0,
            p_nom_extendable=False,
            carrier="AC/DC",
            marginal_cost=1,
        )

        network.links.loc[border, "x"] = 0
        network.links.loc[border, "length"] = np.nan
        network.links.loc[border, "num_parallel"] = 1
        network.links.loc[border, "capital_cost"] = np.nan
        network.links.loc[border, "under_construction"] = 1
        network.links.loc[border, "border"] = border
        network.links.loc[border, "country_border"] = (
            bus0.split("_")[0] + "->" + bus1.split("_")[0]
        )
        network.links.loc[border, "max_cc_flow_entsoe"] = abs(flows[flow]).max()
        network.links.loc[border, "backup_id"] = False
        network.links.loc[border, "entsoe_id"] = True
        network.links.loc[border, "missing_entose_id"] = True
        network.links.loc[border, "crossborder_id"] = crossborder_id

    # Attach backup link capacities with high cost for all network links
    network.madd(
        "Link",
        names=network.links.index,
        suffix="_backup",
        p_nom=10**6,
        p_nom_extendable=False,
        p_min_pu=-1,
        p_max_pu=1,
        capital_cost=0,
        x=10**6,
        **network.links.drop(
            ["p_nom", "capital_cost", "p_max_pu", "p_min_pu", "p_nom_extendable", "x"],
            axis=1,
        ),
    )

    network.links.loc[network.links.index.str.contains("_backup"), "backup_id"] = True

    # Attach generation data
    # rename columns to remove suffix _generation
    generation.columns = ["_".join(col) for col in generation.columns]
    generation = generation.groupby(generation.columns, axis=1).sum()

    # Mask for required generation data
    mask = []
    for gen in generation.columns:
        if gen.split("_")[0] in network.buses.country.unique():
            mask.append(gen)

    # Remove generation data that is not required in network file
    generation = generation[mask]

    # Split negative and positiv generation to account for storage units as loads properly
    positive_generation = generation[generation > 0].fillna(0)
    negative_generation = generation[generation <= 0].fillna(0)

    # Normalize generation and negative generation to use it as kind of capacity factor
    positive_generation_normalized = (
        positive_generation / positive_generation.max()
    ).fillna(0)
    negative_generation_normalized = (
        negative_generation / negative_generation.min()
    ).fillna(0)

    # Remove all Entries with only positiv or negative values
    positive_generation_normalized = positive_generation_normalized[
        positive_generation_normalized.columns[
            positive_generation_normalized.max() > 0.0
        ]
    ]
    negative_generation_normalized = negative_generation_normalized[
        negative_generation_normalized.columns[
            negative_generation_normalized.max() > 0.0
        ]
    ]

    positive_generation = positive_generation[positive_generation_normalized.columns]

    negative_generation = negative_generation[negative_generation_normalized.columns]

    # Generation Carriers
    positive_gen_carrier = pd.Series(
        positive_generation.columns.str.split("_").str[-1:].str[0]
    )
    negative_gen_carrier = (
        pd.Series(negative_generation.columns.str.split("_").str[-1:].str[0])
        + " "
        + "Demand"
    )

    # Filter for the buses where to attach the generation with (generation.columns.str.split("_").str[:-1].str[0])
    positive_generation_buses = (
        positive_generation.columns.str.split("_").str[:-1].str.join("_")
    )
    negative_generation_buses = (
        negative_generation.columns.str.split("_").str[:-1].str.join("_")
    )

    # network.madd('Load', names = generation.columns, bus = bus_generation, p_set = generation, sign = 1, carrier=carrier)
    network.madd(
        "Generator",
        names=positive_generation.columns,
        bus=positive_generation_buses,
        p_nom=positive_generation.max(),
        p_max_pu=positive_generation_normalized,
        p_min_pu=positive_generation_normalized,
        carrier=positive_gen_carrier.values,
    )

    # Add load from storage units
    network.madd(
        "Load",
        names=negative_generation.columns,
        bus=negative_generation_buses,
        p_set=abs(negative_generation),
        carrier=negative_gen_carrier.values,
    )

    # Attach carrier data
    network.madd("Carrier", names=pd.Series(positive_gen_carrier).unique())
    network.carriers["co2_emissions"] = config["emission_factors"]

    # Attach storage carrier data
    network.madd("Carrier", names=pd.Series(negative_gen_carrier).unique())

    # Attach demand data
    # Add missing buses to demand as colums with value 0
    for bus in network.buses.index:
        if bus not in demand.columns.get_level_values(0).unique():
            demand[(bus, "demand")] = 0

    # Remove demand data that is not required in network file
    demand = demand[
        list(
            set(network.buses.index).intersection(
                set(demand.columns.get_level_values(0))
            )
        )
    ]
    demand_countries = demand.columns.get_level_values(0)
    demand.columns = demand_countries

    demand_carrier = pd.Series("Load", index=demand.columns)

    network.madd(
        "Load",
        names=demand_countries,
        suffix="_demand",
        p_set=demand,
        carrier=demand_carrier.values,
        **n_ref.loads.loc[demand_countries].drop(["p_set", "carrier"], axis=1),
    )

    return network


def _quadexpr(*tuples, as_pandas=True, return_axes=False):
    """
    Elementwise concatenation of tuples in the form (coefficient, variables).
    Coefficient and variables can be arrays, series or frames. Per default
    returns a pandas.Series or pandas.DataFrame of strings. If return_axes is
    set to True the return value is split into values and axes, where values
    are the numpy.array and axes a tuple containing index and column if
    present.

    Parameters
    ----------
    tuples: tuple of tuples
        Each tuple must of the form (coeff, var), where
        * weighting is a numerical  value, or a numerical array, series, frame
        * coeff is a numerical  value, or a numerical array, series, frame
        * var is a str or a array, series, frame of variable strings
    as_pandas : bool, default True
        Whether to return to resulting array as a series, if 1-dimensional, or
        a frame, if 2-dimensional. Supersedes return_axes argument.
    return_axes: Boolean, default False
        Whether to return index and column (if existent)

    Example
    -------
    Initialize coefficients and variables

    >>> coeff1 = 1
    >>> var1 = pd.Series(['a1', 'a2', 'a3'])
    >>> coeff2 = pd.Series([-0.5, -0.3, -1])
    >>> var2 = pd.Series(['b1', 'b2', 'b3'])

    Create the linear expression strings

    >>> linexpr((coeff1, var1), (coeff2, var2))
    0    +1.0 a1 -0.5 b1
    1    +1.0 a2 -0.3 b2
    2    +1.0 a3 -1.0 b3
    dtype: object

    For a further step the resulting frame can be used as the lhs of
    :func:`pypsa.linopt.define_constraints`

    For retrieving only the values:

    >>> linexpr((coeff1, var1), (coeff2, var2), as_pandas=False)
    array(['+1.0 a1 -0.5 b1', '+1.0 a2 -0.3 b2', '+1.0 a3 -1.0 b3'], dtype=object)
    """
    axes, shape = broadcasted_axes(*tuples)
    expr = np.repeat("", np.prod(shape)).reshape(shape).astype(object)
    if np.prod(shape):
        for weighting, cost, var in tuples:
            newexpr = (
                _str_array(weighting)
                + " [ "
                + _str_array(cost)
                + " x"
                + _str_array(var, True)
                + " * "
                + "x"
                + _str_array(var, True)
                + " ]"
                + "\n"
            )
            if isinstance(expr, np.ndarray):
                isna = (
                    np.isnan(weighting) | np.isnan(cost) | np.isnan(var) | (var == -1)
                )
                newexpr = np.where(isna, "", newexpr)
            expr = expr + newexpr
    if return_axes:
        return (expr, *axes)
    return to_pandas(expr, *axes) if as_pandas else expr


# Solve the flow pattern of a network based on its generation, demand and crossborder flow balance
# The problem contains a quadratic objectiv function
def solve_network_transport(
    snapshots, network, entsoe_flows, delta=0.001, solver="gurobi", **kwargs
):
    # new objective function to allow power flow calculation based on generation load balance
    def add_objective(network, snapshots):
        # snapshot weightings
        weighting = pd.concat(
            [network.snapshot_weightings.objective.loc[snapshots]]
            * len(network.links.index),
            axis=1,
        )
        weighting.columns = network.links.index

        # Add susceptance as optimization cost
        cost = weighting * network.links.x

        terms = _quadexpr(
            (weighting, cost, get_var(network, "Link", "p").loc[snapshots])
        )

        write_objective(network, terms)

    # additional constraint to cuarantee power flows over parallel lines do not exceed given flow
    def add_flow_constraint(network, snapshots, entsoe_flows, delta):
        # Limit all network links with the flows given by Entsoe
        # All other flows are unconstrained and only taken care of by the optimization
        entsoe_links = network.links[network.links["entsoe_id"]]

        # Limit all links that have unique ENTSOE flows with the corresponding ENTSOE flow
        # This is necessary in case the specific line flow is known
        unique_links = entsoe_links[
            entsoe_links.index.str.split("_backup").str[0].isin(entsoe_flows.columns)
        ]
        unique_lhs = (
            linexpr(
                (1, get_var(network, "Link", "p").loc[snapshots, unique_links.index])
            )
            .groupby(unique_links["border"], axis=1)
            .sum()
        )
        unique_flows = entsoe_flows.loc[snapshots, unique_lhs.columns]

        # Limit all links that have a non-unique ENTSOE flow with the sum of the corresponding ENTSOE flow
        # This is necessary in case the total flow is known, but not the specific flow over each line
        nonunique_links = entsoe_links[~entsoe_links.index.isin(unique_links.index)]
        nonunique_lhs = (
            linexpr(
                (1, get_var(network, "Link", "p").loc[snapshots, nonunique_links.index])
            )
            .groupby(nonunique_links["country_border"], axis=1)
            .sum()
        )
        nonunique_flows = entsoe_flows.loc[snapshots, nonunique_lhs.columns]

        # Merge the additionally constrained links and its left hand side into a dataframe
        flows = pd.concat([unique_flows, nonunique_flows], axis=1)
        lhs = pd.concat([unique_lhs, nonunique_lhs], axis=1)

        # Limit all links for which ENTSOE data exists with an upper and lower flow limit with a certain delta for accuracy
        define_constraints(
            network,
            lhs,
            "<=",
            flows[lhs.columns] + abs(delta * flows[lhs.columns]),
            "Link",
            "lower_flow_limit",
        )
        define_constraints(
            network,
            lhs,
            ">=",
            flows[lhs.columns] - abs(delta * flows[lhs.columns]),
            "Link",
            "upper_flow_limit",
        )

    def extra_functionality(network, snapshots):
        add_objective(network, snapshots)
        add_flow_constraint(network, snapshots, entsoe_flows.loc[snapshots], delta)

    status, termination_condition = network.lopf(
        snapshots,
        solver_name=solver,
        skip_objective=True,
        extra_functionality=extra_functionality,
        **kwargs,
    )

    return status, termination_condition
