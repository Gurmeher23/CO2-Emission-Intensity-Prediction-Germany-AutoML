import logging

import pandas as pd
import xarray as xr
import yaml

import cosema.pypsa_scripts as pf
import netallocation as ntl
from cosema.cross_border_scripts import query_cross_border_flows
from cosema.demand_scripts import query_demand_data, query_reg_demand_data
from cosema.per_type_scripts import query_per_type_data, query_reg_per_type_data
from cosema.prepare_data_scripts import (
    internal_sigma_approach,
    make_balance,
    make_bilateral,
    normalization_F,
    normalization_g_d,
    prepare_cross_border_flows,
    renormalization_F,
    renormalization_g_d,
)

# Set up logging
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "config.yaml"
TECHNOLOGIES_AND_EMISSION_FACTORS_PATH = (
    "inputs/generation_data/gen_types_and_emission_factors.csv"
)
NETWORK_INPUT_PATH = "inputs/networks/elec_s_37"
LEVEL = "DE_federal"


# Load configurations
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config(CONFIG_PATH)
nuts2_to_state = config["nuts2_to_state"]
state_to_nuts2 = {v: k for k, v in nuts2_to_state.items()}

flow_tracing_method = config["flow_tracing_method"]


# Load technologies and emission factors
def load_technologies_and_ef():
    technologies_df = pd.read_csv(TECHNOLOGIES_AND_EMISSION_FACTORS_PATH)
    technologies = technologies_df["entsoe"].unique()

    conv_technologies = (
        technologies_df.drop_duplicates(subset="entsoe")
        .set_index("entsoe")["converted"]
        .to_dict()
    )

    emission_factors_df = pd.read_csv(TECHNOLOGIES_AND_EMISSION_FACTORS_PATH)
    # set index to converted and drop entssoe column
    emission_factors_df = emission_factors_df.set_index("converted").drop(
        columns="entsoe"
    )
    # drop duplicates
    emission_factors_df = emission_factors_df[~emission_factors_df.index.duplicated()]

    return technologies, conv_technologies, emission_factors_df


technologies, conv_technologies, emission_factors_df = load_technologies_and_ef()


def collect_and_prepare_data(start, end, db_client, mode="with_per_unit"):
    # Query generation, demand and crossborder flows from the database
    # (gap filling is done automatically)
    gen_per_country = query_per_type_data(start=start, end=end, db_client=db_client)
    gen_per_country = gen_per_country.fillna(0.0)

    demand_per_country = query_demand_data(start=start, end=end, db_client=db_client)

    cross_border_flows = query_cross_border_flows(
        start=start, end=end, db_client=db_client
    )

    # Format cross_border_flows
    cross_border_flows = prepare_cross_border_flows(cross_border_flows)

    # Query generation and demand data per region
    gen_per_region = query_reg_per_type_data(
        start=start, end=end, db_client=db_client, mode=mode
    )

    demand_per_region = query_reg_demand_data(start=start, end=end, db_client=db_client)

    cross_regions_flow = pd.DataFrame(index=gen_per_region.index)

    # Find indexes where the sum across rows is less than 20000 for both DataFrames
    indexes_to_drop_gen = gen_per_region.index[gen_per_region.sum(axis=1) < 20000]
    indexes_to_drop_demand = demand_per_region.index[
        demand_per_region.sum(axis=1) < 20000
    ]

    # Combine the indexes from both conditions using union to ensure no duplicates
    indexes_to_drop = indexes_to_drop_gen.union(indexes_to_drop_demand)

    if len(indexes_to_drop) == 0:
        logger.debug("All data complete. No need to shorten the data.")
    else:
        logger.warning(
            f"Not all data is complete. Filling all empty rows with 0.0 and shortening the data."
        )
        gen_per_country.loc[indexes_to_drop] = 0.0
        demand_per_country.loc[indexes_to_drop] = 0.0
        cross_border_flows.loc[indexes_to_drop] = 0.0
        gen_per_region.loc[indexes_to_drop] = 0.0
        demand_per_region.loc[indexes_to_drop] = 0.0
        cross_regions_flow.loc[indexes_to_drop] = 0.0

        # also get all zero rows in gen_per_region at the end of the dataframe
        # Identify all-zero rows
        all_zeros = gen_per_region.eq(0).all(axis=1)
        # Find the first valid index
        first_non_zero_from_bottom = gen_per_region[~all_zeros].index[-1]

        # Drop trailing zero rows. Keep rows from the beginning to the first_non_zero_from_bottom (inclusive in original order)
        gen_per_country = gen_per_country.loc[:first_non_zero_from_bottom]
        demand_per_country = demand_per_country.loc[:first_non_zero_from_bottom]
        cross_border_flows = cross_border_flows.loc[:first_non_zero_from_bottom]
        gen_per_region = gen_per_region.loc[:first_non_zero_from_bottom]
        demand_per_region = demand_per_region.loc[:first_non_zero_from_bottom]
        cross_regions_flow = cross_regions_flow.loc[:first_non_zero_from_bottom]

    return (
        gen_per_country,
        demand_per_country,
        cross_border_flows,
        gen_per_region,
        demand_per_region,
        cross_regions_flow,
    )


def balance_data(gen_per_country, demand_per_country, cross_border_flows):
    # Apply correction method to the countries data
    (
        gen_per_country_delta,
        demand_per_country_delta,
        cross_border_flows_delta,
        time_map,
    ) = internal_sigma_approach(
        generation=gen_per_country,
        demand=demand_per_country,
        cross_border_flows=cross_border_flows,
        weights=config["internal_sigma_weights"],
        eta=config["internal_sigma_thresholds"],
    )

    # Create reconciled data set from raw data and correction factors
    (
        gen_per_country_balanced,
        demand_per_country_balanced,
        cross_border_flows_balanced,
    ) = make_balance(
        gen=gen_per_country,
        demand=demand_per_country,
        cross_border_flows=cross_border_flows,
        g_delta=gen_per_country_delta,
        d_delta=demand_per_country_delta,
        F_delta=cross_border_flows_delta,
        t_map=time_map,
    )

    return (
        gen_per_country_balanced,
        demand_per_country_balanced,
        cross_border_flows_balanced,
    )


def refactor_demand_column_names(demand_per_country, demand_per_region):
    cols = []
    for x in demand_per_country.columns:
        cols.append((x, "Demand"))
    demand_per_country.columns = pd.MultiIndex.from_tuples(cols)

    cols = []
    for x in demand_per_region.columns:
        cols.append((state_to_nuts2[x], "Demand"))
    demand_per_region.columns = pd.MultiIndex.from_tuples(cols)

    return demand_per_country, demand_per_region


def aggregate_technologies(gen_per_country, gen_per_region):
    # aggregate technologies for final data set
    gen_per_country.columns = pd.MultiIndex.from_tuples(
        [(x[0], conv_technologies.get(x[1], x[1])) for x in gen_per_country.columns]
    )
    # and merge columns with same technology
    temp = {}
    for country in gen_per_country.columns.get_level_values(0).unique():
        temp[country] = gen_per_country[country].groupby(level=0, axis=1).sum()
    gen_per_country = pd.concat(temp, axis=1)

    # rename columns in all_gen_balanced using the state_to_nuts2 dict
    gen_per_region.columns = pd.MultiIndex.from_tuples(
        [(state_to_nuts2.get(x[0], x[0]), x[1]) for x in gen_per_region.columns]
    )

    # rename technologies for final data set
    gen_per_region.columns = pd.MultiIndex.from_tuples(
        [(x[0], conv_technologies.get(x[1], x[1])) for x in gen_per_region.columns]
    )
    # and merge columns with same technology
    temp = {}
    for region in gen_per_region.columns.get_level_values(0).unique():
        temp[region] = gen_per_region[region].groupby(level=0, axis=1).sum()
    gen_per_region = pd.concat(temp, axis=1)

    return gen_per_country, gen_per_region


def apply_regionalization_factors(
    gen_per_country,
    demand_per_country,
    gen_per_region,
    demand_per_region,
):
    # Calculate Generation and Demand regionalization factors for the regions
    g_reg_norm = normalization_g_d(gen_per_region)
    d_reg_norm = normalization_g_d(demand_per_region)

    # Recalculated Generation and Demand of additional zones based on the regionalization factors from STEP 6
    all_gen = renormalization_g_d(gen_per_country, g_reg_norm)
    all_demand = renormalization_g_d(demand_per_country, d_reg_norm)

    all_gen = all_gen.tz_convert("UTC").tz_localize(None)
    all_demand = all_demand.tz_convert("UTC").tz_localize(None)

    return all_gen, all_demand


def reconcile_cross_border_flows(cross_border_flows, cross_regions_flow):
    # Transform  reconciled unilateral crossborder flows to bilateral net_crossborder flows
    if not cross_regions_flow.empty:
        F_reg_norm = normalization_F(cross_regions_flow)
        all_cross_border_flows = renormalization_F(cross_border_flows, F_reg_norm)
    else:
        print(
            "INFO: No Flow Regionalization is used. Flows in regions are determined by Optimization."
        )
        all_cross_border_flows = cross_border_flows.copy()

    all_cross_border_flows = make_bilateral(all_cross_border_flows)

    all_cross_border_flows = all_cross_border_flows.tz_convert("UTC").tz_localize(None)

    return all_cross_border_flows


def save_balanced_reg_data(gen_df, demand_df, mode, db_client):
    # preprocess generation data
    temp_gen_df = gen_df.copy()
    temp_gen_df.columns = pd.MultiIndex.from_tuples(
        [(nuts2_to_state.get(x[0], x[0]), x[1]) for x in temp_gen_df.columns]
    )
    # get only those columns with DE_ in them
    temp_gen_df = temp_gen_df.filter(regex="DE_")
    # remove DE_ from columns in the first level
    temp_gen_df.columns = pd.MultiIndex.from_tuples(
        [(x[0].replace("DE_", ""), x[1]) for x in temp_gen_df.columns]
    )

    # get technologies and calculate total generation per technology for DE
    technologies = temp_gen_df.columns.get_level_values(1).unique()
    for technology in technologies:
        temp_gen_df[("DE", technology)] = temp_gen_df.filter(regex=technology).sum(
            axis=1
        )

    # convert multiindex to single state_technology column names
    temp_gen_df.columns = temp_gen_df.columns.map("_".join)

    # preprocess demand data
    temp_demand_df = demand_df.copy()
    temp_demand_df.columns = pd.MultiIndex.from_tuples(
        [(nuts2_to_state.get(x[0], x[0]), x[1]) for x in temp_demand_df.columns]
    )

    # get only those columns with DE_ in them
    temp_demand_df = temp_demand_df.filter(regex="DE_")
    # drop second level
    temp_demand_df.columns = temp_demand_df.columns.droplevel(1)
    # remove DE_ from columns
    temp_demand_df.columns = temp_demand_df.columns.str.replace("DE_", "")

    # add DE column as sum of all columns
    temp_demand_df["DE"] = temp_demand_df.sum(axis=1)

    # fill nan values with 0
    temp_gen_df = temp_gen_df.fillna(0)
    temp_demand_df = temp_demand_df.fillna(0)

    # round all values to 0 decimals and convert dtype to int
    temp_gen_df = temp_gen_df.round(0).astype(int)
    temp_demand_df = temp_demand_df.round(0).astype(int)

    # write balanced regionalized data to database
    db_client.write_reg_generation_data(df=temp_gen_df, mode=mode, balanced=True)
    db_client.write_reg_demand_data(df=temp_demand_df, balanced=True)

    # also save balanced data with mode "web_version" for the web application
    # these data will be initially with per_type and will be overwritten with per_unit

    db_client.write_reg_generation_data(
        df=temp_gen_df, mode="web_version", balanced=True
    )


def prepare_and_solve_network(start, end, generation, demand, cross_border_flows):
    # Prepare the network for the optimization
    n_ref = pf.get_reference_network(NETWORK_INPUT_PATH, level=LEVEL)
    countries, regions = pf.get_regions(network=n_ref, subset=[])

    # select emission factors for the year of the start date
    # select previous year if not available
    if f"year_{start.year}" in emission_factors_df.columns:
        emission_factors = emission_factors_df.loc[:, f"year_{start.year}"].to_dict()
    elif f"year_{start.year - 1}" in emission_factors_df.columns:
        emission_factors = emission_factors_df.loc[:, f"year_{start.year - 1}"].to_dict()
    else:
        raise ValueError(
            f"No emission factors available for the year {start.year} or {start.year - 1}."
        )

    network_config = {
        "start": start,
        "end": end,
        "frequency": "H",
        "regions": regions,
        "countries": countries,
        "emission_factors": emission_factors,
    }

    network = pf.prepare_network(
        generation=generation,
        demand=demand,
        flows=cross_border_flows,
        config=network_config,
        n_ref=n_ref,
    )

    # STEP 10: Solve the network with a specifically desigend pypsa transport model and the reconciled data set
    snapshots = network.snapshots
    kwargs = {
        "keep_references": True,
        "keep_shadowprices": True,
        "solver_options": {
            "ResultFile": "model.ilp",
            "method": 2,  # barrier
            "crossover": 0,
            "BarConvTol": 1.0e-4,
            "Seed": 123,
            "AggFill": 0,
            "PreDual": 0,
            "GURO_PAR_BARDENSETHRESH": 200,
            "FeasibilityTol": 1.0e-5,
            "BarHomogeneous": 1,
        },
    }
    try:
        pf.solve_network_transport(
            snapshots=snapshots,
            network=network,
            entsoe_flows=cross_border_flows,
            delta=10**-5,
            **kwargs,
        )
    except Exception as e:
        logger.warning(
            f"Optimization failed with error: {e}. Retrying with a higher delta value."
        )

        pf.solve_network_transport(
            snapshots=snapshots,
            network=network,
            entsoe_flows=cross_border_flows,
            delta=10**-4,
            **kwargs,
        )

    return network, network_config


def run_flow_tracing(network):
    # perform flow tracing using the chosen method
    # this piece of code is prone to failing due to unknown reasons
    # and work fine when run again
    try:
        allocated_flows = ntl.allocate_flow(
            n=network,
            snapshots=network.snapshots,
            to_netcdf=True,
            dims=config["flow_tracing_methods"][flow_tracing_method]["dims"],
            method=config["flow_tracing_methods"][flow_tracing_method]["method"],
            aggregated=config["flow_tracing_methods"][flow_tracing_method][
                "aggregated"
            ],
            sparse=True,
            round=2,
            include_self_consumption=True,
        )
    except Exception as e:
        allocated_flows = ntl.allocate_flow(
            n=network,
            snapshots=network.snapshots,
            to_netcdf=True,
            dims=config["flow_tracing_methods"][flow_tracing_method]["dims"],
            method=config["flow_tracing_methods"][flow_tracing_method]["method"],
            aggregated=config["flow_tracing_methods"][flow_tracing_method][
                "aggregated"
            ],
            sparse=True,
            round=2,
            include_self_consumption=True,
        )

    allocated_flows = ntl.breakdown.by_carriers(allocated_flows, network, chunksize=100)
    allocated_flows = allocated_flows.rename_vars({"peer_to_peer": flow_tracing_method})

    if "branch" in config["flow_tracing_methods"][flow_tracing_method]["dims"]:
        allocated_flows = allocated_flows.rename_vars(
            {"peer_on_branch_to_peer": f"{flow_tracing_method}_branch"}
        )

    allocated_flows = xr.merge([xr.Dataset(), allocated_flows])

    return allocated_flows


def extract_regional_data(allocated_flows, network, network_config):
    # Calculate CO2 signals for each region
    # Get generation based on input data
    generation = allocated_flows[flow_tracing_method]

    # Get co2 emissions based on input emission factors
    emissions = ntl.cost.allocate_carrier_attribute(
        allocated_flows[flow_tracing_method], network, attr="co2_emissions", sparse=True
    )
    emissions.name = flow_tracing_method
    emissions = emissions.transpose(*generation.dims, transpose_coords=True)

    # Get total generation with exports per region
    total_regions_generation_w_export = generation.sel(
        {"source": network_config["regions"]}
    ).sum(["sink", "source_carrier", "sink_carrier"])

    total_regions_generation_w_export = (
        ntl.utils.as_dense(total_regions_generation_w_export)
        .to_dataframe()
        .reset_index()
    )

    total_regions_generation_w_export = total_regions_generation_w_export.pivot(
        columns="source", index="snapshot", values=flow_tracing_method
    )

    # Get total emissions with exports per region
    total_regions_generation_w_export_emissions = emissions.sel(
        {"source": network_config["regions"]}
    ).sum(["sink", "source_carrier", "sink_carrier"])

    total_regions_generation_w_export_emissions = (
        ntl.utils.as_dense(total_regions_generation_w_export_emissions)
        .to_dataframe()
        .reset_index()
    )

    total_regions_generation_w_export_emissions = (
        total_regions_generation_w_export_emissions.pivot(
            columns="source", index="snapshot", values=flow_tracing_method
        )
    )

    # Get total generation including imports per region
    total_regions_generation_w_import = generation.sel(
        {"sink": network_config["regions"]}
    ).sum(["source", "source_carrier", "sink_carrier"])

    # Get total emissions per region
    total_regions_generation_w_import_emissions = emissions.sel(
        {"sink": network_config["regions"]}
    ).sum(["source", "source_carrier", "sink_carrier"])

    # Merge xarray to dataframes
    total_regions_generation_w_import = (
        ntl.utils.as_dense(total_regions_generation_w_import)
        .to_dataframe()
        .reset_index()
    )

    total_regions_generation_w_import_emissions = (
        ntl.utils.as_dense(total_regions_generation_w_import_emissions)
        .to_dataframe()
        .reset_index()
    )

    # Dataframe snapshots as index, regions as columns and imports/ emissions as values
    total_regions_generation_w_import = total_regions_generation_w_import.pivot(
        columns="sink", index="snapshot", values=flow_tracing_method
    )

    total_regions_generation_w_import_emissions = (
        total_regions_generation_w_import_emissions.pivot(
            columns="sink", index="snapshot", values=flow_tracing_method
        )
    )

    return (
        total_regions_generation_w_export,
        total_regions_generation_w_export_emissions,
        total_regions_generation_w_import,
        total_regions_generation_w_import_emissions,
    )


def get_reg_intensities(
    total_consumption_emissions,
    total_consumption,
    total_generation_emissions,
    total_generation,
):
    # Calculate consumption CO2 intensity for each region and write to database
    cons_intensity = total_consumption_emissions / total_consumption

    # Calculate production based CO2 intensity for each region and write to database
    gen_intensity = total_generation_emissions / total_generation

    # rename columns in temp accroding to nuts2_to_state dict
    cons_intensity.columns = cons_intensity.columns.map(nuts2_to_state)
    gen_intensity.columns = gen_intensity.columns.map(nuts2_to_state)

    # remove DE_ from columns
    cons_intensity.columns = cons_intensity.columns.str.replace("DE_", "")
    gen_intensity.columns = gen_intensity.columns.str.replace("DE_", "")

    # localize to UTC
    cons_intensity = cons_intensity.tz_localize("UTC")
    gen_intensity = gen_intensity.tz_localize("UTC")

    return cons_intensity, gen_intensity


def get_country_intensities(
    total_consumption_emissions,
    total_consumption,
    total_generation_emissions,
    total_generation,
):
    # Calculate consumption CO2 intensity for each region and write to database
    cons_intensity = total_consumption_emissions.sum(axis=1) / total_consumption.sum(
        axis=1
    )

    # Calculate production based CO2 intensity for each region and write to database
    gen_intensity = total_generation_emissions.sum(axis=1) / total_generation.sum(
        axis=1
    )

    # localize to UTC
    cons_intensity = cons_intensity.tz_localize("UTC")
    gen_intensity = gen_intensity.tz_localize("UTC")

    return cons_intensity, gen_intensity


def calculate_intensities(start, end, db_client, mode="with_per_unit"):
    # STEP 1: Query generation, demand and crossborder flows from the database
    logger.info("Querying and preparing data from database.")
    (
        gen_per_country,
        demand_per_country,
        cross_border_flows,
        gen_per_region,
        demand_per_region,
        cross_regions_flow,
    ) = collect_and_prepare_data(start=start, end=end, db_client=db_client, mode=mode)

    # get the end date of the data in case it was shortened
    new_end = gen_per_country.index[-1]
    if new_end != end:
        logger.warning(f"Data was shortened. The end date of the data is {new_end}.")
        end = new_end

    # STEP 2: Balance the data
    logger.info("Balancing data.")
    (
        gen_per_country_balanced,
        demand_per_country_balanced,
        cross_border_flows_balanced,
    ) = balance_data(
        gen_per_country=gen_per_country,
        demand_per_country=demand_per_country,
        cross_border_flows=cross_border_flows,
    )

    # STEP 3: Refactor demand column names
    demand_per_country_balanced, demand_per_region = refactor_demand_column_names(
        demand_per_country=demand_per_country_balanced,
        demand_per_region=demand_per_region,
    )

    # STEP 4: Aggregate technologies
    logger.info("Aggregating technologies.")
    gen_per_country_balanced, gen_per_region = aggregate_technologies(
        gen_per_country=gen_per_country_balanced, gen_per_region=gen_per_region
    )

    # STEP 5: Apply regionalization factors
    all_gen_balanced, all_demand_balanced = apply_regionalization_factors(
        gen_per_country=gen_per_country_balanced,
        demand_per_country=demand_per_country_balanced,
        gen_per_region=gen_per_region,
        demand_per_region=demand_per_region,
    )

    # write balanced regionalized data to database
    save_balanced_reg_data(
        gen_df=all_gen_balanced,
        demand_df=all_demand_balanced,
        mode=mode,
        db_client=db_client,
    )

    # STEP 6: Reconcile cross border flows
    all_cross_border_flows_balanced = reconcile_cross_border_flows(
        cross_border_flows=cross_border_flows_balanced,
        cross_regions_flow=cross_regions_flow,
    )

    # STEP 7: Prepare and solve network
    logger.info("Preparing and solving network.")
    network, network_config = prepare_and_solve_network(
        start=start,
        end=end,
        generation=all_gen_balanced,
        demand=all_demand_balanced,
        cross_border_flows=all_cross_border_flows_balanced,
    )

    # STEP 8: Run flow tracing
    logger.info("Running flow tracing.")
    allocated_flows = run_flow_tracing(network=network)

    # STEP 9: Extract regional data
    (
        total_generation,
        total_generation_emissions,
        total_consumption,
        total_consumption_emissions,
    ) = extract_regional_data(
        allocated_flows=allocated_flows, network=network, network_config=network_config
    )

    # STEP 10: Calculate CO2 signals for each region
    reg_cons_intensity, reg_gen_intensity = get_reg_intensities(
        total_consumption_emissions=total_consumption_emissions,
        total_consumption=total_consumption,
        total_generation_emissions=total_generation_emissions,
        total_generation=total_generation,
    )

    # STEP 11: Calculate CO2 signals for Germany
    country_cons_intensity, country_gen_intensity = get_country_intensities(
        total_consumption_emissions=total_consumption_emissions,
        total_consumption=total_consumption,
        total_generation_emissions=total_generation_emissions,
        total_generation=total_generation,
    )

    # Step 12: Add CO2 signals for germany to regionalized data
    reg_cons_intensity["DE"] = country_cons_intensity
    reg_gen_intensity["DE"] = country_gen_intensity

    # fill inf and nan values with 0
    reg_cons_intensity = reg_cons_intensity.fillna(0)
    reg_gen_intensity = reg_gen_intensity.fillna(0)

    # multiply values by 1000 to get gCO2/kWh
    # and round to 0 decimals and convert dtype to int
    reg_cons_intensity = (reg_cons_intensity * 1000).round(0).astype(int)
    reg_gen_intensity = (reg_gen_intensity * 1000).round(0).astype(int)

    # STEP 11: Write CO2 intensities to database
    db_client.write_reg_intensities(
        intensity=reg_cons_intensity, type="consumption", mode=mode
    )
    db_client.write_reg_intensities(
        intensity=reg_gen_intensity, type="production", mode=mode
    )

    # also save intensities with mode "web_version" for the web application
    # these data will be initially with per_type and will be overwritten with per_unit

    db_client.write_reg_intensities(
        intensity=reg_cons_intensity, type="consumption", mode="web_version"
    )

    db_client.write_reg_intensities(
        intensity=reg_gen_intensity, type="production", mode="web_version"
    )

    logger.info(
        f"Intensities calculated and written to database, period {start}-{end}."
    )
