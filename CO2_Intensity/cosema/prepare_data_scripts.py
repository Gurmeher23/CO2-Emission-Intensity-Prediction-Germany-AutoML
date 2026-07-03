import logging

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

logger = logging.getLogger(__name__)


def prepare_cross_border_flows(cross_border_flows):
    """
    Parameters
    ----------
    cross_border_flows :  original cross-border data

    Returns
    -------
    cross_border_flows: pd.dataframe
        Processed cross-border data
    """

    # Create net_flows from the bidirectional flows
    cross_border_flows.iloc[:, 1::2] = -1 * cross_border_flows.iloc[:, 1::2]

    # Net crossborder flows
    cross_border_flows = cross_border_flows.groupby(level=0, axis=1).sum()

    # Remove prefix flow_ and Rename columns <country1->country2>
    cross_border_flows.columns = [
        "_".join(col.split("_")[1:]).replace("-", "->")
        for col in cross_border_flows.columns
    ]

    # Create unidirectional flows from the bidirectional ones
    cross_border_flows = make_unilateral(cross_border_flows)

    return cross_border_flows


def calcualte_weighting_factors(weights, g, d, F):
    A_g = pd.DataFrame(
        weights["default"]["generation"], index=g.columns, columns=["weight"]
    )
    for country, weight in weights["specific"].items():
        g_mask = A_g.index.get_level_values(0).isin([country])
        A_g.loc[A_g.index[g_mask]] = weight["generation"]

    A_d = pd.DataFrame(
        weights["default"]["demand"], index=d.columns, columns=["weight"]
    )
    for country, weight in weights["specific"].items():
        d_mask = A_d.index.get_level_values(0).isin([country])
        A_d.loc[A_d.index[d_mask]] = weight["demand"]

    A_F = pd.DataFrame(weights["default"]["flow"], index=F.columns, columns=["weight"])
    for country, weight in weights["specific"].items():
        priorities = pd.Series(weights["priority"])
        countries_w_higher_priority = priorities[
            priorities < weights["priority"][country]
        ].index

        F_mask = (
            A_F.index.str.split("->").str[0].isin([country])
            & ~(A_F.index.str.split("->").str[1].isin([countries_w_higher_priority]))
        ) | (
            A_F.index.str.split("->").str[1].isin([country])
            & ~(A_F.index.str.split("->").str[0].isin([countries_w_higher_priority]))
        )

        A_F.loc[A_F.index[F_mask]] = weight["flow"]

    return {"generation": A_g, "demand": A_d, "flow": A_F}


def calcualte_threshold_factors(eta, g, d, F):
    T_g = pd.DataFrame(
        eta["default"]["generation"], index=g.columns, columns=["threshold"]
    )
    for country, threshold in eta["specific"].items():
        g_mask = T_g.index.get_level_values(0).isin([country])
        T_g.loc[T_g.index[g_mask]] = threshold["generation"]

    T_d = pd.DataFrame(eta["default"]["demand"], index=d.columns, columns=["threshold"])
    for country, threshold in eta["specific"].items():
        d_mask = T_d.index.get_level_values(0).isin([country])
        T_d.loc[T_d.index[d_mask]] = threshold["demand"]

    T_F = pd.DataFrame(eta["default"]["flow"], index=F.columns, columns=["threshold"])
    for country, threshold in eta["specific"].items():
        priorities = pd.Series(eta["priority"])
        countries_w_higher_priority = priorities[
            priorities < eta["priority"][country]
        ].index

        F_mask = (
            T_F.index.str.split("->").str[0].isin([country])
            & ~(T_F.index.str.split("->").str[1].isin([countries_w_higher_priority]))
        ) | (
            T_F.index.str.split("->").str[1].isin([country])
            & ~(T_F.index.str.split("->").str[0].isin([countries_w_higher_priority]))
        )

        T_F.loc[T_F.index[F_mask]] = threshold["flow"]

    return {"generation": T_g, "demand": T_d, "flow": T_F}


def calculate_internal_sigma(g, d, F, A, T):
    # Create temporaty dataframes
    g_clean = g.copy()
    d_clean = d.copy()
    F_clean = F.copy()

    sigma_g = {}
    # Construct sigma for generation
    for country, df in g_clean.items():
        threshold = T["generation"].at[country, "threshold"]
        weight = A["generation"].at[country, "weight"]
        df_mean = df.rolling("10d").mean()
        df_mean = df_mean.fillna(df)
        df_mean = df_mean.where(~(df_mean < threshold), threshold)
        df_mean = weight / df_mean
        sigma_g[country] = df_mean

    sigma_d = {}
    # Construct sigma for demand
    for country, df in d_clean.items():
        threshold = T["demand"].at[country, "threshold"]
        weight = A["demand"].at[country, "weight"]
        df_mean = df.rolling("10d").mean()
        df_mean = df_mean.fillna(df)
        df_mean = df_mean.where(~(df_mean < threshold), threshold)
        df_mean = weight / df_mean
        sigma_d[country] = df_mean

    # Construct sigma values for cross-border flows

    threshold = T["flow"]["threshold"]
    weight = A["flow"]["weight"]

    F_mean = F_clean.rolling("10d").mean()
    F_mean = F_mean.fillna(F)
    F_mean = F_mean.where(~(F_mean < threshold), threshold, axis=1)
    F_mean = weight / F_mean

    sigma_F = F_mean
    sigma_g = pd.concat(sigma_g, axis=1)
    sigma_d = pd.concat(sigma_d, axis=1)

    return (sigma_g, sigma_d, sigma_F)


def make_balance(gen, demand, cross_border_flows, g_delta, d_delta, F_delta, t_map):
    # Create temporaty dataframes
    g_clean = gen.copy()
    d_clean = demand.copy()
    F_clean = cross_border_flows.copy()
    g_delta_temp = g_delta.copy()
    d_delta_temp = d_delta.copy()
    F_delta_temp = F_delta.copy()

    # Correct original data with deltas
    g_bal = g_clean + (g_delta_temp).set_axis(t_map)
    d_bal = d_clean + (d_delta_temp).set_axis(t_map)
    F_bal = F_clean + (F_delta_temp).set_axis(t_map)

    return g_bal, d_bal, F_bal


def make_bilateral(F_uni):
    F_uni_temp = F_uni.copy()

    borders = F_uni_temp.columns
    borders = pd.Series(
        tuple(sorted(border.split("->"))) for border in borders
    ).unique()

    F_bi = {}
    for border in borders:
        border_e = "->".join([border[0], border[1]])
        border_i = "->".join([border[1], border[0]])
        F_bi[border_e] = F_uni_temp[border_e] - F_uni_temp[border_i]

    F_bi = pd.concat(F_bi, axis=1)

    return F_bi


def make_unilateral(F_bi):
    F_bi_temp = F_bi.copy()

    borders = F_bi_temp.columns
    borders = pd.Series(
        tuple(sorted(border.split("->"))) for border in borders
    ).unique()

    F_uni = {}
    for border in borders:
        border_e = "->".join([border[0], border[1]])
        border_i = "->".join([border[1], border[0]])

        F_bi_temp[border_i] = 0
        F_bi_temp.loc[F_bi_temp[border_e] < 0, border_i] = abs(
            F_bi_temp.loc[F_bi_temp[border_e] < 0, border_e]
        )
        F_bi_temp.loc[F_bi_temp[border_e] < 0, border_e] = 0

        F_uni[border_i] = F_bi_temp[border_i]
        F_uni[border_e] = F_bi_temp[border_e]

    F_uni = pd.concat(F_uni, axis=1)

    return F_uni


def make_country_flow_balance(F_uni):
    F_uni_temp = F_uni.copy()

    countries = pd.Series(
        country for col in F_uni_temp.columns for country in col.split("->")
    ).unique()

    borders = F_uni_temp.columns
    borders = pd.DataFrame(
        [border.split("->") for border in borders], columns=["bus0", "bus1"]
    )

    F_country = {}
    for country in countries:
        border_e = borders.loc[borders["bus0"] == country]
        border_e = border_e["bus0"] + "->" + border_e["bus1"]
        border_i = borders.loc[borders["bus1"] == country]
        border_i = border_i["bus0"] + "->" + border_i["bus1"]

        F_e = -1 * F_uni_temp[border_e]
        F_i = F_uni_temp[border_i]

        F_country[country] = pd.concat([F_e, F_i], axis=1).sum(axis=1)

    F_country = pd.concat(F_country, axis=1)

    return F_country


def normalization_g_d(df):
    df_temp = df.copy()
    label_dict = {
        col: col.split("_")[0] for col in df_temp.columns.get_level_values(0).unique()
    }
    total_df = (
        df_temp.rename(columns=label_dict, level=0).groupby(level=[0, 1], axis=1).sum()
    )
    for col in df_temp.columns:
        country = col[0].split("_")[0]
        technology = col[1]
        technology_mapping = pd.DataFrame(
            df_temp.columns[
                df_temp.columns.get_level_values(1) == technology
            ].to_list(),
            columns=["region", "technology"],
        )
        country_mapping = pd.DataFrame(
            label_dict.items(), columns=["region", "country"]
        )
        mapping = country_mapping[
            country_mapping["region"].isin(technology_mapping["region"])
        ]
        uniform = 1 / mapping.country[mapping["country"] == country].count()
        df_temp[col] = (
            df_temp[col]
            .divide(total_df[country][technology], axis=0)
            .replace([np.inf, -np.inf, np.nan], uniform)
        )
    return df_temp


def normalization_F(df):
    df_temp = df.copy()
    label_dict = {
        col: "->".join(
            [col.split("->")[0].split("_")[0], col.split("->")[1].split("_")[0]]
        )
        for col in df_temp.columns.get_level_values(0).unique()
    }
    df_temp.columns = pd.MultiIndex.from_tuples(
        [(label_dict[col], col) for col in df_temp.columns]
    )
    total_df = df_temp.groupby(level=[0], axis=1).sum()
    for col in df_temp.columns:
        country = col[0]
        mapping = pd.DataFrame(label_dict.items(), columns=["region", "country"])
        uniform = 1 / mapping.country[mapping["country"] == country].count()
        df_temp[col] = (
            df_temp[col]
            .divide(total_df[country], axis=0)
            .replace([np.inf, -np.inf, np.nan], uniform)
        )
    return df_temp


def renormalization_g_d(df, factors):
    df_temp = df.copy()
    factors_temp = factors.copy()

    regions = factors_temp.columns[
        factors_temp.columns.get_level_values(0)
        .str.split("_")
        .str[0]
        .isin(df_temp.columns.get_level_values(0).unique())
    ]
    countries = pd.Series(list(zip(*regions))[0]).str.split("_").str[0].unique()

    df_insert = {}
    for region in regions:
        country = region[0].split("_")[0]
        technology = region[1]
        df_insert[region] = factors_temp[region] * df_temp[country][technology]
    df_insert = pd.concat(df_insert, axis=1)

    for country in countries:
        for technology in df_temp[country].columns:
            country_identifier = (
                df_insert.columns.get_level_values(0)
                .str.split("_")
                .str[0]
                .isin([country])
            )
            technology_identifier = df_insert.columns.get_level_values(1).isin(
                [technology]
            )
            if df_insert[
                df_insert.columns[(country_identifier) & (technology_identifier)]
            ].empty:
                country_regions = (
                    df_insert.columns[country_identifier].get_level_values(0).unique()
                )
                uniform = 1 / country_regions.shape[0]
                for reg in country_regions:
                    df_insert[(reg, technology)] = (
                        uniform * df_temp[country][technology]
                    )

    df_temp = pd.concat([df_temp, df_insert], axis=1)
    df_temp = df_temp.drop(columns=countries, level=0)

    return df_temp


def renormalization_F(df, factors):
    df_temp = df.copy()
    factors_temp = factors.copy()
    flows = factors_temp.columns[
        factors_temp.columns.get_level_values(0)
        .str.split("_")
        .str[0]
        .isin(df_temp.columns)
    ]
    country_flows = list(list(zip(*flows))[0])
    df_insert = {col[1]: factors_temp[col] * df_temp[col[0]] for col in flows}
    df_insert = pd.concat(df_insert, axis=1)
    df_temp = pd.concat([df_temp, df_insert], axis=1)
    df_temp = df_temp.drop(columns=country_flows)
    return df_temp


def get_data(country, generation, demand, flows):
    # Get temporary dataframes
    g_clean = generation.copy()
    d_clean = demand.copy()
    F_clean = flows.copy()

    # Get dataframe data columns
    g_cols = g_clean.columns[
        g_clean.columns.get_level_values(0).str.split("_").str[0] == country
    ]
    d_cols = d_clean.columns[
        d_clean.columns.get_level_values(0).str.split("_").str[0] == country
    ]
    F_cc_cols = F_clean.columns[
        (
            F_clean.columns.get_level_values(0)
            .str.split("->")
            .str[0]
            .str.split("_")
            .str[0]
            == country
        )
        & (
            F_clean.columns.get_level_values(0)
            .str.split("->")
            .str[1]
            .str.split("_")
            .str[0]
            != country
        )
        | (
            F_clean.columns.get_level_values(0)
            .str.split("->")
            .str[0]
            .str.split("_")
            .str[0]
            != country
        )
        & (
            F_clean.columns.get_level_values(0)
            .str.split("->")
            .str[1]
            .str.split("_")
            .str[0]
            == country
        )
    ]

    F_reg_cols = F_clean.columns[
        (
            F_clean.columns.get_level_values(0)
            .str.split("->")
            .str[0]
            .str.split("_")
            .str[0]
            == country
        )
        & (
            F_clean.columns.get_level_values(0)
            .str.split("->")
            .str[1]
            .str.split("_")
            .str[0]
            == country
        )
    ]

    # Calculate generation, demand and residual load
    g_tech = g_clean[g_cols].sum(axis=0)
    d_tech = d_clean[d_cols].sum(axis=0)

    g_total = g_clean[g_cols].groupby(level=0, axis=1).sum()

    g_mix = {}
    for region, tech in g_tech.index:
        g_mix[(region, tech)] = g_clean[(region, tech)] / g_total[region]
    g_mix = pd.concat(g_mix, axis=1).mean(axis=0)

    # Calculate crossboder imports, exports and difference
    F_cc = F_clean[F_cc_cols]
    outgoing_cc_lines = F_cc_cols[
        F_cc_cols.str.split("->").str[0].str.split("_").str[0] == country
    ]
    incoming_cc_lines = F_cc_cols[
        F_cc_cols.str.split("->").str[0].str.split("_").str[0] != country
    ]

    F_cc_i_line = abs(F_cc[F_cc[outgoing_cc_lines] <= 0].sum(axis=0)) + abs(
        F_cc[F_cc[incoming_cc_lines] >= 0].sum(axis=0)
    )
    F_cc_e_line = abs(F_cc[F_cc[incoming_cc_lines] <= 0].sum(axis=0)) + abs(
        F_cc[F_cc[outgoing_cc_lines] >= 0].sum(axis=0)
    )
    F_cc_res_line = F_cc_i_line - F_cc_e_line

    # Calculate region imports, exports and difference
    F_reg = F_clean[F_reg_cols]
    regions = pd.concat(
        [
            pd.Series(
                F_reg_cols.get_level_values(0)
                .str.strip("_backup")
                .str.split("->")
                .str[0]
            ),
            pd.Series(
                F_reg_cols.get_level_values(0)
                .str.strip("_backup")
                .str.split("->")
                .str[1]
            ),
        ]
    ).unique()

    F_reg_i_line = {}
    F_reg_e_line = {}
    for region in regions:
        outgoing_reg_lines = F_reg_cols[F_reg_cols.str.split("->").str[0] == region]
        incoming_reg_lines = F_reg_cols[F_reg_cols.str.split("->").str[1] == region]

        F_reg_i_line[region] = abs(
            F_reg[F_reg[outgoing_reg_lines] <= 0].sum(axis=0)
        ) + abs(F_reg[F_reg[incoming_reg_lines] >= 0].sum(axis=0))
        F_reg_e_line[region] = abs(
            F_reg[F_reg[incoming_reg_lines] <= 0].sum(axis=0)
        ) + abs(F_reg[F_reg[outgoing_reg_lines] >= 0].sum(axis=0))

    try:
        F_reg_i_line = pd.concat(F_reg_i_line, axis=0)
        F_reg_e_line = pd.concat(F_reg_e_line, axis=0)
        F_reg_res_line = F_reg_i_line - F_reg_e_line
    except ValueError:
        F_reg_i_line = abs(F_reg[F_reg[F_reg_cols] <= 0].sum(axis=0))
        F_reg_e_line = abs(F_reg[F_reg[F_reg_cols] >= 0].sum(axis=0))
        F_reg_res_line = F_reg_i_line - F_reg_e_line

    # Build a dataframe from all the results using the unique regions as columns

    # Build dataframe for generation, demand and residual load
    g_tech_labels = []
    for region, tech in g_tech.index:
        g_tech_labels.append((region, tech))
    g_tech.index = pd.MultiIndex.from_tuples(
        g_tech_labels, names=["region", "technology"]
    )

    d_tech_labels = []
    for region, tech in d_tech.index:
        d_tech_labels.append((region, tech))
    d_tech.index = pd.MultiIndex.from_tuples(
        d_tech_labels, names=["region", "technology"]
    )

    g_mix_labels = []
    for region, tech in g_mix.index:
        g_mix_labels.append((region, tech))
    g_mix.index = pd.MultiIndex.from_tuples(
        g_mix_labels, names=["region", "technology"]
    )

    gd_res = g_tech.groupby(level=0).sum() - d_tech.groupby(level=0).sum()
    gd_res_labels = []
    for idx in gd_res.index:
        gd_res_labels.append((idx, "total_residual_load"))
    gd_res.index = pd.MultiIndex.from_tuples(gd_res_labels, names=["region", "value"])

    # Build dataframe for crossborder imports, exports and difference
    F_cc_i_line_labels = []
    for idx in F_cc_i_line.index:
        idx = idx.strip("_backup")
        if idx.split("->")[0].split("_")[0] == country:
            region_to = idx.split("->")[0]
            region_from = idx.split("->")[1]
        else:
            region_to = idx.split("->")[1]
            region_from = idx.split("->")[0]
        F_cc_i_line_labels.append((region_to, region_from))
    F_cc_i_line.index = pd.MultiIndex.from_tuples(
        F_cc_i_line_labels, names=["from", "to"]
    )

    F_cc_e_line_labels = []
    for idx in F_cc_e_line.index:
        idx = idx.strip("_backup")
        if idx.split("->")[0].split("_")[0] == country:
            region_from = idx.split("->")[0]
            region_to = idx.split("->")[1]
        else:
            region_from = idx.split("->")[1]
            region_to = idx.split("->")[0]
        F_cc_e_line_labels.append((region_from, region_to))
    F_cc_e_line.index = pd.MultiIndex.from_tuples(
        F_cc_e_line_labels, names=["from", "to"]
    )

    F_cc_res_line_labels = []
    for idx in F_cc_res_line.index:
        idx = idx.strip("_backup")
        if idx.split("->")[0].split("_")[0] == country:
            region1 = idx.split("->")[0]
            region2 = idx.split("->")[1]
        else:
            region1 = idx.split("->")[1]
            region2 = idx.split("->")[0]
        F_cc_res_line_labels.append((region1, region2))
    F_cc_res_line.index = pd.MultiIndex.from_tuples(
        F_cc_res_line_labels, names=["from", "to"]
    )

    F_cc_i_line = -1 * F_cc_i_line.groupby(level=[0, 1]).sum().sort_index(level=0)
    F_cc_e_line = F_cc_e_line.groupby(level=[0, 1]).sum().sort_index(level=0)
    F_cc_res_line = F_cc_res_line.groupby(level=[0, 1]).sum().sort_index(level=0)

    # Build dataframe for region imports, exports and difference
    F_reg_i_line_labels = []
    for idx in F_reg_e_line.index:
        region_to = idx[1].strip("_backup").split("->")[1]
        if region_to == idx[0]:
            region_from = idx[1].strip("_backup").split("->")[0]
            F_reg_i_line_labels.append((region_to, region_from))
        else:
            region_to = "to_drop"
            region_from = idx[1].strip("_backup").split("->")[1]
            F_reg_i_line_labels.append((region_to, region_from))

    F_reg_i_line.index = pd.MultiIndex.from_tuples(
        F_reg_i_line_labels, names=["from", "to"]
    )
    try:
        F_reg_i_line = F_reg_i_line.drop("to_drop")
    except KeyError:
        pass

    F_reg_e_line_labels = []
    for idx in F_reg_e_line.index:
        region_from = idx[1].strip("_backup").split("->")[0]
        if region_from == idx[0]:
            region_to = idx[1].strip("_backup").split("->")[1]
            F_reg_e_line_labels.append((region_from, region_to))
        else:
            region_from = "to_drop"
            region_to = idx[1].strip("_backup").split("->")[1]
            F_reg_e_line_labels.append((region_from, region_to))

    F_reg_e_line.index = pd.MultiIndex.from_tuples(
        F_reg_e_line_labels, names=["from", "to"]
    )
    try:
        F_reg_e_line = F_reg_e_line.drop("to_drop")
    except KeyError:
        pass

    F_reg_res_line_labels = []
    for idx in F_reg_res_line.index:
        region_from = idx[1].strip("_backup").split("->")[0]
        if region_from == idx[0]:
            region_to = idx[1].strip("_backup").split("->")[1]
            F_reg_res_line_labels.append((region_from, region_to))
        else:
            region_from = "to_drop"
            region_to = idx[1].strip("_backup").split("->")[1]
            F_reg_res_line_labels.append((region_from, region_to))

    F_reg_res_line.index = pd.MultiIndex.from_tuples(
        F_reg_res_line_labels, names=["from", "to"]
    )
    try:
        F_reg_res_line = F_reg_res_line.drop("to_drop")
    except KeyError:
        pass

    F_reg_i_line = -1 * F_reg_i_line.groupby(level=[0, 1]).sum().sort_index(level=0)
    F_reg_e_line = F_reg_e_line.groupby(level=[0, 1]).sum().sort_index(level=0)
    F_reg_res_line = F_reg_res_line.groupby(level=[0, 1]).sum().sort_index(level=0)

    # Write the data into a pivot table using data['col'].reset_index().pivot(index='level_1', columns='level_0', values='data')
    # Generation and demand data
    g_tech = (
        g_tech.reset_index()
        .pivot(index="technology", columns="region", values=0)
        .fillna(0)
    )
    d_tech = (
        d_tech.reset_index()
        .pivot(index="technology", columns="region", values=0)
        .fillna(0)
    )
    gd_res = (
        gd_res.reset_index().pivot(index="value", columns="region", values=0).fillna(0)
    )
    g_mix = (
        g_mix.reset_index()
        .pivot(index="technology", columns="region", values=0)
        .fillna(0)
    )

    # Crossborder flow data
    F_cc_i_line = (
        F_cc_i_line.reset_index().pivot(index="to", columns="from", values=0).fillna(0)
    )
    F_cc_e_line = (
        F_cc_e_line.reset_index().pivot(index="to", columns="from", values=0).fillna(0)
    )
    F_cc_res_line = (
        F_cc_res_line.reset_index()
        .pivot(index="to", columns="from", values=0)
        .fillna(0)
    )

    # Internal flow data
    F_reg_i_line = (
        F_reg_i_line.reset_index().pivot(index="to", columns="from", values=0).fillna(0)
    )
    F_reg_e_line = (
        F_reg_e_line.reset_index().pivot(index="to", columns="from", values=0).fillna(0)
    )
    F_reg_res_line = (
        F_reg_res_line.reset_index()
        .pivot(index="to", columns="from", values=0)
        .fillna(0)
    )

    data = {
        "generation": g_tech,
        "demand": d_tech,
        "residual_load": gd_res,
        "generation_mix": g_mix,
        "crossborder_import": F_cc_i_line,
        "crossborder_export": F_cc_e_line,
        "crossborder_residual": F_cc_res_line,
        "internal_import": F_reg_i_line,
        "internal_export": F_reg_e_line,
        "internal_residual": F_reg_res_line,
    }

    return data


# Internal sigma approach (SIGI)
# Credits go to Chalendar
def internal_sigma_approach(
    generation,
    demand,
    cross_border_flows,
    weights,
    eta,
    solver="gurobi",
):
    # Create temporaty dataframes
    g_temp = generation.copy()
    d_temp = demand.copy()
    F_temp = cross_border_flows.copy()

    # check if any of inputs has nan values and raise a warning
    if g_temp.isnull().values.any():
        logger.warning("Generation data contains NaN values")
        g_temp = g_temp.fillna(0)
    if d_temp.isnull().values.any():
        logger.warning("Demand data contains NaN values")
        d_temp = d_temp.fillna(0)
    if F_temp.isnull().values.any():
        logger.warning("Cross-border flow data contains NaN values")
        F_temp = F_temp.fillna(0)

    # Calculate weighting factors
    A = calcualte_weighting_factors(weights, g_temp, d_temp, F_temp)

    # Calculate threshold factors
    T = calcualte_threshold_factors(eta, g_temp, d_temp, F_temp)

    # Calculate 10-day rolling average for every segment
    sigma_g, sigma_d, sigma_F = calculate_internal_sigma(g_temp, d_temp, F_temp, A, T)
    sigma_g = sigma_g.astype(float)
    sigma_d = sigma_d.astype(float)
    sigma_F = sigma_F.astype(float)

    # check if any of the calculated sigmas has nan values and raise a warning
    if sigma_g.isnull().values.any():
        logger.warning("Sigma for generation data contains NaN values")
    if sigma_d.isnull().values.any():
        logger.warning("Sigma for demand data contains NaN values")
    if sigma_F.isnull().values.any():
        logger.warning("Sigma for cross-border flow data contains NaN values")

    t_map = g_temp.reset_index(drop=False)["index"]

    sigma_g = sigma_g.reset_index(drop=True)
    g_temp = g_temp.reset_index(drop=True)

    sigma_d = sigma_d.reset_index(drop=True)
    d_temp = d_temp.reset_index(drop=True)

    sigma_F = sigma_F.reset_index(drop=True)
    F_temp = F_temp.reset_index(drop=True)

    # INDEX WORK
    countries = g_temp.columns.get_level_values(0).unique()
    # Index for time steps

    time = list(g_temp.index)

    # Production type indices
    tech_index = {f"{country}": list(g_temp[country].columns) for country in countries}

    # Index for all production types
    technologies = np.array(
        list(set([item for sublist in tech_index.values() for item in sublist]))
    )
    technologies.sort()

    # Prepare the country-dependent indices
    # Direction import
    import_index = {
        f"{country}": F_temp.columns[F_temp.columns.str.split("->").str[1] == country]
        for country in countries
    }
    # Direction export
    export_index = {
        f"{country}": F_temp.columns[F_temp.columns.str.split("->").str[0] == country]
        for country in countries
    }
    # Index for all links
    flows = np.array(F_temp.columns)
    flows.sort()

    # PYOMO WORK
    model = pyo.ConcreteModel()

    # Index declartions
    model.countries = pyo.Set(initialize=countries)
    model.time = pyo.Set(initialize=time)
    model.technologies = pyo.Set(initialize=technologies)
    model.flows = pyo.Set(initialize=flows)

    # Variable declarations
    model.delta_g = pyo.Var(
        model.countries, model.time, model.technologies, bounds=(0.0, None)
    )
    model.delta_d = pyo.Var(model.countries, model.time, bounds=(0.0, None))
    model.delta_F = pyo.Var(model.time, model.flows, bounds=(0.0, None))

    # Rule Definition
    def balance_rule(model, country, t):
        return sum(
            model.delta_g[country, t, tech] + g_temp[country].at[t, tech]
            for tech in tech_index[country]
        ) + sum(
            model.delta_F[t, flow] + F_temp.at[t, flow]
            for flow in import_index[country]
        ) == model.delta_d[
            country, t
        ] + d_temp.at[
            t, country
        ] + sum(
            model.delta_F[t, flow] + F_temp.at[t, flow]
            for flow in export_index[country]
        )

    def ObjRule(model):
        return (
            sum(
                model.delta_g[country, t, tech] ** 2 * sigma_g[country].at[t, tech]
                for country in model.countries
                for t in model.time
                for tech in tech_index[country]
            )
            + sum(
                model.delta_d[country, t] ** 2 * sigma_d.at[t, country]
                for country in model.countries
                for t in model.time
            )
            + sum(
                model.delta_F[t, flow] ** 2 * sigma_F.at[t, flow]
                for t in model.time
                for flow in model.flows
            )
        )

    model.balance_rule = pyo.Constraint(model.countries, model.time, rule=balance_rule)
    model.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize)

    opt = SolverFactory(solver, solver_io="python")

    # Solve the model
    opt.solve(model)

    # Convert the correction variables (deltas) into data frames
    g_delta = {}
    d_delta = {}
    for country in countries:
        df_g = {}
        df_d = {}
        for t in time:
            row_g = {}
            for tech in tech_index[country]:
                row_g[tech] = model.delta_g[country, t, tech].value
            df_g[t] = pd.Series(row_g)
            df_d[t] = model.delta_d[country, t].value

        g_delta[country] = pd.concat(df_g, axis=1)
        d_delta[country] = pd.Series(df_d)

    g_delta = pd.concat(g_delta).T
    d_delta = pd.concat(d_delta, axis=1)

    F_delta = {}
    for l in flows:
        df = {}
        for t in time:
            df[t] = model.delta_F[t, l].value

        F_delta[l] = pd.Series(df)

    F_delta = pd.concat(F_delta, axis=1)

    return g_delta, d_delta, F_delta, t_map
