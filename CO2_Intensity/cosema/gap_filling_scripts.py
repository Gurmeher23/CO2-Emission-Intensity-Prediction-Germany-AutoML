"""
Script for gap filling in pandas DataFrame objects.

Authors: Ramiz Qussous, Robin L. Grether
"""

import numpy as np
import pandas as pd


def default_rules(series: pd.Series, gaps: pd.DataFrame, inferred_freq: pd.Timedelta):
    # use zero as fallback and for negative values
    gaps["method"] = "ZERO"

    # use week before for larger gaps
    MAX_WEEK_BEFORE = pd.Timedelta(weeks=1)
    gaps.loc[
        (gaps["type"] == "nan")
        & (gaps["duration"] * inferred_freq <= MAX_WEEK_BEFORE)
        & (
            gaps["start"] - series.index[0] >= MAX_WEEK_BEFORE
        ),  # ensure there exists a week before to fill with
        "method",
    ] = "WEEK_BEFORE"

    # use linear interpolation for small gaps
    MAX_LINEAR = pd.Timedelta(hours=3)
    gaps.loc[
        (gaps["type"] == "nan")
        & (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] > series.index[0])
        & (gaps["end"] < series.index[-1]),  # ensure we are not on the edge
        "method",
    ] = "LINEAR"

    # use forward fill for edge gap at the end
    gaps.loc[
        (gaps["type"] == "nan")
        & (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] > series.index[0])
        & (gaps["end"] == series.index[-1]),
        "method",
    ] = "FORWARD_FILL"

    # use backward fill for edge gap in the beginning
    gaps.loc[
        (gaps["type"] == "nan")
        & (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] == series.index[0])
        & (gaps["end"] < series.index[-1]),
        "method",
    ] = "BACKWARD_FILL"


# only linear interpolation for internal gaps up to 3h, no gap filling for edge gaps and longer gaps
def germany_rules(series: pd.Series, gaps: pd.DataFrame, inferred_freq: pd.Timedelta):
    # use zero as fallback and for negative values
    gaps["method"] = "ZERO"

    # use linear interpolation for small gaps
    MAX_LINEAR = pd.Timedelta(hours=3)
    gaps.loc[
        (gaps["type"] == "nan")
        & (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] > series.index[0])
        & (gaps["end"] < series.index[-1]),  # ensure we are not on the edge
        "method",
    ] = "LINEAR"


def cross_border_rules_unilateral(
    series: pd.Series, gaps: pd.DataFrame, inferred_freq: pd.Timedelta
):
    """
    Use for unilateral cross-border flows, i.e., when a single border is represented by two series of data for the two flow directions.

    ATTENTION: These rules will fill all negative values with zero!
    """
    # use zero as fallback and for negative values
    gaps["method"] = "ZERO"

    # use linear interpolation for small gaps
    MAX_LINEAR = pd.Timedelta(hours=3)
    gaps.loc[
        (gaps["type"] == "nan")
        & (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] > series.index[0])
        & (gaps["end"] < series.index[-1]),  # ensure we are not on the edge
        "method",
    ] = "LINEAR"

    # use forward fill for edge gap at the end
    gaps.loc[
        (gaps["type"] == "nan")
        & (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] > series.index[0])
        & (gaps["end"] == series.index[-1]),
        "method",
    ] = "FORWARD_FILL"

    # use backward fill for edge gap in the beginning
    gaps.loc[
        (gaps["type"] == "nan")
        & (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] == series.index[0])
        & (gaps["end"] < series.index[-1]),
        "method",
    ] = "BACKWARD_FILL"


def cross_border_rules_bilateral(
    series: pd.Series, gaps: pd.DataFrame, inferred_freq: pd.Timedelta
):
    """
    Use for bilateral cross-border flows, i.e., when a single border is represented by one series of data with +/- sign indicating the flow direction.
    """
    # use zero as fallback
    gaps["method"] = "ZERO"

    # use linear interpolation for small gaps
    MAX_LINEAR = pd.Timedelta(hours=3)
    gaps.loc[
        (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] > series.index[0])
        & (gaps["end"] < series.index[-1]),  # ensure we are not on the edge
        "method",
    ] = "LINEAR"

    # use forward fill for edge gap at the end
    gaps.loc[
        (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] > series.index[0])
        & (gaps["end"] == series.index[-1]),
        "method",
    ] = "FORWARD_FILL"

    # use backward fill for edge gap in the beginning
    gaps.loc[
        (gaps["duration"] * inferred_freq <= MAX_LINEAR)
        & (gaps["start"] == series.index[0])
        & (gaps["end"] < series.index[-1]),
        "method",
    ] = "BACKWARD_FILL"

    # make sure negative values are kept for bilateral flows!
    gaps.loc[gaps["type"] == "negative"] = "UNDEFINED"


def find_gaps(
    df: pd.DataFrame,
    check_negatives: bool = False,
    allow_negatives: list = [],
    fill_gaps: bool = False,
    gap_filling_rules: callable = default_rules,
):
    """
    Find (and optionally fill) gaps in a pandas DataFrame with DatetimeIndex. Finding and filling works per Series (i.e. per column of data).

        Parameters:
            df (pd.DataFrame): The data.
            check_negatives (bool): If true, negative values are treated as gaps.
            allow_negatives (list): A list of series names for which negative values are not to be treated as gaps. (Only applicable if check_negatives is True.)
            fill_gaps (bool): If true, the gaps are filled on the fly; if false, the gaps are only analyzed.
            gap_filling_rules (callable): The callable that shall be used to determine the gap filling method;
                                          must have the same parameter structure as default_rules().

        Returns:
            (pd.DataFrame): The data. (Optionally: gap-filled)
            (dict): The dictionary with information about the gaps. (Optionally: info about the values filled)
    """
    output_dict = {}
    df = df.apply(
        find_gaps_series,
        axis=0,
        output_dict=output_dict,
        check_negatives=check_negatives,
        allow_negatives=allow_negatives,
        fill_gaps=fill_gaps,
        gap_filling_rules=gap_filling_rules,
    )
    return df, output_dict


def find_gaps_series(
    series: pd.Series,
    output_dict: dict = None,
    check_negatives: bool = False,
    allow_negatives: list = [],
    fill_gaps: bool = False,
    gap_filling_rules: callable = None,
):
    """
    Find (and optionally fill) gaps in a pandas Series object.

        Parameters:
            Series (pd.Series): The data.
            output_dict (dict): The dictionary in which to put info about the gaps found (and filled). Pass None to deactivate.
            check_negatives (bool): If true, negative values are treated as gaps.
            allow_negatives (list): A list of series names for which negative values are not to be treated as gaps. (Only applicable if check_negatives is True.)
            fill_gaps (bool): If true, the gaps are filled on the fly; if false, the gaps are only analyzed.
            gap_filling_rules (callable): The callable that shall be used to determine the gap filling method;
                                          must have the same parameter structure as default_rules().

        Returns:
            (pd.Series): The data. (Optionally: gap-filled)
    """
    # create dataframe, so we can work with the data
    # df_series = pd.DataFrame(series)

    # -----------------------------------
    # Find actual gaps (i.e., NaN values)
    # -----------------------------------

    # check if any of the values is larger than 50000 and replace with nan
    series = series.where(series < 100000, np.nan)

    # find mask of nan values
    is_nan = series.isna()

    # identify start and end of gaps
    gap_starts = is_nan & (~is_nan.shift(1, fill_value=False))
    gap_ends = is_nan & (~is_nan.shift(-1, fill_value=False))

    # create DataFrame with gaps
    gaps = pd.DataFrame(
        {"start": series[gap_starts].index, "end": series[gap_ends].index}
    )

    # add column for duration of gaps
    gaps["duration"] = gaps.apply(
        lambda row: is_nan[row["start"] : row["end"]].sum(),
        axis=1,
        result_type="reduce",
    ).astype("int")

    # add column for aggregated value
    gaps["value"] = np.nan

    # add column for gap type
    gaps["type"] = "nan"

    if check_negatives and (str(series.name) not in allow_negatives):
        # --------------------
        # Find negative values
        # --------------------

        # find mask of negative values
        is_neg = series < 0

        # identify start and end of negatives
        neg_starts = is_neg & (~is_neg.shift(1, fill_value=False))
        neg_ends = is_neg & (~is_neg.shift(-1, fill_value=False))

        # create DataFrame with negatives
        negs = pd.DataFrame(
            {"start": series[neg_starts].index, "end": series[neg_ends].index}
        )

        # add column for duration of negs
        negs["duration"] = negs.apply(
            lambda row: is_neg[row["start"] : row["end"]].sum(),
            axis=1,
            result_type="reduce",
        ).astype("int")

        # add column for aggregated value
        negs["value"] = negs.apply(
            lambda row: series[row["start"] : row["end"]].sum(),
            axis=1,
            result_type="reduce",
        )

        # add column for type
        negs["type"] = "negative"

        # concat with nan gaps
        gaps = pd.concat([gaps, negs])

        # sort gaps
        gaps = gaps.sort_values(by="start")

        # re-index
        gaps = gaps.reset_index(drop=True)

    # infer frequency
    inferred_freq = pd.infer_freq(series.index[:3])
    if (inferred_freq is not None) and (len(inferred_freq) == 1):
        inferred_freq = "1" + inferred_freq
        # put an extra '1' at the front if we only have a unit symbol

    # convert inferred frequency to timedelta
    inferred_freq = pd.to_timedelta(inferred_freq)

    # by default, we do not set any gap filling
    gaps["method"] = "UNDEFINED"

    # set gap filling rules
    if gap_filling_rules is not None:
        gap_filling_rules(series, gaps, inferred_freq)

    # fill gaps if required
    if fill_gaps:
        series, gaps = fill_gaps_series(series, gaps)

    # add gap info if required
    if output_dict is not None:
        output_dict[series.name] = gaps

    # return series data
    return series


def fill_gaps_series(series: pd.Series, gaps: pd.DataFrame):
    """
    Fill gaps in a pandas Series object.

        Parameters:
            series (pd.Series): The data.
            gaps (pd.DataFrame): Gap information for the given data series as returned by find_gaps().

        Returns:
            (pd.Series): The gap-filled data.
            (pd.DataFrame): Gap information with gap filling output.
    """

    # add output columns
    gaps["success"] = False
    gaps["filled_values"] = 0
    gaps["filled_quantity"] = 0.0

    # Iterate over the gaps and fill them using the appropriate method according to duration
    for i, gap in gaps.iterrows():
        # input data about gap
        start = gap["start"]
        end = gap["end"]
        duration = gap["duration"]
        method = gap["method"]

        # ---------------------------------
        # Apply set filling method for gaps
        # ---------------------------------

        if method == "ZERO":
            # fill gap with zeros
            series.loc[start:end] = 0

        elif method == "LINEAR":
            # find precursor and successor positions
            pos_start = series.index.get_loc(start)
            pos_precursor = pos_start - 1
            pos_successor = pos_start + duration

            # fill gap with linear interpolation between precursor and successor
            series.loc[start:end] = np.linspace(
                series.iloc[pos_precursor], series.iloc[pos_successor], duration + 2
            )[1:-1]

        elif method == "FORWARD_FILL":
            # find precursor
            pos_start = series.index.get_loc(start)
            pos_precursor = pos_start - 1

            # fill gap with precursor value
            series.loc[start:end] = series.iloc[pos_precursor]

        elif method == "BACKWARD_FILL":
            # find successor
            pos_start = series.index.get_loc(start)
            pos_successor = pos_start + duration

            # fill gap with successor value
            series.loc[start:end] = series.iloc[pos_successor]

        elif method == "WEEK_BEFORE":
            # ------------------------------------------------------------------------------
            # ATTENTION: This can result in gap-filled data be used for filling another gap!
            # ------------------------------------------------------------------------------

            # get positions in week before
            one_week = pd.Timedelta(weeks=1)
            week_before_start = start - one_week
            week_before_end = end - one_week

            # fill gap with data from week before
            series.loc[start:end] = series.loc[week_before_start:week_before_end].values

        elif method == "WEEK_AFTER":
            # get positions in week after
            one_week = pd.Timedelta(weeks=1)
            week_after_start = start + one_week
            week_after_end = end + one_week

            # fill gap with data from week after
            series.loc[start:end] = series.loc[week_after_start:week_after_end].values

        elif method == "RANDOM":
            # ------------------------------------------------------------------------------
            # ATTENTION: Use this method only if you are absolutely sure what you are doing!
            # ------------------------------------------------------------------------------

            # determine min and max in series
            value_min = series.min()
            value_max = series.max()

            # fill gap with random data
            series.loc[start:end] = (
                np.random.random_sample(size=duration) * (value_max - value_min)
                + value_min
            )

        elif method == "SELF_DESTRUCTION":
            # -----------------------------------------------
            # ATTENTION: This method is only for the bravest!
            # -----------------------------------------------

            import time

            print("Initiating SELF_DESTRUCTION gap filling protocol...\n")
            time.sleep(2)  # Dramatic pause

            for i in range(3, 0, -1):
                print(f"Self-destruction in {i}...")
                time.sleep(1)

            print("\nJust kidding! :D Filling gaps with zero instead.")

            # fill gap with zeros
            series.loc[start:end] = 0

        # elif method == 'ARIMA':
        # Potentially in the future: Add ARIMA-based gap filling.

        # output data
        filled_values = series.loc[start:end].count()
        filled_quantity = series.loc[start:end].sum()
        success = filled_values > 0

        gaps.loc[i, "success"] = success
        gaps.loc[i, "filled_values"] = filled_values
        gaps.loc[i, "filled_quantity"] = filled_quantity

    return series, gaps


def evaluate_gap_filling(
    gen_per_country_gaps, demand_per_country_gaps, cross_border_flows_gaps, start
):
    # evaluate and output infos about gap filling
    gap_keys = []
    gap_numbers = []
    gap_filled_values = []
    gap_filled_quantity = []

    for gap_dict in [
        gen_per_country_gaps,
        demand_per_country_gaps,
        cross_border_flows_gaps,
    ]:
        for key in gap_dict.keys():
            gaps = gap_dict[key]
            gap_keys.append(key)
            gap_numbers.append(len(gaps[gaps["end"] >= start]))
            gap_filled_values.append(gaps[gaps["end"] >= start]["filled_values"].sum())
            gap_filled_quantity.append(
                gaps[gaps["end"] >= start]["filled_quantity"].sum()
            )

    gap_evaluation = pd.DataFrame(
        {
            "Data Set": gap_keys,
            "Number of Gaps": gap_numbers,
            "Number of values filled": gap_filled_values,
            "Quantity filled": gap_filled_quantity,
        }
    )

    return gap_evaluation
