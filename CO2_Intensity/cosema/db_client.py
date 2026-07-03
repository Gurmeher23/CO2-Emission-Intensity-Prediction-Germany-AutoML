"""
This file contains the class that writes the results of the simulation to a database or to CSV files.

Implemented by Nick Harder, University of Freiburg, 2023

"""
import logging

logger = logging.getLogger(__name__)

import pandas as pd
from influxdb import DataFrameClient, InfluxDBClient

from cosema.gap_filling_scripts import (
    cross_border_rules_unilateral,
    default_rules,
    find_gaps,
    germany_rules,
)

ALLOW_NEGATIVE_GENERATION = pd.read_csv(
    "inputs/generation_data/gen_types_and_emission_factors.csv"
)
ALLOW_NEGATIVE_GENERATION = list(
    ALLOW_NEGATIVE_GENERATION[ALLOW_NEGATIVE_GENERATION["is_storage"]][
        "entsoe"
    ].unique()
)


class DBClient:
    def __init__(
        self,
        database_name,
        host="localhost",
        port=8086,
        username="root",
        password="root",
        time_zone="UTC",
    ):
        self.time_zone = time_zone

        self.client = InfluxDBClient(
            host=host,
            port=8086,
            username=username,
            password=password,
        )
        if database_name not in self.client.get_list_database():
            self.client.create_database(database_name)

        self.db_client = DataFrameClient(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database_name,
        )

    def write_df(self, df: pd.DataFrame, measurement: str, tags: dict):
        self.db_client.write_points(
            dataframe=df, measurement=measurement, tags=tags, protocol="line"
        )

    def query_per_type_gen(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        technologies: list,
        country: str = "DE",
        fill_gaps: bool = True,
        return_gap_info: bool = False,
        resample: str = "1h",
    ):
        measurement = "per_type_gen"
        per_type_gen = pd.DataFrame()
        orig_start = start

        # for gap filling we require the previous weeks' data as well
        if fill_gaps:
            start = start - pd.Timedelta(weeks=2)

        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        for technology in technologies:
            query = f'select "Generation [MW]" FROM {measurement} WHERE "country" = \'{country}\' AND "technology" = \'{technology}\' AND time >= {start_in_s}s AND time <= {end_in_s}s'
            temp_df = self.db_client.query(query)
            if len(temp_df) == 0:
                continue
            temp_df = temp_df[measurement]
            temp_df = temp_df.rename(columns={"Generation [MW]": technology})
            per_type_gen = pd.concat([per_type_gen, temp_df], axis=1)

        # convert time zone and fill missing rows with nan
        per_type_gen.index = per_type_gen.index.tz_convert(self.time_zone)
        per_type_gen = per_type_gen.reindex(
            pd.date_range(
                start=start, end=end, freq=pd.infer_freq(per_type_gen.index[:3])
            )
        )

        # fill gaps if requested
        if fill_gaps:
            per_type_gen, gap_dict = find_gaps(
                per_type_gen,
                check_negatives=True,
                allow_negatives=ALLOW_NEGATIVE_GENERATION,
                fill_gaps=True,
                gap_filling_rules=(germany_rules if country == "DE" else default_rules),
            )

        # resample data if requested
        if (resample is not None) and (resample is not False):
            per_type_gen = per_type_gen.resample(resample).mean()

        # trim additional data if necessary
        per_type_gen = per_type_gen[orig_start:end]

        if fill_gaps and return_gap_info:
            return per_type_gen, gap_dict
        else:
            return per_type_gen

    def query_per_unit_gen(self, start: pd.Timestamp, end: pd.Timestamp):
        measurement = "per_unit_gen"
        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        query = f'select "Generation [MW]" FROM {measurement} WHERE time >= {start_in_s}s AND time <= {end_in_s}s GROUP BY *'
        per_unit_gen = self.db_client.query(query)

        return per_unit_gen

    def check_per_unit_data(self, start: pd.Timestamp, end: pd.Timestamp):
        measurement = "per_unit_gen"
        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        query = f'select "Generation [MW]" FROM {measurement} WHERE "technology" = \'Hydro Run-of-river and poundage\' AND time >= {start_in_s}s AND time <= {end_in_s}s'
        gen = self.db_client.query(query)
        gen = gen[measurement].dropna()
        gen.index = gen.index.tz_convert(self.time_zone)

        return gen.index.unique()

    def query_vre(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        technology: str = "solar",
        state: str = "BW",
        value: str = "Generation [MW]" or "reg_factor",
        mode: str = "historical" or "forecast",
    ):
        measurement = "vre_gen" if mode == "historical" else "vre_forecast"
        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        query = f'select mean("{value}") FROM {measurement} WHERE "state" = \'{state}\' AND "technology" = \'{technology}\' AND time >= {start_in_s}s AND time <= {end_in_s}s GROUP BY time(1h)'
        vre_cf = self.db_client.query(query)
        vre_cf = vre_cf[measurement]
        vre_cf.index = vre_cf.index.tz_convert(self.time_zone)

        return vre_cf

    def query_demand_data(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        country: str = "DE",
        mode: str = "historical" or "forecast",
        fill_gaps: bool = True,
        return_gap_info: bool = False,
        resample: str = "1h",
    ):
        measurement = "demand" if mode == "historical" else "demand_forecast"
        orig_start = start

        # for gap filling we require the previous weeks' data as well
        if fill_gaps:
            start = start - pd.Timedelta(weeks=2)

        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        query = f'select "Demand [MW]" FROM {measurement} WHERE "country" = \'{country}\' AND time >= {start_in_s}s AND time <= {end_in_s}s'
        demand = self.db_client.query(query)
        demand = demand[measurement]

        # convert time zone and fill missing rows with nan
        demand.index = demand.index.tz_convert(self.time_zone)
        demand = demand.reindex(
            pd.date_range(start=start, end=end, freq=pd.infer_freq(demand.index[:3]))
        )

        # fill gaps if requested
        if fill_gaps:
            demand, gap_dict = find_gaps(
                demand,
                check_negatives=True,
                fill_gaps=True,
                gap_filling_rules=(germany_rules if country == "DE" else default_rules),
            )

        # resample data if requested
        if (resample is not None) and (resample is not False):
            demand = demand.resample(resample).mean()

        # trim additional data if necessary
        demand = demand[orig_start:end]

        if fill_gaps and return_gap_info:
            return demand, gap_dict
        else:
            return demand

    def query_reg_gen_data(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        state: str = "DE",
        mode: str = "with_per_unit",
        technologies: list = None,
        column_name: str = "Generation [MW]",
        balanced: bool = False,
    ):
        measurement = "reg_generation"
        balanced = "yes" if balanced else "no"

        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        per_type_gen = pd.DataFrame()

        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        for technology in technologies:
            query = f'select sum("{column_name}") FROM {measurement} WHERE ("state" = \'{state}\' AND "technology" = \'{technology}\' AND "mode" = \'{mode}\' AND "balanced" = \'{balanced}\') AND time >= {start_in_s}s AND time <= {end_in_s}s GROUP BY time(1h)'
            temp_df = self.db_client.query(query)
            if len(temp_df) == 0:
                continue
            temp_df = temp_df[measurement]
            temp_df = temp_df.rename(columns={"sum": technology})
            per_type_gen[technology] = temp_df[technology]

        per_type_gen.index = per_type_gen.index.tz_convert(self.time_zone)

        return per_type_gen

    def query_reg_demand_data(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        state: str = "DE",
        column_name: str = "Demand [MW]",
        balanced: bool = False,
    ):
        measurement = "reg_demand"
        balanced = "yes" if balanced else "no"

        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        query = f'select sum("{column_name}") FROM {measurement} WHERE ("state" = \'{state}\' AND "balanced" = \'{balanced}\' AND time >= {start_in_s}s) AND time <= {end_in_s}s GROUP BY time(1h)'
        reg_demand = self.db_client.query(query)
        reg_demand = reg_demand[measurement]
        reg_demand.index = reg_demand.index.tz_convert(self.time_zone)

        return reg_demand

    def query_intensities(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        state: str = "DE",
        emission_type: str = "consumption" or "production",
        mode: str = "with_per_unit" or "forecast",
    ):
        measurement = "co2_intensity"
        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        query = f'select mean("Intensity [t/MWh]") FROM {measurement} WHERE "state" = \'{state}\' AND "type" = \'{emission_type}\' AND "mode" = \'{mode}\' AND time >= {start_in_s}s AND time <= {end_in_s}s GROUP BY time(1h)'
        intensity = self.db_client.query(query)
        intensity = intensity[measurement]
        intensity.index = intensity.index.tz_convert(self.time_zone)

        return intensity

    def query_cross_border_flows(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        country_code_from: str,
        country_code_to: str,
        fill_gaps: bool = True,
        return_gap_info: bool = False,
        resample: str = "1h",
    ):
        measurement = "cross_border_flow"
        orig_start = start

        # for gap filling we require the previous weeks' data as well
        if fill_gaps:
            start = start - pd.Timedelta(weeks=2)

        start_in_s = start.value // 10**9
        end_in_s = end.value // 10**9

        query = f'select "Flow [MW]" FROM {measurement} WHERE "from" = \'{country_code_from}\' AND "to" = \'{country_code_to}\' AND time >= {start_in_s}s AND time <= {end_in_s}s'
        flows = self.db_client.query(query)
        flows = flows[measurement]

        # convert time zone and fill missing rows with nan
        flows.index = flows.index.tz_convert(self.time_zone)
        flows = flows.reindex(
            pd.date_range(start=start, end=end, freq=pd.infer_freq(flows.index[:3]))
        )

        # fill gaps if requested
        if fill_gaps:
            flows, gap_dict = find_gaps(
                flows,
                check_negatives=True,
                fill_gaps=True,
                gap_filling_rules=cross_border_rules_unilateral,
            )

        # resample data if requested
        if (resample is not None) and (resample is not False):
            flows = flows.resample(resample).mean()

        # trim additional data if necessary
        flows = flows[orig_start:end]

        if fill_gaps and return_gap_info:
            return flows, gap_dict
        else:
            return flows

    def write_vre_data(
        self,
        df: pd.DataFrame,
        technology: str,
        state: str,
        mode: str = "historical" or "forecast",
        column_name: str = "Generation [MW]" or "reg_factor",
    ):
        measurement = "vre_gen" if mode == "historical" else "vre_forecast"

        tempDF = pd.DataFrame(
            index=df.index,
            columns=[column_name],
            data=df.values,
        ).astype("float32")

        self.write_df(
            df=tempDF,
            measurement=measurement,
            tags={"state": state, "technology": technology},
        )

    def write_reg_intensities(
        self,
        intensity: pd.DataFrame,
        type: str = "generation" or "consumption",
        mode="only_per_type" or "with_per_unit" or "forecast",
    ):
        intensity.index = intensity.index.tz_convert("UTC")

        for state, content in intensity.items():
            tempDF = pd.DataFrame(
                index=content.index, columns=["Intensity [g/kWh]"], data=content.values
            ).astype("int")
            self.write_df(
                df=tempDF,
                measurement="co2_intensity",
                tags={"type": type, "state": state, "mode": mode},
            )

    def write_reg_demand_data(
        self,
        df,
        column_name="Demand [MW]",
        balanced=False,
        extra_tags=None,
    ):
        balanced = "yes" if balanced else "no"
        dtype = "int" if balanced else "float32"

        for state in df:
            tempDF = pd.DataFrame(
                index=df.index,
                columns=[column_name],
                data=df[state].values,
            ).astype(dtype)

            tags = {"state": state, "balanced": balanced}
            if extra_tags is not None:
                tags.update(extra_tags)

            self.write_df(
                df=tempDF,
                measurement="reg_demand",
                tags=tags,
            )

    def write_reg_generation_data(
        self,
        df,
        mode,
        column_name="Generation [MW]",
        balanced=False,
        extra_tags=None,
    ):
        balanced = "yes" if balanced else "no"
        dtype = "int" if balanced else "float32"

        for column in df:
            state, technology = column.split("_")
            tempDF = pd.DataFrame(
                index=df.index,
                columns=[column_name],
                data=df[column].values,
            ).astype(dtype)

            tags = {
                "state": state,
                "technology": technology,
                "mode": mode,
                "balanced": balanced,
            }
            if extra_tags is not None:
                tags.update(extra_tags)

            self.write_df(
                df=tempDF,
                measurement="reg_generation",
                tags=tags,
            )

    # Function to delete databases and simulations
    def delete_series(self, tags: dict = None, measurement: str = None):
        print(f"You are about to delete measurement {measurement} with tags: {tags}")
        reply = input('Are you sure? Type "yes" or "y" to confirm:')
        if reply.lower() in ["yes", "y"]:
            self.db_client.delete_series(measurement=measurement, tags=tags)
            print(f"Measurement {measurement}, tags {tags} deleted")
        else:
            print("Ok, not deleted !")

    # Function to delete intervals
    def delete_interval(self, measurement: str, start: pd.Timestamp, end: pd.Timestamp):
        print(
            f"You are about to delete measurement {measurement} between {start} and {end}"
        )
        reply = input('Are you sure? Type "yes" or "y" to confirm:')
        if reply.lower() in ["yes", "y"]:
            start_s = start.value // 10**9
            end_s = end.value // 10**9

            query = f"DELETE FROM {measurement} WHERE time >= {start_s}s AND time <= {end_s}s"
            self.db_client.query(query)
            print(f"Measurement {measurement} between {start} and {end} deleted")
        else:
            print("Ok, not deleted !")
