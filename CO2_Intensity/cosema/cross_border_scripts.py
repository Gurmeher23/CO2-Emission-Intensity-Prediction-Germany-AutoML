import logging
import traceback

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def query_cross_border_flows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_client,
    country_borders: list = None,
):
    if country_borders is None:
        country_borders = config["country_borders"]

    cross_border_flows = {}
    for border in country_borders:
        country_code_from, country_code_to = border.split("-")
        try:
            flow_from = db_client.query_cross_border_flows(
                country_code_from=country_code_from,
                country_code_to=country_code_to,
                start=start,
                end=end,
            )
            flow_to = db_client.query_cross_border_flows(
                country_code_from=country_code_to,
                country_code_to=country_code_from,
                start=start,
                end=end,
            )
        except Exception as e:
            logger.warning(
                f"No data for {country_code_from}-{country_code_to} for period {start} - {end}. Skipping..."
            )
            continue

        # concat flows and flows_opposite
        flows = pd.concat([flow_from, flow_to], axis=1)
        # set correct column names
        flows.columns = [
            f"{country_code_from}->{country_code_to}",
            f"{country_code_to}->{country_code_from}",
        ]

        cross_border_flows[f"flows_{country_code_from}-{country_code_to}"] = flows

    cross_border_flows = pd.concat(cross_border_flows, axis=1)

    return cross_border_flows
