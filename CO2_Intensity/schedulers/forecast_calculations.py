import argparse
import logging
import time
from datetime import datetime

import pandas as pd
import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from entsoe import EntsoePandasClient

from cosema.calc_forecast import forecast_intensities
from cosema.calc_vre import run_vre_calculations
from cosema.db_client import DBClient
from cosema.download_scripts import download_demand_forecast_data
from cosema.loggers import get_handlers

handlers = get_handlers(log_path="logs/forecast.log")
logging.basicConfig(level=logging.INFO, handlers=handlers)

logger = logging.getLogger(__name__)

# how much time to go back in time to download data
INITIAL_DATE_OFFSET = pd.Timedelta("1h")
# time delta for the calculations
TIME_DELTA = pd.Timedelta("48h")
# when to start the scheduler for the first time
FIRST_RUN_OFFSET = pd.Timedelta("1h")
# how often to run the scheduler [in hours]
TASK_INTERVAL = 12


def init_date(initial_date):
    global timestep
    timestep = initial_date
    logger.info(f"Scheduler initial date set to {timestep} ({timestep.tzinfo})")


def perform_forecast(entsoe_client, db_client, time_delta, interval):
    global timestep

    start = timestep.floor("d")
    end = start + time_delta
    end = end.ceil("d")
    timestep += interval

    # download_demand_forecast_data(
    #     start=start, end=end, entsoe_client=entsoe_client, db_client=db_client
    # )
    run_vre_calculations(start=start, 
                         end=end, 
                         db_client=db_client, 
                         mode="forecast", 
                         level="federal",
                         overwrite=True,
    )

    # forecast_intensities(start=start, end=end, db_client=db_client)

    logger.info("Operation sucessfull")
    logger.info(f"Scheduler time advanced to {timestep} ({timestep.tzinfo})")


def main(
    initial_date: pd.Timestamp = None,
    first_run: datetime = None,
    tasks_interval: int = None,
    time_delta: pd.Timedelta = None,
):
    # load keys.yaml where the database and entsoe keys are stored
    with open("keys.yaml", "r") as f:
        keys = yaml.safe_load(f)

    db_client = DBClient(
        database_name="cosema",
        username=keys["influxdb"]["username"],
        password=keys["influxdb"]["password"],
    )

    entsoe_client = EntsoePandasClient(api_key=keys["entsoe-key"])

    if initial_date is None:
        initial_date = pd.Timestamp("now", tz="UTC").normalize() # normalize to 00:00
    else:
        initial_date = pd.Timestamp(initial_date, tz="UTC")

    if first_run is None:
        first_run = (
            pd.Timestamp("now", tz="Europe/Berlin").ceil("h") #+ FIRST_RUN_OFFSET
        )
    else:
        first_run = pd.Timestamp(first_run, tz="Europe/Berlin")

    time_delta = (
        TIME_DELTA if time_delta is None else pd.Timedelta(f"{time_delta}h", tz="UTC")
    )

    tasks_interval = TASK_INTERVAL if tasks_interval is None else tasks_interval

    logger.info(
        f"Scheduler starts at {first_run}. Scheduler will run every {tasks_interval} hours."
    )

    init_date(initial_date=initial_date)

    scheduler = BackgroundScheduler()

    scheduler.add_job(
        perform_forecast,
        trigger="interval",
        hours=tasks_interval,
        kwargs={
            "entsoe_client": entsoe_client,
            "db_client": db_client,
            "time_delta": time_delta,
            "interval": pd.Timedelta(f"{tasks_interval}h"),
        },
        name="perform_forecast",
        next_run_time=first_run.to_pydatetime(),
        misfire_grace_time=15 * 60,
    )

    scheduler.start()

    try:
        # This is here to simulate application activity (which keeps the main thread alive).
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to initialize the scheduler for initial data downloads."
    )

    parser.add_argument(
        "-st",
        "--start_date",
        type=str,
        required=False,
        help="A starting date for the scheduler in Y-m-d-H:M format.",
    )

    parser.add_argument(
        "-r",
        "--first_run",
        type=str,
        required=False,
        help="An optional argument to when to start the scheduler in Y-m-d-H:M. datetine.now() by default.",
    )

    parser.add_argument(
        "-i",
        "--tasks_interval",
        type=int,
        required=False,
        help="An optional argument to set the interval between tasks [in hours]. 24 by default.",
    )

    parser.add_argument(
        "-t",
        "--time_delta",
        type=str,
        required=False,
        help="An optional argument to set the time delta for the calculation [in hours]. 48h by default.",
    )

    args = parser.parse_args()

    main(
        initial_date=args.start_date,
        first_run=args.first_run,
        tasks_interval=args.tasks_interval,
        time_delta=args.time_delta,
    )
