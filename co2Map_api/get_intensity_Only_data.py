import pandas as pd
import requests
import datetime

def fetch_with_backoff(url, params, max_retries=5, base_delay=2):
    """Simple backoff mechanism for API calls."""
    import time
    delay = base_delay
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 429:
                print(f"Received 429, waiting {delay} seconds (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(delay)
            delay *= 2
    raise Exception("Failed to fetch data after several retries.")

def fetch_co2map_data(start_date, end_date, state="DE", country="DE"):
    """
    Fetches production (generation) and consumption intensity data from co2map.de.
    Returns two DataFrames: one for production and one for consumption.
    """
    prod_url = "https://api.co2map.de/ProductionIntensityHistorical/"
    cons_url = "https://api.co2map.de/ConsumptionIntensityHistorical/"

    # Prepare parameters
    params = {
        "state": state,
        "country": country,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d")
    }

    print(f"[INFO] Fetching Production Data for state {state} from {params['start']} to {params['end']}")
    prod_json = fetch_with_backoff(prod_url, params=params)
    prod_list = prod_json.get("Production-based Intensity (historical)", [])
    df_prod = pd.DataFrame(prod_list, columns=["timestamp", "Production_Intensity"])
    df_prod["timestamp"] = pd.to_datetime(df_prod["timestamp"], utc=True).dt.tz_convert("Europe/Brussels").dt.tz_localize(None)

    print(f"[INFO] Fetching Consumption Data for state {state} from {params['start']} to {params['end']}")
    cons_json = fetch_with_backoff(cons_url, params=params)
    cons_list = cons_json.get("Consumption-based Intensity (historical)", [])
    df_cons = pd.DataFrame(cons_list, columns=["timestamp", "Consumption_Intensity"])
    df_cons["timestamp"] = pd.to_datetime(df_cons["timestamp"], utc=True).dt.tz_convert("Europe/Brussels").dt.tz_localize(None)

    return df_prod, df_cons

def main():
    # Define the parameters for data fetching
    start_date = datetime.datetime(2025, 1, 1)
    end_date = datetime.datetime(2025, 2, 28)
    state = "TH"
    country = "DE"

    # Fetch the production and consumption data from co2map.de
    df_prod, df_cons = fetch_co2map_data(start_date, end_date, state, country)

    # Merge the two DataFrames on the "timestamp" column using an outer join.
    # This will align rows that share similar timestamps.
    df_merged = pd.merge(df_prod, df_cons, on="timestamp", how="outer")
    df_merged.sort_values("timestamp", inplace=True)

    # Optionally, forward-fill and backward-fill any missing values.
    df_merged.ffill(inplace=True)
    df_merged.bfill(inplace=True)

    # Save the merged data to an Excel file
    output_file = "co2map_data.xlsx"
    df_merged.to_excel(output_file, index=False)
    print(f"[INFO] Data saved to {output_file}")

if __name__ == "__main__":
    main()