import os
import sys
import yaml
import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient
from autogluon.tabular import TabularPredictor

# ---------------------------------------------------------------------
# HELPER 1: process_api_data => convert raw ENTSO-E data to hourly timeseries
# ---------------------------------------------------------------------
def process_api_data(df, default_item_id, table_name):
    if df is None or len(df) == 0:
        return None
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # If there's no 'timestamp' column, try to rename index if it's datetime
    if "timestamp" not in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        else:
            print(f"[WARN] {table_name}: no 'timestamp' found and index is not datetime. Skipping.")
            return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_convert("Europe/Brussels").dt.tz_localize(None)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None  # No numeric data

    # Resample to hourly, interpolate, forward fill and backward fill
    df = df.set_index("timestamp").resample("h").mean()
    df = df.interpolate(method='linear').ffill().bfill().reset_index()

    # Add an item_id column if needed
    df["item_id"] = default_item_id
    return df

# ---------------------------------------------------------------------
# HELPER 2: fetch_entsoe_data => gather all needed ENTSO-E data
# ---------------------------------------------------------------------
def fetch_entsoe_data(countries, borders, start, end, api_key):
    client = EntsoePandasClient(api_key=api_key)
    data_dict = {}

    # Expand cross-borders (like your pipeline)
    expanded_borders = []
    for b in borders:
        try:
            c_from, c_to = b.split("-")
            expanded_borders.append((c_from, c_to))
            expanded_borders.append((c_to, c_from))
        except:
            print(f"[WARN] Could not parse border '{b}' -> skip.")
    # Remove duplicates (preserving order)
    expanded_borders = list(dict.fromkeys(expanded_borders))

    # 1) Cross-border queries
    for (country_from, country_to) in expanded_borders:
        # Imports
        tbl_i = f"{country_from}_imports_{country_to}"
        try:
            print(f"[INFO] Query imports: {country_to} -> {country_from}")
            df_i = client.query_scheduled_exchanges(
                country_code_from=country_to,
                country_code_to=country_from,
                start=start,
                end=end,
                dayahead=True
            )
            df_i = process_api_data(df_i, default_item_id=country_from, table_name=tbl_i)
            data_dict[tbl_i] = df_i
        except Exception as e:
            print(f"[WARN] Could not fetch {tbl_i}: {e}")

        # Exports
        tbl_e = f"{country_from}_exports_{country_to}"
        try:
            print(f"[INFO] Query exports: {country_from} -> {country_to}")
            df_e = client.query_scheduled_exchanges(
                country_code_from=country_from,
                country_code_to=country_to,
                start=start,
                end=end,
                dayahead=True
            )
            df_e = process_api_data(df_e, default_item_id=country_from, table_name=tbl_e)
            data_dict[tbl_e] = df_e
        except Exception as e:
            print(f"[WARN] Could not fetch {tbl_e}: {e}")

    # 2) Standard ENTSO-E queries per country
    queries = {
        "query_query_load_forecast": client.query_load_forecast,
        "query_query_generation_forecast": client.query_generation_forecast,
        "query_query_wind_and_solar_forecast": client.query_wind_and_solar_forecast,
        "query_query_day_ahead_prices": client.query_day_ahead_prices,
    }

    for cc in countries:
        for qname, qfunc in queries.items():
            tbl = f"{cc}_{qname}"
            try:
                print(f"[INFO] Query {qname} for {cc}")
                df_q = qfunc(cc, start=start, end=end)
                df_q = process_api_data(df_q, default_item_id=cc, table_name=tbl)
                data_dict[tbl] = df_q
            except Exception as e:
                print(f"[WARN] Could not fetch {tbl}: {e}")

    return data_dict

# ---------------------------------------------------------------------
# HELPER 3: merge all dataframes on 'timestamp'
# ---------------------------------------------------------------------
def merge_all_dataframes(data_dict):
    valid_dfs = []
    for name, df in data_dict.items():
        if df is not None and len(df) > 0:
            valid_dfs.append(df)

    if not valid_dfs:
        print("[ERROR] No valid ENTSO-E data was fetched.")
        return pd.DataFrame()

    # Collect all timestamps
    all_ts = pd.concat([df[['timestamp']] for df in valid_dfs]).drop_duplicates()
    all_ts.sort_values("timestamp", inplace=True)

    # Start the consolidated DataFrame
    consolidated = all_ts.copy().reset_index(drop=True)

    # Merge numeric columns from each table
    for name, df in data_dict.items():
        if df is None or len(df) == 0:
            continue
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ("item_id",)]
        for col in numeric_cols:
            merged_col = f"{name}_{col}"
            consolidated = pd.merge(
                consolidated,
                df[['timestamp', col]],
                how="left",
                on="timestamp"
            ).rename(columns={col: merged_col})

    consolidated.sort_values("timestamp", inplace=True)
    consolidated.ffill(inplace=True)
    consolidated.bfill(inplace=True)
    return consolidated

# ---------------------------------------------------------------------
# HELPER 4: create_features -> drop columns not used by the model
# ---------------------------------------------------------------------
def create_features(df, label_col):
    """
    Drop:
     - 'timestamp', 'item_id'
     - Any columns that contain "Intensity" (except label_col, if present)
     - Any columns that start with 'y_'
    """
    drop_cols = []
    for c in df.columns:
        if c in ["timestamp", "item_id"]:
            drop_cols.append(c)
        if c.startswith("y_"):
            drop_cols.append(c)
        if ("Intensity" in c) and (c != label_col):
            drop_cols.append(c)
    return df.drop(columns=drop_cols, errors="ignore")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # Hard-coded parameters (update as needed)
    model_path = "AutogluonModels/ag-20250331_174228"
    config_file = "config_DE.yaml"
    api_key = "YOUR_ENTSOE_API_KEY"
    start_date = "2025-01-01"
    end_date = "2025-03-31"  # shorter time range for testing

    # 1. Load config file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    countries = config["countries"]         # e.g. ["DE_LU"]
    borders   = config["country_borders"]     # e.g. ["DE_LU-CH", ...]

    # 2. Define date range with timezone
    start_ts = pd.Timestamp(start_date, tz="Europe/Brussels")
    end_ts   = pd.Timestamp(end_date, tz="Europe/Brussels")

    print(f"[INFO] Fetching ENTSO-E data from {start_ts} to {end_ts} for {countries} / {borders}...")
    # 3. Fetch ENTSO-E data
    data_dict = fetch_entsoe_data(
        countries=countries,
        borders=borders,
        start=start_ts,
        end=end_ts,
        api_key=api_key
    )

    # 4. Merge fetched data into a single DataFrame
    print("[INFO] Merging data into a single DataFrame...")
    df_merged = merge_all_dataframes(data_dict)
    print(f"[INFO] Merged shape = {df_merged.shape} columns = {df_merged.columns.tolist()}")

    # --- Drop columns for imports/exports that do not start with "DE_LU" ---
    cols_to_drop = [col for col in df_merged.columns if (("imports" in col.lower() or "exports" in col.lower()) and not col.startswith("DE_LU"))]
    if cols_to_drop:
        print(f"[INFO] Dropping columns: {cols_to_drop}")
        df_merged.drop(columns=cols_to_drop, inplace=True)
    # ------------------------------------------------------------------------

    # --- Add missing INDEX column if not present ---
    if "INDEX" not in df_merged.columns:
        df_merged["INDEX"] = df_merged.index
    # ------------------------------------------------------------------------

    # 5. Save the consolidated features to CSV
    features_csv = "AllFeatures.csv"
    df_merged.to_csv(features_csv, index=False)
    print(f"[INFO] Consolidated features saved to {features_csv}")

    # 6. Load the AutoGluon model
    print(f"[INFO] Loading model from {model_path}")
    predictor = TabularPredictor.load(model_path, require_py_version_match=False)
    label_col = predictor.label  # The label this model was trained to predict

    # 7. Create the feature set for prediction (dropping columns not used by the model)
    df_features = create_features(df_merged, label_col=label_col)

    # 8. Generate predictions
    print(f"[INFO] Generating predictions for label: {label_col}")
    y_pred = predictor.predict(df_features)
    pred_col = f"Predicted_{label_col}"
    df_merged[pred_col] = y_pred

    # Print a preview of predictions in terminal
    print("[INFO] Prediction preview:")
    print(df_merged[['timestamp', pred_col]].head())

    # 9. Save the final CSV with predictions
    output_csv = "AllFeatures_withPredictions.csv"
    df_merged.to_csv(output_csv, index=False)
    print(f"[INFO] Final CSV with predictions saved as {output_csv}")

if __name__ == "__main__":
    main()