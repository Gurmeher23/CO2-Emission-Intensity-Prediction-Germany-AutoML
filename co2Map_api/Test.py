import os
import yaml
import pandas as pd
import numpy as np
import sqlite3
import requests
import time
from entsoe import EntsoePandasClient

# AutoGluon
from autogluon.tabular import TabularPredictor

# Sklearn + splits
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============== Metrics Helper ==============
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def SMAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


def mean_bias(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_pred - y_true)


# ============== Simple backoff for co2map ==============
def fetch_with_backoff(url, params, headers=None, max_retries=10, base_delay=2):
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers)
            if resp.status_code == 429:
                print(f"[WARNING] 429 Too Many Requests. Sleeping {delay}s (Attempt {attempt}/{max_retries})")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[WARNING] Request exception: {e}, attempt {attempt}/{max_retries}, sleeping {delay}s.")
            time.sleep(delay)
            delay *= 2
    raise Exception(f"[ERROR] Exceeded {max_retries} retries for {url} with params={params}.")


# ============== ENTSoE data => resampled ==============
def process_api_data(df, default_item_id, table_name):
    """Convert raw entsoe data to hourly timeseries, fill NA, etc."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if "timestamp" not in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        else:
            print(f"[WARN] {table_name}: no 'timestamp' found.")
            return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert("Europe/Brussels")
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert("Europe/Brussels")
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df  # might be empty or no numeric data

    # resample hourly, fill
    df = df.set_index('timestamp')
    df = df.resample("h").mean()
    df = df.interpolate(method='linear').ffill().bfill()
    df = df.reset_index()
    df["item_id"] = default_item_id
    return df


def main():
    # ============ 1. Load config + Setup ============
    with open("/content/drive/My Drive/Colab Notebooks/config_DE.yaml", "r") as f:
        config = yaml.safe_load(f)

    countries = config["countries"]  # e.g. ["DE_LU"]
    borders = config["country_borders"]  # e.g. ["DE_LU-CH", "BE-DE_LU", ...]
    states = config.get("states", [])  # e.g. ["BW", "BY", ...]

    db_filename = "entsoe_data_DE.db"
    conn = sqlite3.connect(db_filename)

    csv_folder = "entsoe_DB_data_DE"
    os.makedirs(csv_folder, exist_ok=True)

    # For demonstration, we'll still fetch from 2021 to 2025
    start = pd.Timestamp("20210101", tz="Europe/Brussels")
    end = pd.Timestamp("20250328", tz="Europe/Brussels")

    # ENTSoE
    API_KEY = "YOUR_ENTSOE_API_KEY"
    client = EntsoePandasClient(api_key=API_KEY)

    consolidated_csv = os.path.join(csv_folder, "DE_LU_consolidated.csv")
    all_exist = os.path.exists(consolidated_csv)

    if not all_exist:
        print("[INFO] Not all consolidated CSVs exist. Fetching from ENTSO-E & co2map...")

        # -------------- 2. Data Fetch --------------
        expanded_borders = []
        for b in borders:
            try:
                a, c = b.split("-")
                expanded_borders.append((a, c))
                expanded_borders.append((c, a))
            except Exception as e:
                print(f"[WARN] skipping border '{b}': {e}")
        # remove duplicates
        expanded_borders = list(dict.fromkeys(expanded_borders))

        # 2a. cross border
        all_cross_tables = []
        for country_from, country_to in expanded_borders:
            # imports
            try:
                print(f"Fetching imports: {country_to} to {country_from}...")
                df_i = client.query_scheduled_exchanges(country_to, country_from, start=start, end=end, dayahead=True)
                tbl_i = f"{country_from}_imports_{country_to}"
                df_i = process_api_data(df_i, default_item_id=country_from, table_name=tbl_i)
                if df_i is not None:
                    df_i.to_sql(tbl_i, conn, if_exists="replace", index=False)
                    all_cross_tables.append(tbl_i)
            except Exception as e:
                print(f"[WARN] fail cross import {country_to}->{country_from}: {e}")

            # exports
            try:
                print(f"Fetching exports: {country_from} to {country_to}...")
                df_e = client.query_scheduled_exchanges(country_from, country_to, start=start, end=end, dayahead=True)
                tbl_e = f"{country_from}_exports_{country_to}"
                df_e = process_api_data(df_e, default_item_id=country_from, table_name=tbl_e)
                if df_e is not None:
                    df_e.to_sql(tbl_e, conn, if_exists="replace", index=False)
                    all_cross_tables.append(tbl_e)
            except Exception as e:
                print(f"[WARN] fail cross export {country_from}->{country_to}: {e}")

        # 2b. queries
        queries = {
            "query_load_forecast": lambda c: client.query_load_forecast(c, start=start, end=end),
            "query_generation_forecast": lambda c: client.query_generation_forecast(c, start=start, end=end),
            "query_wind_and_solar_forecast": lambda c: client.query_wind_and_solar_forecast(c, start=start, end=end,
                                                                                            psr_type=None),
            "query_day_ahead_prices": lambda c: client.query_day_ahead_prices(c, start=start, end=end),
        }
        for cc in countries:
            for qn, qfunc in queries.items():
                tbl_q = f"{cc}_query_{qn}"
                try:
                    print(f"[INFO] Running {qn} for {cc}...")
                    q_data = qfunc(cc)
                    q_data = process_api_data(q_data, default_item_id=cc, table_name=tbl_q)
                    if q_data is not None:
                        q_data.to_sql(tbl_q, conn, if_exists="replace", index=False)
                except Exception as e:
                    print(f"[WARN] failed {qn} for {cc}: {e}")

        # 2c. co2map for DE if "DE_LU" in countries
        if "DE_LU" in countries:
            prod_url = "https://api.co2map.de/ProductionIntensityHistorical/"
            cons_url = "https://api.co2map.de/ConsumptionIntensityHistorical/"
            segments = pd.date_range(start=start, end=end, freq='Q')
            segments = list(zip([start] + list(segments), list(segments) + [end]))
            p_dfs = []
            c_dfs = []
            for (seg_s, seg_e) in segments:
                st_s = seg_s.strftime("%Y-%m-%d")
                st_e = seg_e.strftime("%Y-%m-%d")
                params_de = {"state": "DE", "country": "DE", "start": st_s, "end": st_e}
                # Production
                try:
                    print(f"Fetching DE production {st_s}->{st_e}")
                    p_json = fetch_with_backoff(prod_url, params=params_de)
                    p_list = p_json.get("Production-based Intensity (historical)", [])
                    df_p = pd.DataFrame(p_list, columns=["timestamp", "Production_Intensity"])
                    df_p["timestamp"] = pd.to_datetime(df_p["timestamp"], utc=True).dt.tz_convert(
                        "Europe/Brussels").dt.tz_localize(None)
                    p_dfs.append(df_p)
                except Exception as e:
                    print(f"[WARN] DE production fail: {e}")
                # Consumption
                try:
                    print(f"Fetching DE consumption {st_s}->{st_e}")
                    c_json = fetch_with_backoff(cons_url, params=params_de)
                    c_list = c_json.get("Consumption-based Intensity (historical)", [])
                    df_c = pd.DataFrame(c_list, columns=["timestamp", "Consumption_Intensity"])
                    df_c["timestamp"] = pd.to_datetime(df_c["timestamp"], utc=True).dt.tz_convert(
                        "Europe/Brussels").dt.tz_localize(None)
                    c_dfs.append(df_c)
                except Exception as e:
                    print(f"[WARN] DE consumption fail: {e}")
            if p_dfs:
                df_pp = pd.concat(p_dfs).drop_duplicates().sort_values("timestamp").reset_index(drop=True)
                df_pp["item_id"] = "DE_LU"
                df_pp.to_sql("DE_LU_Production_Intensity", conn, if_exists="replace", index=False)
            if c_dfs:
                df_cc = pd.concat(c_dfs).drop_duplicates().sort_values("timestamp").reset_index(drop=True)
                df_cc["item_id"] = "DE_LU"
                df_cc.to_sql("DE_LU_Consumption_Intensity", conn, if_exists="replace", index=False)

        # 2c2. co2map for each state
        states_prod_tables = []
        states_cons_tables = []
        if states:
            prod_url = "https://api.co2map.de/ProductionIntensityHistorical/"
            cons_url = "https://api.co2map.de/ConsumptionIntensityHistorical/"
            segments = pd.date_range(start=start, end=end, freq='Q')
            segments = list(zip([start] + list(segments), list(segments) + [end]))
            for st_code in states:
                s_pdfs = []
                s_cdfs = []
                for (sg_s, sg_e) in segments:
                    st_s = sg_s.strftime("%Y-%m-%d")
                    st_e = sg_e.strftime("%Y-%m-%d")
                    params_st = {"state": st_code, "country": "DE", "start": st_s, "end": st_e}
                    try:
                        print(f"Fetching Production {st_code} {st_s}->{st_e}")
                        p_json = fetch_with_backoff(prod_url, params=params_st)
                        p_list = p_json.get("Production-based Intensity (historical)", [])
                        df_p = pd.DataFrame(p_list, columns=["timestamp", "Production_Intensity"])
                        df_p["timestamp"] = pd.to_datetime(df_p["timestamp"], utc=True).dt.tz_convert(
                            "Europe/Brussels").dt.tz_localize(None)
                        df_p["item_id"] = st_code
                        s_pdfs.append(df_p)
                    except Exception as e:
                        print(f"[WARN] production {st_code} fail: {e}")
                    try:
                        print(f"Fetching Consumption {st_code} {st_s}->{st_e}")
                        c_json = fetch_with_backoff(cons_url, params=params_st)
                        c_list = c_json.get("Consumption-based Intensity (historical)", [])
                        df_c = pd.DataFrame(c_list, columns=["timestamp", "Consumption_Intensity"])
                        df_c["timestamp"] = pd.to_datetime(df_c["timestamp"], utc=True).dt.tz_convert(
                            "Europe/Brussels").dt.tz_localize(None)
                        df_c["item_id"] = st_code
                        s_cdfs.append(df_c)
                    except Exception as e:
                        print(f"[WARN] consumption {st_code} fail: {e}")

                if s_pdfs:
                    dfp_all = pd.concat(s_pdfs).drop_duplicates().sort_values("timestamp").reset_index(drop=True)
                    tbl_p = f"{st_code}_Production_Intensity"
                    dfp_all.to_sql(tbl_p, conn, if_exists="replace", index=False)
                    states_prod_tables.append(tbl_p)
                if s_cdfs:
                    dfc_all = pd.concat(s_cdfs).drop_duplicates().sort_values("timestamp").reset_index(drop=True)
                    tbl_c = f"{st_code}_Consumption_Intensity"
                    dfc_all.to_sql(tbl_c, conn, if_exists="replace", index=False)
                    states_cons_tables.append(tbl_c)

        conn.commit()
        print("[INFO] Data fetched & saved. Now creating table 'DE_LU' with cross-border merges...")

        # -------------- 3. CREATE `DE_LU` from cross-border --------------
        if not all_cross_tables:
            print("[WARN] No cross-border tables exist, skipping cross-border merges.")
            try:
                conn.execute("DROP TABLE IF EXISTS DE_LU")
                conn.execute("""
                    CREATE TABLE DE_LU AS
                    SELECT DISTINCT timestamp
                    FROM DE_LU_Production_Intensity
                    ORDER BY timestamp
                """)
            except:
                pass
        else:
            base_table = all_cross_tables[0]
            print(f"[INFO] Using base cross-border table for DE_LU: {base_table}")
            conn.execute("DROP TABLE IF EXISTS DE_LU")
            conn.execute(f"""
                CREATE TABLE DE_LU AS
                SELECT DISTINCT timestamp
                FROM {base_table}
                ORDER BY timestamp
            """)

            def get_numeric_columns(table):
                df_test = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1", conn)
                return [c for c in df_test.columns if c not in ("timestamp", "item_id")]

            for tbl in all_cross_tables:
                numeric_cols = get_numeric_columns(tbl)
                for col in numeric_cols:
                    new_col_name = tbl + "_" + col
                    try:
                        conn.execute(f"ALTER TABLE DE_LU ADD COLUMN [{new_col_name}] REAL")
                    except:
                        pass
                    sql_update = f"""
                    UPDATE DE_LU
                    SET [{new_col_name}] = (
                        SELECT [{col}]
                        FROM {tbl} t
                        WHERE t.timestamp = DE_LU.timestamp
                    )
                    """
                    conn.execute(sql_update)
            conn.commit()

        # -------------- 4. Add ENTSO-E Queries --------------
        c_query = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
              AND name LIKE 'DE_LU_query_%'
        """).fetchall()
        for row in c_query:
            qtbl = row[0]
            numeric_cols = pd.read_sql_query(f"SELECT * FROM {qtbl} LIMIT 1", conn).columns
            numeric_cols = [c for c in numeric_cols if c not in ("timestamp", "item_id")]
            for col in numeric_cols:
                new_col = qtbl + "_" + col
                try:
                    conn.execute(f"ALTER TABLE DE_LU ADD COLUMN [{new_col}] REAL")
                except:
                    pass
                sql_up = f"""
                UPDATE DE_LU
                SET [{new_col}] = (
                    SELECT [{col}]
                    FROM {qtbl} t
                    WHERE t.timestamp = DE_LU.timestamp
                )
                """
                conn.execute(sql_up)
        conn.commit()

        # -------------- 5. Add DE intensities --------------
        try:
            conn.execute("""UPDATE DE_LU_Production_Intensity
                            SET timestamp = DATETIME(timestamp, '-1 hour')""")
            conn.execute("""UPDATE DE_LU_Consumption_Intensity
                            SET timestamp = DATETIME(timestamp, '-1 hour')""")
            conn.commit()
            print("[INFO] SHIFTed DE intensities by -1 hour in DB.")
        except:
            pass

        try:
            conn.execute("ALTER TABLE DE_LU ADD COLUMN Generation_Intensity REAL")
            conn.execute("ALTER TABLE DE_LU ADD COLUMN Consumption_Intensity REAL")
        except:
            pass

        conn.execute("""
            UPDATE DE_LU
            SET Generation_Intensity = (
                SELECT Production_Intensity
                FROM DE_LU_Production_Intensity p
                WHERE p.timestamp = DE_LU.timestamp
            )
        """)
        conn.execute("""
            UPDATE DE_LU
            SET Consumption_Intensity = (
                SELECT Consumption_Intensity
                FROM DE_LU_Consumption_Intensity c
                WHERE c.timestamp = DE_LU.timestamp
            )
        """)
        conn.commit()

        # -------------- 6. Add States intensities --------------
        for st_code in states:
            try:
                conn.execute(f"""
                    UPDATE {st_code}_Production_Intensity
                    SET timestamp = DATETIME(timestamp, '-1 hour')
                """)
                conn.execute(f"""
                    UPDATE {st_code}_Consumption_Intensity
                    SET timestamp = DATETIME(timestamp, '-1 hour')
                """)
                conn.commit()
                print(f"[INFO] SHIFTed {st_code} intensities by -1 hour in DB.")
            except:
                pass

            gen_col = f"{st_code}_GenerationIntensity"
            con_col = f"{st_code}_ConsumptionIntensity"
            try:
                conn.execute(f"ALTER TABLE DE_LU ADD COLUMN [{gen_col}] REAL")
                conn.execute(f"ALTER TABLE DE_LU ADD COLUMN [{con_col}] REAL")
            except:
                pass

            conn.execute(f"""
                UPDATE DE_LU
                SET [{gen_col}] = (
                    SELECT Production_Intensity
                    FROM {st_code}_Production_Intensity sp
                    WHERE sp.timestamp = DE_LU.timestamp
                )
            """)
            conn.execute(f"""
                UPDATE DE_LU
                SET [{con_col}] = (
                    SELECT Consumption_Intensity
                    FROM {st_code}_Consumption_Intensity sc
                    WHERE sc.timestamp = DE_LU.timestamp
                )
            """)
            conn.commit()

        print(
            "[INFO] 'DE_LU' table now has cross-border flows, ENTSO-E queries, DE intensities, and state intensities.")

    else:
        print("[INFO] Found existing consolidated CSV. Skipping fetch & merges in DB.")

    conn.close()

    # ============ 7. Clean up the final DE_LU data, write CSV and update DB ============
    conn_new = sqlite3.connect(db_filename)
    df_de = pd.read_sql_query("SELECT * FROM DE_LU", conn_new, parse_dates=["timestamp"])
    df_de.sort_values("timestamp", inplace=True)
    df_de.ffill(inplace=True)
    df_de.bfill(inplace=True)

    unique_import_cols = [col for col in df_de.columns if "imports" in col.lower() and col.startswith("DE_LU")]
    unique_export_cols = [col for col in df_de.columns if "exports" in col.lower() and col.startswith("DE_LU")]
    if unique_import_cols:
        df_de["Total_Imports"] = df_de[unique_import_cols].sum(axis=1, skipna=True)
    if unique_export_cols:
        df_de["Total_Exports"] = df_de[unique_export_cols].sum(axis=1, skipna=True)

    # Drop redundant import/export columns that do not start with "DE_LU"
    cols_to_drop = [col for col in df_de.columns if (("imports" in col.lower() or "exports" in col.lower())
                                                     and not col.startswith("DE_LU"))]
    df_de.drop(columns=cols_to_drop, inplace=True)

    consolidated_csv = os.path.join(csv_folder, "DE_LU_consolidated.csv")
    df_de.to_csv(consolidated_csv, index=False)
    print(
        f"[INFO] Wrote final cleaned data to {consolidated_csv} with shape={df_de.shape} columns={df_de.columns.tolist()}")

    df_de.to_sql("DE_LU", conn_new, if_exists="replace", index=False)
    conn_new.commit()
    conn_new.close()

    # ============ Helper to drop intensity columns for training ============

    def create_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """
        For the given DataFrame, drop:
         - 'timestamp', 'item_id'
         - ANY column that contains 'Intensity' (including state columns),
           except for the chosen label_col
         - ANY predicted columns that start with 'y_'
        This way we follow the rule: for consumption (or generation) training
        we remove all states' generation/consumption intensities (including DE),
        except the label we are predicting.
        """
        drop_cols = []
        for c in df.columns:
            if c in ["timestamp", "item_id"]:
                drop_cols.append(c)
                continue
            if c.startswith("y_"):
                drop_cols.append(c)
                continue
            if "Intensity" in c and c != label_col:
                drop_cols.append(c)
        return df.drop(columns=drop_cols, errors="ignore")

    # ============ 8. Train models using data from the CSV file ============
    df_de = pd.read_csv(consolidated_csv, parse_dates=["timestamp"])
    df_de.sort_values("timestamp", inplace=True)
    df_de.ffill(inplace=True)
    df_de.bfill(inplace=True)

    # Ensure required intensity columns exist
    if "Generation_Intensity" not in df_de.columns or "Consumption_Intensity" not in df_de.columns:
        print("[ERROR] Missing Generation_Intensity or Consumption_Intensity in final table. Exiting.")
        return
    # ----------- LAST 3 MONTHS as TEST DATA -----------
    last_timestamp = df_de["timestamp"].max()
    test_start = last_timestamp - pd.DateOffset(months=3)

    train_mask = df_de["timestamp"] < test_start
    test_mask = df_de["timestamp"] >= test_start

    # ========== 8A. DE Generation model ==========
    label_gen = "Generation_Intensity"

    df_train_gen = df_de[train_mask].copy()
    df_test_gen = df_de[test_mask].copy()

    y_train_gen = df_train_gen[label_gen]
    y_test_gen = df_test_gen[label_gen]

    # Create input features
    train_features_gen = create_features(df_train_gen, label_gen)
    test_features_gen = create_features(df_test_gen, label_gen)

    # Attach label for training
    train_gen_ag = train_features_gen.copy()
    train_gen_ag[label_gen] = y_train_gen

    print("\n=== Training DE Generation Model ===")
    predictor_gen = TabularPredictor(
        label=label_gen,
        problem_type="regression",
        eval_metric="r2"
    ).fit(
        train_data=train_gen_ag,
        presets="high_quality",
        time_limit=1800
    )

    # Predict on test (last 3 months)
    y_pred_gen_test = predictor_gen.predict(test_features_gen)

    # Metrics on test set
    print(f"\nDE Generation_Intensity Test Metrics (Last 3 Months):")
    print(f"  MSE={mean_squared_error(y_test_gen, y_pred_gen_test):.4f}")
    print(f"  R^2={r2_score(y_test_gen, y_pred_gen_test):.4f}")
    print(f"  RMSE={RMSE(y_test_gen, y_pred_gen_test):.4f}")
    print(f"  SMAPE={SMAPE(y_test_gen, y_pred_gen_test):.4f}")
    print(f"  MeanBias={mean_bias(y_test_gen, y_pred_gen_test):.4f}")

    # Store predictions for test set
    df_test_gen["y_DE_GenerationIntensity"] = y_pred_gen_test

    # Also predict on train for sample in terminal
    y_pred_gen_train = predictor_gen.predict(train_features_gen)
    df_train_gen["y_DE_GenerationIntensity"] = y_pred_gen_train

    # Print a small sample of train data
    print("\n[Train sample - DE Generation]")
    print(df_train_gen[["timestamp", label_gen, "y_DE_GenerationIntensity"]].head(10))

    # Print a small sample of test data
    print("\n[Test sample - DE Generation]")
    print(df_test_gen[["timestamp", label_gen, "y_DE_GenerationIntensity"]].head(10))

    # ========== 8B. DE Consumption model ==========
    label_cons = "Consumption_Intensity"

    df_train_cons = df_de[train_mask].copy()
    df_test_cons = df_de[test_mask].copy()

    y_train_cons = df_train_cons[label_cons]
    y_test_cons = df_test_cons[label_cons]

    train_features_cons = create_features(df_train_cons, label_cons)
    test_features_cons = create_features(df_test_cons, label_cons)

    train_cons_ag = train_features_cons.copy()
    train_cons_ag[label_cons] = y_train_cons

    print("\n=== Training DE Consumption Model ===")
    predictor_cons = TabularPredictor(
        label=label_cons,
        problem_type="regression",
        eval_metric="r2"
    ).fit(
        train_data=train_cons_ag,
        presets="high_quality",
        time_limit=1800
    )

    y_pred_cons_test = predictor_cons.predict(test_features_cons)

    print(f"\nDE Consumption_Intensity Test Metrics (Last 3 Months):")
    print(f"  MSE={mean_squared_error(y_test_cons, y_pred_cons_test):.4f}")
    print(f"  R^2={r2_score(y_test_cons, y_pred_cons_test):.4f}")
    print(f"  RMSE={RMSE(y_test_cons, y_pred_cons_test):.4f}")
    print(f"  SMAPE={SMAPE(y_test_cons, y_pred_cons_test):.4f}")
    print(f"  MeanBias={mean_bias(y_test_cons, y_pred_cons_test):.4f}")

    df_test_cons["y_DE_ConsumptionIntensity"] = y_pred_cons_test

    # Predict on train for sample
    y_pred_cons_train = predictor_cons.predict(train_features_cons)
    df_train_cons["y_DE_ConsumptionIntensity"] = y_pred_cons_train

    # Print samples
    print("\n[Train sample - DE Consumption]")
    print(df_train_cons[["timestamp", label_cons, "y_DE_ConsumptionIntensity"]].head(10))

    print("\n[Test sample - DE Consumption]")
    print(df_test_cons[["timestamp", label_cons, "y_DE_ConsumptionIntensity"]].head(10))

    # Combine predictions for final CSV
    df_de = df_de.merge(df_test_gen[["timestamp", "y_DE_GenerationIntensity"]],
                        on="timestamp", how="left")
    df_de = df_de.merge(df_test_cons[["timestamp", "y_DE_ConsumptionIntensity"]],
                        on="timestamp", how="left")

    # Overwrite predicted values for the training portion
    df_de = df_de.merge(
        df_train_gen[["timestamp", "y_DE_GenerationIntensity"]],
        on="timestamp", how="left", suffixes=("", "_train_gen")
    )
    df_de["y_DE_GenerationIntensity"] = df_de["y_DE_GenerationIntensity"].fillna(
        df_de["y_DE_GenerationIntensity_train_gen"]
    )
    df_de.drop(columns=["y_DE_GenerationIntensity_train_gen"], inplace=True)

    df_de = df_de.merge(
        df_train_cons[["timestamp", "y_DE_ConsumptionIntensity"]],
        on="timestamp", how="left", suffixes=("", "_train_cons")
    )
    df_de["y_DE_ConsumptionIntensity"] = df_de["y_DE_ConsumptionIntensity"].fillna(
        df_de["y_DE_ConsumptionIntensity_train_cons"]
    )
    df_de.drop(columns=["y_DE_ConsumptionIntensity_train_cons"], inplace=True)

    predicted_csv_de = os.path.join(csv_folder, "Predicted_Values_consolidated_DE_LU.csv")
    df_de.to_csv(predicted_csv_de, index=False)
    print(f"\n[INFO] DE predictions saved to CSV '{predicted_csv_de}'.")

    # ========== 9. Train each state model (if states exist) ==========
    if states:
        print("\n=== Now training each state model ===")
        df_pred_de = pd.read_csv(consolidated_csv, parse_dates=["timestamp"])
        df_pred_de.sort_values("timestamp", inplace=True)
        df_pred_de.ffill(inplace=True)
        df_pred_de.bfill(inplace=True)

        # For states, also define last 3 months from df_pred_de
        last_ts_states = df_pred_de["timestamp"].max()
        test_start_states = last_ts_states - pd.DateOffset(months=3)
        train_mask_state = df_pred_de["timestamp"] < test_start_states
        test_mask_state = df_pred_de["timestamp"] >= test_start_states

        for st_code in states:
            gen_col = f"{st_code}_GenerationIntensity"
            con_col = f"{st_code}_ConsumptionIntensity"
            if gen_col not in df_pred_de.columns or con_col not in df_pred_de.columns:
                print(f"[WARN] {st_code} intensities not found, skipping.")
                continue

            df_train_s = df_pred_de[train_mask_state].copy()
            df_test_s = df_pred_de[test_mask_state].copy()

            # === 9A. Generation model for state ===
            y_train_sg = df_train_s[gen_col]
            y_test_sg = df_test_s[gen_col]

            features_sg_train = create_features(df_train_s, gen_col)
            features_sg_test = create_features(df_test_s, gen_col)

            train_sg_ag = features_sg_train.copy()
            train_sg_ag[gen_col] = y_train_sg

            print(f"\n=== {st_code} Generation Model ===")
            pred_sg = TabularPredictor(
                label=gen_col,
                problem_type="regression",
                eval_metric="r2"
            ).fit(
                train_data=train_sg_ag,
                presets="high_quality",
                time_limit=1800
            )

            y_pred_sg_test = pred_sg.predict(features_sg_test)

            print(f"\n{st_code} Gen Test Metrics (Last 3 Months):")
            print(f"  MSE={mean_squared_error(y_test_sg, y_pred_sg_test):.4f}")
            print(f"  R^2={r2_score(y_test_sg, y_pred_sg_test):.4f}")
            print(f"  RMSE={RMSE(y_test_sg, y_pred_sg_test):.4f}")
            print(f"  SMAPE={SMAPE(y_test_sg, y_pred_sg_test):.4f}")
            print(f"  MeanBias={mean_bias(y_test_sg, y_pred_sg_test):.4f}")

            df_test_s[f"y_{st_code}_GenerationIntensity"] = y_pred_sg_test

            # Also predict on train for sample
            y_pred_sg_train = pred_sg.predict(features_sg_train)
            df_train_s[f"y_{st_code}_GenerationIntensity"] = y_pred_sg_train

            print(f"\n[Train sample - {st_code} Generation]")
            print(df_train_s[["timestamp", gen_col, f"y_{st_code}_GenerationIntensity"]].head(10))

            print(f"\n[Test sample - {st_code} Generation]")
            print(df_test_s[["timestamp", gen_col, f"y_{st_code}_GenerationIntensity"]].head(10))

            # === 9B. Consumption model for state ===
            y_train_sc = df_train_s[con_col]
            y_test_sc = df_test_s[con_col]

            features_sc_train = create_features(df_train_s, con_col)
            features_sc_test = create_features(df_test_s, con_col)

            train_sc_ag = features_sc_train.copy()
            train_sc_ag[con_col] = y_train_sc

            print(f"\n=== {st_code} Consumption Model ===")
            pred_sc = TabularPredictor(
                label=con_col,
                problem_type="regression",
                eval_metric="r2"
            ).fit(
                train_data=train_sc_ag,
                presets="high_quality",
                time_limit=1800
            )
            y_pred_sc_test = pred_sc.predict(features_sc_test)

            print(f"\n{st_code} Cons Test Metrics (Last 3 Months):")
            print(f"  MSE={mean_squared_error(y_test_sc, y_pred_sc_test):.4f}")
            print(f"  R^2={r2_score(y_test_sc, y_pred_sc_test):.4f}")
            print(f"  RMSE={RMSE(y_test_sc, y_pred_sc_test):.4f}")
            print(f"  SMAPE={SMAPE(y_test_sc, y_pred_sc_test):.4f}")
            print(f"  MeanBias={mean_bias(y_test_sc, y_pred_sc_test):.4f}")

            df_test_s[f"y_{st_code}_ConsumptionIntensity"] = y_pred_sc_test

            # Also predict on train
            y_pred_sc_train = pred_sc.predict(features_sc_train)
            df_train_s[f"y_{st_code}_ConsumptionIntensity"] = y_pred_sc_train

            print(f"\n[Train sample - {st_code} Consumption]")
            print(df_train_s[["timestamp", con_col, f"y_{st_code}_ConsumptionIntensity"]].head(10))

            print(f"\n[Test sample - {st_code} Consumption]")
            print(df_test_s[["timestamp", con_col, f"y_{st_code}_ConsumptionIntensity"]].head(10))

            # Merge predictions back so we can save
            df_pred_s = pd_pred_merge(df_pred_de, df_test_s, df_train_s,
                                      f"y_{st_code}_GenerationIntensity",
                                      gen_col)
            df_pred_s = pd_pred_merge(df_pred_s, df_test_s, df_train_s,
                                      f"y_{st_code}_ConsumptionIntensity",
                                      con_col)

            # Save final CSV for this state
            state_csv = os.path.join(csv_folder, f"Predicted_Values_consolidated_DE_LU_{st_code}.csv")
            df_pred_s.to_csv(state_csv, index=False)
            print(f"[INFO] Wrote {st_code} predictions => CSV '{state_csv}'")

    print("\nAll done.")


def pd_pred_merge(df_main, df_test, df_train, pred_col, true_col):
    """
    Helper to combine predictions from test and train back into df_main (by timestamp).
    Takes whichever is non-null for that timestamp.
    """
    import pandas as pd

    # test
    df_test_small = df_test[["timestamp", pred_col]]
    # train
    df_train_small = df_train[["timestamp", pred_col]]


    # Merge test first
    merged = df_main.merge(df_test_small, on="timestamp", how="left")
    # Then train with suffix
    merged = merged.merge(df_train_small, on="timestamp", how="left", suffixes=("", "_trainT"))

    # If test portion is null for the training set, fill with training predictions
    merged[pred_col] = merged[pred_col].fillna(merged[f"{pred_col}_trainT"])
    merged.drop(columns=[f"{pred_col}_trainT"], inplace=True, errors="ignore")
    return merged


if __name__ == "__main__":
    main()