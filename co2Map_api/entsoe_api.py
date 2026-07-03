from entsoe import EntsoePandasClient
import pandas as pd

# Replace with your ENTSO-E API key
API_KEY = "YOUR_ENTSOE_API_KEY"
client = EntsoePandasClient(api_key=API_KEY)

# Time range for data
start = pd.Timestamp('20220101', tz='Europe/Brussels')  # Start date
end = pd.Timestamp('20220107', tz='Europe/Brussels')   # End date (1 week for faster queries)

# Country code for Germany
country_code = 'DE_LU'  # Germany-Luxembourg

# Fetching generation import data
generation_import_data = client.query_generation_import(country_code, start=start, end=end)
print("\nGeneration Import Data (Imports into Germany):")
print(generation_import_data)

# Optionally save results to CSV
generation_import_data.to_csv("generation_import_data.csv")

"""
# Fetching physical cross-border electricity flows for all borders (exports)
crossborder_data = client.query_physical_crossborder_allborders(country_code, start=start, end=end, export=False)
print("\nPhysical Cross-Border Electricity Flows (Exports):")
print(crossborder_data)

# Optionally save results to CSV
crossborder_data.to_csv("crossborder_data.csv")

# Commenting out other API calls for now

# Fetching actual load data
load_data = client.query_load(country_code, start=start, end=end)
print("\nActual Load Data:")
print(load_data)

# Fetching forecasted load data
load_forecast_data = client.query_load_forecast(country_code, start=start, end=end)
print("\nLoad Forecast Data:")
print(load_forecast_data)

# Fetching both actual and forecasted load data
load_and_forecast_data = client.query_load_and_forecast(country_code, start=start, end=end)
print("\nActual and Forecasted Load Data:")
print(load_and_forecast_data)

# Optionally save results to CSV files
load_data.to_csv("load_data.csv")
load_forecast_data.to_csv("load_forecast_data.csv")
load_and_forecast_data.to_csv("load_and_forecast_data.csv")

# Fetching actual generation data
generation_data = client.query_generation(country_code, start=start, end=end)
print("Actual Generation Data:")
print(generation_data)

# Fetching forecasted generation data
generation_forecast_data = client.query_generation_forecast(country_code, start=start, end=end)
print("\nGeneration Forecast Data:")
print(generation_forecast_data)

# Fetching wind and solar generation forecast
wind_solar_forecast_data = client.query_wind_and_solar_forecast(country_code, start=start, end=end)
print("\nWind and Solar Forecast Data:")
print(wind_solar_forecast_data)

# Optionally save results to CSV files
generation_data.to_csv("generation_data.csv")
generation_forecast_data.to_csv("generation_forecast_data.csv")
wind_solar_forecast_data.to_csv("wind_solar_forecast_data.csv")
"""
