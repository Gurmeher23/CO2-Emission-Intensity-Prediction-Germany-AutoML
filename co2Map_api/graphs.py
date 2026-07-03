import os
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Set the folder containing all CSV files
DATA_DIR = "/Users/gurmehersinghpuri/Documents/Masters Project/co2Map_api/Predictions"

# Create a dictionary of available files based on naming convention
files = os.listdir(DATA_DIR)
data_options = {}

for f in files:
    if f.endswith(".csv"):
        try:
            state, dtype = f.replace(".csv", "").split("_")
            if state not in data_options:
                data_options[state] = {}
            data_options[state][dtype.lower()] = f
        except ValueError:
            pass  # Skip files not matching pattern

# Define the state list in the desired order
state_list = ["DE", "BB", "BW", "BY", "HE", "MV", "NI", "NW", "RP", "SH", "SL", "SN", "ST", "TH"]

# Dash App
app = Dash(__name__)
app.layout = html.Div([
    html.H2("Select State and Data Type"),

    html.Label("State:"),
    dcc.Dropdown(
        id="state-dropdown",
        options=[{"label": state, "value": state} for state in state_list],
        value=state_list[0]
    ),

    html.Label("Data Type:"),
    dcc.Dropdown(
        id="type-dropdown",
        options=[
            {"label": "Generation", "value": "generation"},
            {"label": "Consumption", "value": "consumption"}
        ],
        value="consumption"
    ),

    dcc.Graph(id="intensity-graph")
])

@app.callback(
    Output("intensity-graph", "figure"),
    Input("state-dropdown", "value"),
    Input("type-dropdown", "value")
)
def update_graph(selected_state, selected_type):
    # Look up file using the state selected in the fixed order list
    file_name = data_options.get(selected_state, {}).get(selected_type)
    if not file_name:
        return px.line(title="No data available for this selection.")

    file_path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(file_path)

    # Convert and clean timestamp (dayfirst handles dd/mm/yyyy)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # Debug info
    print(f"\n📁 {file_name}")
    print(f"📅 Date Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print("🧾 Columns:", df.columns.tolist())

    # Dynamically choose the "true" column based on type
    true_col = "True_Consumption" if selected_type == "consumption" else "True_Generation"

    # Dynamically find predicted column (any column that includes 'predicted')
    predicted_col = next((col for col in df.columns if "predicted" in col.lower()), None)

    # Check if both required columns exist
    if true_col not in df.columns or not predicted_col:
        return px.line(title="Required columns not found in the selected file.")

    # Convert to numeric
    df[true_col] = pd.to_numeric(df[true_col], errors="coerce")
    df[predicted_col] = pd.to_numeric(df[predicted_col], errors="coerce")
    df = df.dropna(subset=[true_col, predicted_col])

    # Plot
    fig = px.line(
        df,
        x="timestamp",
        y=[true_col, predicted_col],
        labels={"value": "Intensity", "timestamp": "Time"},
        title=f"Actual vs Predicted - {selected_state} ({selected_type.capitalize()})"
    )
    fig.update_layout(xaxis_rangeslider_visible=True, hovermode="x unified")
    return fig

if __name__ == "__main__":
    app.run(debug=True)