import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.stats import norm

# ---------------- Configuration -------------------
# File path of your combined CSV (make sure it contains both columns)
FILE_PATH = "/Users/gurmehersinghpuri/Documents/Masters Project/co2Map_api/Predictions/BB_Generation.csv"

# Name of the columns – update these to match your CSV
TRUE_COL = "True_Generation"  # Column with true intensity values
PRED_COL = "Predicted_BB_GenerationIntensity"  # Column with predicted intensity values


# ---------------------------------------------------

def load_and_prepare_data(file_path, true_col, pred_col):
    # Load data
    df = pd.read_csv(file_path)

    # Convert the specified columns to numeric (if necessary)
    df[true_col] = pd.to_numeric(df[true_col], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")

    # Drop rows where either true or predicted values are missing
    df = df.dropna(subset=[true_col, pred_col])
    return df


def plot_distribution(df, true_col, pred_col):
    # Fit a normal distribution to each column
    mu_true, sigma_true = norm.fit(df[true_col])
    mu_pred, sigma_pred = norm.fit(df[pred_col])

    # Define an x-axis range based on the minimum and maximum across both sets of values
    x_min = min(df[true_col].min(), df[pred_col].min())
    x_max = max(df[true_col].max(), df[pred_col].max())
    x = np.linspace(x_min, x_max, 100)

    # Calculate the probability density functions for each
    pdf_true = norm.pdf(x, mu_true, sigma_true)
    pdf_pred = norm.pdf(x, mu_pred, sigma_pred)

    # Create overlaid histogram traces for true and predicted values
    trace_hist_true = go.Histogram(
        x=df[true_col],
        histnorm="probability density",
        opacity=0.6,
        name="True",
        marker=dict(color="blue")
    )
    trace_hist_pred = go.Histogram(
        x=df[pred_col],
        histnorm="probability density",
        opacity=0.6,
        name="Predicted",
        marker=dict(color="red")
    )

    # Create density (curve) traces from the fitted normal distribution
    trace_density_true = go.Scatter(
        x=x,
        y=pdf_true,
        mode="lines",
        name=f"True Fit (μ={mu_true:.2f}, σ={sigma_true:.2f})",
        line=dict(color="blue", width=3)
    )
    trace_density_pred = go.Scatter(
        x=x,
        y=pdf_pred,
        mode="lines",
        name=f"Predicted Fit (μ={mu_pred:.2f}, σ={sigma_pred:.2f})",
        line=dict(color="red", width=3)
    )

    # Combine all traces into one figure
    fig = go.Figure(data=[trace_hist_true, trace_hist_pred, trace_density_true, trace_density_pred])

    # Update layout settings
    fig.update_layout(
        title="Distribution: True vs Predicted Intensity",
        xaxis_title="Intensity",
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
        xaxis_rangeslider_visible=True,
        hovermode="x unified"
    )

    fig.show()


def main():
    df = load_and_prepare_data(FILE_PATH, TRUE_COL, PRED_COL)
    if df.empty:
        print("No data found or the specified columns are missing!")
        return

    plot_distribution(df, TRUE_COL, PRED_COL)


if __name__ == "__main__":
    main()