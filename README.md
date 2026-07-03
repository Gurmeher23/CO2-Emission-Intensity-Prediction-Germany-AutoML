# CO2 Emission Intensity Prediction for Germany Using AutoML

**Electricity Carbon Intensity Prediction for Germany Using AutoML** — Master's Project, M.Sc. Computer Science (Artificial Intelligence), Albert-Ludwigs-Universität Freiburg.

**Authors:** Gurmeher Singh Puri, Udit Chawla
**Supervisors:** Nick Harder, André Biedenkapp

📄 **[Read the full project report](Final/Master_Project_Final_Report.pdf)**

## Overview

This project develops a machine learning system that predicts the CO2 emission intensity (gCO2/kWh) of the German electricity grid — nationally and for each federal state — using automated machine learning (AutoML). Germany's growing share of renewables causes large daily and seasonal swings in grid carbon intensity, and accurate forecasts enable carbon-aware computing, energy management optimization, and environmental impact assessment.

The system:

- Integrates data from the **ENTSO-E Transparency Platform** and the **CO2map.de** carbon tracking service, covering generation forecasts, cross-border trading flows, and real-time carbon intensity measurements for Germany and 11 neighboring countries (22+ cross-border connections, 4 API endpoints).
- Uses **AutoGluon** ensembles of three gradient boosting algorithms (**LightGBM, CatBoost, XGBoost**), optimized to shrink models from several gigabytes down to **100–130 MB per model** while retaining strong accuracy — suitable for real-time deployment.
- Predicts both **production-based and consumption-based** carbon intensity at the **national and federal-state level**.

## Key Results

| Target | Test R² | Test SMAPE |
|---|---|---|
| Germany (national), consumption-based | **0.91** | 7.6% |
| Germany (national), generation-based | **0.93** | 8.1% |
| Federal states (16 states × 2 targets) | mostly 0.77–0.87 | ~8–20% |

Training-set R² was above 0.90 for nearly all states, with the train–test comparison in the report showing the models learned generalizable patterns rather than memorizing.

## Repository Structure

| Path | Contents |
|---|---|
| `Final/` | **Final deliverables**: project report (PDF), the two main notebooks (`Main_Pipeline-2.ipynb` for data collection/training, `Get_Predictions-2.ipynb` for inference), step-by-step usage guides (PDF), and the pipeline configuration (`config_DE.yaml`) |
| `co2Map_api/` | Prediction pipeline scripts: ENTSO-E data fetching, AutoGluon training and all-states prediction (`ALLState_Autoglueon_Prediction.py`, `Prediction_model_Autogluon.py`), analysis/plotting utilities, and sample prediction outputs (`Predictions/`) |
| `CO2_Intensity/` | Carbon intensity calculation stack (based on [INATECH-CIG/CO2_Intensity](https://github.com/INATECH-CIG/CO2_Intensity), see its `LICENSE`): the `cosema` package for intensity computation, regionalization, gap filling and forecasting, plus Docker/Grafana configs used during the project |
| `Docs/` | Supporting documentation: dataset approach, ENTSO-E API coverage, TRACE summary, and sample manually-exported ENTSO-E data |

## How It Works

1. **Data collection** — hourly generation-per-type, load, and cross-border physical flows are pulled from ENTSO-E for Germany and its neighbors; measured carbon intensities come from CO2map.de.
2. **Feature engineering** — seasonal/time features, cross-border trading relationships, and renewable-share signals are built from the raw series, with gap filling and quality control.
3. **Model training** — AutoGluon trains and ensembles LightGBM, CatBoost, and XGBoost per target (national + each federal state, generation- and consumption-based), with presets tuned to balance accuracy against model size.
4. **Prediction** — `Final/Get_Predictions-2.ipynb` loads the trained models and produces day-ahead carbon intensity forecasts.

See `Final/Main_Pipeline_Usage_Guide.pdf` and `Final/Get_Predictions_Usage_Guide.pdf` for step-by-step instructions.

## Running It Yourself

You need a free [ENTSO-E Transparency Platform API key](https://transparency.entsoe.eu/). Replace the `YOUR_ENTSOE_API_KEY` placeholder in the scripts/notebooks (or set it in `CO2_Intensity/keys.yaml`, see `keys.yaml.example`) before running.

```bash
pip install autogluon entsoe-py pandas numpy matplotlib
```

Then follow the usage guides in `Final/`.

> **Note:** Trained model artifacts (~1.1 GB, 30 AutoGluon models) and the raw training datasets (~600 MB of Excel exports) are not included in this repository to keep it lightweight — they can be regenerated with `Final/Main_Pipeline-2.ipynb`, or shared on request.

## Acknowledgments

Thanks to the [CO2map.de](https://co2map.de) team and [ENTSO-E](https://transparency.entsoe.eu/) for data access, the INATECH-CIG group for the CO2_Intensity framework, and our supervisors Nick Harder and André Biedenkapp for their guidance.
