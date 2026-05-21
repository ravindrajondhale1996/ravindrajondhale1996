# Real Estate Demand Forecasting

This project provides a minimal, reproducible pipeline for demand forecasting in real estate pricing using synthetic data, logistic regression (demand classification), and linear regression (price prediction). It also generates basic EDA outputs and visualizations.

## What it does
- Generates a synthetic real estate dataset (structured features + text descriptions).
- Performs lightweight EDA and saves summary statistics.
- Trains a logistic regression model to forecast demand.
- Trains a linear regression model to estimate prices.
- Evaluates using Accuracy, F1, and RMSE.
- Produces simple plots for results.

## Running the pipeline

```bash
python -m real_estate_forecasting.pipeline
# or
run-forecasting
```

Optional arguments:

```bash
python -m real_estate_forecasting.pipeline \
  --samples 1500 \
  --test-size 0.2 \
  --random-state 42 \
  --output-dir artifacts \
  --plots-dir reports
```

## Outputs
- `artifacts/synthetic_real_estate.csv`: generated dataset
- `artifacts/eda_summary.json`: EDA summary
- `artifacts/metrics.json`: model metrics
- `reports/demand_confusion_matrix.png`: demand classification plot
- `reports/price_predictions.png`: regression plot

## Data sources
The pipeline uses a synthetic dataset to remain self-contained. Replace the synthetic data generator with data from AWS Public Datasets or domain-specific APIs as needed for production use.

## Notes
The models and feature engineering are intentionally lightweight and meant to be a starting point for experimentation with additional data sources and feature enrichment.
