# Blood Glucose Prediction Regression

This project builds a regression model to predict future blood glucose levels using CGM time-series data and clinical intervention data.

## Overview

The goal of this project is to predict blood glucose values 30 minutes ahead based on:

- Continuous Glucose Monitor (CGM) data
- Clinical baseline information
- Daily intervention-related features such as nutrition and insulin use

The project also evaluates model performance across different patient subgroups.

## Features

The model uses several types of features, including:

- Recent CGM glucose readings
- Glucose change and slope
- Rolling statistics from recent glucose values
- Baseline clinical variables
- Daily nutrition features
- Daily insulin features
- Other intervention-related indicators

## Model

This project uses a LightGBM regression model for blood glucose prediction.

## Evaluation Metrics

Model performance is evaluated with:

- MAE
- RMSE
- NRMSE
- R²

Subgroup analysis is also performed to compare performance across different intervention groups.

## Dataset

The project uses hospital CGM and clinical data from Excel files, including:

- `Master Clarity log 1-101 for Dexcom FINAL.xlsx`
- `Full REDCap Data Intervention for Dexcom FINAL.xlsx`
- `Final Secondary CGM cohort pull_uncleaned.xlsx`

## Project Structure

```text
data/
  Master Clarity log 1-101 for Dexcom FINAL.xlsx
  Full REDCap Data Intervention for Dexcom FINAL.xlsx
  Final Secondary CGM cohort pull_uncleaned.xlsx

output/
  run_timestamp/
    train.log
    metrics.txt
    overall_metrics.json
    subset_metrics_table.csv
    test_predictions_with_subsets.csv
