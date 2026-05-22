# Walmart M5 - Sales Demand Forecasting

ML-powered demand forecasting system built using LightGBM and Streamlit.

## Project Overview
Developed an end-to-end machine learning pipeline to predict daily store-level sales across multiple Walmart stores and SKUs.

## Tech Stack
- Python, Pandas, NumPy
- LightGBM
- Scikit-learn
- Matplotlib
- Streamlit

## Features
- Data preprocessing and feature engineering (lag features, rolling mean)
- LightGBM model training and evaluation
- Interactive Streamlit dashboard with store-wise filters
- RMSE: 3.414

## How to Run
pip install -r requirements.txt
streamlit run app.py

## Results
- Trained on 3,285 records across 3 stores and 3 items
- Achieved RMSE of 3.414
- Interactive dashboard with Actual vs Predicted visualization
