# Stock Price Prediction

## Overview

This project utilizes machine learning models to predict stock prices of Tesla (TSLA) and Google (GOOGL) based on historical stock market data. The goal is to develop an accurate predictive model that can forecast stock prices and provide insights into market trends.

## Features

Data Collection: Utilized historical stock price data.

Exploratory Data Analysis (EDA): Analyzed trends, moving averages, and volatility.

Machine Learning Models: Implemented Linear Regression & LSTM for prediction.

Model Evaluation: Used Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

Visualization: Plotted stock trends and predictions using Matplotlib & Seaborn.

## Technologies Used

Programming Language: Python

Libraries: TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

Platform: Jupyter Notebook / Google Colab

## Installation

To run this project locally, install the required dependencies:
```
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
```
## Usage

Load the Dataset: Import stock data from Yahoo Finance or a CSV file.

Preprocess Data: Handle missing values and normalize stock prices.

Train the Model: Choose between Linear Regression & LSTM.

Evaluate Performance: Check prediction accuracy using MAE & RMSE.

Visualize Predictions: Plot actual vs predicted prices.

## Results

Achieved up to 92% accuracy using LSTM for stock price predictions.

Linear Regression performed well for short-term trends but struggled with volatility.

## Project Structure
```
Stock_Market_Prediction(Tesla, Google)/
│── data/                   # Contains stock market datasets
│── models/                 # Trained models
│── notebooks/              # Jupyter Notebooks with implementation
│── src/                    # Source code
│── README.md               # Project documentation
```
## Future Enhancements

Implement Sentiment Analysis using financial news data.

Experiment with Recurrent Neural Networks (RNNs) for improved long-term predictions.

Include real-time stock data integration.

## Contributors

Khushal Joshi 

## License

This project is open-source under the MIT License.















