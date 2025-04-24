# Bitcoin Price Prediction Project

This project aims to analyze historical Bitcoin price data and predict future prices using a Linear Regression model. It includes data preprocessing, exploratory data analysis, and machine learning to forecast Bitcoin's closing prices.

## Project Overview
- **Goal**: Predict Bitcoin's closing prices based on historical data.
- **Approach**: Use Linear Regression with features like opening, high, and low prices to predict the closing price.
- **Tools**: Python, pandas, scikit-learn, matplotlib/seaborn for visualization.

## Steps

### 1. Data Loading and Preprocessing
- Loaded the Bitcoin historical dataset.
- Converted the "date" column to datetime format for time-series analysis.
- Set the "date" column as the index for easier trend visualization.
- Selected features (X) and target (y):
  - **Features (X)**: Open, High, Low prices
  - **Target (y)**: Close price (the value to predict)

### 2. Exploratory Data Analysis (EDA)
- **Visualizations**:
  - **Bitcoin Closing Price Over Time**: Displayed price fluctuations.
  - **High & Low Prices Over Time**: Compared daily high and low prices.
  - **Trading Volume Over Time**: Analyzed trading activity.
  - **Correlation Heatmap**: Examined relationships between features (open, high, low, close, volume).
- **Key Insights**:
  - Bitcoin prices are highly volatile with clear up-and-down trends.
  - High and low prices strongly correlate with closing prices.
  - Trading volume often spikes during significant price movements.

### 3. Linear Regression Model
- **Why Linear Regression?**: It’s a simple yet effective method to predict the closing price based on input features (Open, High, Low).
- **Process**:
  - Split data into training (80%) and testing (20%) sets.
  - Trained a Linear Regression model using scikit-learn.
  - Made predictions on the test set.
  - Evaluated model performance using:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
- **Code**:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
```

### 4. Model Interpretation
- **Low MAE/MSE**: Indicates accurate predictions.
- **High MAE/MSE**: Suggests the need for improvement, such as:
  - Using advanced models (e.g., LSTM neural networks).
  - Including additional features (e.g., trading volume, moving averages).

## Key Learnings
- Bitcoin prices exhibit significant volatility, making prediction challenging.
- Linear Regression provides a baseline for price prediction but may not capture complex patterns.
- More sophisticated models or additional features could improve accuracy.

## Future Improvements
- Experiment with advanced models like LSTM or ARIMA for time-series forecasting.
- Incorporate more features, such as technical indicators (e.g., RSI, MACD) or external factors (e.g., market sentiment).
- Perform hyperparameter tuning to optimize the model.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## How to Run
1. Clone the repository.
2. Install required libraries: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python script to preprocess data, visualize trends, and train the model.