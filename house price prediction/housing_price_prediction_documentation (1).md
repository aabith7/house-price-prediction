
# Housing Price Prediction using Linear Regression

## Project Overview
This project aims to predict housing prices based on various features such as lot size, number of bedrooms, bathrooms, number of stories, and garage space. The prediction model employs linear regression, a fundamental statistical technique, to analyze the relationship between the features and the target variable (price).

## Table of Contents
1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Methodology](#methodology)
4. [Implementation](#implementation)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction
Predicting housing prices is a common problem in data science and real estate analytics. Accurate price predictions can aid buyers and sellers in making informed decisions. In this project, we use synthetic housing data to train a linear regression model, evaluate its performance, and visualize the results.

## Data Description
The dataset consists of the following features:

- **price**: The target variable representing the house price.
- **lotsize**: The size of the lot (in square feet).
- **bedrooms**: The number of bedrooms in the house.
- **bathrms**: The number of bathrooms in the house.
- **stories**: The number of stories in the house.
- **garagepl**: The number of garage spaces available.

## Methodology
The following steps were followed to build the linear regression model:

1. **Data Loading**: The dataset is loaded into a Pandas DataFrame.
2. **Feature Selection**: The features are separated from the target variable.
3. **Data Splitting**: The data is split into training (90%) and test (10%) sets using `train_test_split` from `scikit-learn`.
4. **Feature Normalization**: The features are normalized using `StandardScaler` to improve model performance.
5. **Model Training**: A linear regression model is trained using the training data.
6. **Prediction**: The model predicts housing prices on the test set.
7. **Evaluation**: Several metrics are computed to evaluate the model's performance, including:
   - Mean Absolute Percentage Error (MAPE)
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
8. **Visualization**: A scatter plot is generated to compare predicted prices with actual prices.

## Implementation
The following Python code implements the above methodology using `numpy`, `pandas`, `matplotlib`, and `scikit-learn` libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Load data
data = pd.read_csv('/content/synthetic_housing_data.csv')

# Separate features (X) and target (Y)
X = data.drop('price', axis=1)
Y = data['price']

# Split the data into training and test sets (90% train, 10% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model using Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

# Predict the test data
Y_pred = lr_model.predict(X_test)

# Compute Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(Y_test, Y_pred)
print(f"Linear Regression MAPE: {mape * 100:.2f}%")

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, Y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Compute Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Compute Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot predictions vs actual prices with different colors
plt.scatter(Y_test, Y_pred, color='red', label='Predicted Prices')  # Red for predicted
plt.scatter(Y_test, Y_test, color='blue', label='Actual Prices')    # Blue for actual

# Add labels, title, and legend
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices - Linear Regression')
plt.legend()  # Show the legend to distinguish the points
plt.show()
```

## Results
After executing the code, the following evaluation metrics are produced:

- **Linear Regression MAPE:** Indicates the percentage error of the predictions.
- **Mean Absolute Error (MAE):** The average magnitude of the errors.
- **Mean Squared Error (MSE):** The average of the squared errors.
- **Root Mean Squared Error (RMSE):** The square root of the MSE, providing error in the same unit as the target variable.

The scatter plot illustrates the relationship between the actual prices and predicted prices, where:
- Red points represent the predicted prices.
- Blue points represent the actual prices.

## Conclusion
This project demonstrates how to predict housing prices using linear regression. The model's performance can be evaluated using various metrics, and the results can be visualized to understand the prediction accuracy. Future work could include using more advanced regression techniques or incorporating additional features for improved predictions.

## References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
