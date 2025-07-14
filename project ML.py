import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv(treamlit\8th- slr with streamlit\Salary_Data.csr'C:\Users\LENOVO\Downloads\8th- slr with sv')

# Split the data into independent (X) and dependent (y) variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the dataset into 80% training and 20% testing
from sklearn.model_selection import train_test_split
# Import and fit the Simple Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test set results
y_pred = regressor.predict(X_test)

# Visualize the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Same line for test set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Get the model parameters (slope and intercept)
m = regressor.coef_[0]
c = regressor.intercept_

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Predict salary for 12 and 20 years of experience
y_12 = m * 12 + c
y_20 = m * 20 + c

print(f"Predicted salary for 12 years experience: {y_12}")
print(f"Predicted salary for 20 years experience: {y_20}")

# Model accuracy
bias = regressor.score(X_train, y_train)
variance = regressor.score(X_test, y_test)

print(f"Training Accuracy (Bias): {bias}")
print(f"Testing Accuracy (Variance): {variance}")

# Save the model using pickle
import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

print("Model has been pickled and saved as linear_regression_model.pkl")

# Fix for last line
import os
print("Current working directory:", os.getcwd())  # If you want to show the path where the model is saved



