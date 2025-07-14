import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
dataset=pd.read_csv(r'C:\Users\LENOVO\Downloads\10th- mlr\10th- mlr\MLR\Investment.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]



X=pd.get_dummies(X,dtype=int)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
 
 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)

bias =regressor.score(X_train, y_train)
bias

variance = regressor.score(X_test,y_test)
variance

slope =regressor.coef_
print(slope)
intercept =regressor.coef_
print(intercept)
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = X[:,[1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
