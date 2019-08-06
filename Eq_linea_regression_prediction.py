"""
The goal of this project is to check how well we can predict the values of magnitude
for the given input dataset of years and depth and see if Date or Depth are
good predictor for magnitudes of earthquakes.	
"""
import pandas as pd
import numpy as np
from dateutil import parser
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Save all data from "database.csv" file to a DataFrame.
data = pd.read_csv('database.csv', converters={0: lambda x: parser.parse(x).year})
# Save all "Earthquakes" inside the data DataFrame.
data = data[data['Type'] == 'Earthquake']

# Save all years inside "year" DataFrame.
Date = data['Date']
# Change the axis of the "Date" column.
Date = Date[:, np.newaxis]
# Save "magnitude" column from "data" DataFrame to "Magnitude".
Magnitude = data['Magnitude']
# Split the data to two part so we can compare our porediction and calculate the Error.
Date1, Date2, Magnitude1, Magnitude2 = train_test_split(Date, Magnitude, train_size=0.5)
# Create a linear regression model where Magnitude is a function of Date.
model = LinearRegression(fit_intercept=True).fit(Date1, Magnitude1)
print("\n-*-*-*-*-*-*-*-*-*-*-*-*-*")
print("Regression coefficient and model intercept:")
print(f"     {model.coef_[0]},     {model.intercept_}")
print("-*-*-*-*-*-*-*-*-*-*-*-*-*\n")

# Predict the values of magnitude according to "Date" and also calculate the mean square error.
Magnitude_pred = model.predict(Date2)
MSE_date = np.sum((Magnitude2 - Magnitude_pred)**2)/Magnitude.shape[0]
print("\n-*-*-*-*-*-*-*-*-*-*-*-*-*")
print(f"The Mean Square Error for Magnitude-Date prediction is: {MSE_date}")
print("-*-*-*-*-*-*-*-*-*-*-*-*-*\n")

# Plot the first prediction.
plt.subplot(2, 1, 1)
plt.scatter(Date2, Magnitude2)
plt.plot(Date2, Magnitude_pred, color='r')
plt.title('Magnitude-Date')
plt.ylabel('Magnitude')
plt.xlabel('Years')
plt.legend(loc='best')


# Split our data and predict the values of magnitude according to "Date" 
# and calculates the mean square error.
Depth = data['Depth']
Depth = Depth[:, np.newaxis]
Depth1, Depth2, Magnitude21, Magnitude22 = train_test_split(Depth, Magnitude, train_size=0.5)
model2 = LinearRegression().fit(Depth1, Magnitude21)
Magnitude_pred2 = model.predict(Depth2)
MSE_depth = np.sum((Magnitude22 - Magnitude_pred2)**2)/Magnitude.shape[0]

# Plot the second prediction.
plt.subplot(2, 1, 2)
plt.scatter(Depth2, Magnitude22)
plt.plot(Depth2, Magnitude_pred2, color='r')
plt.title('Magnitude-Depth')
plt.ylabel('Magnitude')
plt.xlabel('Depth')
plt.legend(loc='best')

print("\n-*-*-*-*-*-*-*-*-*-*-*-*-*")
print(f"The Mean Square Error for Magnitude-Depth prediction is: {MSE_depth}")
print("-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
