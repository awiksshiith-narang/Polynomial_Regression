import pandas as p
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics

#Gathering data:
data = p.read_csv( 'C:/Users/AWIKSSHIITH/OneDrive/Desktop/housing.csv/housing.csv' )
x = np.array( data[ 'median_income' ] )
y = np.array( data[ 'median_house_value' ] )
x = x.reshape( len(x), 1 )
y = y.reshape( len(x), 1 )

#Splitting the data:
x_train = x[ : 14447 ]
y_train = y[ : 14447 ]
x_test = x[ 14447 : ]
y_test = y[ 14447 : ]

#Training and testing the data in linear regression:
regr = linear_model.LinearRegression()
regr.fit( x_train, y_train )
y_pred = regr.predict( x_test )

#Preparing the data for Polynomial regression:
polynom = preprocessing.PolynomialFeatures( degree = 3 ) #Polynomial regression = Cubic regression
x_train_poly = polynom.fit_transform( x_train )
x_test_poly = polynom.fit_transform( x_test )

#Training and testing the data in Polynomial regression:
poly_regr = linear_model.LinearRegression()
poly_regr.fit( x_train_poly, y_train )
y_pred_poly = poly_regr.predict( x_test_poly )

#Plotting the linear regression curve, polynomial regression curve and comparing with actual outputs:
plt.scatter( x_test, y_test, color = 'green' )
plt.plot( x_test, y_pred, color = 'red' )
plt.plot( x_test, y_pred_poly, color = 'blue' )
plt.title( 'California Housing Prices' )
plt.xlabel( 'Median income' )
plt.ylabel( 'Median house value' )
plt.show()

#Computing the root square error in linear regression and polynomial regression:
r_s_e = metrics.r2_score( y_test, y_pred )
p_r_s_e = metrics.r2_score( y_test, y_pred_poly )
print( 'The root square error in linear regression is: ', r_s_e )
print( 'The root square error in polynomial regression is: ', p_r_s_e )