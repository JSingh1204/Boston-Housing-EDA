import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

df=pd.read_csv('/Users/jashan/Desktop/Development/ML/Boston-Housing-EDA/0304/housing.Data',delim_whitespace=True, header=None)
# print(df.describe())
# print(df.info())
col=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
col_study=['CRIM','ZN','INDUS','CHAS','MEDV']
df.columns=col
# print(df.head())
#print(df.describe())
pd.options.display.float_format='{:,.2f}'.format 


# 7. Attribute Information:

#     1. CRIM      per capita crime rate by town
#     2. ZN        proportion of residential land zoned for lots over 
#                  25,000 sq.ft.
#     3. INDUS     proportion of non-retail business acres per town
#     4. CHAS      Charles River dummy variable (= 1 if tract bounds 
#                  river; 0 otherwise)
#     5. NOX       nitric oxides concentration (parts per 10 million)
#     6. RM        average number of rooms per dwelling
#     7. AGE       proportion of owner-occupied units built prior to 1940
#     8. DIS       weighted distances to five Boston employment centres
#     9. RAD       index of accessibility to radial highways
#     10. TAX      full-value property-tax rate per $10,000
#     11. PTRATIO  pupil-teacher ratio by town
#     12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
#                  by town
#     13. LSTAT    % lower status of the population
#     14. MEDV     Median value of owner-occupied homes in $1000's

# sns.pairplot(df[col_study],height=2) 
# plt.show()
# print(df[col_study].corr())
# plt.figure(figsize=(16,10))
# sns.heatmap(df.corr(),annot=True)
# plt.show()

#############################  LINEAR REGRESSION ##############################

# X=df['RM'].values.reshape(-1,1)
# y=df['MEDV']
# model=LinearRegression(fit_intercept=True)
# #print(model)
# model.fit(X,y)
# print(model.coef_)
# print(model.intercept_)
# plt.figure(figsize=(10,8))
# sns.regplot(X,y,color='r')
# plt.xlabel('Average number of rooms per dewelling')
# plt.ylabel('Median value of owner-occupied homes in $1000 s')
# plt.show()


# X=df['LSTAT'].values.reshape(-1,1)
# y=df['MEDV']

# model=LinearRegression(fit_intercept=True)
# model.fit(X,y)
# print(model.coef_)
# print(model.intercept_)
# plt.figure(figsize=(10,8))
# sns.regplot(X,y,color='g')
# plt.ylabel('Median value of owner-occupied homes in $1000 s')
# plt.xlabel(' percentage of lower status of the population')
# plt.show()

######################### ROBUST REGRESSION ######################################

# X=df['RM'].values.reshape(-1,1)
# y=df['MEDV']

# ransac=RANSACRegressor()
# print(ransac)
# print('----------')
# print(ransac.fit(X,y))

# inlier=ransac.inlier_mask_
# outlier=np.logical_not(inlier)



# line_X=np.arange(3,10,2)
# line_y_ransac=ransac.predict(line_X.reshape(-1,1))

# sns.set(style='darkgrid',context='notebook')
# plt.figure(figsize=(12,10))
# plt.scatter(X[inlier],y[inlier],c='Blue',marker='o', label='inliers')
# plt.scatter(X[outlier],y[outlier],c='Green', marker='x', label='outliers')
# plt.plot(line_X,line_y_ransac,color='Red')
# plt.xlabel('Average number of rooms dewelling')
# plt.ylabel('Median value of owner-occupied homes in 1000 s')
# plt.legend(loc='upper left')
# print(ransac.estimator_.coef_)
# print(ransac.estimator_.intercept_)
# plt.show()


################################# Practice for Robust Regression #################################

# X=df['LSTAT'].values.reshape(-1,1)
# y=df['MEDV']

# random= RANSACRegressor()

# print(random.fit(X,y))

# inlier=random.inlier_mask_
# outlier=np.logical_not(inlier)

# X_line=np.arange(0,40,1)
# y_line=random.predict(X_line.reshape(-1,1))

# sns.set(style='darkgrid',context='notebook')
# plt.figure(figsize=(10,8))
# plt.scatter(X[inlier],y[inlier],color='purple',marker='x')
# plt.scatter(X[outlier],y[outlier],color='brown', marker='o')
# plt.plot(X_line,y_line,color='Green')
# plt.xlabel('percentage of lower status of the population')
# plt.ylabel('Median value of owner-occupied homes in 1000 s')
# plt.legend(loc='upper left')
# plt.show()


############################# Performance Evaluation of Regression Model ##########################

######### Residual Analysis #################

from sklearn.model_selection import train_test_split

X=df['LSTAT'].values.reshape(-1,1)

y=df['MEDV'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_train_predict=lr.predict(X_train)
y_test_predict=lr.predict(X_test)
# plt.figure(figsize=(10,8))
# plt.scatter(y_train_predict,y_train_predict - y_train, color='Green', marker='x',label='Training Data')
# plt.scatter(y_test_predict,y_test_predict - y_test, color='Red', marker='o', label='Test Data')
# plt.xlabel('Predicted values')
# plt.ylabel('Residuals')
# plt.legend(loc='upper left')
# plt.hlines(y=0,xmin=-10, xmax=50,lw=2,color='K')
# plt.xlim([-10,50])
# plt.show()



############ Mean Squared Error ##############

# from sklearn.metrics import mean_squared_error

# print(mean_squared_error(y_train,y_train_predict))
# print(mean_squared_error(y_test,y_test_predict))


######### Coefficient of Determination ###########

from sklearn.metrics import r2_score

print(r2_score(y_train,y_train_predict))
print(r2_score(y_test,y_test_predict))
















































