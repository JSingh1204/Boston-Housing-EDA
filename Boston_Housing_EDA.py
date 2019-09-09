import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

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

sns.pairplot(df[col_study],height=2) 
plt.show()
print(df[col_study].corr())
plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot=True)
plt.show()

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



X=df['LSTAT'].values.reshape(-1,1)
y=df['MEDV']

model=LinearRegression(fit_intercept=True)
model.fit(X,y)
print(model.coef_)
print(model.intercept_)
plt.figure(figsize=(10,8))
sns.regplot(X,y,color='g')
plt.ylabel('Median value of owner-occupied homes in $1000 s')
plt.xlabel(' percentage of lower status of the population')
plt.show()


