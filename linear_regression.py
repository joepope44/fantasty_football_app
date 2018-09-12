# from nfl_madden.py import df

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
sns.set(style="whitegrid")

tmp.head()

y = tmp['Passing Yards']
X = tmp.drop(['Passing Yards'], axis=1)

labels = X[['Team', 'Name']]

X.drop(['Team', 'Name', 'Opponent', 'Fantasy Points', 'Total Salary', 'Total Salary_DST'], axis=1, inplace=True)


regr = LinearRegression()

model = regr.fit(X, y)

# Create decision tree classifer object
clf = RandomForestClassifier(random_state=0, n_jobs=-1)

# Train model
tree = clf.fit(X, y)
importances = tree.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X.columns[i] for i in indices]

imp = pd.DataFrame(list(zip(names, importances[indices])))

imp.sort_values(1, ascending=False)[:10]

# LINEAR REGRESSION

X_train, X_test, y_train, y_test = \
	train_test_split(X, y, test_size=0.2, random_state=44)

# Create a dummy regressor
dummy_mean = DummyRegressor(strategy='mean')

# "Train" dummy regressor
dummy_mean.fit(X_train, y_train)

# Get R-squared score
dummy_mean.score(X_test, y_test) # -0.11 R2 using mean

# vanilla linear regression - R2 is -4!
model = regr.fit(X_train, y_train)

model.score(X_test, y_test)

# LASSO

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X_train)

# Create lasso regression with alpha value
lasso = Lasso(alpha=0.9)

# Fit the linear regression
lasso_model = lasso.fit(X_std, y_train)

lasso_model.score(X_test, y_test)

# Create ridge regression with an alpha value
ridge_model = Ridge(alpha=0.3)

# Fit the linear regression
ridge_model = regr.fit(X_std, y_train)

ridge_model.score(X_test, y_test)


model = sm.OLS(y_test, X_test)
fit = model.fit()
fit.summary()

data = pd.DataFrame()

data['predict']=fit.predict(X_std)
data['resid']=y_train - data.predict
with sns.axes_style('white'):
    plot=data.plot(kind='scatter',
                  x='predict', y='resid', figsize=(10,6))


X.to_csv('data/regr.csv')

sns.distplot(y)

