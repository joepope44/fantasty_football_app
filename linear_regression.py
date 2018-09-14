# from nfl_madden.py import df

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
sns.set(style="whitegrid")

tmp.head()
tmp.to_csv('data/tmp.csv')

y = tmp['Passing Yards']
X = tmp.drop(['Passing Yards'], axis=1)
labels = X[['Team_x', 'Name']]

X.drop([
	'Team_x', 'Team_y', 'Name', 'Opponent', 'Total Salary', 'Fantasy Points_y',
	'Fantasy Points_x'], axis=1, inplace=True)


regr = LinearRegression()

model = regr.fit(X, y)
model.score(X, y)


rf = RandomForestRegressor(n_estimators=1800, max_features=3)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

RMSE(rf.predict(X_test), y_test)

gbm = GradientBoostingRegressor(n_estimators=1600, max_depth=3, learning_rate=.01)
gbm.fit(X_train, y_train)
gbm.score(X_test, y_test)
RMSE(gbm.predict(X_test),y_test)

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

feat_imps = list(zip(X_train.columns,rf.feature_importances_))
feat_imps = sorted(feat_imps, key = lambda x: x[1], reverse=False)
feat_imps = pd.DataFrame(feat_imps, columns=['feature','importance']).head()

feat_imps.plot(x='feature',y='importance',kind='barh')



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
lasso = Lasso(alpha=0.1)

# Fit the linear regression
lasso_model = lasso.fit(X_std, y_train)

lasso_model.score(X_test, y_test)

dict(zip((X_train).columns, lasso_model.coef_))


# Create ridge regression with an alpha value
ridge_model = Ridge(alpha=0.3)

# Fit the linear regression
ridge_model = regr.fit(X_std, y_train)

ridge_model.score(X_test, y_test)

# R2 .914
model = sm.OLS(y_train, X_train, hasconst=100)
fit = model.fit()
fit.summary()

# rslt = model.fit_regularized()
# rslt.summary()


data = pd.DataFrame()

data['predict']=fit.predict(X_train)
data['resid']=y_train - data.predict
with sns.axes_style('white'):
	plot=data.plot(
		kind='scatter',
		x='predict', y='resid', figsize=(10,6)
	)


from  sklearn.metrics import mean_squared_error

def RMSE(actual, predicted):
    return np.sqrt(mean_squared_error(actual,predicted))

print('OLS regression score val RMSE: %.3f \n' % RMSE(regr.predict(X_test), y_test))


X.to_csv('data/regr.csv')

df9 = X.join(y)
df9.head()

sns.distplot(y)

y.mean() # 211 yards

# DECISION TREE
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=0)
tree_model = tree.fit(X_train, y_train)

