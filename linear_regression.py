# from nfl_madden.py import df

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
sns.set(style="whitegrid")

qbr = df.drop(['Fantasy Points', 'Season', 'Game', 'Players', 'Position_x', 'Position_y', 'Name', 'Team', 'Handedness', 'Week'], axis=1)

y = qbr['Passing Yards']
X = qbr.drop('Passing Yards', axis=1)

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

imp.sort_values(1, ascending=False)

