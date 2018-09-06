import pandas as pd

madden = pd.read_csv('data/madden3.csv')

nfl = pd.read_csv('data/nfl_2017_clean.csv', index_col='Index')

# QB MERGE BETWEEN NFL DATA AND MADDEN RATINGS 2017

nfl_qb = nfl[['Fantasy Points', 'Game', 'Pass Attempts', \
	'Pass Completions', 'Passes Intercepted', 'Passing Conversions', \
	'Passing Yards', 'Players', 'Rushing Attempts', \
	'Rushing Conversions', 'Rushing Touchdowns', 'Rushing Yards', \
	'Touchdown Passes', 'Position', 'Week']]

nfl_qb = nfl_qb[nfl_qb['Position'] == 'QB']

nfl_qb.columns

mad_qb = madden[['Position', 'Age', 'Acceleration', 'Agility', 'Awareness', 'Ball Carrier Vision',\
		'Carrying', 'Elusiveness', 'Handedness', 'Height', 'Injury', 'Name', \
		'Overall Rating', 'Play Action', 'Speed', 'Stamina', 'Stiff Arm', \
		'Strength', 'Team', 'Throw Accuracy Deep', 'Throw Accuracy Mid', \
		'Throw Accuracy Short', 'Throw Power', 'Throw on the Run', \
		'Total Salary', 'Toughness', 'Weight', 'Years Pro', 'Season'
		]]

# filter to QB and 2017-2018 season
mad_qb = mad_qb[mad_qb['Position'] == 'QB']
mad_qb = mad_qb[mad_qb['Season'] == '2017-2018']
mad_qb.head()

mad_qb.Name.head()
nfl.Players.head()

df = pd.merge(nfl_qb, mad_qb, how='inner', left_on='Players', right_on='Name')

df.to_csv('data/qb_clean.csv')

## TEST OUT THE BIG DATA



