import pandas as pd
import numpy as np

qb_nfl_fields = [
	'Fantasy Points', 'Game', 'Pass Attempts', 'Pass Completions',
	'Passes Intercepted', 'Passing Conversions', 'Passing Yards', 'Players',
	'Rushing Attempts', 'Rushing Conversions', 'Rushing Touchdowns',
	'Rushing Yards', 'Touchdown Passes', 'Position', 'Week'
]

rb_nfl_fields = [
	'Fantasy Points', 'Game', 'Players', 'Receiving Conversions',
	'Receiving Yards', 'Receptions', 'Rushing Attempts', 'Rushing Touchdowns',
	'Rushing Yards', 'Touchdown Receptions', 'Position', 'Week'
]

wr_nfl_fields = te_nfl_fields = rb_nfl_fields

dst_nfl_fields = [
	'Fantasy Points', 'Interceptions', 'Passing Yards Allowed', 'Points Against',
	'Rushing Yards Allowed', 'Sacks', 'Safeties', 'Total Yards Allowed',
	'Touchdowns', 'Position', 'Week', 'Team'
]


def filter_nfl_data(pos, fields, year):
	"""
	function filters nfl data by position and fields. data should be pre-filtered
	by season.
	:param pos: NFL Position, choice are 'QB','RB','TE,'WR','DST','K'
	:param fields: List of fields that should be kept and treated as features
	:return: pandas dataframe filtered to position and fields
	"""
	nfl = pd.read_csv('data/nfl_' + str(year) + '_clean.csv')
	nfl_by_pos = nfl[nfl['Position'] == pos]
	return nfl_by_pos[fields]

# 2017 NFL data

nfl_qb = filter_nfl_data('QB', qb_nfl_fields)
nfl_rb = filter_nfl_data('RB', rb_nfl_fields)
nfl_wr = filter_nfl_data('WR', wr_nfl_fields)
nfl_te = filter_nfl_data('TE', te_nfl_fields)
nfl_dst = filter_nfl_data('DST', dst_nfl_fields)
nfl_dst_agg = nfl_dst.groupby('Team').mean().reset_index()

# 2018 NFL data

nfl_qb_2018 = filter_nfl_data('QB', qb_nfl_fields, 2018)


# MADDEN FIELDS PER POSITION

mad_qb_fields = [
	'Position', 'Age', 'Acceleration', 'Agility', 'Awareness',
	'Ball Carrier Vision', 'Carrying', 'Elusiveness', 'Handedness',
	'Height', 'Injury', 'Name', 'Overall Rating', 'Play Action', 'Speed',
	'Stamina', 'Stiff Arm', 'Strength', 'Team', 'Throw Accuracy Deep',
	'Throw Accuracy Mid', 'Throw Accuracy Short', 'Throw Power',
	'Throw on the Run', 'Total Salary', 'Toughness', 'Weight', 'Years Pro',
	'Season'
]

mad_dst_fields = [
	'Age', 'Acceleration', 'Agility', 'Awareness', 'Block Shedding',
	'Finesse Moves', 'Height', 'Hit Power', 'Impact Blocking',
	'Man Coverage', 'Name', 'Overall Rating', 'Play Recognition',
	'Position', 'Power Moves', 'Speed', 'Spin Move', 'Stamina', 'Strength',
	'Tackle', 'Team', 'Weight', 'Years Pro', 'Zone Coverage', 'Season'
]

mad_offline_fields = [
	'Height', 'Overall Rating', 'Pass Block',
	'Position', 'Stamina', 'Strength', 'Team',
	'Weight', 'Years Pro', 'Season'
]

mad_rb_fields = [
	'Age', 'Acceleration', 'Agility', 'Awareness', 'Ball Carrier Vision',
	'Carrying', 'Catch', 'Catch in Traffic', 'Elusiveness', 'Height',
	'Injury', 'Juke Move', 'Jumping', 'Name', 'Overall Rating', 'Position',
	'Speed', 'Stamina', 'Stiff Arm', 'Team', 'Toughness', 'Trucking', 'Weight',
	'Years Pro', 'Total Salary', 'Years Pro', 'Season'
]
# same fields relevant for WRs, TEs and RBs
mad_wr_fields = mad_te_fields = mad_rb_fields


######

def filter_madden_data(pos, fields, season):
	"""
	:param pos: NFL Position, choice are 'QB','RB','TE,'WR','DST','K'
	:param fields: List of fields that should be kept and treated as features
	:param season: '2017-2018'
	:return:
	"""
	madden = pd.read_csv('data/madden3.csv')

	# handle DST and OL positions
	dst_list = ['LOLB', 'MLB', 'RE', 'LE', 'CB', 'FS', 'SS', 'ROLB', 'DT']
	ol_list = ['LT', 'LG', 'C', 'RG', 'RT']

	if pos == 'DST':
		temp = madden[madden['Position'].isin(dst_list)]
	elif pos == 'OL':
		temp = madden[madden['Position'].isin(ol_list)]
	else:
		temp = madden[madden['Position'] == pos]

	temp = temp[temp['Season'] == season]
	temp = temp[fields]

	return temp

# 2017 Madden filtered data

mad_qb = filter_madden_data('QB', mad_qb_fields, '2017-2018')
mad_rb = filter_madden_data('HB', mad_rb_fields, '2017-2018')
mad_te = filter_madden_data('TE', mad_te_fields, '2017-2018')
mad_wr = filter_madden_data('WR', mad_wr_fields, '2017-2018')
mad_dst = filter_madden_data('DST', mad_dst_fields, '2017-2018')
mad_ol = filter_madden_data('OL', mad_offline_fields, '2017-2018')

# 2018 Madden filtered data

mad_qb_2018 = filter_madden_data('QB', mad_qb_fields, '2018-2019')

# take the max rating for each offensive line position, ignore all other ratings
mad_ol_agg = mad_ol.pivot_table(
	values='Overall Rating',
	index='Team',
	columns='Position',
	aggfunc=np.max).reset_index()

# create average of the OL
mad_ol_agg['OL_AVG'] = mad_ol_agg.mean(axis=1)

# take max rating for each defensive player
mad_dst_agg = mad_dst.pivot_table(
	values='Overall Rating',
	index='Team',
	columns='Position',
	aggfunc=np.max).reset_index()

mad_dst_agg['DST_AVG'] = mad_dst_agg.mean(axis=1).round(1)

mad_dst_agg['Backfield_AVG'] = mad_dst_agg[['CB', 'FS', 'SS']].mean(axis=1).round(1)
mad_dst_agg['Linebackers AVG'] = mad_dst_agg[['LOLB', 'MLB', 'ROLB']].mean(axis=1).round(1)
mad_dst_agg['DL_AVG'] = mad_dst_agg[['DT', 'LE', 'RE']].mean(axis=1).round(1)


def merge_nfl_madden(nfl_pos_df, mad_pos_df):
	"""
	:param nfl_pos_df: nfl positional dataframe, if DST use diff fields to merge
	:param mad_pos_df: madden positional dataframe
	:return: merged dataframe to be used for further processing
	"""
	return pd.merge(nfl_pos_df, mad_pos_df, how='inner', left_on='Players', right_on='Name')


def merge_nfl_madden_dst(nfl_pos_df, mad_pos_df):
	"""

	:param nfl_pos_df: DST NFL data
	:param mad_pos_df: DST madden data
	:return: merged dataframe, aggregate DST data
	"""
	return pd.merge(nfl_pos_df, mad_pos_df, how='inner', left_on='Team', right_on='Team')

# 2017 merged NFL and Madden data

dst_data = merge_nfl_madden_dst(nfl_dst_agg, mad_dst_agg)
qb_data = merge_nfl_madden(nfl_qb, mad_qb)
rb_data = merge_nfl_madden(nfl_rb, mad_rb)
wr_data = merge_nfl_madden(nfl_wr, mad_wr)
te_data = merge_nfl_madden(nfl_te, mad_te)

# 2018 merged NFL and Madden data
qb_data_2018 = merge_nfl_madden(nfl_qb_2018, mad_qb_2018)


# preprocess before regression analysis. look at player, their opponents DST
# their offensive line.

sched = pd.read_excel('data/2017_NFL_Schedule.xlsx')
sched2018 = pd.read_excel('data/2018_NFL_schedule.xlsx')

sched2018['Week'] = sched2018['Week'].astype(float)
sched['Week'] = sched['Week'].astype(float)


def regr_preprocess(pos_data, sched):
	"""
	:param pos_data: qb, rb, wr, te data
	:return: dataframe to be used for modeling
	"""

	# merge all home and away game schedule data with positional NFL and Madden data
	away = pd.merge(pos_data, sched, left_on=['Week', 'Team'], right_on=['Week', 'Away'])
	home = pd.merge(pos_data, sched, left_on=['Week', 'Team'], right_on=['Week', 'Home'])
	df = pd.concat([away, home])

	df.drop([
		'Game', 'Position_x', 'Position_y', 'Handedness', 'Players', 'Season', 'Day',
		'Date', 'Time', 'Winner/tie', 'At', 'Loser/tie', 'Away Team', 'Home Team ',
		'PtsW', 'PtsL', 'TOW', 'YdsL', 'TOL'
	], axis=1, inplace=True)

	# input one if players team is home team, else zero for away team.
	df['isHomeTeam'] = df.apply(lambda x: x['Team'] in x['Home'], axis=1).astype(int)
	# identify opponent
	df['Opponent'] = df.apply(lambda x: x['Away'] if x['isHomeTeam'] == 1 else x['Home'], axis=1)

	# now drop away and home fields.
	df.drop(['Home', 'Away'], axis=1, inplace=True)

	# merge offensive line stats
	df = pd.merge(df, mad_ol_agg, on='Team')

	# merge defensive stats
	df = pd.merge(df, dst_data, left_on='Opponent', right_on='Team')

	return df


# 2017 data, used to build regression model
tmp = regr_preprocess(qb_data, sched)
tmp.shape

qb2018 = regr_preprocess(qb_data_2018, sched2018)
qb2018.shape
qb2018.to_csv('data/qb2018.csv', index=False)


tmp.to_csv('data/check.csv')

#
# def test_data(player, position, opponent):
# 	tmp = regr_preprocess(qb_data)
#
# 	tmp['Name'] == player


# df.to_csv('data/qb_clean.csv')

## TEST OUT THE BIG DATA AND SAVE TO CSV FOR LATER ANALYSIS
## https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values/notebook


# big = pd.read_csv('data/NFL Play by Play 2009-2017 (v4).csv', low_memory=False)
#
# pbp2017 = big[big.Season == 2017]
#
# pbp2017.to_csv('data/pbp2017.csv')
