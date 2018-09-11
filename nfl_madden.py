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
	'Blocked Kicks', 'Fantasy Points', 'Fumble Recoveries', 'Game',
	'Interceptions', 'Passing Yards Allowed', 'Players', 'Points Against',
	'Rushing Yards Allowed', 'Sacks', 'Safeties', 'Total Yards Allowed',
	'Touchdowns', 'Position', 'Week', 'Team'
	]


def filter_nfl_data(pos, fields):
	"""
	function filters nfl data by position and fields. data should be pre-filtered
	by season.
	:param pos: NFL Position, choice are 'QB','RB','TE,'WR','DST','K'
	:param fields: List of fields that should be kept and treated as features
	:return: pandas dataframe filtered to position and fields
	"""
	nfl = pd.read_csv('data/nfl_2017_clean.csv')
	nfl_by_pos = nfl[nfl['Position'] == pos]
	temp = nfl_by_pos[fields]

	return temp


nfl_qb = filter_nfl_data('QB', qb_nfl_fields)
nfl_rb = filter_nfl_data('RB', rb_nfl_fields)
nfl_wr = filter_nfl_data('WR', wr_nfl_fields)
nfl_te = filter_nfl_data('TE', te_nfl_fields)
nfl_dst = filter_nfl_data('DST', dst_nfl_fields)

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
	'Finesse Moves', 'Height', 'Hit Power', 'Impact Blocking', 'Injury',
	'Jumping', 'Man Coverage', 'Name', 'Overall Rating', 'Play Recognition',
	'Position', 'Power Moves', 'Speed', 'Spin Move', 'Stamina', 'Strength',
	'Tackle', 'Team', 'Total Salary', 'Toughness', 'Weight', 'Years Pro',
	'Zone Coverage', 'Season'
	]

mad_offline_fields = [
	'Age', 'Acceleration', 'Agility', 'Awareness', 'Height', 'Injury', 'Name',
	'Overall Rating', 'Pass Block', 'Position', 'Speed', 'Stamina', 'Strength',
	'Team', 'Total Salary', 'Toughness', 'Weight', 'Years Pro', 'Season'
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


	#handle DST and OL positions
	DST_list = ['LOLB', 'MLB', 'RE', 'LE', 'CB', 'FS', 'SS', 'ROLB', 'DT']
	OL_list = ['LT', 'LG', 'C', 'RG', 'RT']

	if pos == 'DST':
		temp = madden[madden['Position'].isin(DST_list)]
	elif pos == 'OL':
		temp = madden[madden['Position'].isin(OL_list)]
	else:
		temp = madden[madden['Position'] == pos]

	temp = temp[temp['Season'] == season]
	temp = temp[fields]

	return temp


mad_qb = filter_madden_data('QB', mad_qb_fields, '2017-2018')
mad_rb = filter_madden_data('HB', mad_rb_fields, '2017-2018')
mad_te = filter_madden_data('TE', mad_te_fields, '2017-2018')
mad_wr = filter_madden_data('WR', mad_wr_fields, '2017-2018')
mad_dst = filter_madden_data('DST', mad_dst_fields, '2017-2018')
mad_ol = filter_madden_data('OL', mad_offline_fields, '2017-2018')


# need to figure out how to handle DST and OL, position doesn't match up with
# NFL data

def merge_nfl_madden(nfl_pos_df, mad_pos_df):
	"""
	function merges nfl_df and madden_df by position and season
	:param nfl_df: name of nfl dataframe, filtered by position
	:param madden_df: name of madden dataframe filtered by position
	:param dst: default to False. if merging defense, set to True
	:return:
	"""
	if nfl_pos_df.Position.iloc[1] == 'DST':
		temp = pd.merge(nfl_pos_df, mad_pos_df, how='inner', left_on='Team', right_on='Team')
	else:
		temp = pd.merge(nfl_pos_df, mad_pos_df, how='inner', left_on='Players', right_on='Name')
	return temp


dst_data = merge_nfl_madden(nfl_dst, mad_dst)
qb_data = merge_nfl_madden(nfl_qb, mad_qb)
rb_data = merge_nfl_madden(nfl_rb, mad_rb)
wr_data = merge_nfl_madden(nfl_wr, mad_wr)
te_data = merge_nfl_madden(nfl_te, mad_te)

# dst_data.to_csv('data/dst_data_clean.csv')

# preprocess before regression analysis. look at player, their opponents DST
# their offensive line.

sched = pd.read_excel('data/2017_NFL_Schedule.xlsx')
sched['Week'] = sched['Week'].astype(float)


def regr_preprocess(pos_data):

	global dst_data
	global mad_ol
	global sched

	# merge all home and away game schedule data with positional NFL and Madden data
	away = pd.merge(pos_data, sched, left_on=['Week','Team'], right_on=['Week','Away'])
	home = pd.merge(pos_data, sched, left_on=['Week','Team'], right_on=['Week','Home'])
	df = pd.concat([away, home])

	df.drop([
		'Game', 'Position_x', 'Position_y', 'Handedness', 'Players', 'Season', 'Day',\
		'Date', 'Time', 'Winner/tie', 'At', 'Loser/tie', 'Away Team', 'Home Team ',\
		'PtsW', 'PtsL', 'TOW', 'YdsL','TOL'\
	], axis=1, inplace=True)

	# input one if players team is home team, else zero for away team.
	df['isHomeTeam'] = df.apply(lambda x: x['Team'] in x['Home'], axis=1).astype(int)
	# identify opponent
	df['Opponent'] = df.apply(lambda x: x['Away'] if x['isHomeTeam'] == 1 else x['Home'],
							  axis=1)

	# now drop away and home fields.
	df.drop(['Home', 'Away'], axis=1, inplace=True)

	# merge offensive line stats
	# consider dropping off most of these stats
	# df = pd.merge(df, mad_ol, on='Team', suffixes='_OL')

	# merge defensive stats

	return df


tmp = regr_preprocess(qb_data)







# df.to_csv('data/qb_clean.csv')

## TEST OUT THE BIG DATA AND SAVE TO CSV FOR LATER ANALYSIS
## https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values/notebook


# big = pd.read_csv('data/NFL Play by Play 2009-2017 (v4).csv', low_memory=False)
#
# pbp2017 = big[big.Season == 2017]
#
# pbp2017.to_csv('data/pbp2017.csv')
