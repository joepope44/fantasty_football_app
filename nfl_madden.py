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
	:param year: year that the season begins. 2018-2019, should be 2018.
	:return: pandas DataFrame filtered to position and fields
	"""
	nfl = pd.read_csv('data/nfl_' + str(year) + '_clean.csv')
	nfl_by_pos = nfl[nfl['Position'] == pos]
	nfl_by_pos = nfl_by_pos[fields]
	nfl_by_pos['Year'] = float(year)
	return nfl_by_pos


# 2017 NFL data

nfl_qb = filter_nfl_data('QB', qb_nfl_fields, 2017)
nfl_rb = filter_nfl_data('RB', rb_nfl_fields, 2017)
nfl_wr = filter_nfl_data('WR', wr_nfl_fields, 2017)
nfl_te = filter_nfl_data('TE', te_nfl_fields, 2017)

# DST is handled differently
nfl_dst = filter_nfl_data('DST', dst_nfl_fields, 2017)
nfl_dst_agg = nfl_dst.groupby('Team').mean().reset_index()

# 2018 NFL data

nfl_qb_2018 = filter_nfl_data('QB', qb_nfl_fields, 2018)
nfl_rb_2018 = filter_nfl_data('RB', rb_nfl_fields, 2018)
nfl_wr_2018 = filter_nfl_data('WR', wr_nfl_fields, 2018)
nfl_te_2018 = filter_nfl_data('TE', te_nfl_fields, 2018)

nfl_dst_2018 = filter_nfl_data('DST', dst_nfl_fields, 2018)
nfl_dst_agg_2018 = nfl_dst_2018.groupby('Team').mean().reset_index()


# 2017-2018 data

nfl_qb_alltime = pd.concat([nfl_qb, nfl_qb_2018])
nfl_qb_alltime['Completion Rate'] = nfl_qb_alltime['Pass Completions'] / nfl_qb_alltime['Pass Attempts']
nfl_qb_alltime['Yards Per Attempt'] = nfl_qb_alltime['Passing Yards'] / nfl_qb_alltime['Pass Attempts']
nfl_qb_alltime['Yards Per Completion'] = nfl_qb_alltime['Passing Yards'] / nfl_qb_alltime['Pass Completions']

nfl_rb_alltime = pd.concat([nfl_rb, nfl_rb_2018])
nfl_te_alltime = pd.concat([nfl_te, nfl_te_2018])
nfl_wr_alltime = pd.concat([nfl_wr, nfl_te_2018])
nfl_dst_alltime = pd.concat([nfl_dst_agg, nfl_dst_agg_2018])



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

def filter_madden_data(pos, fields):
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

	temp = temp[fields]
	temp['Year'] = temp['Season'].apply(lambda x: x[:4])

	return temp


# Collect Madden filtered data by position, all seasons

mad_qb = filter_madden_data('QB', mad_qb_fields)
mad_rb = filter_madden_data('HB', mad_rb_fields)
mad_te = filter_madden_data('TE', mad_te_fields)
mad_wr = filter_madden_data('WR', mad_wr_fields)
mad_dst = filter_madden_data('DST', mad_dst_fields)
mad_ol = filter_madden_data('OL', mad_offline_fields)

# take the max rating for each offensive line position, ignore all other ratings
mad_ol_agg = mad_ol.pivot_table(
	values='Overall Rating',
	index='Team',
	columns='Position',
	aggfunc=np.max).reset_index()

# create average of the OL
mad_ol_agg['OL_AVG'] = mad_ol_agg.mean(axis=1)


# take max rating for each defensive player

def setup_mad_dst(df):
	"""
	:param df: input madden DST DataFrame
	:return: aggregated DataFrame, with average ratings of top defenders
	"""
	mad_dst_agg = df.pivot_table(
		values='Overall Rating',
		index=['Team', 'Year'],
		columns='Position',
		aggfunc=np.max).reset_index()

	mad_dst_agg['DST_AVG'] = mad_dst_agg.mean(axis=1).round(1)

	mad_dst_agg['Backfield_AVG'] = mad_dst_agg[['CB', 'FS', 'SS']].mean(axis=1).round(1)
	mad_dst_agg['Linebackers AVG'] = mad_dst_agg[['LOLB', 'MLB', 'ROLB']].mean(axis=1).round(1)
	mad_dst_agg['DL_AVG'] = mad_dst_agg[['DT', 'LE', 'RE']].mean(axis=1).round(1)

	return mad_dst_agg


mad_dst_agg = setup_mad_dst(mad_dst)
mad_dst_agg.Year = mad_dst_agg.Year.astype(float)

def merge_nfl_madden(nfl_pos_df, mad_pos_df):
	"""
	:param nfl_pos_df: nfl positional DataFrame, if DST use diff fields to merge
	:param mad_pos_df: madden positional DataFrame
	:return: merged DataFrame to be used for further processing
	"""
	tmp = pd.merge(
		nfl_pos_df, mad_pos_df, how='inner', left_on=['Players', 'Year', 'Position'],
		right_on=['Name', 'Year', 'Position']
	)
	tmp.Year = tmp.Year.astype(float)

	return tmp


def merge_nfl_madden_dst(nfl_pos_df, mad_pos_df):
	"""
	:param nfl_pos_df: DST NFL data
	:param mad_pos_df: DST madden data
	:return: merged dataframe, aggregate DST data
	"""
	tmp = pd.merge(nfl_pos_df, mad_pos_df, how='inner', left_on=['Team', 'Year'], right_on=['Team', 'Year'])
	tmp.Year = tmp.Year.astype(float)
	return tmp


qb_data_alltime = merge_nfl_madden(nfl_qb_alltime, mad_qb)
rb_data_alltime = merge_nfl_madden(nfl_rb_alltime, mad_rb)
wr_data_alltime = merge_nfl_madden(nfl_wr_alltime, mad_wr)
te_data_alltime = merge_nfl_madden(nfl_te_alltime, mad_te)
dst_data_alltime = merge_nfl_madden_dst(nfl_dst_alltime, mad_dst_agg)
dst_data_alltime.drop('Week', inplace=True, axis=1)

# preprocess before regression analysis. look at player, their opponents DST
# their offensive line.

sched = pd.read_excel('data/2017_NFL_Schedule.xlsx')
sched2018 = pd.read_excel('data/2018_NFL_schedule.xlsx')

sched2018['Week'] = sched2018['Week'].astype(float)
sched['Week'] = sched['Week'].astype(float)

sched2018['Year'] = 2018
sched['Year'] = 2017

supersched = pd.concat([sched, sched2018])
supersched.Year = supersched.Year.astype(float)
supersched.to_csv('data/full_schedule.csv')

def regr_preprocess(pos_data):
	"""
	:param pos_data: qb, rb, wr, te data
	:return: DataFrame to be used for modeling
	"""
	global supersched  # 2017 and 2018 schedule with week and year

	# merge all home and away game schedule data with positional NFL and Madden data
	away = pd.merge(pos_data, supersched, left_on=['Week', 'Team', 'Year'], right_on=['Week', 'Away', 'Year'])
	home = pd.merge(pos_data, supersched, left_on=['Week', 'Team', 'Year'], right_on=['Week', 'Home', 'Year'])
	df = pd.concat([away, home])

	df.drop([
		'Game', 'Position', 'Handedness', 'Players', 'Season', 'Day',
		'Date', 'Time', 'Winner/tie', 'At', 'Loser/tie', 'Away Team', 'Home Team ',
		'PtsW', 'PtsL', 'TOW', 'YdsL', 'TOL', 'YdsW'
	], axis=1, inplace=True)

	# input one if players team is home team, else zero for away team.
	df['isHomeTeam'] = df.apply(lambda x: x['Team'] in x['Home'], axis=1).astype(int)
	# identify opponent
	df['Opponent'] = df.apply(lambda x: x['Away'] if x['isHomeTeam'] == 1 else x['Home'], axis=1)

	# now drop away and home fields.
	df.drop(['Home', 'Away'], axis=1, inplace=True)

	# merge offensive line stats
	mad_ol_alltime = pd.read_excel('data/mad_ol_alltime.xlsx')
	df = pd.merge(df, mad_ol_alltime, on=['Team', 'Year'])

	# merge defensive stats. these are aggregates for the year, not weekly.
	df = pd.merge(
		df, dst_data_alltime, left_on=['Opponent', 'Year'],
		right_on=['Team', 'Year'], how='left', suffixes=('_x', '_Opponent'),
	)

	return df


qb_all = regr_preprocess(qb_data_alltime)

qb_all.to_csv('data/qb_all.csv', index=False)



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
