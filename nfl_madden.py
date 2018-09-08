import pandas as pd

qb_nfl_fields = [
	'Fantasy Points', 'Game', 'Pass Attempts', 'Pass Completions',
	'Passes Intercepted', 'Passing Conversions', 'Passing Yards', 'Players',
	'Rushing Attempts', 'Rushing Conversions', 'Rushing Touchdowns',
	'Rushing Yards', 'Touchdown Passes', 'Position', 'Week'
	]

def filter_nfl_data(pos, fields):
	"""
	function filters nfl data by position and fields. data should be pre-filtered
	by season.
	:param pos: NFL Position, choice are 'QB','RB','TE,'WR','DST','K'
	:param fields: List of fields that should be kept and treated as features
	:return: pandas dataframe filtered to position and fields
	"""
	nfl = pd.read_csv('data/nfl_2017_clean.csv', index_col='Index')
	nfl_by_pos = nfl[nfl['Position'] == pos]
	temp = nfl_by_pos[fields]

	return temp


nfl_qb = filter_nfl_data('QB', qb_nfl_fields)

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
	'Age', 'Acceleration', 'Agility', 'Awareness','Block Shedding',
	'Finesse Moves', 'Height', 'Hit Power', 'Impact Blocking', 'Injury',
	'Jumping', 'Man Coverage', 'Name', 'Overall Rating', 'Play Recognition',
	'Position', 'Power Moves', 'Speed', 'Spin Moves', 'Stamina', 'Strength',
	'Tackle', 'Team', 'Total Salary', 'Toughness', 'Weight', 'Years Pro',
	'Zone Coverage', 'Season'
	]

mad_offline_fields = [
	'Age', 'Acceleration', 'Agility', 'Awareness', 'Height', 'Injury',
	'Overall Rating', 'Pass Block', 'Position', 'Speed', 'Stamina', 'Strength',
	'Team', 'Total Salary', 'Toughness', 'Weight', 'Years Pro', 'Season'
]





def filter_madden_data(pos, fields, season):
	"""
	:param pos: NFL Position, choice are 'QB','RB','TE,'WR','DST','K'
	:param fields: List of fields that should be kept and treated as features
	:param season: '2017-2018'
	:return:
	"""
	madden = pd.read_csv('data/madden3.csv')
	temp = madden[madden['Position'] == pos]
	temp = temp[temp['Season'] == season]
	temp = temp[fields]
	return temp


mad_qb = filter_madden_data('QB', mad_qb_fields, '2017-2018')


def merge_nfl_madden(nfl_df, madden_df):
	"""
	function merges nfl_df and madden_df by position and season
	:param nfl_df:
	:param madden_df:
	:return:
	"""
	df = pd.merge(nfl_df, madden_df, how='inner', left_on='Players', right_on='Name')
	return df

df = merge_nfl_madden()

# df.to_csv('data/qb_clean.csv')

## TEST OUT THE BIG DATA AND SAVE TO CSV FOR LATER ANALYSIS
## https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values/notebook


# big = pd.read_csv('data/NFL Play by Play 2009-2017 (v4).csv', low_memory=False)
#
# pbp2017 = big[big.Season == 2017]
#
# pbp2017.to_csv('data/pbp2017.csv')
