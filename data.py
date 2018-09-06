import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import numpy as np

###### DATA COLLECTION

### NFL SAVANT - play by play NFL data

# files = ['http://www.nflsavant.com/pbp_data.php?year=2017',
# 		 'http://www.nflsavant.com/pbp_data.php?year=2016',
# 		 'http://www.nflsavant.com/pbp_data.php?year=2015',
# 		 'http://www.nflsavant.com/pbp_data.php?year=2014']
#
# df = pd.concat(pd.read_csv(f) for f in files)
#
# df.to_csv('~josephpope/GitHub/Kojak/data/play_by_play_2014_2017.csv')

### MADDEN PLAYER RATINGS

path = '/Users/josephpope/GitHub/Kojak/data/'

df1 = pd.read_excel('data/Full Madden 19 Ratings.xlsx')
df2 = pd.read_excel('data/Madden 18 Player Ratings.xlsx')
df3 = pd.read_excel('madden_nfl_16_-_full_player_ratings.xlsx')
df4 = pd.read_excel('madden_nfl_17_-_full_player_ratings.xlsx')

#clean up dataframe columns

df1.rename(columns={'BC Vision': 'Ball Carrier Vision'}, inplace=True)
df1.rename(columns={'Overall': 'Overall Rating'}, inplace=True)
df1.rename(columns={'Playaction': 'Play Action'}, inplace=True)

df2.rename(columns={'Catching': 'Catch'}, inplace=True)
df2.rename(columns={'Catch In Traffic': 'Catch in Traffic'}, inplace=True)

# merge all dataframe based on team, name (first name last name) , position
common_keys = ['Team', 'Name', 'Position']

fields = [
			'Age', 'Acceleration', 'Agility', 'Awareness', 'Ball Carrier Vision', 'Block Shedding',
			'Carrying', 'Catch', 'Catch in Traffic', 'Elusiveness', 'Finesse Moves',
			'Handedness', 'Height', 'Hit Power', 'Impact Blocking', 'Injury', 'Juke Move', 'Jumping',
			'Kick Accuracy', 'Kick Power', 'Kick Return', 'Man Coverage', 'Name', 'Overall Rating', 'Pass Block',
			'Play Recognition', 'Play Action', 'Position', 'Power Moves', 'Speed', 'Spin Move', 'Stamina', 'Stiff Arm',
			'Strength', 'Tackle', 'Team', 'Throw Accuracy Deep', 'Throw Accuracy Mid', 'Throw Accuracy Short',
			'Throw Power', 'Throw on the Run', 'Total Salary', 'Toughness', 'Trucking', 'Weight', 'Years Pro',
			'Zone Coverage', 'Season'
		]

df1 = df1[fields]

df2 = df2[fields]

madden_df = pd.concat([df1, df2])

madden_df.to_csv('data/madden3.csv', index=False)

# link = 'https://www.footballdb.com/fantasy-football/index.html?pos=QB%2CRB%2CWR%2CTE&yr=2017&wk=1&rules=1'

### SCRAPE NFL 2017 STATS PER PLAYER, PER GAME FROM FOOTBALLDB.COM

### OFFENSE STATS

def scrape_fballdb(year=2017, week):
	"""

	insert year and week of season to scrape

	:param year:
	:param weeks:
	:return:
	"""

	weeks = range(1, 18)
	positions = ['QB','RB','WR','TE','K','DST']
	user_agent = 'Mozilla/5.0'
	headers = {'User-Agent': user_agent}

	# table is limited to 100 rows. so need to cycle through each week and each
	# position to capture as much as possible

	nfl_df = pd.DataFrame()

	for week in weeks:

		for position in positions:

			url = 'https://www.footballdb.com/fantasy-football/index.html?pos=' + \
				  str(position) + '&yr=' + str(year) + '&wk=' + str(week) + '&rules=1'
			print(url)

			response = requests.get(url, headers=headers)

			soup = BeautifulSoup(response.text, 'lxml')

			table = soup.find_all('table')[0]
			table

			# capture headers from table

			headers_ = ['Players','Game']

			for row in table.find_all('tr', class_="header right"):
				for th in row.find_all('a', href=True):
					headers_.append(th['title'])

			# capture data from table
			table = table.find('tbody')

			# find number of rows
			n_columns = 0
			n_rows = 0

			for row in table.find_all('tr'):

				# Determine the number of rows in the table
				td_tags = row.find_all('td')
				if len(td_tags) > 0:
					n_rows += 1
					if n_columns == 0:
						# Set the number of columns for our table
						n_columns = len(td_tags)

			# initialize temp dataframe
			temp = pd.DataFrame(columns=headers_, index=range(0, n_rows))

			row_marker = 0
			for row in table.find_all('tr'):
				column_marker = 0
				columns = row.find_all('td')
				for column in columns:
					temp.iat[row_marker, column_marker] = column.get_text()
					column_marker += 1
				if len(columns) > 0:
					row_marker += 1

			#insert week number as field in dataframe
			temp['week'] = week
			temp['position'] = position

			for col in temp:
				try:
					temp[col] = temp[col].astype(float)
				except ValueError:
					pass


			nfl_df = nfl_df.append(temp, ignore_index=True)

	nfl_df.to_csv('data/nfl_test.csv')

	nfl_df2 = nfl_df.copy()

	return nfl_df

# scrape_fballdb()

nfl_df.Players.head()

# nfl_df.Players.replace('(.\.\S*)', " ", regex=True)

nfl_df.Players.str.replace("&nbsp", " ")
nfl_df.Players.replace('[^.\w]+', " ", regex=True)
nfl_df.Players.replace('(.\.\S*)', " ", regex=True).head()

nfl_df.Players.head()

# newtext = t.replace("&nbsp", "")
#    t.replace_with(newtext)