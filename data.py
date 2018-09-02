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

df1 = pd.read_excel('Full Madden 19 Ratings.xlsx')
df2 = pd.read_excel('Madden 18 Player Ratings.xlsx')
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
			'Play Recognition', 'Play Action', 'Power Moves', 'Speed', 'Spin Move', 'Stamina', 'Stiff Arm',
			'Strength', 'Tackle', 'Team', 'Throw Accuracy Deep', 'Throw Accuracy Mid', 'Throw Accuracy Short',
			'Throw Power', 'Throw on the Run', 'Total Salary', 'Toughness', 'Trucking', 'Weight', 'Years Pro',
			'Zone Coverage'
			]

df1 = df1[fields]

df2 = df2[fields]

madden_df = pd.concat([df1, df2])

