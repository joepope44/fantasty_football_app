import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import numpy as np

nfl_df = pd.read_csv('data/nfl_test.csv')

nfl_df.rename(columns = {'position':'Position', 'week': 'Week'}, inplace=True)

nfl_df['Players'] = nfl_df['Players'].apply(lambda x: str(x).rsplit(sep='.', maxsplit=1)[0][:-1])

def clean_teams(team):
	if team[:-2][0].isupper():
		return team[:-2]
	else:
		return team[:-1]

# dfmi.loc[:,('one','second')]


nfl_df.loc[nfl_df['Position'] == 'DST','Players'] = \
nfl_df[nfl_df['Position'] == 'DST']['Players']\
		.apply(lambda x: clean_teams(x))

nfl_df.drop(axis=1, columns="Unnamed: 0", inplace=True)


nfl_df['Team'] = nfl_df[nfl_df['Position'] == 'DST']['Players'].apply(lambda x: x.rsplit(sep=' ', maxsplit=1)[-1])

nfl_df['Team'] = nfl_df['Team'].apply(lambda x: str(x) if str(x).endswith('s') else str(x) + 's')

nfl_df.to_csv('data/nfl_2017_clean.csv')

