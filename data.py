import pandas as pd
from bs4 import beautifulsoup4
import requests
import numpy as np

# DATA COLLECTION

# NFL SAVANT - play by play NFL data

files = ['http://www.nflsavant.com/pbp_data.php?year=2017',
		 'http://www.nflsavant.com/pbp_data.php?year=2016',
		 'http://www.nflsavant.com/pbp_data.php?year=2015',
		 'http://www.nflsavant.com/pbp_data.php?year=2014']

df = pd.concat(pd.read_csv(f) for f in files)

df.head(5)

df.Description[:5]

# MADDEN PLAYER RATINGS

# will need to scrape all .xlsx files and ignore non nfl-team sheets.

# https://maddenratings.weebly.com/madden-nfl-19.html

def get_madden():
