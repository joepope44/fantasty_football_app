import requests
import pandas as pd
import json
import datetime as dt
import time
from dateutil import parser
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from string import punctuation
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np


# initialize comments and submissions as c, s

c = 'https://api.pushshift.io/reddit/search/comment/'
s = 'https://api.pushshift.io/reddit/search/submission/'

# review nfl schedule and QB data

sched = pd.read_csv('data/full_schedule.csv')
qb = pd.read_csv('data/qb_all.csv')


# set schedule to 2018
sched[sched['Year'] == 2018.0].reset_index(drop=True, inplace=True)


def find_dt_game(player, week):

	global qb
	global sched

	# return team name given QB name
	try:
		team = qb[(qb['Name'] == player) & (qb['Year'] == 2018) & (qb['Week'] == week)]['Team_x'].iloc[0]

		# filter schedule
		w = sched['Week'] == week
		f_sched = sched[w]

		result = f_sched[(f_sched['Home'] == team) | (f_sched['Away'] == team)]

		return result['Date'].iloc[0]

	except:
		pass


def convert_to_utc(d):
	if d is not None:
		return int(parser.parse(d).timestamp())


def before_utc(d):
	new = dt.datetime.fromtimestamp(d) - dt.timedelta(days=1)
	return new.timestamp()


def after_utc(d):
	new = dt.datetime.fromtimestamp(d) + dt.timedelta(days=1)
	return new.timestamp()


def pull_submissions():

	s = 'https://api.pushshift.io/reddit/search/submission/'

	params = {
		'filter': ['id', 'title', 'created_utc'],
		'size': 50,
		'subreddit': 'fantasyfootball',
		'after': "7d"
	}

	response = requests.get(s, params)
	if response is not None:
		content = response.json()

	sub_results = []
	for entry in content['data']:
		sub_results.append(entry['id'])

	return sub_results


pull_submissions()


def create_filter_list(player):
	firstname = player.split(' ')[0]
	lastname = player.split(' ')[1]

	return [firstname, lastname, player]


# def compile_comments(player, week):
#
# 	global c
#
# 	filter_list = create_filter_list(player)
# 	filter_list = '|'.join(filter_list)
#
# 	params = {
# 		'q': filter_list,
# 		'size': 100,
# 		'link_id': '9gh3uz',
# 		'filter': 'body'
# 	}
#
# 	sub_ids = pull_submissions()
#
# 	results = []
#
# 	for id_ in sub_ids:
# 		params['id'] = id_
#
# 		response = requests.get(c, params)
# 		content = response.json()
#
# 		for comment in content['data']:
# 			results.append(comment['body'])
#
# 	df = pd.DataFrame(results)
# 	df['QB'] = player
#
# 	return df


def compile_comments(player):

	global c

	# filter_list = create_filter_list(player)
	# filter_list = '|'.join(filter_list)

	params = {
		'q': player,
		'size': 500,
		'filter': 'body',
		'subreddit': 'fantasyfootball',
		'before': '7d'
	}

	results = []

	try:
		response = requests.get(c, params)
		if response is not None:
			content = response.json()

			for comment in content['data']:
				results.append(comment['body'])

			df = pd.DataFrame(results)
			df['QB'] = player

			return df
	except:
		pass


compile_comments('Ryan Fitzpatrick')

qbs = qb['Name'].unique()


def create_qb_df():

	global qbs

	df = pd.DataFrame()

	for qb in qbs:
		df = pd.concat([df, compile_comments(qb)])
		print(f"Collecting data on {qb}...")

	return df


df1 = create_qb_df()

df1 = df1.rename(columns={0: 'Text'})
df1.head()

df1.to_csv('data/qb_reddit_data2.csv', index=False)

df2 = df1.groupby('QB').Text.unique().reset_index()


df2.to_csv('data/anothertest2.csv', index=False)
df2.shape

# clean text

# pull in csv file if needed to restart process
df2 = pd.read_csv('data/anothertest2.csv')

# remove non-word characters and convert all to lowercase
df2.Text = df2.Text.str.replace('[^\w\s]','')
df2.Text = df2.Text.apply(lambda x: x.lower())

# remove newline characters
df2.Text = df2.Text.replace(r'\\n','', regex=True)
df2.Text = df2.Text.replace(r'\n','', regex=True)



# WORDCLOUD

def wordclouder(qb, df):

	stopwords = set(STOPWORDS)

	filter_list = create_filter_list(qb)

	stopwords |= set(filter_list)
	stopwords.add('game')
	stopwords.add('qb')

	# iterate through the DataFrame
	df = df[df['QB'] == qb]

	count_vec = CountVectorizer(stop_words="english", analyzer='word',
								ngram_range=(1, 2), max_df=1, min_df=1, max_features=60)

	# typecaste each val to string
	# val = str(val)
	#
	# # split the value
	# tokens = val.split()

	vec = count_vec.fit_transform(df.Text.astype(str))
	comment_words = count_vec.vocabulary_

	dfv = dict(zip(comment_words.keys(), pd.DataFrame(vec.toarray()).sum()))

	wordcloud = WordCloud(
		width=800,
		height=800,
		background_color='white',
		stopwords=stopwords,
		min_font_size=10,
		max_words=30)\
		.generate_from_frequencies(dfv)

	# plot the WordCloud image
	plt.figure(figsize=(8, 8), facecolor=None)
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.tight_layout(pad=0)

	plt.show()


wordclouder('Eli Manning', df2)


s = df2.Text.iloc[0]

test = [s.split('.') for line in s]


def wordclouder_sentiment(qb, df):

	stopwords = set(STOPWORDS)

	filter_list = create_filter_list(qb)

	stopwords |= set(filter_list)
	stopwords.add('game')
	stopwords.add('qb')

	# iterate through the DataFrame
	df = df[df['QB'] == qb]

	count_vec = CountVectorizer(stop_words="english", analyzer='word',
								ngram_range=(1, 2), max_df=1, min_df=1, max_features=60)

	analyser = SentimentIntensityAnalyzer()





	def sentiment_finder_vader(comment):
		analysis = analyser.polarity_scores(comment)
		return analysis['compound']

	df['Vader'] = np.array([sentiment_finder_vader(comment) for comment in df['Text']])

	vec = count_vec.fit_transform(df.Text.astype(str))
	comment_words = count_vec.vocabulary_

	dfv = dict(zip(comment_words.keys(), pd.DataFrame(vec.toarray()).sum()))

	wordcloud = WordCloud(
		width=800,
		height=800,
		background_color='white',
		stopwords=stopwords,
		min_font_size=10,
		max_words=30)\
		.generate_from_frequencies(dfv)

	# plot the WordCloud image
	plt.figure(figsize=(8, 8), facecolor=None)
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.tight_layout(pad=0)

	plt.show()

	return df.Vader

wordclouder_sentiment('Eli Manning', df2)
