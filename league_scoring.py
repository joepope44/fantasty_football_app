cbs = {
	'site': 'cbs',
	'pass_yds': 0.04,  # Pass Yards
	'pass_tds': 4,  # Pass Touchdowns
	'pass_bonus': 2,  # when pass_yds > 300, bonus 2 pts
	'int': -2,  # Interceptions
	'rush_yds': 0.1,  # Rush Yards
	'rush_bonus': 2,  # when rush_yds > 100, bonus 2 pts
	'rush_tds': 6,  # Rush Touchdowns
	'rec_yds': 0.1,  # Reception Yards
	'rec_bonus': 2,  # when rec_yds > 100, bonus 2 pts
	'rec_tds': 6,  # Reception Touchdowns
	'fum': -2,  # Fumbles
	'10-19_fgm': 3,  # 10-19 Yard Field Goal
	'20-29_fgm': 3,  # 20-29 Yard Field Goal
	'30-39_fgm': 3,  # 30-39 Yard Field Goal
	'40-49_fgm': 3,  # 40-49 Yard Field Goal
	'50+_fgm': 5,  # 50+ Yard Field Goal
	'xpm': 1,  # Extra Point
}


class Scoring():
	def __init__(self, site, pass_yds, pass_tds, int):
		self.site = site
		self.pass_yds = pass_yds
		self.pass_tds = pass_tds
		self.int = int


# https://stackoverflow.com/questions/1639174/creating-class-instance-properties-from-a-dictionary