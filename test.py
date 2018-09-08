from fantasty_football_app.nfl_madden import merge_nfl_madden
import seaborn as sns
% pylab inline



sns.set(style="ticks")

sns.pairplot(df, hue="species")