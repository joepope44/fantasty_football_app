import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

"""

for histogram comparisons - will need historical yards, mean and std. 
https://plot.ly/pandas/histograms/



"""

df = pd.read_csv('/Users/josephpope/GitHub/Kojak/data/qb_dash2.csv')

df.rename(
    columns={
    'Unnamed: 0': 'QB',
    'qb_pred': 'Predicted Yards',
    'qb_mean': 'Average Yards Per Game',
    'qb_std': 'Std. Dev. Per Game'},
    inplace=True)

x = np.random.randn(1000)
hist_data = [x]
group_labels = ['distplot']

df.sort_values(by='QB', inplace=True)

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

app = dash.Dash()

app.layout = html.Div(
    children=[

    html.H4(children='Fantasy Football Passing Yards Predictor'),

    dcc.Dropdown(id='dropdown',
                 options=[
        {'label': i, 'value': i} for i in sorted(df.QB.unique())],
        multi=True, placeholder='Filter by QB...'),

    html.Div(id='table-container'),


    ])


@app.callback(
    dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])
def display_table(dropdown_value):
    if dropdown_value is None:
        return generate_table(df)

    dff = df[df.QB.str.contains('|'.join(dropdown_value))]
    return generate_table(dff)

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


# import plotly.plotly as py
# import plotly.figure_factory as ff
#
# import numpy as np
#
# x1 = np.random.randn(200) - 1
# x2 = np.random.randn(200)
# x3 = np.random.randn(200) + 1
#
# hist_data = [x1, x2, x3]
#
# group_labels = ['Group 1', 'Group 2', 'Group 3']
# colors = ['#333F44', '#37AA9C', '#94F3E4']
#
# # Create distplot with curve_type set to 'normal'
# fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
#
# # Add title
# fig['layout'].update(title='Curve and Rug Plot')
#
# # Plot!
# py.iplot(fig, filename='Curve and Rug')

# def normcurve(pred, std):
#     mu = pred
#     variance = std
#     sigma = math.sqrt(variance)
#     x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#     plt.plot(x,mlab.normpdf(x, mu, sigma))
#     plt.show()




if __name__ == '__main__':
    app.run_server(debug=True)
