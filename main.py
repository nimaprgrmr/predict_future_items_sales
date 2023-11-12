import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
# import plotly.express as px
import plotly.graph_objs as go
from utils import make_period_time
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# import dash_bootstrap_components as dbc

# Loading model and scaler
FILE_NAME_MODEL = 'rfr_model.pkl'
FILE_NAME_SCALER = 'scaler.pk'
model = pickle.load(open(FILE_NAME_MODEL, 'rb'))
scaler = pickle.load(open(FILE_NAME_SCALER, 'rb'))

start_time = '1402- 1- 1'
end_time = '1402- 2- 1'

LABELS = ['اورآل', 'بادی', 'بارانی', 'بافت', 'بلوز', 'بلوز و شلوار کودک', 'بلوز کودک', 'تاپ',
          'تونیک', 'تی شرت', 'دامن', 'سارافون', 'سایر', 'ست بلوز و شلوار', 'سویی شرت', 'شال و روسری',
          'شلوار', 'شلوار کودک', 'شومیز', 'ماسک', 'مانتو', 'هودی کودک', 'پارچه', 'پالتو', 'پیراهن', 'ژاکت', 'کاپشن',
          'کت', 'کت شلوار', 'کفش و صندل', 'کلاه، هدبند، پاپوش', 'کیف']

# Define your custom color palette
custom_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
    '#5254a3', '#6b6ecf', '#3182bd', '#e6550d', '#31a354'
]

# Define the available options for year, month, and day
YEARS = [str(year) for year in range(1405, 1396, -1)]
MONTHS = [str(month).zfill(2) for month in range(1, 13)]
DAYS = [str(day).zfill(2) for day in range(1, 32)]


# create a function that takes 2 datetimes and make sales prediction between this 2 times.
def predict_period_time(start_date: str, end_date: str, model: RandomForestRegressor, scaler: StandardScaler):
    """
    :param start_date: the start date that you want to predict in the `str` format for example: '1402-03-01'
    :param end_date: the end date that you want to predict in the `str` format for example: '1402-04-01'
    :param model: The pre-trained model that will use for predictions
    :param scaler: the pre-trained scaler that will use for normalize input data
    :return: the total sales between start date and end date in each category
    """

    start_date = list(start_date.split("-"))
    end_date = list(end_date.split("-"))

    start_date = [int(x) for x in start_date]
    end_date = [int(x) for x in end_date]

    period_time = make_period_time(start_date, end_date)
    inputs = scaler.transform(period_time)
    predictions = model.predict(inputs)
    predictions += (0.13 * predictions).astype(int)

    total_predictions = 0
    for i in predictions:
        total_predictions += i
    total_predictions += (0.13 * total_predictions)
    total_predictions = np.round(total_predictions)
    return total_predictions


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Creating my dashboard application
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the font type you want to use for buttons
font_style = {
    'font-family': 'Helvetica, Arial, sans-serif',
    'font-size': '13px',  # Adjust the font size as needed
    'color': 'Black'  # Change 'Arial' to the desired font family
}

app.layout = html.Div([
    html.Div([
        html.Label('Select Start Year'),
        dcc.Dropdown(id='start-year-dropdown', options=[{'label': year, 'value': year} for year in YEARS],
                     value='1402', style={'background-color': 'rgba(221,235,241,0.7)'}),
    ], style={'display': 'inline-block', 'margin-right': '15px', 'margin-left': '20px'}),

    html.Div([
        html.Label('Start Month'),
        dcc.Dropdown(id='start-month-dropdown', options=[{'label': month, 'value': month} for month in MONTHS],
                     value='01', style={'background-color': 'rgba(221,235,241,0.7)'}),
    ], style={'display': 'inline-block', 'margin-right': '15px'}),

    html.Div([
        html.Label('Start Day'),
        dcc.Dropdown(id='start-day-dropdown', options=[{'label': day, 'value': day} for day in DAYS], value='01',
                     style={'background-color': 'rgba(221,235,241,0.7)'}),
    ], style={'display': 'inline-block', 'margin-right': '30px'}),

    html.Div([
        html.Label('Select End Year'),
        dcc.Dropdown(id='end-year-dropdown', options=[{'label': year, 'value': year} for year in YEARS], value='1402'
                     , style={'background-color': 'rgba(248,194,127,0.6)'}),
    ], style={'display': 'inline-block', 'margin-right': '15px'}),

    html.Div([
        html.Label('End Month'),
        dcc.Dropdown(id='end-month-dropdown', options=[{'label': month, 'value': month} for month in MONTHS],
                     value='02', style={'background-color': 'rgba(248,194,127,0.6)'}),
    ], style={'display': 'inline-block', 'margin-right': '15px'}),

    html.Div([
        html.Label('End Day'),
        dcc.Dropdown(id='end-day-dropdown', options=[{'label': day, 'value': day} for day in DAYS], value='01',
                     style={'background-color': 'rgba(248,194,127,0.6)'}),
    ], style={'display': 'inline-block', 'margin-right': '30px'}),

    html.Div([
        html.Button('Predict Sales', id='predict-button', n_clicks=0,
                    style={'background-color': 'green', 'color': 'white'}),
    ], style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-top': '55px'}),

    # # Spacer to push the Help button to the right
    # html.Div(style={'flex': 0.8}),

    html.Div([
        html.Button('Help', id='help-button', n_clicks=0,
                    style={'background-color': 'rgb(13,152,186)', 'color': 'white'}),
    ], style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-top': '55px', 'margin-right': '20px',
              'margin-left': '550px'}),

    html.Div([
        dcc.Graph(id='prediction-plot'), ], style={'margin-top': '20px'}),

    html.Div(id='help-modal', style={'display': 'none'}, children=[
        html.Div([
            html.H4("Dashboard Guide"),
            dcc.Markdown("""
                    This is a guide for using your dashboard.
    
                    1. Use the dropdowns to select start and end dates.
                    2. Click on the "Predict Sales" button to generate predictions.
                    3. Click on the "Help" button to close this guide.
    
                    Enjoy using the dashboard!
                """, style={'color': 'black', 'background-color': 'rgba(13,152,186, 0.3)', 'padding': '10px',
                            'border-radius': '10px'}),

        ], style={'position': 'fixed', 'top': '15%', 'left': '65%', 'transform': 'translate(-23%, -50%)'}),
        html.Div(id='modal-background',
                 style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '80%',
                        'background-color': 'rgba(0, 0, 0, 0.2)', 'display': 'none'}),
    ]),
])


@app.callback(
    Output('prediction-plot', 'figure'),
    Input('predict-button', 'n_clicks'),
    [
        State('start-year-dropdown', 'value'),
        State('start-month-dropdown', 'value'),
        State('start-day-dropdown', 'value'),
        State('end-year-dropdown', 'value'),
        State('end-month-dropdown', 'value'),
        State('end-day-dropdown', 'value')
    ]
)
def update_prediction_plot(n_clicks, start_year, start_month, start_day, end_year, end_month, end_day):
    if n_clicks > 0:
        # Combine the selected start date and end date into strings
        start_date = f"{start_year}-{start_month}-{start_day}"
        end_date = f"{end_year}-{end_month}-{end_day}"
        # Convert the input strings to the appropriate format if needed
        # Call your sales prediction function with start_time and end_time
        predictions = predict_period_time(start_date, end_date, model, scaler)

        # Create a bar plot with the predictions
        # Create a Plotly bar plot
        fig = go.Figure(data=[go.Bar(x=LABELS, y=predictions, marker=dict(color=custom_palette))])
        fig.update_layout(
            height=600,
            title="Sales Predictions of `Bamland` branch",
            title_font=dict(size=20),
            xaxis_title="Categories",
            xaxis_title_font=dict(size=16),
            yaxis_title="Counts",
            yaxis_title_font=dict(size=16),
            # font=dict(family="Arial", size=18, color="black"),
            paper_bgcolor="white",
            plot_bgcolor="rgba(221,235,241,0.7)",
            xaxis=dict(tickangle=45, tickfont=dict(size=15), gridcolor='white'),
        )
        # Change the theme to "plotly_dark"
        # fig.update_layout(template="plotly_dark") # for changing the background of plot
        return fig

    return dash.no_update


@app.callback(
    Output('help-modal', 'style'),
    Output('modal-background', 'style'),
    Input('help-button', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_modal(help_clicks):
    if help_clicks is None:
        help_clicks = 0
    if help_clicks % 2 == 1:  # Odd number of clicks on Help or Close button
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True)
