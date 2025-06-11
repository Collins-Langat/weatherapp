
import requests
import pandas as pd
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output, State
import joblib
from datetime import datetime

# Load model
xgb_model = joblib.load('rain_predictor_xgb.pkl')
if xgb_model is None:
    raise ValueError("Failed to load model.")

def fetch_weather_data(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_mean", "cloud_cover_min", "dew_point_2m_mean",
            "relative_humidity_2m_max", "cape_mean", "rain_sum"
        ],
        "timezone": "America/New_York"
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    data = response.json()
    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)
    return df

initial_lat, initial_lon = 38.2542, -85.7594
df = fetch_weather_data(initial_lat, initial_lon)

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Rainfall Prediction Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Img(src='/assets/logo.jpeg', style={
            'height': '80px', 'margin': '0 auto', 'display': 'block', 'paddingBottom': '10px'
        }))
    ]),
    dbc.Row([dbc.Col(html.H1("Rainfall Prediction Dashboard", style={'textAlign': 'center', 'color': '#FFFFFF'}))]),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📍 Location Selector"),
            dbc.CardBody([
                html.Label("Latitude:", style={'color': 'white'}),
                dcc.Input(id='input-lat', type='number', value=initial_lat, step=0.01),
                html.Label("Longitude:", style={'color': 'white', 'marginLeft': '20px'}),
                dcc.Input(id='input-lon', type='number', value=initial_lon, step=0.01),
                html.Br(), html.Br(),
                html.Button('🔄 Update Location', id='submit-coordinates', n_clicks=0, className='btn btn-info')
            ])
        ]))
    ], style={'marginBottom': '20px'}),
    html.Div(id='forecast-table')
], fluid=True, style={'backgroundColor': '#1c1c1e', 'padding': '30px'})

@app.callback(
    Output('forecast-table', 'children'),
    Input('submit-coordinates', 'n_clicks'),
    State('input-lat', 'value'),
    State('input-lon', 'value')
)
def update_table(n_clicks, lat, lon):
    df_updated = fetch_weather_data(lat, lon)
    features = df_updated[['cloud_cover_min', 'dew_point_2m_mean', 'relative_humidity_2m_max', 'cape_mean']]
    df_updated["Rain_Probability"] = (xgb_model.predict_proba(features)[:, 1] * 100).round(2)
    df_updated["Will It Rain?"] = df_updated["Rain_Probability"].apply(lambda x: "Yes" if x > 50 else "No")
    df_updated["date"] = df_updated["date"].dt.strftime('%b %d, %Y')
    table = dbc.Table.from_dataframe(df_updated[['date', 'rain_sum', 'Rain_Probability', 'Will It Rain?']], striped=True, bordered=True, hover=True, style={"color": "white"})
    return table

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=10000)
