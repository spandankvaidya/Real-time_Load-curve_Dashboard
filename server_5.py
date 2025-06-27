import sys
import os
import polars as pl
import lightgbm as lgb
import numpy as np
from datetime import datetime
from dateutil import parser
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import time

# === CLI ARG ===
if len(sys.argv) < 2:
    print("❌ Date argument missing! Run: python server_5.py YYYY-MM-DD")
    sys.exit(1)

selected_date = sys.argv[1]
print(f"📅 Selected date: {selected_date}")

# === Load Model ===
model = lgb.Booster(model_file='lightgbm_power_model_2.txt')

# === Load Data ===
file_path = f"test/{selected_date}.csv"
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    sys.exit(1)

df = pl.read_csv(file_path)
y_true = df["PowerConsumption"]
X = df.drop("PowerConsumption")

# === Transform datetime features ===
def transform_datetime_column(pl_df):
    dt_series = pl_df['Datetime'].to_list()
    dt_objects = [parser.parse(dt) for dt in dt_series]

    months = [dt.month for dt in dt_objects]
    minutes = [dt.hour * 60 + dt.minute for dt in dt_objects]

    month_sin = [np.sin(2 * np.pi * m / 12) for m in months]
    month_cos = [np.cos(2 * np.pi * m / 12) for m in months]
    time_sin = [np.sin(2 * np.pi * t / 1440) for t in minutes]
    time_cos = [np.cos(2 * np.pi * t / 1440) for t in minutes]

    pl_df = pl_df.with_columns([
        pl.Series('Month_sin', month_sin),
        pl.Series('Month_cos', month_cos),
        pl.Series('Time_sin', time_sin),
        pl.Series('Time_cos', time_cos)
    ])
    return pl_df.drop('Datetime'), [dt.strftime("%H:%M") for dt in dt_objects]

X_transformed, time_labels = transform_datetime_column(X)
X_transformed = X_transformed.to_pandas()[[
    'Month_sin', 'Month_cos', 'Time_sin', 'Time_cos',
    'Temperature', 'Humidity', 'WindSpeed',
    'GeneralDiffuseFlows', 'DiffuseFlows'
]]

predicted = model.predict(X_transformed)

# === Dash App ===
app = dash.Dash(__name__)
app.title = "Live Power Prediction Dashboard"

predicted_values = []
actual_values = []
time_ticks = []
index = 0

app.layout = html.Div([
    html.Div(f"📅 Forecast for {selected_date}", id="date-display", style={"fontSize": "20px", "marginBottom": "20px", "color": "#444", "textAlign": "center"}),
    dcc.Graph(id='live-graph', style={"height": "80vh"}),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    global index
    if index < len(predicted):
        predicted_values.append(predicted[index])
        actual_values.append(y_true[index])
        time_ticks.append(time_labels[index])
        index += 1

    return {
        'data': [
            go.Scatter(x=time_ticks, y=predicted_values, mode='lines+markers', name='Predicted', line=dict(color='blue')),
            go.Scatter(x=time_ticks, y=actual_values, mode='lines+markers', name='Actual', line=dict(color='red'))
        ],
        'layout': go.Layout(
            xaxis={'title': 'Time (HH:MM)'},
            yaxis={'title': 'Power Consumption'},
            margin={'l': 60, 'r': 10, 't': 40, 'b': 60},
            legend={'x': 0, 'y': 1},
            height=600
        )
    }

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8050, debug=True)
    print("✅ Dash launched")
    while True:
        time.sleep(1)

