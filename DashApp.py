import pandas as pd
import numpy as np
import joblib
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# Load dataset
file_path = "healthcare_dataset_cp.csv"  # Ensure this file is in the same directory
data = pd.read_csv(file_path)

# Extract feature names
feature_names = ['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']

# Load pre-trained model
model_path = "random_forest_model.pkl"
try:
    rf_model = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Train the model and save it as 'random_forest_model.pkl'.")

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# App layout
app.layout = html.Div([
    html.H1("Healthcare Risk Prediction System", style={'text-align': 'center', 'color': 'white'}),

    html.Div([
        html.Label("Enter Values for Each Feature:", style={'color': 'white'}),

        # Input fields for user entry
        *[
            dcc.Input(id=f'input-{col}', type='number', placeholder=col, style={'margin': '5px', 'width': '200px','color':'black'})
            for col in feature_names
        ],
        

        # Predict button
        html.Button("Predict Risk Level", id="predict-button", n_clicks=0, style={'margin-top': '10px'}),

        # Prediction output display
        html.Div(id="prediction-output", style={'text-align': 'center', 'color': 'white', 'margin-top': '20px'})
    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),

    # Dropdown and graph for visualization
    html.Div([
        dcc.Dropdown(id="feature-dropdown",
                     options=[{"label": col, "value": col} for col in feature_names],
                     placeholder="Select a feature to visualize",
                     style={'color': 'black', 'margin-top': '20px'}),

        dcc.Graph(id="feature-plot")
    ])
])

# Prediction callback
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [State(f"input-{col}", "value") for col in feature_names]
)
def predict_risk(n_clicks, *values):
    if n_clicks > 0:
        # Handle missing inputs and reshape for prediction
        input_data = [float(v) if v is not None else 0 for v in values]
        input_data = np.array(input_data).reshape(1, -1)

        # Perform prediction using the trained model
        prediction = rf_model.predict(input_data)[0]
        return f"Predicted Risk Level: {prediction}"
    return "Enter values and click 'Predict Risk Level' to see the result."

# Visualization callback
@app.callback(
    Output("feature-plot", "figure"),
    Input("feature-dropdown", "value")
)
def update_graph(selected_feature):
    if selected_feature:
        fig = px.histogram(data, x=selected_feature, color='Type', barmode='overlay',
                           title=f"{selected_feature} Distribution by Risk Level")
        return fig
    return {}

# Run the app
if __name__ == "__main__":
    app.run(port=8260, debug=True)


