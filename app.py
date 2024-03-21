# Copyright 2024 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc, html, Input, Output, State
from dash.dcc import Dropdown

import plotly.graph_objects as go

import datetime
import matplotlib.pyplot as plt
import numpy as np

from dwave.cloud import Client

from helpers.tooltips import tool_tips
from helpers.layouts import *
from helpers.plots import *
from helpers.kb_calcs import *

#from kz import (build_bqm)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

job_bar = {"READY": [0, "link"],
#          "WAITING": [0, "dark"],     Placeholder, to remember the color
           "NO_SOLVER": [100, "danger"],
           "SUBMITTED": [10, "info"],
           "PENDING": [50, "warning"],
           "IN_PROGRESS": [75 ,"primary"],
           "COMPLETED": [100, "success"],
           "CANCELLED": [100, "light"],
           "FAILED": [100, "danger"], }


# Problem-submission card
solver_card = dbc.Card([
    html.H4("Job Submission", className="card-title",
        style={"color":"rgb(243, 120, 32)"}),
    dbc.Col([
        dbc.Button("Send", id="btn_solve_cqm", color="primary", className="me-1",
            style={"marginBottom":"5px"}),
        dcc.Interval(id="wd_job", interval=None, n_intervals=0, disabled=True, max_intervals=1),
        dbc.Progress(id="bar_job_status", value=job_bar[init_job_status][0],
            color=job_bar[init_job_status][1], className="mb-3",
            style={"width": "60%"}),
        html.P(id="job_submit_state", children=f"Status: {init_job_status}",
            style={"color": "white", "fontSize": 12}),
        html.P(id="job_submit_time", children="", style = dict(display="none")),
        html.P(id="job_id", children="", style = dict(display="none"))],
        width=12)],
    color="dark", body=True)

# Configuration section
kz_config = dbc.Card([
    dbc.Row([
        dbc.Col([
            html.H4("Configuration", className="card-title",
                style={"color": "rgb(243, 120, 32)"})
            ])
            ],id="tour_settings_row"),
    dbc.Row([
        dbc.Col([
            html.P(f"QPU",
                style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
                ), 
            html.Div([
                config_qpu_selection
                ]), 
        ], width=9),
        dbc.Col([
            html.P(f"Chain Length",
                style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
                ), 
            html.Div([
                config_chain_length
                ]), 
        ], width=3),
    ]),
    dbc.Row([
        dbc.Col([
            html.P(f"Coupling Strength",
                style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
            ), 
            html.Div([
                config_coupling_strength
            ]),
        ]),
        dbc.Col([
            html.P(f"Anneal Duration [ns]",
                style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
            ),
            html.Div([
                config_anneal_duration
                
            ]), 
        ]),
    ]),
], body=True, color="dark")


# Graph section
graphs = dbc.Card([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="sample_vs_theory", figure=go.Figure())
        ], width=6),
        dbc.Col([
            dcc.Graph(id=f"sample_kinks")
        ], width=6),
    ]),
], color="dark"),
    


# Page-layout section

app_layout = [
    dbc.Row([
        dbc.Col(
            kz_config,
            width=6
            ),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    solver_card
                ])
            ]),
        ], width=3),
    ], justify="left"),
    dbc.Row([
        dbc.Col(
        graphs,   
        width=12),
    ], justify="left"),
    ]

# tips = [dbc.Tooltip(
#             message, target=target, id=f"tooltip_{target}", style = dict())
#             for target, message in tool_tips.items()]
# app_layout.extend(tips)


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Coherent Annealing: KZ Simulation", style={"textAlign": "left", "color": "white"})],
            width=9),
        dbc.Col([
            html.Img(src="assets/dwave_logo.png", height="25px",
                style={"textAlign": "left"})],
            width=3)]),
    dbc.Container(app_layout, fluid=True,
        style={"color": "rgb(3, 184, 255)",
            "paddingLeft": 10, "paddingRight": 10})],
    style={"backgroundColor": "black",
        "background-image": "url('assets/electric_squids.png')",
        "background-size": "cover",
        "paddingLeft": 100, "paddingRight": 100,
        "paddingTop": 25, "paddingBottom": 50}, fluid=True)

server = app.server
app.config["suppress_callback_exceptions"] = True

# Callbacks Section

@app.callback(
    Output('coupling_strength_display', 'children'), 
    Input('coupling_strength', 'value'))
def update_output(value):
    return f"J={value}"

@app.callback(
    Output("sample_vs_theory", "figure"),
    Input("coupling_strength", "value"),
    State("anneal_duration", "min"),
    State("anneal_duration", "max"))
def display_graphics_left(J, ta_min, ta_max):
    """Generate graphics for theory and samples."""

    trigger = dash.callback_context.triggered
    trigger_id = trigger[0]["prop_id"].split(".")[0]

    if trigger_id == "coupling_strength":
        
        samples = {(J, ta_min): np.NaN, (J, ta_max): np.NaN}
        fig = plot_kink_densities(samples, [J])

        return fig
    
    samples = {(J, ta_min): np.NaN, (J, ta_max): np.NaN}
    fig = plot_kink_densities(samples, [J])
    return fig

@app.callback(
    Output("anneal_duration", "disabled"),
    Output("coupling_strength", "disabled") ,
    Output("chain_length", "options"),
    Input("job_submit_state", "children"),
    State("chain_length", "options"))
def disable_buttons(job_submit_state, chain_length_options):
    """Disable user input during job submissions."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id !="job_submit_state":
        return dash.no_update, dash.no_update, dash.no_update

    if any(job_submit_state == status for status in ["SUBMITTED", "PENDING", "IN_PROGRESS"]):
        
        chain_length_disable = chain_length_options
        for inx, option in enumerate(chain_length_disable): 
            chain_length_disable[inx]['disabled'] = True
        
        return  True, True, chain_length_disable

    elif any(job_submit_state == status for status in ["COMPLETED", "CANCELLED", "FAILED"]):

        chain_length_enable = chain_length_options
        for inx, option in enumerate(chain_length_enable): 
            chain_length_enable[inx]['disabled'] = False
        
        return False, False, chain_length_enable

    else:
        return dash.no_update, dash.no_update, dash.no_update

if __name__ == "__main__":
    app.run_server(debug=True)
