# Copyright 2024 D-Wave 
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

from dash import dcc, html, State
import dash_bootstrap_components as dbc 

import plotly.graph_objects as go

from helpers.layouts_components import *

__all__ = ["graphs_card", "kz_config", "simulation_card"]

# Configuration card
def kz_config(solvers={}): 
    return dbc.Card([
        dbc.Row([
            dbc.Col([
                html.H4(
                    "Configuration", 
                    className="card-title",
                    style={"color": "rgb(243, 120, 32)"}
                )
            ])
        ],
            id="tour_settings_row"
        ),
        dbc.Row([
            dbc.Col([
                html.P(
                    "QPU",
                    style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
                ), 
                html.Div([
                    config_qpu_selection(solvers),
                    html.P(
                        id="embedding", 
                        children="", 
                        style = dict(display="none")
                    )
                ]), 
            ], 
                width=9
            ),
            dbc.Col([
                html.P(
                    "Spins",
                    style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
                ), 
                html.Div([
                    config_chain_length
                ]), 
            ], 
                width=3
            ),
        ]),
        dbc.Row([
            dbc.Col([
                html.P(
                    "Coupling Strength",
                    style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
                ), 
                html.Div([
                    config_coupling_strength
                ]),
            ]),
            dbc.Col([
                html.P(
                    "Quench Duration [ns]",
                    style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
                ),
                html.Div([
                    config_anneal_duration
                    
                ]), 
            ]),
        ]),
    ], 
        body=True, 
        color="dark"
    )

# Plots card
def graphs_card():
    return dbc.Card([
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id="spin_orientation", 
                    figure=go.Figure()
                )
            ], 
                width=6
            ),
            dbc.Col([
                dcc.Graph(
                    id="sample_vs_theory",
                    figure=go.Figure()
                )
            ], 
                width=6
            ),
        ]),
    ], 
        color="dark"
    )

# Simulation card
def simulation_card(init_job_status="READY"):

    if init_job_status == "NO SOLVER":
        job_status_color = "red"
    else:  
        job_status_color = "white"

    return dbc.Card([
        html.H4(
            "Simulation", 
            className="card-title",
            style={"color":"rgb(243, 120, 32)"}
        ),
        dbc.Col([
            dbc.Button(
                "Simulate", 
                id="btn_simulate", 
                color="primary", 
                className="me-1",
                style={"marginBottom":"5px"}
            ),
            dcc.Interval(
                id="wd_job", 
                interval=None, 
                n_intervals=0, 
                disabled=True, 
                max_intervals=1
            ),
            dbc.Progress(
                id="bar_job_status", 
                value=0,
                color="link", 
                className="mb-3",
                style={"width": "60%"}
            ),
            html.P(
                id="job_submit_state", 
                children=f"Status: {init_job_status}",
                style={"color": job_status_color, "fontSize": 12}
            ),
            html.P(
                id="job_submit_time", 
                children="", 
                style = dict(display="none")
            ),
            status_solver,
            html.P(
                id="job_id", 
                children="", 
                style = dict(display="none")
            )
        ],
            width=12)
    ],
        color="dark", body=True
    )