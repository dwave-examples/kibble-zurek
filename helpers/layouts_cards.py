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

from dash import dcc, html
import dash_bootstrap_components as dbc 

import plotly.graph_objects as go

from helpers.layouts_components import *

__all__ = ["control_card", "graphs_card", ]

# Configuration card
def control_card(
    solvers={}, 
    init_job_status="READY"): 

    if init_job_status == "NO SOLVER":
        job_status_color = "red"
    else:  
        job_status_color = "white"

    return dbc.Card([
        dbc.Row(
            [
            dbc.Col(
                [
                html.H4(
                    "Coherent Annealing: KZ Simulation", 
                    className="card-title",
                    style={"color": "rgb(243, 120, 32)"}
                ),
                html.P([
                    "Simulate the Kibble-Zurek mechanism of a 1D ring of magnetic spins.", 
                ],
                    style={"color": "white", "fontSize": 12}),
                html.H5(
                        "Spins",
                        style={"color": "rgb(3, 184, 255)", "marginTop": "20px"}
                ),
                html.Div([
                    config_chain_length
                ]),
                html.H5(
                    "Coupling Strength",
                    style={"color": "rgb(3, 184, 255)", "marginTop": "20px"}
                ), 
                html.Div([
                    config_coupling_strength
                ]),
                html.H5(
                    "Quench Duration [ns]",
                    style={"color": "rgb(3, 184, 255)", "marginTop": "20px"}
                ),
                html.Div([
                    config_anneal_duration
                ]),
                html.H5(
                    "QPU",
                    style={"color": "rgb(3, 184, 255)", "marginTop": "20px"}
                ), 
                html.Div([
                    config_qpu_selection(solvers),
                    html.P(
                        id="embedding", 
                        children="", 
                        style = dict(display="none")
                    )
                ]),
                html.P([
                    "Quench Schedule: ",
                    html.Span(
                        id="quench_schedule_filename", 
                        children="",
                        style={"color": "white", "fontSize": 12}
                    ),
                ],
                    style={"color": "white", "marginTop": "10px"}
                ),
                html.H5(
                    "Cached Embeddings",
                    style={"color": "rgb(3, 184, 255)", "marginTop": "20px"}
                ),
                embeddings, 
                html.H5(
                    "Simulation",
                    style={"color": "rgb(3, 184, 255)", "marginTop": "20px"}
                ),
                dbc.Button(
                    "Run", 
                    id="btn_simulate", 
                    color="primary", 
                    className="me-1",
                    style={"marginTop":"5px"}
                ),
                dbc.Progress(
                    id="bar_job_status", 
                    value=0,
                    color="link", 
                    className="mb-3",
                    style={"width": "60%"}
                ),
                html.P([
                    "Status: ",
                    html.Span(
                        id="job_submit_state", 
                        children=f"{init_job_status}",
                        style={"color": job_status_color, "fontSize": 12}
                    ),
                ], 
                    style={"color": "white", "marginTop": "5px"}
                ),
                # Used for storing status. Can probably be replaced with dcc.Store. 
                dcc.Interval(
                    id="wd_job", 
                    interval=None, 
                    n_intervals=0, 
                    disabled=True, 
                    max_intervals=1
                ),
                html.P(
                    id="job_submit_time", 
                    children="", 
                    style = dict(display="none")
                ),
                html.P(
                    id="job_id", 
                    children="", 
                    style = dict(display="none")
                ),
                dcc.Store(
                    id='embeddings_cached',
                    storage_type='memory',
                    data={},
                ),
                dcc.Store(
                    id='embeddings_found',
                    storage_type='memory',
                    data={},
                ),
                ]
            ),
            ],
            id="tour_settings_row"
        ),
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
                    figure=go.Figure(),
                    style={'height': '40vh'},
                ),
            ], 
                width=12,
            ),
        ], ),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id="sample_vs_theory",
                    figure=go.Figure(),
                    
                )
            ], 
                width=12
            ),
        ]),
    ], 
        color="dark",
    )
