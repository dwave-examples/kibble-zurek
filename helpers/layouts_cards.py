# Copyright 2025 D-Wave
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

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from demo_configs import DESCRIPTION, MAIN_HEADER
from helpers.layouts_components import *
from src.demo_enums import ProblemType

__all__ = [
    "control_card",
    "graphs_card",
]


def control_card(solvers={}, init_job_status="READY"):
    """Lay out the configuration and job-submission card.

    Args:
        solvers: Dict of QPUs in the format {name: solver}.
        init_job_status: Initial status of the submission progress bar.

    Returns:
        Dash card.
    """

    job_status_color = "red" if init_job_status == "NO SOLVER" else "white"

    return dbc.Card(
        [
            html.H1(MAIN_HEADER, id="main-header"),
            html.P(DESCRIPTION, id="main-description"),
            html.Label("Spins"),
            html.Div(config_spins),
            html.Label("Coupling Strength (J)"),
            html.Div(get_coupling_strength_slider(ProblemType.KZ), id="coupling-strength-slider"),
            html.Label("Quench/Anneal Duration [ns]"),
            html.Div(get_anneal_duration_setting(ProblemType.KZ), id="anneal-duration-dropdown"),
            html.Label("QPU"),
            html.Div(config_qpu_selection(solvers)),
            html.P(
                [
                    "Quench Schedule: ",
                    html.Span(
                        id="quench_schedule_filename",
                        style={"color": "white", "fontSize": 10},
                    ),
                ],
                style={"marginTop": "10px"},
            ),
            html.P(
                ["Cached Embeddings: ", html.Span(id="embedding_is_cached")],
                style={"marginTop": 10},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Run Simulation",
                            id="btn_simulate",
                            color="primary",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        [
                            dbc.Progress(
                                id="bar_job_status",
                                value=0,
                                color="link",
                                style={"width": "60%"},
                            ),
                            html.P(
                                [
                                    "Status: ",
                                    html.Span(
                                        id="job_submit_state",
                                        children=f"{init_job_status}",
                                        style={
                                            "color": job_status_color,
                                            "fontSize": 12,
                                        },
                                    ),
                                ],
                                style={"margin": "0"},
                            ),
                        ]
                    ),
                ],
                justify="start",  # Aligns buttons to the left
                align="end",
                style={"marginTop": 40},
            ),
        ],
        body=True,
        color="dark",
        style={"height": "100%", "minHeight": "50rem"},
    )

def default_graph(title, id, load_radio=False):
    return [
        html.H3(title),
        html.Div(get_graph_radio_options(), id="graph-radio-options") if load_radio else "",
        dcc.Graph(
            id=f"{id}-graph",
            figure=go.Figure(),
            style={"height": "40vh", "minHeight": "20rem"},
        ),
    ]


def graphs_card():
    return dbc.Card(
        [
            html.Div([
                *default_graph("Extrapolating Zero-Noise Density", "kink-v-noise"),
                *default_graph("Measured and Extrapolated Kink Densities", "kink-v-anneal"),
            ], id="kz-nm-graphs", className="display-none"),
            html.Div([
                *default_graph("Spin States of Qubits in a 1D Ring", "spin-orientation"),
                *default_graph("QPU Samples vs Kibble-Zurek Prediction", "sample-v-theory", True),
            ], id="kz-graphs"),
        ],
        color="white",
        style={"height": "100%", "minHeight": "50rem"},
    )
