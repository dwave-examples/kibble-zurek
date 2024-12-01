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

__all__ = [
    "control_card",
    "graphs_card",
]

control_header_style = {"color": "rgb(3, 184, 255)", "marginTop": "10px"}


def control_card(solvers={}, init_job_status="READY"):
    """Lay out the configuration and job-submission card.

    Args:

        solvers: Dict of QPUs in the format {name: solver}.

        init_job_status: Initial status of the submission progress bar.

    Returns:

        Dash card.
    """

    if init_job_status == "NO SOLVER":
        job_status_color = "red"
    else:
        job_status_color = "white"

    return dbc.Card(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                "Coherent Annealing: KZ Simulation",
                                className="card-title",
                                style={"color": "rgb(243, 120, 32)"},
                            ),
                            html.P(
                                """
Use a quantum computer to simulate the formation of topological defects in a 1D ring 
of spins undergoing a phase transition, described by the Kibble-Zurek mechanism.  
""",
                                style={"color": "white", "fontSize": 14},
                            ),
                            html.H5("Spins", style=control_header_style),
                            html.Div([config_spins]),
                            html.H5("Coupling Strength", style=control_header_style),
                            html.Div([config_coupling_strength]),
                            html.H5("Quench Duration [ns]", style=control_header_style),
                            html.Div([config_anneal_duration]),
                            html.H5("QPU", style=control_header_style),
                            html.Div(
                                [
                                    config_qpu_selection(solvers),
                                ]
                            ),
                            html.P(
                                [
                                    "Quench Schedule: ",
                                    html.Span(
                                        id="quench_schedule_filename",
                                        children="",
                                        style={"color": "white", "fontSize": 10},
                                    ),
                                ],
                                style={"color": "white", "marginTop": "10px"},
                            ),
                            html.H5("Cached Embeddings", style=control_header_style),
                            embeddings,
                            html.H5("Simulation", style=control_header_style),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Run",
                                            id="btn_simulate",
                                            color="primary",
                                            className="me-2",  # Adds spacing between buttons
                                            style={
                                                "marginTop": "10px"
                                            },  # Adds some vertical spacing
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Reset",
                                            id="btn_reset",
                                            color="danger",
                                            style={"marginTop": "10px"},
                                        ),
                                        width="auto",
                                    ),
                                ],
                                justify="start",  # Aligns buttons to the left
                                align="center",  # Vertically centers buttons
                            ),
                            dbc.Progress(
                                id="bar_job_status",
                                value=0,
                                color="link",
                                className="mb-3",
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
                                            "marginTop": "10px",
                                        },
                                    ),
                                ],
                                style={"color": "white", "marginTop": "5px"},
                            ),
                            html.P(
                                "Tooltips (hover over fields for descriptions)",
                                style={
                                    "color": "white",
                                    "fontSize": 12,
                                    "marginBottom": 5,
                                    "marginTop": "10px",
                                },
                            ),
                            tooltips_activate,
                            # Non-displayed section
                            dcc.Interval(
                                id="wd_job",
                                interval=None,
                                n_intervals=0,
                                disabled=True,
                                max_intervals=1,
                            ),
                            # Used for storing job status. Can probably be replaced with dcc.Store.
                            html.P(
                                id="job_submit_time",
                                children="",
                                style=dict(display="none"),
                            ),
                            html.P(
                                id="job_id", children="", style=dict(display="none")
                            ),
                            dcc.Store(
                                id="embeddings_cached",
                                storage_type="memory",
                                data={},
                            ),
                            dcc.Store(
                                id="embeddings_found",
                                storage_type="memory",
                                data={},
                            ),
                        ]
                    ),
                ],
                id="tour_settings_row",
            ),
        ],
        body=True,
        color="dark",
        style={"height": "100%", "minHeight": "50rem"},
    )


graphic_header_style = {
    "color": "rgb(243, 120, 32)",
    "margin": "15px 0px 0px 15px",
    "backgroundColor": "white",
}


def graphs_card():
    return dbc.Card(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5(
                                "Spin States of Qubits in a 1D Ring",
                                style=graphic_header_style,
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="spin_orientation",
                                figure=go.Figure(),
                                style={"height": "40vh", "minHeight": "20rem"},
                            ),
                        ],
                        width=12,
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5(
                                "QPU Samples Vs. Kibble-Zurek Prediction",
                                style=graphic_header_style,
                            ),
                            html.Div([config_kz_graph]),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="sample_vs_theory",
                                figure=go.Figure(),
                                style={"height": "40vh", "minHeight": "20rem"},
                            )
                        ],
                        width=12,
                    ),
                ]
            ),
        ],
        color="white",
        style={"height": "100%", "minHeight": "50rem"},
    )
