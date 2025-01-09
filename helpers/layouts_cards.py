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

from dash import dcc, html
import dash_bootstrap_components as dbc

from demo_configs import DESCRIPTION, MAIN_HEADER
from src.demo_enums import ProblemType
import plotly.graph_objects as go

from helpers.layouts_components import *

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
            html.Label("Quench Duration [ns]"),
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
            html.Label("Cached Embeddings"),
            embeddings,
            html.Label("Simulation"),
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
                            },
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
                style={"marginTop": "5px"},
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
            html.P(id="job_submit_time", style={"display": "none"}),
            html.P(id="job_id", style={"display": "none"}),
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
        ],
        body=True,
        color="dark",
        style={"height": "100%", "minHeight": "50rem"},
    )


def graphs_card(problem_type=ProblemType.KZ):
    return dbc.Card(
        [
            html.H3("Spin States of Qubits in a 1D Ring"),
            dcc.Graph(
                id="spin_orientation",
                figure=go.Figure(),
                style={"height": "40vh", "minHeight": "20rem"},
            ),
            html.H3("QPU Samples Vs. Kibble-Zurek Prediction"),
            html.Div(get_kz_graph_radio_options(problem_type), id="graph-radio-options"),
            dcc.Graph(
                id="sample_vs_theory",
                figure=go.Figure(),
                style={"height": "40vh", "minHeight": "20rem"},
            )
        ],
        color="white",
        style={"height": "100%", "minHeight": "50rem"},
    )
