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

from itertools import chain

import dash_bootstrap_components as dbc
from dash import dcc, html

from demo_configs import DEFAULT_QPU, J_OPTIONS
from src.demo_enums import ProblemType

__all__ = [
    "get_anneal_duration_setting",
    "get_graph_radio_options",
    "config_spins",
    "get_coupling_strength_slider",
    "config_qpu_selection",
    "dbc_modal",
    "job_bar_display",
    "ring_lengths",
]

ring_lengths = [512, 1024, 2048]


def get_anneal_duration_setting(problem_type):
    if problem_type is ProblemType.KZ_NM:
        return dcc.Dropdown(
            id="anneal_duration",
            options=[
                {"label": "5 ns", "value": 5},
                {"label": "10 ns", "value": 10},
                {"label": "20 ns", "value": 20},
                {"label": "40 ns", "value": 40},
                {"label": "80 ns", "value": 80},
                {"label": "160 ns", "value": 160},
                {"label": "320 ns", "value": 320},
                {"label": "640 ns", "value": 640},
                {"label": "1280 ns", "value": 1280},
            ],
            value=80,  # default value
            clearable=False,
        )

    return dbc.Input(
        id="anneal_duration",
        type="number",
        min=5,
        max=100,
        step=1,
        value=7,
    )


def get_graph_radio_options():

    return dcc.RadioItems(
        id="graph-selection-radio",
        options=[
            {"label": "Both", "value": "both"},
            {"label": "Kink Density", "value": "kink_density"},
            {"label": "Schedule", "value": "schedule"},
        ],
        value="both",
        inputStyle={"marginRight": "10px"},
        inline=True,
    )


config_spins = dcc.RadioItems(
    id="spins",
    options=[{"label": f"{length}", "value": length} for length in ring_lengths],
    value=512,
    inputStyle={"marginRight": "10px"},
    inline=True,
)

def get_coupling_strength_slider(problem_type):

    if problem_type is ProblemType.KZ_NM:
        marks = J_OPTIONS
        value = -1.8
    else:
        marks = [
            round(0.1 * val) if val % 10 == 0 else round(0.1 * val, 1)
            for val in chain(range(-20, 0, 2), range(2, 12, 2))
        ]
        value = -1.4

    return html.Div(
        [
            dcc.Slider(
                id="coupling_strength",
                value=value,
                marks={mark: f"{mark}" for mark in marks},
                step=None,
                tooltip={"placement": "bottom", "always_visible": True},
            )
        ]
    )


def config_qpu_selection(solvers):
    return dcc.Dropdown(
        id="qpu_selection",
        options=[{"label": qpu_name, "value": qpu_name} for qpu_name in solvers],
        placeholder="Select a quantum computer",
        value=DEFAULT_QPU if DEFAULT_QPU in solvers else list(solvers.keys())[0],
        clearable=False,
    )


job_bar_display = {
    "READY": [0, "link"],
    "EMBEDDING": [20, "warning"],
    "NO SOLVER": [100, "danger"],
    "SUBMITTED": [40, "info"],
    "PENDING": [60, "primary"],
    "IN_PROGRESS": [85, "#2A7DE1"],
    "COMPLETED": [100, "success"],
    "CANCELLED": [100, "light"],
    "FAILED": [100, "danger"],
}

model_contents = [
    "Leap's Quantum Computers Inaccessible",
    [
        html.Div(
            [
                html.Div("Could not connect to a Leap quantum computer."),
                html.Div(
                    [
                        """
                        If you are running locally, set environment variables or a
                        dwave-cloud-client configuration file as described in the
                        """,
                        dcc.Link(
                            children=[html.Div(" Ocean")],
                            href="https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html",
                            style={"display": "inline-block"},
                        ),
                        "documentation.",
                    ],
                    style={"display": "inline-block"},
                ),
                html.Div(
                    [
                        "If you are running in an online IDE, see the ",
                        dcc.Link(
                            children=[html.Div("system documentation")],
                            href="https://docs.dwavesys.com/docs/latest/doc_leap_dev_env.html",
                            style={"display": "inline-block"},
                        ),
                        " on supported IDEs.",
                    ],
                    style={"display": "inline-block"},
                ),
            ]
        )
    ],
]


def dbc_modal():
    return [
        html.Div(
            [
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle(model_contents[0])),
                        dbc.ModalBody(model_contents[1]),
                    ],
                    size="sm",
                )
            ]
        )
    ]
