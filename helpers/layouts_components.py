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

import dash_bootstrap_components as dbc
from dash.dcc import Checklist, Dropdown, Link, RadioItems, Slider
from dash import html, dcc

__all__ = [
    "config_anneal_duration",
    "config_kz_graph",
    "config_spins",
    "config_coupling_strength",
    "config_qpu_selection",
    "dbc_modal",
    "embeddings",
    "job_bar_display",
    "ring_lengths",
    "tooltips_activate",
]

ring_lengths = [512, 1024, 2048]

config_anneal_duration = dcc.Dropdown(
    id="anneal_duration",
    options=[
        {"label": "5 ns", "value": 5},
        {"label": "10 ns", "value": 10},
        {"label": "20 ns", "value": 20},
        {"label": "40 ns", "value": 40},
        {"label": "80 ns", "value": 80},
        {"label": "160 ns", "value": 160},
        {"label": "320 ns", "value": 320},
    ],
    value=80,  # default value
    style={"max-width": "95%"},
)

config_kz_graph = RadioItems(
    id="kz_graph_display",
    options=[
        {"label": "Both", "value": "both", "disabled": False},
        {"label": "Kink density vs Anneal time", "value": "kink_density", "disabled": False},
        {"label": "Schedule", "value": "schedule", "disabled": False},
        {"label": "Kink density vs Noise level", "value": "coupling", "disabled": False},
    ],
    value="both",
    inputStyle={"margin-right": "10px", "margin-bottom": "5px"},
    labelStyle={
        "color": "rgb(3, 184, 255)",
        "font-size": 12,
        "display": "inline-block",
        "marginLeft": 20,
    },
    inline=True,  # Currently requires above 'inline-block'
)

config_spins = RadioItems(
    id="spins",
    options=[
        {"label": f"{length}", "value": length, "disabled": False}
        for length in ring_lengths
    ],
    value=512,
    inputStyle={"margin-right": "10px", "margin-bottom": "10px"},
    labelStyle={
        "color": "white",
        "font-size": 12,
        "display": "inline-block",
        "marginLeft": 20,
    },
    inline=True,  # Currently requires above 'inline-block'
)

j_marks = {
    round(0.1 * val, 1): (
        {"label": f"{round(0.1*val, 1)}", "style": {"color": "blue"}}
        if round(0.1 * val, 0) != 0.1 * val
        else {"label": f"{round(0.1*val)}", "style": {"color": "blue"}}
    )
    for val in range(-18, 0, 2)
}
j_marks.update(
    {
        round(0.1 * val, 1): (
            {"label": f"{round(0.1*val, 1)}", "style": {"color": "red"}}
            if round(0.1 * val, 0) != 0.1 * val
            else {"label": f"{round(0.1*val)}", "style": {"color": "red"}}
        )
        for val in range(2, 10, 2)
    }
)
# Dash Slider has some issue with int values having a zero after the decimal point
j_marks[-2] = {"label": "-2", "style": {"color": "blue"}}
del j_marks[-1.0]
j_marks[-1] = {"label": "-1", "style": {"color": "blue"}}
j_marks[1] = {"label": "1", "style": {"color": "red"}}
config_coupling_strength = dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    Slider(
                        id="coupling_strength",
                        value=-1.8,
                        marks=j_marks,
                        step=None,
                        min=-1.8,
                        max=-0.6,
                    )
                ]
            ),
        ),
    ]
)


def config_qpu_selection(solvers, default="mock_dwave_solver"):
    default = "mock_dwave_solver" if "mock_dwave_solver" in solvers else None
    return Dropdown(
        id="qpu_selection",
        options=[{"label": qpu_name, "value": qpu_name} for qpu_name in solvers],
        placeholder="Select a quantum computer",
        # value=default
    )


job_bar_display = {
    "READY": [0, "link"],
    "EMBEDDING": [20, "warning"],
    "NO SOLVER": [100, "danger"],
    "SUBMITTED": [40, "info"],
    "PENDING": [60, "primary"],
    "IN_PROGRESS": [85, "dark"],
    "COMPLETED": [100, "success"],
    "CANCELLED": [100, "light"],
    "FAILED": [100, "danger"],
}

modal_texts = {
    "solver": [
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
                            Link(
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
                            Link(
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
    ],
}


def dbc_modal(name):
    name = name.split("_")[1]
    return [
        html.Div(
            [
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle(modal_texts[name][0])),
                        dbc.ModalBody(modal_texts[name][1]),
                    ],
                    id=f"{name}_modal",
                    size="sm",
                )
            ]
        )
    ]


embeddings = Checklist(
    options=[
        {
            "label": html.Div(
                [f"{length}"],
                style={"color": "white", "font-size": 10, "marginRight": 10},
            ),
            "value": length,
            "disabled": True,
        }
        for length in ring_lengths
    ],
    value=[],
    id=f"embedding_is_cached",
    style={"color": "white"},
    inline=True,
)

tooltips_activate = RadioItems(
    id="tooltips_show",
    options=[
        {
            "label": "On",
            "value": "on",
        },
        {
            "label": "Off",
            "value": "off",
        },
    ],
    value="on",
    inputStyle={"margin-right": "10px", "margin-bottom": "10px"},
    labelStyle={
        "color": "white",
        "font-size": 12,
        "display": "inline-block",
        "marginLeft": 20,
    },
    inline=True,  # Currently requires above 'inline-block'
)
