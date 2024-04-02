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
from dash.dcc import Checklist, Dropdown, Input, Link, RadioItems
from dash_daq import Knob
from dash import html

__all__ = ["config_anneal_duration", "config_spins", 
    "config_coupling_strength", "config_qpu_selection", "dbc_modal", "embeddings",
    "job_bar_display", "ring_lengths", ]

ring_lengths = [512, 1024, 2048]

config_anneal_duration = Input(
    id="anneal_duration",
    type="number",
    min=5,
    max=100,
    step=1,
    value=7,
    style={"max-width": "95%"}
)

config_spins = RadioItems(
    id="spins",
    options=[
        {
            "label": f"{length}", 
            "value": length, 
            "disabled": False
        } for length in ring_lengths
    ],
    value=512,
    inputStyle={"margin-right": "10px", "margin-bottom": "10px"},
    labelStyle={"color": "white", "font-size": 12, 'display': 'inline-block', 'marginLeft': 20},
    inline=True,    # Currently requires above "inline-block"
)

config_coupling_strength = dbc.Row([
    dbc.Col([
        html.Div([
            Knob(
                id="coupling_strength",
                color={"default": "rgb(243, 120, 32)"},
                size=50,
                scale={"custom": {
                        0: {"label": "-2", "style": {"fontSize": 20}}, 
                        2: {"label": "0", "style": {"fontSize": 20}}, 
                        3: {"label": "1", "style": {"fontSize": 20}}
                                }, 
                        },
                style={"marginBottom": 0},
                min=0,
                max=3,
                value=-1.4+2
            )
        ]),
    ],
    width=4,
    ),
    dbc.Col([        
        html.Div(
            id='coupling_strength_display',
            style={'width': '100%', "color": "white", "marginLeft": 40, "marginTop": 0}
        ),
    ],
    width=8,
    ),
])

def config_qpu_selection(solvers):
    return Dropdown(
        id="qpu_selection",
        options=[{"label": qpu_name, "value": qpu_name} for qpu_name in solvers],
        placeholder="Select a quantum computer"
    )

job_bar_display = {
    "READY": [0, "link"],
    "EMBEDDING": [20, "warning"],     
    "NO SOLVER": [100, "danger"],
    "SUBMITTED": [40, "info"],
    "PENDING": [60, "primary"],
    "IN_PROGRESS": [85 ,"dark"],
    "COMPLETED": [100, "success"],
    "CANCELLED": [100, "light"],
    "FAILED": [100, "danger"], 
}

modal_texts = {
    "solver": ["Leap's Quantum Computers Inaccessible",
    [
        html.Div([
        html.Div("Could not connect to a Leap quantum computer."),
        html.Div(["""
    If you are running locally, set environment variables or a
    dwave-cloud-client configuration file as described in the
    """,
        Link(children=[html.Div(" Ocean")],
            href="https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html",
            style={"display":"inline-block"}),
        "documentation."],
            style={"display":"inline-block"}),
        html.Div(["If you are running in an online IDE, see the ",
        Link(children=[html.Div("system documentation")],
            href="https://docs.dwavesys.com/docs/latest/doc_leap_dev_env.html",
            style={"display":"inline-block"}),
        " on supported IDEs."],
            style={"display":"inline-block"}),])]
    ],
}

def dbc_modal(name):
    name = name.split("_")[1]
    return [
        html.Div([
            dbc.Modal([
                dbc.ModalHeader(
                    dbc.ModalTitle(
                        modal_texts[name][0]
                    )
                ),
                dbc.ModalBody(
                    modal_texts[name][1]
                ),
            ],
            id=f"{name}_modal", size="sm")
        ])
        ]

embeddings = Checklist(
    options=[{
        "label": 
            html.Div([
                f"{length}"], 
                style={'color': 'white', 'font-size': 10, "marginRight": 10}
            ), 
        "value": length,
        "disabled": True,
    } for length in ring_lengths], 
    value=[], 
    id=f"embedding_is_cached",
    style={"color": "white"},
    inline=True
)
