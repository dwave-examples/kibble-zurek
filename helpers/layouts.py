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

import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dcc import Dropdown, Input, RadioItems
from dash import dcc, html

__all__ = ["config_anneal_duration", "config_chain_length", 
    "config_coupling_strength", "config_qpu_selection", "job_bar",
    "ring_lengths", "status_solver",]

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

config_chain_length = RadioItems(
    id="chain_length",
    options=[
        {
            "label": f"{length}", 
            "value": length, 
            "disabled": False
        } for length in ring_lengths
    ],
    value=512,
    inline=True,
    inputStyle={"margin-right": "10px", "margin-bottom": "10px"},
    labelStyle={"color": "white", "font-size": 12, "display": "flex"}
)

config_coupling_strength = html.Div([
    daq.Knob(
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
    ),
    html.Div(
    id='coupling_strength_display',
    style={'width': '100%', "color": "white", "marginLeft": 40, "marginTop": 0}
    )
])

def config_qpu_selection(solvers):
    return Dropdown(
        id="qpu_selection",
        options=[{"label": qpu_name, "value": qpu_name} for qpu_name in solvers],
        placeholder="Select a quantum computer"
    )

job_bar = {"READY": [0, "link"],
#   "WAITING": [0, "dark"],     Placeholder, to remember the color
    "NO_SOLVER": [100, "danger"],
    "SUBMITTED": [10, "info"],
    "PENDING": [50, "warning"],
    "IN_PROGRESS": [75 ,"primary"],
    "COMPLETED": [100, "success"],
    "CANCELLED": [100, "light"],
    "FAILED": [100, "danger"], 
    }

status_solver = dbc.Row([
    dbc.Col([
        html.P(
            "Cached Embeddings",
            style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
        ),
        dcc.Checklist(
            options=[{
                "label": 
                    html.Div([f"{length}"], 
                    style={'color': 'white', 'font-size': 10, "marginRight": 10}), 
                "value": length,
                "disabled": True} for length in ring_lengths   
            ], 
            value=[], 
            id=f"embedding_is_cached",
            style={"color": "white"},
            inline=True
        ),
    ]),
    dbc.Col([
        html.P(
            "Quench Schedule",
            style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
        ),
        html.P(
            id="quench_schedule_filename", 
            children="",
            style={"color": "white", "fontSize": 12}
        ),
    ])
])