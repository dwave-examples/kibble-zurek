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
from dash.dcc import Checklist, Dropdown, Input, RadioItems
from dash_daq import Knob
from dash import dcc, html, Input, Output, State

import plotly.graph_objects as go

from helpers.layouts import *

__all__ = ["graphs_card", ]

# Plots card
graphs_card = dbc.Card([
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

