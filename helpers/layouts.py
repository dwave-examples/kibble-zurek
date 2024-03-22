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

from dash.dcc import Dropdown, Input, Link, Slider, RadioItems
import dash_bootstrap_components as dbc
from dash import dcc, html
import dash_daq as daq

__all__ = ["config_anneal_duration", "config_chain_length", 
    "config_coupling_strength", "status_solver"]

config_anneal_duration = Input(
    id="anneal_duration",
    type="number",
    min=5,
    max=100,
    step=1,
    value=7,
    style={"max-width": "95%"})

config_chain_length = RadioItems(
    id="chain_length",
    options=[
    {"label": "512", "value": 512, "disabled": False},
    {"label": "1024", "value": 1024, "disabled": False },
    {"label": "2048", "value": 2048, "disabled": False},],
    value=512,
    inline=True,
    inputStyle={"margin-right": "10px", "margin-bottom": "10px"},
    labelStyle={"color": "white", "font-size": 12, "display": "flex"})

config_coupling_strength = html.Div([
    daq.Knob(
        id="coupling_strength",
        color={"default": "rgb(243, 120, 32)"},
        size=50,
        scale={"custom": {0: {"label": "-2", "style": {"color": "blue", "fontSize": 20}}, 
                            2: {"label": "0", "style": {"fontColor": "blue", "fontSize": 20}}, 
                            3: {"label": "1", "style": {"fontColor": "blue", "fontSize": 20}}}, 
                },
        style={"marginBottom": 0},
        min=0,
        max=3,
        value=-1.4+2),
    html.Div(
    id='coupling_strength_display',
    style={'width': '100%', "color": "white", "marginLeft": 40, "marginTop": 0}
    )
])

status_solver = dbc.Row([
    dbc.Col([
        html.P(f"Cached Embeddings",
            style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
            ),
        dcc.Checklist([
            {"label": html.Div(["512"], style={'color': 'white', 'font-size': 10, "marginRight": 10}), "value": 512,},
            {"label": html.Div(["1024"], style={'color': 'white', 'font-size': 10, "marginRight": 10}), "value": 1024,},
            {"label": html.Div(["2048"], style={'color': 'white', 'font-size': 10, "marginRight": 10}), "value": 2048,}], 
            value=[], 
            id=f"embedding_is_cached",
            style={"color": "white"},
            inline=True
        ),
    ])
])