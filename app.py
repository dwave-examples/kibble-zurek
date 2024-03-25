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

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc, html, Input, Output, State
from dash.dcc import Dropdown

import plotly.graph_objects as go

import datetime
import numpy as np
import os

from dwave.cloud import Client
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.system import DWaveSampler

from helpers.tooltips import tool_tips
from helpers.layouts import *
from helpers.plots import *
from helpers.kb_calcs import *
from helpers.cached_embeddings import cached_embeddings
from helpers.qa import *
#from kz import build_bqm

from zzz_TMP import placeholder_params      # TEMPORARY UNTIL SAPI ADDS FEATURE PARAMS

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

try:
    client = Client.from_config(client="qpu")
    # TODO: change to "fast_anneal_time_range"
    qpus = {qpu.name: qpu for qpu in client.get_solvers(anneal_schedule=True)}     
    if len(qpus) < 1:
        client.close()
        init_job_status = "NO_SOLVER"
        job_status_color = dict(color="red")
    else:
        init_job_status = "READY"
        job_status_color = dict()
except Exception as client_err:
    client = None
    init_job_status = "NO_SOLVER"
    job_status_color = dict(color="red")

schedules = [file for file in os.listdir('helpers') if ".csv" in file]
best_schedules = {"NO SCHEDULE FOUND: Using generic schedule": "FALLBACK_SCHEDULE.csv"}
for qpu_name in qpus:
    for schedule_name in schedules:
        if qpu_name.split(".")[0] in schedule_name:
            best_schedules[qpu_name] = schedule_name

# Simulation panel
simulation_card = dbc.Card([
    html.H4(
        "Simulation", 
        className="card-title",
        style={"color":"rgb(243, 120, 32)"}
    ),
    dbc.Col([
        dbc.Button(
            "Simulate", 
            id="btn_simulate", 
            color="primary", 
            className="me-1",
            style={"marginBottom":"5px"}
        ),
        dcc.Interval(
            id="wd_job", 
            interval=None, 
            n_intervals=0, 
            disabled=True, 
            max_intervals=1
        ),
        dbc.Progress(
            id="bar_job_status", 
            value=0,
            color="link", 
            className="mb-3",
            style={"width": "60%"}
        ),
        html.P(
            id="job_submit_state", 
            children=f"Status: {init_job_status}",
            style={"color": "white", "fontSize": 12}
        ),
        html.P(
            id="job_submit_time", 
            children="", 
            style = dict(display="none")
        ),
        status_solver,
        html.P(
            id="job_id", 
            children="", 
            style = dict(display="none")
        )
    ],
        width=12)
],
    color="dark", body=True
)

# Configuration panel
kz_config = dbc.Card([
    dbc.Row([
        dbc.Col([
            html.H4(
                "Configuration", 
                className="card-title",
                style={"color": "rgb(243, 120, 32)"}
            )
        ])
    ],
        id="tour_settings_row"
    ),
    dbc.Row([
        dbc.Col([
            html.P(
                "QPU",
                style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
            ), 
            html.Div([
                config_qpu_selection(qpus),
                html.P(
                    id="embedding", 
                    children="", 
                    style = dict(display="none")
                )
            ]), 
        ], 
            width=9
        ),
        dbc.Col([
            html.P(
                "Spins",
                style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
            ), 
            html.Div([
                config_chain_length
            ]), 
        ], 
            width=3
        ),
    ]),
    dbc.Row([
        dbc.Col([
            html.P(
                "Coupling Strength",
                style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
            ), 
            html.Div([
                config_coupling_strength
            ]),
        ]),
        dbc.Col([
            html.P(
                "Quench Duration [ns]",
                style={"color": "rgb(3, 184, 255)", "marginBottom": 0}
            ),
            html.Div([
                config_anneal_duration
                
            ]), 
        ]),
    ]),
], 
    body=True, 
    color="dark"
)

# Graph panel
graphs = dbc.Card([
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id="sample_vs_theory", 
                figure=go.Figure()
            )
        ], 
            width=6
        ),
        dbc.Col([
            dcc.Graph(
                id="sample_kinks"
            )
        ], 
            width=6
        ),
    ]),
], 
    color="dark"
)

# Page-layout section
app_layout = [
    dbc.Row([
        dbc.Col(
            kz_config,
            width=6
        ),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    simulation_card
                ])
            ]),
        ], 
            width=5
        ),
    ], 
        justify="left"
    ),
    dbc.Row([
        dbc.Col(
            graphs,   
            width=12
        ),
    ], 
        justify="left"
    ),
]

# tips = [dbc.Tooltip(
#             message, target=target, id=f"tooltip_{target}", style = dict())
#             for target, message in tool_tips.items()]
# app_layout.extend(tips)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(
                "Coherent Annealing: KZ Simulation", 
                style={"textAlign": "left", "color": "white"}
            )
        ],
            width=9
        ),
        dbc.Col([
            html.Img(
                src="assets/dwave_logo.png", 
                height="25px",
                style={"textAlign": "left"}
            )
        ],
            width=3
        )
    ]),
    dbc.Container(
        app_layout, 
        fluid=True,
        style={"color": "rgb(3, 184, 255)",
                "paddingLeft": 10, 
                "paddingRight": 10}
    )
],
    style={
        "backgroundColor": "black",
        "background-image": "url('assets/electric_squids.png')",
        "background-size": "cover",
        "paddingLeft": 100, 
        "paddingRight": 100,
        "paddingTop": 25, 
        "paddingBottom": 50
    }, 
    fluid=True
)

server = app.server
app.config["suppress_callback_exceptions"] = True

# Callbacks Section

@app.callback(
    Output("solver_modal", "is_open"),
    Input("btn_simulate", "n_clicks"),)
def alert_no_solver(btn_simulate):
    """Notify if no Leap hybrid CQM solver is accessible."""

    trigger = dash.callback_context.triggered
    trigger_id = trigger[0]["prop_id"].split(".")[0]

    if trigger_id == "btn_simulate":
        if not client:
            return True

    return False

@app.callback(
    Output('embedding_is_cached', 'options'), 
    Output('embedding_is_cached', 'value'),
    Output('quench_schedule_filename', 'children'),
    Output('quench_schedule_filename', 'style'),
    Input('qpu_selection', 'value'))
def select_qpu(qpu_name):
    """Ensure embeddings and schedules"""

    if qpu_name and (qpu_name in cached_embeddings.keys()):
        
        embedding_lengths =  list(cached_embeddings[qpu_name].keys()) 

        options = [     # Display checklist for cached embeddings
            {"label": 
                html.Div([f"{length}"], 
                style={'color': 'white', 'font-size': 10, "marginRight": 10}), 
            "value": length,
            "disabled": length not in embedding_lengths
            }
            for length in ring_lengths 
        ]

        if qpu_name in schedule_name:   # Red if old version of schedule 
            style = {"color": "white", "fontSize": 12} 
            schedule = best_schedules[qpu_name]
        else:
           style = {"color": "red", "fontSize": 12} 
           schedule = next(iter(best_schedules)) 

        return options, embedding_lengths, schedule, style

    options = [     # Default: disable embeddings for all lengths
            {"label": 
                html.Div([f"{length}"], 
                style={'color': 'white', 'font-size': 10, "marginRight": 10}), 
            "value": length,
            "disabled": True
            }
            for length in ring_lengths  
        ]
    return options, [], "", dash.no_update

@app.callback(
    Output('coupling_strength_display', 'children'), 
    Input('coupling_strength', 'value'))
def update_j_output(value):
    J = value - 2
    return f"J={J:.1f}"

@app.callback(
    Output("sample_vs_theory", "figure"),
    Input("coupling_strength", "value"),
    Input("job_submit_state", "children"),
    State("job_id", "children"),
    State("anneal_duration", "min"),
    State("anneal_duration", "max"),
    State("anneal_duration", "value"),
    State('qpu_selection', 'value'),
    State('chain_length', 'value'),
    State("sample_vs_theory", "figure"),)
def display_graphics_left(J, job_submit_state, job_id, ta_min, ta_max, ta, qpu_name, spins, figure):
    """Generate graphics for theory and samples."""

    trigger = dash.callback_context.triggered
    trigger_id = trigger[0]["prop_id"].split(".")[0]

    if trigger_id == "coupling_strength":
        
        fig = plot_kink_densities_bg([ta_min, ta_max], J)

        return fig
    
    if trigger_id == "job_submit_state":

        if job_submit_state == "COMPLETED":
            
            sampleset = client.retrieve_answer(job_id).sampleset
            
            bqm = create_bqm(num_spins=spins, coupling_strength=J)
            embedding = cached_embeddings[qpu_name][spins]
            sampleset_unembedded = unembed_sampleset(sampleset, embedding, bqm)
            
            kink_density = avg_kink_density(sampleset_unembedded, J)
            
            fig = plot_kink_density(figure, kink_density, ta)

            return fig
        
        else:
            return dash.no_update
        
    fig = plot_kink_densities_bg([ta_min, ta_max], J)
    return fig

@app.callback(
    Output("anneal_duration", "disabled"),
    Output("coupling_strength", "disabled") ,
    Output("chain_length", "options"),
    Input("job_submit_state", "children"),
    State("chain_length", "options"))
def disable_buttons(job_submit_state, chain_length_options):        # Add cached embeddings
    """Disable user input during job submissions."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id !="job_submit_state":
        return dash.no_update, dash.no_update, dash.no_update

    if any(job_submit_state == status for status in ["EMBEDDING", "SUBMITTED", "PENDING", "IN_PROGRESS"]):
        
        chain_length_disable = chain_length_options
        for inx, option in enumerate(chain_length_disable): 
            chain_length_disable[inx]['disabled'] = True
        
        return  True, True, chain_length_disable

    elif any(job_submit_state == status for status in ["COMPLETED", "CANCELLED", "FAILED"]):

        chain_length_enable = chain_length_options
        for inx, option in enumerate(chain_length_enable): 
            chain_length_enable[inx]['disabled'] = False
        
        return False, False, chain_length_enable

    else:
        return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("job_id", "children"),
    Input("job_submit_time", "children"),
    State('qpu_selection', 'value'),
    State('chain_length', 'value'),
    State('coupling_strength', 'value'),
    State("anneal_duration", "value"),)
def submit_job(job_submit_time, qpu_name, spins, J, ta_ns):
    """Submit job and provide job ID."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id =="job_submit_time":

        solver = qpus[qpu_name]

        print(qpu_name, solver.name)

        bqm = create_bqm(num_spins=spins, coupling_strength=J)

        embedding = cached_embeddings[qpu_name][spins]

        bqm_embedded = embed_bqm(bqm, embedding, DWaveSampler(solver=solver.name).adjacency)

        param_dict = {
            "bqm": bqm_embedded,
            "anneal_time": 0.001 * ta_ns,
            "auto_scale": False, 
            "answer_mode": "raw",
            "num_reads": 100, 
            "label": f"Examples - KZ Simulation, submitted: {job_submit_time}",}
        param_dict = placeholder_params(param_dict)
        computation = solver.sample_bqm(**param_dict)      # Need final SAPI interface

        return computation.wait_id()

    return dash.no_update

@app.callback(
    Output("btn_simulate", "disabled"),
    Output("wd_job", "disabled"),
    Output("wd_job", "interval"),
    Output("wd_job", "n_intervals"),
    Output("job_submit_state", "children"),
    Output("job_submit_time", "children"),
    Input("btn_simulate", "n_clicks"),
    Input("wd_job", "n_intervals"),
    State("job_id", "children"),
    State("job_submit_state", "children"),
    State("job_submit_time", "children"),)
def simulate(n_clicks, n_intervals, job_id, job_submit_state, job_submit_time):
    """Manage simulation: embedding, job submission."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if not any(trigger_id == input for input in ["btn_simulate", "wd_job"]):
        return dash.no_update, dash.no_update, dash.no_update, \
            dash.no_update, dash.no_update, dash.no_update

    if trigger_id == "btn_simulate":

        submit_time = datetime.datetime.now().strftime("%c")
        disable_btn = True
        disable_watchdog = False

        return disable_btn, disable_watchdog, 0.5*1000, 0, "EMBEDDING", submit_time
    
    # if job_submit_state == "EMBEDDING":

    #     return True, False, 0.2*1000, 0, job_submit_state, dash.no_update

    if any(job_submit_state == status for status in
        ["EMBEDDING", "SUBMITTED", "PENDING", "IN_PROGRESS"]):

        job_submit_state = get_job_status(client, job_id, job_submit_time)
        if not job_submit_state:
            job_submit_state = "SUBMITTED"
            wd_time = 0.2*1000
        else:
            wd_time = 1*1000

        return True, False, wd_time, 0, job_submit_state, dash.no_update

    if any(job_submit_state == status for status in ["COMPLETED", "CANCELLED", "FAILED"]):

        disable_btn = False
        disable_watchdog = True

        return disable_btn, disable_watchdog, 0.1*1000, 0, dash.no_update, dash.no_update

    else:   # Exception state: should only ever happen in testing
        return False, True, 0, 0, "ERROR", dash.no_update, 

@app.callback(
    Output("bar_job_status", "value"),
    Output("bar_job_status", "color"),
    Input("job_submit_state", "children"),)
def set_progress_bar(job_submit_state):
    """Update progress bar for job submissions."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id != "job_submit_state":
        return job_bar_display["READY"][0], job_bar_display["READY"][1]
    else:
        state = job_submit_state
        return job_bar_display[state][0], job_bar_display[state][1]

if __name__ == "__main__":
    app.run_server(debug=True)
