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

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import datetime
import json
import plotly.graph_objects as go
import numpy as np
import os

import dimod
from dwave.cloud import Client
from dwave.embedding import embed_bqm, is_valid_embedding
from dwave.system import DWaveSampler

from helpers.kz_calcs import *
from helpers.layouts_cards import *
from helpers.layouts_components import *
from helpers.plots import *
from helpers.qa import *
from helpers.tooltips import tool_tips
#from kz import build_bqm

from zzz_TMP import placeholder_params      # TEMPORARY UNTIL SAPI ADDS FEATURE PARAMS

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

try:
    client = Client.from_config(client="qpu")
    # TODO: change to "fast_anneal_time_range"
    qpus = {qpu.name: qpu for qpu in client.get_solvers(anneal_schedule=True)}
    if len(qpus) < 1:
        raise Exception    
    init_job_status = "READY"
except Exception:
    qpus = {}
    client = None
    init_job_status = "NO SOLVER"

# Dashboard-organization section

app.layout = dbc.Container([
    # Logo
    dbc.Row([
        dbc.Col([
            html.Img(
                src="assets/dwave_logo.png", 
                height="25px",
                style={"textAlign": "left"}
            )
        ],
            width=3,
        )
    ]),
    dbc.Row([
        # Left: Control panel 
        dbc.Col(
            [
            control_card(
                solvers=qpus, 
                init_job_status=init_job_status
            ),
            *dbc_modal("modal_solver"),
            # [dbc.Tooltip(
            # message, target=target, id=f"tooltip_{target}", style = dict())
            # for target, message in tool_tips.items()]
            ],
            width=4,   
        ),
        # Right: Display area
        dbc.Col(
            graphs_card(),
            width=8,
        ),
    ]),
],
    style={
        "color": "rgb(3, 184, 255)",
        "backgroundColor": "black",
        "background-size": "cover",
        "paddingLeft": 10, 
        "paddingRight": 10,
        "paddingTop": 25, 
        "paddingBottom": 100
    }, 
    fluid=True,
)

server = app.server
app.config["suppress_callback_exceptions"] = True

# Callbacks Section

@app.callback(
    Output("solver_modal", "is_open"),
    Input("btn_simulate", "n_clicks"),)
def alert_no_solver(btn_simulate):
    """Notify if no quantum computer is accessible."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "btn_simulate":
        if not client:
            return True

    return False

@app.callback(
    Output('quench_schedule_filename', 'children'),
    Output('quench_schedule_filename', 'style'),
    Input('qpu_selection', 'value'),)
def set_schedule(qpu_name):
    """Set the schedule for the selected QPU.
    """

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    schedule_filename = "FALLBACK_SCHEDULE.csv"  
    schedule_filename_style = {"color": "red", "fontSize": 12}
 
    if trigger_id == 'qpu_selection':

        for filename in [file for file in os.listdir('helpers') if "schedule.csv" in file.lower()]:

            if qpu_name.split(".")[0] in filename:  # Accepts & reddens older major versions
            
                schedule_filename = filename

                if qpu_name in filename:

                    schedule_filename_style = {"color": "white", "fontSize": 12} 

    return schedule_filename, schedule_filename_style

@app.callback(
    Output('embeddings_cached', 'data'),
    Output('embedding_is_cached', 'value'),
    Input('qpu_selection', 'value'),
    Input("embeddings_found", "data"),
    State("embeddings_cached", "data"),)
def cache_embeddings(qpu_name, embeddings_found, embeddings_cached):
    """Cache embeddings for the selected QPU.
    """

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
 
    if trigger_id == 'qpu_selection':

        embeddings_cached = {}  # Wipe out previous QPU's embeddings
          
        for filename in [file for file in os.listdir('helpers') if ".json" in file and "emb_" in file]:

            if qpu_name.split(".")[0] in filename:

                with open(f'helpers/{filename}', 'r') as fp:
                    embeddings_cached = json.load(fp)

                embeddings_cached = json_to_dict(embeddings_cached)
                            
                # Validate that loaded embeddings' edges are still available on the selected QPU
                for length in list(embeddings_cached.keys()):
                    
                    source_graph = dimod.to_networkx_graph(create_bqm(num_spins=length)).edges 
                    target_graph = qpus[qpu_name].edges
                    emb = embeddings_cached[length]

                    if not is_valid_embedding(emb, source_graph, target_graph):

                        del embeddings_cached[length]

    if trigger_id == 'embeddings_found':

        if embeddings_found != "needed":

            embeddings_cached = json_to_dict(embeddings_cached)
            embeddings_found = json_to_dict(embeddings_found)
            embeddings_cached[list(embeddings_found.keys())[0]] = embeddings_found[list(embeddings_found.keys())[0]]

    return embeddings_cached, list(embeddings_cached.keys())

@app.callback(
    Output('coupling_strength_display', 'children'), 
    Input('coupling_strength', 'value'))
def update_j_output(J_offset):
    J = J_offset - 2
    return f"J={J:.1f}"

@app.callback(
    Output("sample_vs_theory", "figure"),
    Input("coupling_strength", "value"),
    Input("quench_schedule_filename", "children"),
    Input("job_submit_state", "children"),
    State("job_id", "children"),
    State("anneal_duration", "min"),
    State("anneal_duration", "max"),
    State("anneal_duration", "value"),
    State('chain_length', 'value'),
    State('embeddings_cached', 'data'),
    State("sample_vs_theory", "figure"),)
def display_graphics_left(J_offset, schedule_filename, job_submit_state, job_id, ta_min, ta_max, ta, \
    spins, embeddings_cached, figure):
    """Generate graphics for theory and samples."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    J = J_offset - 2

    if trigger_id in ["coupling_strength", "quench_schedule_filename"] :
        
        fig = plot_kink_densities_bg([ta_min, ta_max], J, schedule_filename)

        return fig
    
    if trigger_id == "job_submit_state":

        if job_submit_state == "COMPLETED":

            embeddings_cached = embeddings_cached = json_to_dict(embeddings_cached)

            sampleset_unembedded = get_samples(client, job_id, spins, J, embeddings_cached[spins])              
            _, kink_density = kink_stats(sampleset_unembedded, J)
            
            fig = plot_kink_density(figure, kink_density, ta)
            return fig
        
        else:
            return dash.no_update
        
    fig = plot_kink_densities_bg([ta_min, ta_max], J, schedule_filename)
    return fig

@app.callback(
    Output("spin_orientation", "figure"),
    Input('chain_length', 'value'),
    Input("job_submit_state", "children"),
    State("job_id", "children"),
    State("coupling_strength", "value"),
    State('embeddings_cached', 'data'),)
def display_graphics_right(spins, job_submit_state, job_id, J_offset, embeddings_cached):
    """Generate graphics for spin display."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    J = J_offset - 2
   
    if trigger_id == "job_submit_state": 
    
        if job_submit_state == "COMPLETED":

            embeddings_cached = embeddings_cached = json_to_dict(embeddings_cached)
            sampleset_unembedded = get_samples(client, job_id, spins, J, embeddings_cached[spins])
            kinks_per_sample, kink_density = kink_stats(sampleset_unembedded, J)
            best_indx = np.abs(kinks_per_sample - kink_density).argmin()
            best_sample = sampleset_unembedded.record.sample[best_indx]

            fig = plot_spin_orientation(num_spins=spins, sample=best_sample)
            return fig
        
        else:

            return dash.no_update

    fig = plot_spin_orientation(num_spins=spins, sample=None)
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
    State("anneal_duration", "value"),
    State('embeddings_cached', 'data'),)
def submit_job(job_submit_time, qpu_name, spins, J_offset, ta_ns, embeddings_cached):
    """Submit job and provide job ID."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    J = J_offset - 2

    if trigger_id =="job_submit_time":

        solver = qpus[qpu_name]

        bqm = create_bqm(num_spins=spins, coupling_strength=J)

        embeddings_cached = embeddings_cached = json_to_dict(embeddings_cached)
        embedding = embeddings_cached[spins]

        bqm_embedded = embed_bqm(bqm, embedding, DWaveSampler(solver=solver.name).adjacency)

        param_dict = {
            "bqm": bqm_embedded,
            "annealing_time": 0.001 * ta_ns,
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
    Output("embeddings_found", "data"),
    Input("btn_simulate", "n_clicks"),
    Input("wd_job", "n_intervals"),
    State("job_id", "children"),
    State("job_submit_state", "children"),
    State("job_submit_time", "children"),
    State('embedding_is_cached', 'value'),
    State('chain_length', 'value'),
    State('qpu_selection', 'value'),
    State("embeddings_found", "data"),)
def simulate(dummy1, dummy2, job_id, job_submit_state, job_submit_time, \
             cached_embedding_lengths, spins, qpu_name, embeddings_found):
    """Manage simulation: embedding, job submission."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if not any(trigger_id == input for input in ["btn_simulate", "wd_job"]):
        return dash.no_update, dash.no_update, dash.no_update, \
            dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if trigger_id == "btn_simulate":

        if spins in cached_embedding_lengths:   

            submit_time = datetime.datetime.now().strftime("%c")
            job_submit_state = "SUBMITTED"
            embedding = dash.no_update

        else:

            submit_time = dash.no_update
            job_submit_state = "EMBEDDING"
            embedding = "needed"

        disable_btn = True
        disable_watchdog = False

        return disable_btn, disable_watchdog, 0.5*1000, 0, job_submit_state, submit_time, embedding
    
    if job_submit_state == "EMBEDDING":

        submit_time = dash.no_update
        embedding = dash.no_update

        if embeddings_found == "needed":

            try:
                embedding = find_one_to_one_embedding(spins, qpus[qpu_name].edges)
                if embedding:
                    job_submit_state = "EMBEDDING"  # Stay another WD to allow caching the embedding
                    embedding = {spins: embedding}
            except Exception:
                job_submit_state = "FAILED"

        else:   # Found embedding last WD, so is cached, so now can submit job
            
            submit_time = datetime.datetime.now().strftime("%c")
            job_submit_state = "SUBMITTED"

        return True, False, 0.2*1000, 0, job_submit_state, submit_time, embedding

    if any(job_submit_state == status for status in
        ["SUBMITTED", "PENDING", "IN_PROGRESS"]):

        job_submit_state = get_job_status(client, job_id, job_submit_time)
        if not job_submit_state:
            job_submit_state = "SUBMITTED"
            wd_time = 0.2*1000
        else:
            wd_time = 1*1000

        return True, False, wd_time, 0, job_submit_state, dash.no_update, dash.no_update

    if any(job_submit_state == status for status in ["COMPLETED", "CANCELLED", "FAILED"]):

        disable_btn = False
        disable_watchdog = True

        return disable_btn, disable_watchdog, 0.1*1000, 0, dash.no_update, dash.no_update, dash.no_update

    else:   # Exception state: should only ever happen in testing
        return False, True, 0, 0, "ERROR", dash.no_update, dash.no_update

@app.callback(
    Output("bar_job_status", "value"),
    Output("bar_job_status", "color"),
    Input("job_submit_state", "children"),)
def set_progress_bar(job_submit_state):
    """Update progress bar for job submissions."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "job_submit_state":
 
        return job_bar_display[job_submit_state][0], job_bar_display[job_submit_state][1]
    
    return job_bar_display["READY"][0], job_bar_display["READY"][1]

if __name__ == "__main__":
    app.run_server(debug=True)
