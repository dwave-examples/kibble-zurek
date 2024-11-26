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
from dash import html, Input, Output, State
import datetime
import json
import numpy as np
import os

from dash import dcc
from collections import defaultdict
from numpy.polynomial.polynomial import Polynomial

import dimod
from dwave.cloud import Client
from dwave.embedding import embed_bqm, is_valid_embedding
from dwave.system import DWaveSampler
from MockKibbleZurekSampler import MockKibbleZurekSampler
from dwave.samplers import SimulatedAnnealingSampler

from helpers.kz_calcs import *
from helpers.layouts_cards import *
from helpers.layouts_components import *
from helpers.plots import *
from helpers.qa import *
from helpers.tooltips import tool_tips

import networkx as nx
from minorminer.subgraph import find_subgraph
from plotly.subplots import make_subplots
import plotly.graph_objects as go

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# global variable for a default J value
J_baseline = -1.8

# Initialize: available QPUs, initial progress-bar status 
try:
    client = Client.from_config(client='qpu')
    qpus = {qpu.name: qpu for qpu in client.get_solvers(fast_anneal_time_range__covers=[0.005, 0.1])}
    if len(qpus) < 1:
        raise Exception    
    init_job_status = 'READY'
except Exception:
    qpus = {}
    client = None
    init_job_status = 'NO SOLVER'
if os.getenv('ZNE') == "YES":
    qpus['mock_dwave_solver'] = MockKibbleZurekSampler(topology_type='pegasus', topology_shape=[16]) # Change sampler to mock
    init_job_status = 'READY'
    if not client:
        client = 'dummy'


# Dashboard-organization section
app.layout = dbc.Container([
    dbc.Row([                       # Top: logo
        dbc.Col([
            html.Img(
                src='assets/dwave_logo.png', 
                height='25px',
                style={'textAlign': 'left', 'margin': '10px 0px 15px 0px'}
            )
        ],
            width=3,
        )
    ]),
    dbc.Row([                        
        dbc.Col(                    # Left: control panel
            [
            control_card(
                solvers=qpus, 
                init_job_status=init_job_status
            ),
            *dbc_modal('modal_solver'),
            *[dbc.Tooltip(
            message, target=target, id=f'tooltip_{target}', style = dict())
            for target, message in tool_tips.items()]
            ],
            width=4,
            style={'minWidth': "30rem"},
        ),
        dbc.Col(                    # Right: display area
            graphs_card(),
            width=8,
            style={'minWidth': "60rem"},
        ),
    ]),
    # store coupling data points 
    dcc.Store(id='coupling_data', data={}),
    dcc.Store(id='kink_density_data', data={}),
    # store zero noise extrapolation
    dcc.Store(id='zne_estimates', data={}),
],
    fluid=True,
)

server = app.server
app.config['suppress_callback_exceptions'] = True

# Callbacks Section

@app.callback(
    Output('solver_modal', 'is_open'),
    Input('btn_simulate', 'n_clicks'),)
def alert_no_solver(dummy):
    """Notify if no quantum computer is accessible."""

    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'btn_simulate':
        if not client:
            return True

    return False

@app.callback(
    Output('anneal_duration', 'disabled'),
    Output('coupling_strength', 'disabled'),
    Output('spins', 'options'),
    Output('qpu_selection', 'disabled'),
    Input('job_submit_state', 'children'),
    State('spins', 'options'))
def disable_buttons(job_submit_state, spins_options):        
    """Disable user input during job submissions."""

    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if trigger_id !='job_submit_state':
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if job_submit_state in ['EMBEDDING', 'SUBMITTED', 'PENDING', 'IN_PROGRESS']:
        
        for inx, _ in enumerate(spins_options): 
            
            spins_options[inx]['disabled'] = True
        
        return  True, True, spins_options, True

    elif job_submit_state in ['COMPLETED', 'CANCELLED', 'FAILED']:

        for inx, _ in enumerate(spins_options): 
            spins_options[inx]['disabled'] = False
        
        return False, False, spins_options, False

    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('quench_schedule_filename', 'children'),
    Output('quench_schedule_filename', 'style'),
    Input('qpu_selection', 'value'),)
def set_schedule(qpu_name):
    """Set the schedule for the selected QPU."""

    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    schedule_filename = 'FALLBACK_SCHEDULE.csv'  
    schedule_filename_style = {'color': 'red', 'fontSize': 12}
 
    if trigger_id == 'qpu_selection':

        for filename in [file for file in os.listdir('helpers') if 
                         'schedule.csv' in file.lower()]:
                        
            if qpu_name.split('.')[0] in filename:  # Accepts & reddens older versions
            
                schedule_filename = filename

                if qpu_name in filename:

                    schedule_filename_style = {'color': 'white', 'fontSize': 12} 

    return schedule_filename, schedule_filename_style

@app.callback(
    Output('embeddings_cached', 'data'),
    Output('embedding_is_cached', 'value'),
    Input('qpu_selection', 'value'),
    Input('embeddings_found', 'data'),
    State('embeddings_cached', 'data'),
    State('spins', 'value'))
def cache_embeddings(qpu_name, embeddings_found, embeddings_cached, spins):
    """Cache embeddings for the selected QPU."""

    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'qpu_selection':

        if qpu_name == 'mock_dwave_solver':
            embeddings_cached = {}
            L = spins
            edges = [(i, (i + 1)%L) for i in range(L)]
            emb = find_subgraph(target=qpus['mock_dwave_solver'].to_networkx_graph(), source=nx.from_edgelist(edges))
            emb = {u: [v] for u, v in emb.items()}  # Wrap target nodes in lists
            embeddings_cached[spins] = emb  # Store embedding in cache
            return embeddings_cached, [spins]
            # filename = [file for file in os.listdir('helpers') if 
            #                 '.json' in file and 'emb_' in file][0]
            # with open(f'helpers/{filename}', 'r') as fp:
            #             embeddings_cached = json.load(fp)
            # embeddings_cached = json_to_dict(embeddings_cached)
            # return embeddings_cached, list()

        embeddings_cached = {}  # Wipe out previous QPU's embeddings

        for filename in [file for file in os.listdir('helpers') if 
                         '.json' in file and 'emb_' in file]:

            if qpu_name.split('.')[0] in filename:

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

        if not isinstance(embeddings_found, str): # embeddings_found != 'needed' or 'not found'

            embeddings_cached = json_to_dict(embeddings_cached)
            embeddings_found = json_to_dict(embeddings_found)
            new_embedding = list(embeddings_found.keys())[0]
            embeddings_cached[new_embedding] = embeddings_found[new_embedding]

        else:
            return dash.no_update, dash.no_update

    return embeddings_cached, list(embeddings_cached.keys())

@app.callback(
    Output('sample_vs_theory', 'figure'),
    Output('coupling_data', 'data'), # store data using dcc
    Output('zne_estimates', 'data'),  # update zne_estimates
    Output('kink_density_data', 'data'),  # update kink density data
    Input('kz_graph_display', 'value'),
    State('coupling_strength', 'value'), # previously input 
    Input('quench_schedule_filename', 'children'),
    Input('job_submit_state', 'children'),
    State('job_id', 'children'),
    # State('anneal_duration', 'min'),
    # State('anneal_duration', 'max'),
    State('anneal_duration', 'value'),
    State('spins', 'value'),
    State('embeddings_cached', 'data'),
    State('sample_vs_theory', 'figure'),
    State('coupling_data', 'data'), # access previously stored data 
    State('zne_estimates', 'data'),  # Access ZNE estimates
    State('kink_density_data', 'data'),
    )
def display_graphics_kink_density(kz_graph_display, J, schedule_filename, \
    job_submit_state, job_id, ta, \
    spins, embeddings_cached, figure, coupling_data, zne_estimates, kink_density_data):
    """Generate graphics for kink density based on theory and QPU samples."""

    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if trigger_id in ['kz_graph_display', 'coupling_strength', 'quench_schedule_filename'] :
        
        ta_min = 2
        ta_max = 350

       # Use global J
        fig = plot_kink_densities_bg(kz_graph_display, [ta_min, ta_max], J_baseline, schedule_filename)

        # reset couplingd ata storage if other plot are displayed
        if kz_graph_display != 'coupling':
            coupling_data = {}
            zne_estimates = {}

        return fig, coupling_data, zne_estimates, kink_density_data
    
    if trigger_id == 'job_submit_state':

        if job_submit_state == 'COMPLETED':

            embeddings_cached = embeddings_cached = json_to_dict(embeddings_cached)

            sampleset_unembedded = get_samples(client, job_id, spins, J, embeddings_cached[spins])              
            _, kink_density = kink_stats(sampleset_unembedded, J)
            
            fig = plot_kink_density(kz_graph_display, figure, kink_density, ta, J)

            if kz_graph_display == 'coupling':
                # Calculate kappa
                kappa = -1.8 / J
                
                # Initialize the list for this anneal_time if not present
                ta_str = str(ta)
                if ta_str not in coupling_data:
                    coupling_data[ta_str] = []
                
                # Append the new data point
                coupling_data[ta_str].append({'kappa': kappa, 'kink_density': kink_density})
                
                # Check if more than two data points exist for this anneal_time
                if len(coupling_data[ta_str]) > 2:
                    # Perform a polynomial fit (e.g., linear)
                    data_points = coupling_data[ta_str]
                    x = np.array([point['kappa'] for point in data_points])
                    y = np.array([point['kink_density'] for point in data_points])
                    
                    # Ensure there are enough unique x values for fitting
                    if len(np.unique(x)) > 1:
                        # Fit a 1st degree polynomial (linear fit)
                        coeffs = Polynomial.fit(x, y, deg=1).convert().coef
                        p = Polynomial(coeffs)
                        
                        a = p(0)  # p(kappa=0) = a + b*0 = a
                        zne_estimates[ta_str] = a

                        # Generate fit curve points
                        x_fit = np.linspace(min(x), max(x), 100)
                        y_fit = p(x_fit)
                        
                        # Remove existing fitting curve traces to prevent duplication
                        fig.data = [trace for trace in fig.data if trace.name != 'Fitting Curve']
                        # Remove existing ZNE Estimate traces to prevent duplication
                        fig.data = [trace for trace in fig.data if trace.name != 'ZNE Estimate']
                        
                        # Add the new fitting curve
                        fit_trace = go.Scatter(
                            x=x_fit,
                            y=y_fit,
                            mode='lines',
                            name='Fitting Curve',
                            line=dict(color='green', dash='dash'),
                            showlegend=True,
                            xaxis='x3',  
                            yaxis='y1', 
                        )
                        
                        fig.add_trace(fit_trace)

                        # Add the ZNE point at kappa=0
                        zne_trace = go.Scatter(
                            x=[0],
                            y=[a],
                            mode='markers',
                            name='ZNE Estimate',
                            marker=dict(size=12, color='purple', symbol='diamond'),
                            showlegend=False,
                            xaxis='x3',
                            yaxis='y1',
                        )
                        
                        fig.add_trace(zne_trace)

            elif kz_graph_display == 'kink_density':
                # Initialize the list for this anneal_time if not present
                ta_str = str(ta)
                if ta_str not in kink_density_data:
                    kink_density_data[ta_str] = []
                
                # Append the new data point
                kink_density_data[ta_str].append({'ta': ta, 'kink_density': kink_density})
                ta_str = str(ta)
                # Check if more than two data points exist for this anneal_time
                if len(kink_density_data[ta_str]) > 2:
                    # Perform a polynomial fit (e.g., linear)
                    data_points = kink_density_data[ta_str]
                    x = np.array([point['ta'] for point in data_points])
                    y = np.array([point['kink_density'] for point in data_points])
                    coeffs = Polynomial.fit(x, y, deg=1).convert().coef
                    p = Polynomial(coeffs)
                    
                    a = p(0)  # p(kappa=0) = a + b*0 = a
                    zne_estimates[ta_str] = a
                    # Generate fit curve points
                    x_fit = np.linspace(min(x), max(x), 100)
                    y_fit = p(x_fit)
                    
                   # Add the ZNE point at kappa=0
                    zne_trace = go.Scatter(
                        x=[0],
                        y=[a],
                        mode='markers',
                        name='ZNE Estimate',
                        marker=dict(size=12, color='purple', symbol='diamond'),
                        xaxis='x1',
                        yaxis='y1',
                        showlegend=False,
                    )
                    
                    fig.add_trace(zne_trace)
            
            return fig, coupling_data, zne_estimates, kink_density_data
        
        else:
            return dash.no_update
        
        # use global J value
    fig = plot_kink_densities_bg(kz_graph_display, [ta_min, ta_max], J_baseline, schedule_filename)
    return fig, coupling_data, zne_estimates, kink_density

@app.callback(
    Output('spin_orientation', 'figure'),
    Input('spins', 'value'),
    Input('job_submit_state', 'children'),
    State('job_id', 'children'),
    State('coupling_strength', 'value'),
    State('embeddings_cached', 'data'),)
def display_graphics_spin_ring(spins, job_submit_state, job_id, J, embeddings_cached):
    """Generate graphics for spin-ring display."""

    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
   
    if trigger_id == 'job_submit_state': 
    
        if job_submit_state == 'COMPLETED':

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
    Output('job_id', 'children'),
    Input('job_submit_time', 'children'),
    State('qpu_selection', 'value'),
    State('spins', 'value'),
    State('coupling_strength', 'value'),
    State('anneal_duration', 'value'),
    State('embeddings_cached', 'data'),)
def submit_job(job_submit_time, qpu_name, spins, J, ta_ns, embeddings_cached):
    """Submit job and provide job ID."""

    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if trigger_id =='job_submit_time':

        solver = qpus[qpu_name]

        bqm = create_bqm(num_spins=spins, coupling_strength=J)

        if qpu_name == 'mock_dwave_solver':
            embedding = embeddings_cached
            emb = find_subgraph(
                    target=qpus['mock_dwave_solver'].to_networkx_graph(), 
                    source=dimod.to_networkx_graph(bqm))
            emb = {u: [v] for u, v in emb.items()}
            bqm_embedded = embed_bqm(bqm, emb, MockKibbleZurekSampler(topology_type='pegasus', topology_shape=[16]).adjacency)
            # Calculate annealing_time in microseconds as per your setup
            annealing_time = ta_ns / 1000  # ta_ns is in nanoseconds
            sampleset = qpus['mock_dwave_solver'].sample(bqm_embedded, annealing_time=annealing_time)
            return json.dumps(sampleset.to_serializable())

        else:

            embeddings_cached = json_to_dict(embeddings_cached)
            embedding = embeddings_cached[spins]

            bqm_embedded = embed_bqm(bqm, embedding, DWaveSampler(solver=solver.name).adjacency)

            computation = solver.sample_bqm(
                bqm=bqm_embedded,
                fast_anneal=True,
                annealing_time=0.001*ta_ns,     # SAPI anneal time units is microseconds
                auto_scale=False, 
                answer_mode='raw',              # Easier than accounting for num_occurrences
                num_reads=100, 
                label=f'Examples - Kibble-Zurek Simulation, submitted: {job_submit_time}',)   

        return computation.wait_id()

    return dash.no_update

@app.callback(
    Output('btn_simulate', 'disabled'),
    Output('wd_job', 'disabled'),
    Output('wd_job', 'interval'),
    Output('wd_job', 'n_intervals'),
    Output('job_submit_state', 'children'),
    Output('job_submit_time', 'children'),
    Output('embeddings_found', 'data'),
    Input('btn_simulate', 'n_clicks'),
    Input('wd_job', 'n_intervals'),
    State('job_id', 'children'),
    State('job_submit_state', 'children'),
    State('job_submit_time', 'children'),
    State('embedding_is_cached', 'value'),
    State('spins', 'value'),
    State('qpu_selection', 'value'),
    State('embeddings_found', 'data'),)
def simulate(dummy1, dummy2, job_id, job_submit_state, job_submit_time, \
             cached_embedding_lengths, spins, qpu_name, embeddings_found):
    """Manage simulation: embedding, job submission."""

    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if not any(trigger_id == input for input in ['btn_simulate', 'wd_job']):
        return dash.no_update, dash.no_update, dash.no_update, \
            dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if trigger_id == 'btn_simulate':

        if spins in cached_embedding_lengths or qpu_name == 'mock_dwave_solver':

            submit_time = datetime.datetime.now().strftime('%c')
            if qpu_name == 'mock_dwave_solver':    # Hack to fix switch from SA to QPU
                submit_time = 'SA'
            job_submit_state = 'SUBMITTED'
            embedding = dash.no_update

        else:

            submit_time = dash.no_update
            job_submit_state = 'EMBEDDING'
            embedding = 'needed'

        disable_btn = True
        disable_watchdog = False

        return disable_btn, disable_watchdog, 0.5*1000, 0, job_submit_state, submit_time, embedding
    
    if job_submit_state == 'EMBEDDING':

        submit_time = dash.no_update
        embedding = dash.no_update

        if embeddings_found == 'needed':

            try:
                embedding = find_one_to_one_embedding(spins, qpus[qpu_name].edges)
                if embedding:
                    job_submit_state = 'EMBEDDING'  # Stay another WD to allow caching the embedding
                    embedding = {spins: embedding}
                else:
                    job_submit_state = 'FAILED'
                    embedding = 'not found'
            except Exception:
                job_submit_state = 'FAILED'
                embedding = 'not found'

        else:   # Found embedding last WD, so is cached, so now can submit job
            
            submit_time = datetime.datetime.now().strftime('%c')
            job_submit_state = 'SUBMITTED'

        return True, False, 0.2*1000, 0, job_submit_state, submit_time, embedding

    if any(job_submit_state == status for status in
        ['SUBMITTED', 'PENDING', 'IN_PROGRESS']):

        job_submit_state = get_job_status(client, job_id, job_submit_time)
        if not job_submit_state:
            job_submit_state = 'SUBMITTED'
            wd_time = 0.2*1000
        else:
            wd_time = 1*1000

        return True, False, wd_time, 0, job_submit_state, dash.no_update, dash.no_update

    if any(job_submit_state == status for status in ['COMPLETED', 'CANCELLED', 'FAILED']):

        disable_btn = False
        disable_watchdog = True

        return disable_btn, disable_watchdog, 0.1*1000, 0, dash.no_update, dash.no_update, dash.no_update

    else:   # Exception state: should only ever happen in testing
        return False, True, 0, 0, 'ERROR', dash.no_update, dash.no_update

@app.callback(
    Output('bar_job_status', 'value'),
    Output('bar_job_status', 'color'),
    Input('job_submit_state', 'children'),)
def set_progress_bar(job_submit_state):
    """Update progress bar for job submissions."""

    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'job_submit_state':
 
        return job_bar_display[job_submit_state][0], job_bar_display[job_submit_state][1]
    
    return job_bar_display['READY'][0], job_bar_display['READY'][1]

@app.callback(
    *[Output(f'tooltip_{target}', component_property='style') for target in tool_tips.keys()],
    Input('tooltips_show', 'value'),)
def activate_tooltips(tooltips_show):
    """Activate or hide tooltips."""

    trigger = dash.callback_context.triggered
    trigger_id = trigger[0]['prop_id'].split('.')[0]

    if trigger_id == 'tooltips_show':
        if tooltips_show == 'off':
            return dict(display='none'), dict(display='none'), dict(display='none'), \
dict(display='none'), dict(display='none'), dict(display='none'), \
dict(display='none'), dict(display='none'), dict(display='none'), 

    return dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()


if __name__ == "__main__":
    app.run_server(debug=True)
