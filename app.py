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
from dash import dcc
import datetime
import json
import numpy as np
import os

import dimod
from dwave.cloud import Client
from dwave.embedding import embed_bqm, is_valid_embedding
from dwave.system import DWaveSampler
from MockKibbleZurekSampler import MockKibbleZurekSampler

from helpers.kz_calcs import *
from helpers.layouts_cards import *
from helpers.layouts_components import *
from helpers.plots import *
from helpers.qa import *
from helpers.tooltips import tool_tips_demo1, tool_tips_demo2

import yaml

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# global variable for a default J value
J_baseline = -1.8

# Initialize: available QPUs, initial progress-bar status
try:
    client = Client.from_config(client="qpu")
    qpus = {
        qpu.name: qpu
        for qpu in client.get_solvers(fast_anneal_time_range__covers=[0.005, 0.1])
    }
    if len(qpus) < 1:
        raise Exception
    init_job_status = "READY"
except Exception:
    qpus = {}
    client = None
    init_job_status = "NO SOLVER"
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    if config["ZNE"]:
        qpus["Diffusion [Classical]"] = globals()[config["sampler"]["type"]](
            topology_type=config["sampler"]["topology_type"],
            topology_shape=config["sampler"]["topology_shape"],
        )
        init_job_status = config["init_job_status"]
        if not client:
            client = config["client"]

tool_tips = tool_tips_demo1
def demo_layout(demo_type):

    if demo_type == "Kibble-Zurek":
        tool_tips = tool_tips_demo1
    else:
        tool_tips = tool_tips_demo2

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(  # Left: control panel
                        [
                            control_card(solvers=qpus, init_job_status=init_job_status, demo_type=demo_type),
                            *dbc_modal("modal_solver"),
                            *[
                                dbc.Tooltip(
                                    message,
                                    target=target,
                                    id=f"tooltip_{target}",
                                    style=dict(),
                                )
                                for target, message in tool_tips.items()
                            ],
                        ],
                        width=4,
                        style={"minWidth": "30rem"},
                    ),
                    dbc.Col(  # Right: display area
                        graphs_card(demo_type=demo_type),
                        width=8,
                        style={"minWidth": "60rem"},
                    ),
                ]
            ),
            # store coupling data points
            dcc.Store(id="coupling_data", data={}),
            # store zero noise extrapolation
            dcc.Store(id="zne_estimates", data={}),
            dcc.Store(id="modal_trigger", data=False),
            dcc.Store(id="initial_warning", data=False), 
            dcc.Store(id="kz_data", data={}),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Error")),
                    dbc.ModalBody(
                        "Fitting function failed likely due to ill conditioned data, please collect more."
                    ),
                ],
                id="error-modal",
                is_open=False,
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Warning", style={"color": "orange", "fontWeight": "bold"})),
                    dbc.ModalBody(
                        "The Classical [diffusion] option executes a Markov Chain method locally for purposes of testing the demo interface. Kinks diffuse to annihilate, but are also created/destroyed by thermal fluctuations.  The number of updates performed is set proportional to the annealing time. In the limit of no thermal noise, kinks diffuse to eliminate producing a power law, this process produces a power-law but for reasons independent of the Kibble-Zurek mechanism. In the noise mitigation demo we fit the impact of thermal fluctuations with a mixture of exponentials, by contrast with the quadratic fit appropriate to quantum dynamics.",
                        style={"color": "black", "fontSize": "16px"}, 
                    ),
                ],
                id="warning-modal",
                is_open=False,
            ),
        ],
        fluid=True,
    )

# Define the Navbar with two tabs
navbar = dbc.Navbar(
    dbc.Container(
        [
            # Navbar Brand/Logo
            dbc.NavbarBrand(
                [
                    html.Img(
                        src="assets/dwave_logo.png",
                        height="30px",
                        style={"margin-right": "10px"},
                    ),
                ],
                href="/demo1",  # Default route
            ),

            # Navbar Tabs
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Kibble-Zurek Mechanism", href="/demo1", active="exact")),
                    dbc.NavItem(dbc.NavLink("Kibble-Zurek Mechanism with Noise Mitigation", href="/demo2", active="exact")),
                ],
                pills=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    sticky="top",
)

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),  # Tracks the URL
        navbar,  # Includes the Navbar at the top
        html.Div(id="page-content", style={"paddingTop": "20px"}),  # Dynamic page content
    ],
    fluid=True,
)

server = app.server
app.config["suppress_callback_exceptions"] = True

# Callbacks Section

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    # If the user goes to the "/demo1" route
    if pathname == "/demo1":
       
        return demo_layout("Kibble-Zurek")
    # If the user goes to the "/demo2" route
    elif pathname == "/demo2":
       
        return demo_layout("Zero-Noise")
    else:
        return demo_layout("Kibble-Zurek")


@app.callback(
    Output("solver_modal", "is_open"),
    Input("btn_simulate", "n_clicks"),
)
def alert_no_solver(dummy):
    """Notify if no quantum computer is accessible."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "btn_simulate":
        if not client:
            return True

    return False


@app.callback(
    Output("anneal_duration", "disabled"),
    Output("coupling_strength", "disabled"),
    Output("spins", "options"),
    Output("qpu_selection", "disabled"),
    Input("job_submit_state", "children"),
    State("spins", "options"),
)
def disable_buttons(job_submit_state, spins_options):
    """Disable user input during job submissions."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id != "job_submit_state":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if job_submit_state in ["EMBEDDING", "SUBMITTED", "PENDING", "IN_PROGRESS"]:

        for inx, _ in enumerate(spins_options):

            spins_options[inx]["disabled"] = True

        return True, True, spins_options, True

    elif job_submit_state in ["COMPLETED", "CANCELLED", "FAILED"]:

        for inx, _ in enumerate(spins_options):
            spins_options[inx]["disabled"] = False

        return False, False, spins_options, False

    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("quench_schedule_filename", "children"),
    Output("quench_schedule_filename", "style"),
    Input("qpu_selection", "value"),
)
def set_schedule(qpu_name):
    """Set the schedule for the selected QPU."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    schedule_filename = "FALLBACK_SCHEDULE.csv"
    schedule_filename_style = {"color": "red", "fontSize": 12}

    if trigger_id == "qpu_selection":

        for filename in [
            file for file in os.listdir("helpers") if "schedule.csv" in file.lower()
        ]:

            if qpu_name.split(".")[0] in filename:  # Accepts & reddens older versions

                schedule_filename = filename

                if qpu_name in filename:

                    schedule_filename_style = {"color": "white", "fontSize": 12}

    return schedule_filename, schedule_filename_style


@app.callback(
    Output("embeddings_cached", "data"),
    Output("embedding_is_cached", "value"),
    Input("qpu_selection", "value"),
    Input("embeddings_found", "data"),
    State("embeddings_cached", "data"),
    State("spins", "value"),
)
def cache_embeddings(qpu_name, embeddings_found, embeddings_cached, spins):
    """Cache embeddings for the selected QPU."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "qpu_selection":

        embeddings_cached = {}  # Wipe out previous QPU's embeddings

        for filename in [
            file for file in os.listdir("helpers") if ".json" in file and "emb_" in file
        ]:

            if qpu_name == "Diffusion [Classical]":
                _qpu_name = "Advantage_system6.4"
            else:
                _qpu_name = qpu_name

            # splitting seemed unsafe since the graph can change between versions
            if _qpu_name in filename:

                with open(f"helpers/{filename}", "r") as fp:
                    embeddings_cached = json.load(fp)

                embeddings_cached = json_to_dict(embeddings_cached)

                # Validate that loaded embeddings' edges are still available on the selected QPU
                for length in list(embeddings_cached.keys()):

                    source_graph = dimod.to_networkx_graph(
                        create_bqm(num_spins=length)
                    ).edges
                    target_graph = qpus[_qpu_name].edges
                    emb = embeddings_cached[length]

                    if not is_valid_embedding(emb, source_graph, target_graph):

                        del embeddings_cached[length]
    if trigger_id == "embeddings_found":

        if not isinstance(
            embeddings_found, str
        ):  # embeddings_found != 'needed' or 'not found'

            embeddings_cached = json_to_dict(embeddings_cached)
            embeddings_found = json_to_dict(embeddings_found)
            new_embedding = list(embeddings_found.keys())[0]
            embeddings_cached[new_embedding] = embeddings_found[new_embedding]

        else:
            return dash.no_update, dash.no_update

    return embeddings_cached, list(embeddings_cached.keys())


@app.callback(
    Output("sample_vs_theory", "figure"),
    Output("coupling_data", "data"),  # store data using dcc
    Output("zne_estimates", "data"),  # update zne_estimates
    Output("modal_trigger", "data"),
    Output("kz_data", "data"),
    Input("qpu_selection", "value"),
    #Input("zne_graph_display", "value"),
    Input("graph_display", "value"),
    Input("coupling_strength", "value"),  # previously input
    Input("quench_schedule_filename", "children"),
    Input("job_submit_state", "children"),
    Input("job_id", "children"),
    #Input("anneal_duration_zne", "value"),
    Input("anneal_duration", "value"),
    Input("spins", "value"),
    Input("url", "pathname"),
    State("embeddings_cached", "data"),
    State("sample_vs_theory", "figure"),
    State("coupling_data", "data"),  # access previously stored data
    State("zne_estimates", "data"),  # Access ZNE estimates
    State("kz_data", "data") # get kibble zurek data point
)
def display_graphics_kink_density(
    qpu_name,
    graph_display,
    J,
    schedule_filename,
    job_submit_state,
    job_id,
    ta,
    spins,
    pathname,
    embeddings_cached,
    figure,
    coupling_data,
    zne_estimates,
    kz_data,
):
    """Generate graphics for kink density based on theory and QPU samples."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    ta_min = 2
    ta_max = 350

    if pathname == "/demo2":

        # update the maximum anneal time for zne demo
        ta_max = 1500

        if (
            trigger_id == "qpu_selection" or trigger_id == "spins"
        ):
            coupling_data = {}
            zne_estimates = {}
            fig = plot_kink_densities_bg(
                graph_display,
                [ta_min, ta_max],
                J_baseline,
                schedule_filename,
                coupling_data,
                zne_estimates,
                url="Demo2",
            )

            return fig, coupling_data, zne_estimates, False, kz_data

        if trigger_id in [
            "zne_graph_display",
            "coupling_strength",
            "quench_schedule_filename",
        ]:

            fig = plot_kink_densities_bg(
                graph_display,
                [ta_min, ta_max],
                J_baseline,
                schedule_filename,
                coupling_data,
                zne_estimates,
                url="Demo2",
            )

            return fig, coupling_data, zne_estimates, False, kz_data

        if trigger_id == "job_submit_state":

            if job_submit_state == "COMPLETED":

                embeddings_cached = embeddings_cached = json_to_dict(embeddings_cached)

                sampleset_unembedded = get_samples(
                    client, job_id, spins, J, embeddings_cached[spins]
                )
                _, kink_density = kink_stats(sampleset_unembedded, J)

                 # Calculate kappa
                kappa = calc_kappa(J, J_baseline)

                fig = plot_kink_density(graph_display, figure, kink_density, ta, J)

                # Initialize the list for this anneal_time if not present
                ta_str = str(ta)
                if ta_str not in coupling_data:
                    coupling_data[ta_str] = []
                # Append the new data point
                coupling_data[ta_str].append(
                    {"kappa": kappa, "kink_density": kink_density, "coupling_strength": J}
                )

                zne_estimates, modal_trigger = plot_zne_fitted_line(
                    fig, coupling_data, qpu_name, zne_estimates, graph_display, ta_str
                )

                if graph_display == "kink_density":
                    fig = plot_kink_densities_bg(
                        graph_display,
                        [ta_min, ta_max],
                        J_baseline,
                        schedule_filename,
                        coupling_data,
                        zne_estimates,
                        url='Demo2'
                    )

                return fig, coupling_data, zne_estimates, modal_trigger, kz_data

            else:
                return dash.no_update

            # use global J value
        fig = plot_kink_densities_bg(
            graph_display,
            [ta_min, ta_max],
            J_baseline,
            schedule_filename,
            coupling_data,
            zne_estimates,
            url='Demo2'
        )
        return fig, coupling_data, zne_estimates, False, kz_data
    else:
        if trigger_id == "qpu_selection" or trigger_id == "spins" or trigger_id == "coupling_strength":
            
            kz_data = {"k":[]}
            fig = plot_kink_densities_bg(graph_display, [ta_min, ta_max], J, schedule_filename, coupling_data, zne_estimates, kz_data=kz_data, url="Demo1")

            return fig, coupling_data, zne_estimates, False, kz_data
        
        if trigger_id == 'job_submit_state':

            if job_submit_state == 'COMPLETED':

                embeddings_cached = embeddings_cached = json_to_dict(embeddings_cached)

                sampleset_unembedded = get_samples(client, job_id, spins, J, embeddings_cached[spins])              
                _, kink_density = kink_stats(sampleset_unembedded, J)
                
                
                # Append the new data point
                kz_data["k"].append(
                    (kink_density, ta)
                )
                fig = plot_kink_density(graph_display, figure, kink_density, ta, J, url="Demo1")
                return fig, coupling_data, zne_estimates, False, kz_data
            
            else:
                return dash.no_update
            
        fig = plot_kink_densities_bg(graph_display, [ta_min, ta_max], J, schedule_filename, coupling_data, zne_estimates, kz_data, url="Demo1")
        return fig, coupling_data, zne_estimates, False, kz_data

@app.callback(
    Output("spin_orientation", "figure"),
    Input("spins", "value"),
    Input("job_submit_state", "children"),
    State("job_id", "children"),
    State("coupling_strength", "value"),
    State("embeddings_cached", "data"),
)
def display_graphics_spin_ring(spins, job_submit_state, job_id, J, embeddings_cached):
    """Generate graphics for spin-ring display."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "job_submit_state":

        if job_submit_state == "COMPLETED":

            embeddings_cached = embeddings_cached = json_to_dict(embeddings_cached)
            sampleset_unembedded = get_samples(
                client, job_id, spins, J, embeddings_cached[spins]
            )
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
    Output("job_id", "children"),
    Output("initial_warning", "data"),
    Output("warning-modal", "is_open"),
    Input("job_submit_time", "children"),
    State("qpu_selection", "value"),
    State("spins", "value"),
    State("coupling_strength", "value"),
    State("anneal_duration", "value"),
    State("embeddings_cached", "data"),
    State("url", "pathname"),
    State("quench_schedule_filename", "children"),
    State("initial_warning", "data")
)
def submit_job(job_submit_time, qpu_name, spins, J, ta_ns, embeddings_cached, pathname, filename, initial_warning):

    """Submit job and provide job ID."""
    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "job_submit_time":

        solver = qpus[qpu_name]

        bqm = create_bqm(num_spins=spins, coupling_strength=J)

        embeddings_cached = json_to_dict(embeddings_cached)
        embedding = embeddings_cached[spins]
        annealing_time = (ta_ns / 1000)

        if qpu_name == "Diffusion [Classical]":

            bqm_embedded = embed_bqm(
                bqm,
                embedding,
                qpus["Diffusion [Classical]"].adjacency,
            )

            sampleset = qpus["Diffusion [Classical]"].sample(
                bqm_embedded, annealing_time=annealing_time
            )
            if not initial_warning:
                return json.dumps(sampleset.to_serializable()), True, True
            return json.dumps(sampleset.to_serializable()), True, False

        else:

            bqm_embedded = embed_bqm(
                bqm, embedding, DWaveSampler(solver=solver.name).adjacency
            )
            # ta_multiplier should be 1, unless (withNoiseMitigation and [J or schedule]) changes, shouldn't change for MockSampler. In which case recalculate as ta_multiplier=calc_lambda(coupling_strength, schedule, J_baseline=-1.8) as a function of the correct schedule
            # State("ta_multiplier", "value") ? Should recalculate when J or schedule changes IFF noise mitigation tab?
            ta_multiplier = 1

            if pathname == "/demo2":
                ta_multiplier = calc_lambda(J, schedule_name=filename, J_baseline=J_baseline)

            print(f'{ta_multiplier}: qpu_name')

            computation = solver.sample_bqm(
                bqm=bqm_embedded,
                fast_anneal=True,
                annealing_time=annealing_time*ta_multiplier,
                auto_scale=False,
                answer_mode="raw",  # Easier than accounting for num_occurrences
                num_reads=100,
                label=f"Examples - Kibble-Zurek Simulation, submitted: {job_submit_time}",
            )

        return computation.wait_id(), False, False

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
    State("embedding_is_cached", "value"),
    State("spins", "value"),
    State("qpu_selection", "value"),
    State("embeddings_found", "data"),
)
def simulate(
    dummy1,
    dummy2,
    job_id,
    job_submit_state,
    job_submit_time,
    cached_embedding_lengths,
    spins,
    qpu_name,
    embeddings_found,
):
    """Manage simulation: embedding, job submission."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if not any(trigger_id == input for input in ["btn_simulate", "wd_job"]):
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    if trigger_id == "btn_simulate":

        if spins in cached_embedding_lengths or qpu_name == "Diffusion [Classical]":

            submit_time = datetime.datetime.now().strftime("%c")
            if qpu_name == "Diffusion [Classical]":  # Hack to fix switch from SA to QPU
                submit_time = "SA"
            job_submit_state = "SUBMITTED"
            embedding = dash.no_update

        else:

            submit_time = dash.no_update
            job_submit_state = "EMBEDDING"
            embedding = "needed"

        disable_btn = True
        disable_watchdog = False

        return (
            disable_btn,
            disable_watchdog,
            0.5 * 1000,
            0,
            job_submit_state,
            submit_time,
            embedding,
        )

    if job_submit_state == "EMBEDDING":

        submit_time = dash.no_update
        embedding = dash.no_update

        if embeddings_found == "needed":

            try:
                embedding = find_one_to_one_embedding(spins, qpus[qpu_name].edges)
                if embedding:
                    job_submit_state = (
                        "EMBEDDING"  # Stay another WD to allow caching the embedding
                    )
                    embedding = {spins: embedding}
                else:
                    job_submit_state = "FAILED"
                    embedding = "not found"
            except Exception:
                job_submit_state = "FAILED"
                embedding = "not found"

        else:  # Found embedding last WD, so is cached, so now can submit job

            submit_time = datetime.datetime.now().strftime("%c")
            job_submit_state = "SUBMITTED"

        return True, False, 0.2 * 1000, 0, job_submit_state, submit_time, embedding

    if any(
        job_submit_state == status for status in ["SUBMITTED", "PENDING", "IN_PROGRESS"]
    ):

        job_submit_state = get_job_status(client, job_id, job_submit_time)
        if not job_submit_state:
            job_submit_state = "SUBMITTED"
            wd_time = 0.2 * 1000
        else:
            wd_time = 1 * 1000

        return True, False, wd_time, 0, job_submit_state, dash.no_update, dash.no_update

    if any(
        job_submit_state == status for status in ["COMPLETED", "CANCELLED", "FAILED"]
    ):

        disable_btn = False
        disable_watchdog = True

        return (
            disable_btn,
            disable_watchdog,
            0.1 * 1000,
            0,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    else:  # Exception state: should only ever happen in testing
        return False, True, 0, 0, "ERROR", dash.no_update, dash.no_update


@app.callback(
    Output("bar_job_status", "value"),
    Output("bar_job_status", "color"),
    Input("job_submit_state", "children"),
)
def set_progress_bar(job_submit_state):
    """Update progress bar for job submissions."""

    trigger_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "job_submit_state":

        return (
            job_bar_display[job_submit_state][0],
            job_bar_display[job_submit_state][1],
        )

    return job_bar_display["READY"][0], job_bar_display["READY"][1]


@app.callback(
    *[
        Output(f"tooltip_{target}", component_property="style")
        for target in tool_tips.keys()
    ],
    Input("tooltips_show", "value"),
)
def activate_tooltips(tooltips_show):
    """Activate or hide tooltips."""

    trigger = dash.callback_context.triggered
    trigger_id = trigger[0]["prop_id"].split(".")[0]
        
    if trigger_id == "tooltips_show":
        if tooltips_show == "off":
            return (
                dict(display="none"),
                dict(display="none"),
                dict(display="none"),
                dict(display="none"),
                dict(display="none"),
                dict(display="none"),
                dict(display="none"),
                dict(display="none"),
                dict(display="none"),
            )

    return (
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
        dict(),
    )


@app.callback(
    Output("error-modal", "is_open"),
    Input("modal_trigger", "data"),
    State("error-modal", "is_open"),
)
def toggle_modal(trigger, is_open):
    if trigger:
        return True
    return is_open

if __name__ == "__main__":
    app.run_server(debug=True)
