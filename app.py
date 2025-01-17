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

import datetime
import json
import os
from typing import NamedTuple, Union

import dash
import dash_bootstrap_components as dbc
import dimod
import numpy as np
from dash import ALL, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dwave.cloud import Client
from dwave.embedding import embed_bqm, is_valid_embedding
from dwave.system import DWaveSampler

from demo_configs import (
    DESCRIPTION,
    DESCRIPTION_NM,
    J_BASELINE,
    MAIN_HEADER,
    MAIN_HEADER_NM,
    THUMBNAIL,
    USE_CLASSICAL,
)
from helpers.kz_calcs import *
from helpers.layouts_cards import *
from helpers.layouts_components import *
from helpers.plots import *
from helpers.qa import *
from helpers.tooltips import tool_tips_demo1, tool_tips_demo2
from mock_kz_sampler import MockKibbleZurekSampler
from src.demo_enums import ProblemType

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Initialize: available QPUs, initial progress-bar status
try:
    client = Client.from_config(client="qpu")
    qpus = {
        qpu.name: qpu for qpu in client.get_solvers(fast_anneal_time_range__covers=[0.005, 0.1])
    }
    if len(qpus) < 1:
        raise Exception
    init_job_status = "READY"
except Exception:
    qpus = {}
    client = None
    init_job_status = "NO SOLVER"

# Load base coupling strength and user configuration for mock sampler
if USE_CLASSICAL:
    qpus["Diffusion [Classical]"] = MockKibbleZurekSampler(
        topology_type="pegasus", topology_shape=[16]
    )

init_job_status = "READY"

if not client:
    client = "dummy"


# Define the Navbar with two tabs
navbar = dbc.Navbar(
    dbc.Container(
        [
            # Navbar Brand/Logo
            dbc.NavbarBrand(
                [
                    html.Img(
                        src=THUMBNAIL,
                        height="30px",
                        style={"marginRight": "10px"},
                    ),
                ],
            ),
            # Navbar Tabs
            dbc.Nav(
                [
                    dbc.NavItem(
                        dbc.NavLink(
                            problem_type.label,
                            id={"type": "problem-type", "index": index},
                            active="exact",
                        )
                    )
                    for index, problem_type in enumerate(ProblemType)
                ],
                pills=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    sticky="top",
)


def tooltips(problem_type: Union[ProblemType, int]) -> list[dbc.Tooltip]:
    """Tooltip generator.

    Args:
        problem_type: Either ProblemType.KZ or ProblemType.KZ_NM.
    """
    tool_tips = tool_tips_demo1 if problem_type is ProblemType.KZ else tool_tips_demo2

    return [
        dbc.Tooltip(
            message,
            target=target,
            id=f"tooltip_{target}",
            style=dict(),
        )
        for target, message in tool_tips.items()
    ]


app.layout = html.Div(
    [
        dcc.Store(id="coupling_data", data={}),
        # store zero noise extrapolation
        dcc.Store(id="zne_estimates", data={}),
        dcc.Store(id="modal_trigger", data=False),
        dcc.Store(id="initial_warning", data=False),
        dcc.Store(id="kz_data", data={}),
        dcc.Store(id="selected-problem"),
        dcc.Store(id="job_submit_time"),
        dcc.Store(id="job_id"),
        dcc.Store(id="embeddings_cached", data={}),
        dcc.Interval(
            id="wd_job",
            interval=500,
            n_intervals=0,
            disabled=True,
            max_intervals=1,
        ),
        navbar,  # Includes the Navbar at the top
        html.Div(
            [
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(  # Left: control panel
                                    [
                                        control_card(
                                            solvers=qpus,
                                            init_job_status=init_job_status,
                                        ),
                                        *dbc_modal("modal_solver"),
                                        html.Div(tooltips(ProblemType.KZ), id="tooltips"),
                                    ],
                                    width=4,
                                    style={"minWidth": "30rem"},
                                ),
                                dbc.Col(  # Right: display area
                                    graphs_card(problem_type=ProblemType.KZ),
                                    width=8,
                                    style={"minWidth": "60rem"},
                                ),
                            ]
                        ),
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
                                dbc.ModalHeader(
                                    dbc.ModalTitle(
                                        "Warning", style={"color": "orange", "fontWeight": "bold"}
                                    )
                                ),
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
            ],
            style={"paddingTop": "20px"},
        ),
    ],
)

server = app.server
app.config["suppress_callback_exceptions"] = True


@dash.callback(
    Output({"type": "problem-type", "index": ALL}, "className"),
    Output("selected-problem", "data"),
    Output("graph-radio-options", "children"),
    Output("tooltips", "children"),
    Output("anneal-duration-dropdown", "children"),
    Output("coupling-strength-slider", "children"),
    Output("main-header", "children"),
    Output("main-description", "children"),
    inputs=[
        Input({"type": "problem-type", "index": ALL}, "n_clicks"),
        State("selected-problem", "data"),
    ],
)
def update_selected_problem_type(
    problem_options: list[int],
    selected_problem: Union[ProblemType, int],
) -> tuple[str, int, list, list]:
    """Updates the problem that is selected (KZ or KZ_NM), hides/shows settings accordingly,
        and updates the navigation options to indicate the currently active problem option.

    Args:
        problem_options: A list containing the number of times each problem option has been clicked.
        selected_problem: The currently selected problem.

    Returns:
        problem-type-class (list): A list of classes for the header problem navigation options.
        selected-period (int): Either KZ (``0`` or ``ProblemType.KZ``) or
            KZ_NM (``1`` or ``ProblemType.KZ_NM``).
        graph-radio-options: The radio options for the graph.
        tooltips: The tooltips for the settings form.
        anneal-duration-dropdown: The duration dropdown setting.
        coupling-strength-slider: The coupling strength slider setting.
        main-header: The main header of the problem in the left column.
        main-description: The description of the problem in the left column.
    """
    if ctx.triggered_id and selected_problem == ctx.triggered_id["index"]:
        raise PreventUpdate

    nav_class_names = [""] * len(problem_options)
    problem_type_value = ctx.triggered_id["index"] if ctx.triggered_id else ProblemType.KZ.value
    problem_type = ProblemType(problem_type_value)

    nav_class_names[problem_type_value] = "active"

    return (
        nav_class_names,
        problem_type_value,
        get_kz_graph_radio_options(problem_type),
        tooltips(problem_type),
        get_anneal_duration_setting(problem_type),
        get_coupling_strength_slider(problem_type),
        MAIN_HEADER if problem_type is ProblemType.KZ else MAIN_HEADER_NM,
        DESCRIPTION if problem_type is ProblemType.KZ else DESCRIPTION_NM,
    )


@app.callback(
    Output("solver_modal", "is_open"),
    Input("btn_simulate", "n_clicks"),
)
def alert_no_solver(dummy):
    """Notify if no quantum computer is accessible."""

    return ctx.triggered_id == "btn_simulate" and not client


@app.callback(
    Output("anneal_duration", "disabled"),
    Output("coupling_strength", "disabled"),
    Output("spins", "options"),
    Output("qpu_selection", "disabled"),
    inputs=[
        Input("job_submit_state", "children"),
        State("spins", "options"),
    ],
    prevent_initial_call=True,
)
def disable_buttons(job_submit_state, spins_options):
    """Disable user input during job submissions."""
    running_states = ["EMBEDDING", "SUBMITTED", "PENDING", "IN_PROGRESS"]
    done_states = ["COMPLETED", "CANCELLED", "FAILED"]
    is_running = job_submit_state in running_states

    if job_submit_state not in running_states + done_states:
        raise PreventUpdate

    for inx, _ in enumerate(spins_options):
        spins_options[inx]["disabled"] = is_running

    return is_running, is_running, spins_options, is_running


@app.callback(
    Output("quench_schedule_filename", "children"),
    Output("quench_schedule_filename", "style"),
    Input("qpu_selection", "value"),
)
def set_schedule(qpu_name):
    """Set the schedule for the selected QPU."""

    schedule_filename = "FALLBACK_SCHEDULE.csv"
    schedule_filename_style = {"color": "#FFA143", "fontSize": 12}

    if ctx.triggered_id:
        for filename in [file for file in os.listdir("helpers") if "schedule.csv" in file.lower()]:

            if qpu_name.split(".")[0] in filename:  # Accepts & reddens older versions
                schedule_filename = filename

                if qpu_name in filename:
                    schedule_filename_style = {"color": "white", "fontSize": 12}

    return schedule_filename, schedule_filename_style


@app.callback(
    Output("embeddings_cached", "data"),
    Output("embedding_is_cached", "children"),
    inputs=[
        Input("qpu_selection", "value"),
    ],
    prevent_initial_call=True,
)
def load_cached_embeddings(qpu_name):
    """Cache embeddings for the selected QPU."""

    embeddings_cached = {}  # Wipe out previous QPU's embeddings

    for filename in [file for file in os.listdir("helpers") if ".json" in file and "emb_" in file]:

        if qpu_name == "Diffusion [Classical]":
            qpu_name = "Advantage_system6.4"

        if qpu_name.split(".")[0] in filename:
            with open(f"helpers/{filename}", "r") as fp:
                embeddings_cached = json.load(fp)

            embeddings_cached = json_to_dict(embeddings_cached)

            # Validate that loaded embeddings' edges are still available on the selected QPU
            for length in list(embeddings_cached.keys()):
                source_graph = dimod.to_networkx_graph(create_bqm(num_spins=length)).edges
                target_graph = qpus[qpu_name].edges
                emb = embeddings_cached[length]

                if not is_valid_embedding(emb, source_graph, target_graph):
                    del embeddings_cached[length]

    return embeddings_cached, ", ".join(str(embedding) for embedding in embeddings_cached.keys())


@app.callback(
    Output("sample_vs_theory", "figure"),
    Output("coupling_data", "data"),  # store data using dcc
    Output("zne_estimates", "data"),  # update zne_estimates
    Output("modal_trigger", "data"),
    Output("kz_data", "data"),
    inputs=[
        Input("qpu_selection", "value"),
        Input("graph_display", "value"),
        Input("coupling_strength", "value"),  # previously input
        Input("quench_schedule_filename", "children"),
        Input("job_submit_state", "children"),
        Input("job_id", "data"),
        Input("anneal_duration", "value"),
        Input("spins", "value"),
        Input("selected-problem", "data"),
        State("embeddings_cached", "data"),
        State("sample_vs_theory", "figure"),
        State("coupling_data", "data"),  # access previously stored data
        State("zne_estimates", "data"),  # Access ZNE estimates
        State("kz_data", "data"),  # get kibble zurek data point
    ],
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
    problem_type,
    embeddings_cached,
    figure,
    coupling_data,
    zne_estimates,
    kz_data,
):
    """Generate graphics for kink density based on theory and QPU samples."""
    if ctx.triggered_id == "job_submit_state" and job_submit_state != "COMPLETED":
        raise PreventUpdate

    ta_min = 2
    ta_max = 350
    problem_type = ProblemType(problem_type)

    if problem_type is ProblemType.KZ_NM:
        # update the maximum anneal time for zne demo
        ta_max = 1500

        if ctx.triggered_id == "job_submit_state":
            embeddings_cached = json_to_dict(embeddings_cached)

            sampleset_unembedded = get_samples(client, job_id, spins, J, embeddings_cached[spins])
            _, kink_density = kink_stats(sampleset_unembedded, J)

            # Calculate lambda (previously kappa)
            # Added _ to avoid keyword restriction
            lambda_ = calclambda_(
                J=J, qpu_name=qpu_name, schedule_name=schedule_filename, J_baseline=J_BASELINE
            )

            fig = plot_kink_density(graph_display, figure, kink_density, ta, J, lambda_)

            # Initialize the list for this anneal_time if not present
            ta_str = str(ta)
            if ta_str not in coupling_data:
                coupling_data[ta_str] = []
            # Append the new data point
            coupling_data[ta_str].append(
                {
                    "lambda": lambda_,
                    "kink_density": kink_density,
                    "coupling_strength": J,
                }
            )

            zne_estimates, modal_trigger = plot_zne_fitted_line(
                fig, coupling_data, qpu_name, zne_estimates, graph_display, ta_str
            )

            if graph_display == "kink_density":
                fig = plot_kink_densities_bg(
                    graph_display,
                    [ta_min, ta_max],
                    J_BASELINE,
                    schedule_filename,
                    coupling_data,
                    zne_estimates,
                    problem_type=problem_type,
                )

            return fig, coupling_data, zne_estimates, modal_trigger, kz_data

        if ctx.triggered_id == "qpu_selection" or ctx.triggered_id == "spins":
            coupling_data = {}
            zne_estimates = {}

        fig = plot_kink_densities_bg(
            graph_display,
            [ta_min, ta_max],
            J_BASELINE,
            schedule_filename,
            coupling_data,
            zne_estimates,
            problem_type=problem_type,
        )

        if (
            ctx.triggered_id
            in ["zne_graph_display", "coupling_strength", "quench_schedule_filename"]
            and graph_display == "coupling"
        ):
            zne_estimates, modal_trigger = plot_zne_fitted_line(
                fig, coupling_data, qpu_name, zne_estimates, graph_display, str(ta)
            )

        return fig, coupling_data, zne_estimates, False, kz_data

    if ctx.triggered_id == "job_submit_state":
        embeddings_cached = json_to_dict(embeddings_cached)

        sampleset_unembedded = get_samples(client, job_id, spins, J, embeddings_cached[spins])
        _, kink_density = kink_stats(sampleset_unembedded, J)

        # Append the new data point
        kz_data["k"].append((kink_density, ta))
        fig = plot_kink_density(
            graph_display, figure, kink_density, ta, J, problem_type=problem_type
        )
        return fig, coupling_data, zne_estimates, False, kz_data

    if ctx.triggered_id in ["qpu_selection", "spins", "coupling_strength"]:
        kz_data = {"k": []}

    fig = plot_kink_densities_bg(
        graph_display,
        [ta_min, ta_max],
        J,
        schedule_filename,
        coupling_data,
        zne_estimates,
        kz_data,
        problem_type=problem_type,
    )
    return fig, coupling_data, zne_estimates, False, kz_data


@app.callback(
    Output("spin_orientation", "figure"),
    inputs=[
        Input("spins", "value"),
        Input("job_submit_state", "children"),
        State("job_id", "data"),
        State("coupling_strength", "value"),
        State("embeddings_cached", "data"),
    ],
)
def display_graphics_spin_ring(spins, job_submit_state, job_id, J, embeddings_cached):
    """Generate graphics for spin-ring display."""
    best_sample = None

    if ctx.triggered_id == "job_submit_state":
        if job_submit_state != "COMPLETED":
            raise PreventUpdate

        embeddings_cached = json_to_dict(embeddings_cached)
        sampleset_unembedded = get_samples(client, job_id, spins, J, embeddings_cached[spins])
        kinks_per_sample, kink_density = kink_stats(sampleset_unembedded, J)
        best_indx = np.abs(kinks_per_sample - kink_density).argmin()
        best_sample = sampleset_unembedded.record.sample[best_indx]

    fig = plot_spin_orientation(num_spins=spins, sample=best_sample)
    return fig


class SubmitJobReturn(NamedTuple):
    """Return type for the ``submit_job`` callback function."""

    job_id: str = dash.no_update
    initial_warning: bool = False
    warning_modal_open: bool = False
    wd_job_n_intervals: int = 0


@app.callback(
    Output("job_id", "data"),
    Output("initial_warning", "data"),
    Output("warning-modal", "is_open"),
    Output("wd_job", "n_intervals"),
    inputs=[
        Input("job_submit_time", "data"),
        State("qpu_selection", "value"),
        State("spins", "value"),
        State("coupling_strength", "value"),
        State("anneal_duration", "value"),
        State("embeddings_cached", "data"),
        State("selected-problem", "data"),
        State("quench_schedule_filename", "children"),
        State("initial_warning", "data"),
    ],
    prevent_initial_call=True,
)
def submit_job(
    job_submit_time,
    qpu_name,
    spins,
    J,
    ta_ns,
    embeddings_cached,
    problem_type,
    filename,
    initial_warning,
) -> SubmitJobReturn:
    """Submit job and provide job ID."""

    solver = qpus[qpu_name]

    bqm = create_bqm(num_spins=spins, coupling_strength=J)

    embeddings_cached = json_to_dict(embeddings_cached)
    embedding = embeddings_cached[spins]
    annealing_time = ta_ns / 1000

    if qpu_name == "Diffusion [Classical]":
        bqm_embedded = embed_bqm(bqm, embedding, qpus["Diffusion [Classical]"].adjacency)

        sampleset = qpus["Diffusion [Classical]"].sample(
            bqm_embedded, annealing_time=annealing_time
        )

        return SubmitJobReturn(
            job_id=json.dumps(sampleset.to_serializable()),
            initial_warning=True,
            warning_modal_open=not initial_warning,
        )

    bqm_embedded = embed_bqm(bqm, embedding, DWaveSampler(solver=solver.name).adjacency)

    # ta_multiplier should be 1, unless (withNoiseMitigation and [J or schedule]) changes,
    # shouldn't change for MockSampler. In which case recalculate as
    # ta_multiplier=calclambda_(coupling_strength, schedule, J_baseline=-1.8) as a function of the
    # correct schedule
    # State("ta_multiplier", "value") ? Should recalculate when J or schedule changes IFF noise mitigation tab?
    ta_multiplier = 1

    if problem_type is ProblemType.KZ_NM.value:
        ta_multiplier = calclambda_(J, schedule_name=filename, J_baseline=J_BASELINE)

    computation = solver.sample_bqm(
        bqm=bqm_embedded,
        fast_anneal=True,
        annealing_time=annealing_time * ta_multiplier,
        auto_scale=False,
        answer_mode="raw",  # Easier than accounting for num_occurrences
        num_reads=100,
        label=f"Examples - Kibble-Zurek Simulation, submitted: {job_submit_time}",
    )

    return SubmitJobReturn(job_id=computation.wait_id())


class RunButtonClickReturn(NamedTuple):
    """Return type for the ``run_button_click`` callback function."""

    btn_simulate_disabled: bool = True
    wd_job_disabled: bool = False
    wd_job_n_intervals: int = 0
    job_submit_state: str = dash.no_update
    job_submit_time: str = dash.no_update


@app.callback(
    Output("btn_simulate", "disabled"),
    Output("wd_job", "disabled"),
    Output("wd_job", "n_intervals", allow_duplicate=True),
    Output("job_submit_state", "children"),
    Output("job_submit_time", "data"),
    inputs=[
        Input("btn_simulate", "n_clicks"),
        State("embedding_is_cached", "children"),
        State("spins", "value"),
        State("qpu_selection", "value"),
    ],
    prevent_initial_call=True,
)
def run_button_click(
    run_btn_click,
    cached_embeddings,
    spins,
    qpu_name,
) -> RunButtonClickReturn:
    """Manage simulation: embedding, job submission."""
    if qpu_name == "Diffusion [Classical]":
        return RunButtonClickReturn(
            job_submit_state="SUBMITTED",
            job_submit_time="SA",  # Hack to fix switch from SA to QPU
        )

    if str(spins) in cached_embeddings.split(", "):  # If we have a cached embedding
        return RunButtonClickReturn(
            job_submit_state="SUBMITTED",
            job_submit_time=datetime.datetime.now().strftime("%c"),
        )

    return RunButtonClickReturn(job_submit_state="EMBEDDING")


class SimulateReturn(NamedTuple):
    """Return type for the ``simulate`` callback function."""

    btn_simulate_disabled: bool = dash.no_update
    wd_job_disabled: bool = dash.no_update
    wd_job_interval: int = dash.no_update
    wd_job_n_intervals: int = dash.no_update
    job_submit_state: str = dash.no_update
    job_submit_time: str = dash.no_update
    embeddings_cached: dict = dash.no_update
    embedding_is_cached: str = dash.no_update


@app.callback(
    Output("btn_simulate", "disabled", allow_duplicate=True),
    Output("wd_job", "disabled", allow_duplicate=True),
    Output("wd_job", "interval"),
    Output("wd_job", "n_intervals", allow_duplicate=True),
    Output("job_submit_state", "children", allow_duplicate=True),
    Output("job_submit_time", "data", allow_duplicate=True),
    Output("embeddings_cached", "data", allow_duplicate=True),
    Output("embedding_is_cached", "children", allow_duplicate=True),
    inputs=[
        Input("wd_job", "n_intervals"),
        State("job_id", "data"),
        State("job_submit_state", "children"),
        State("job_submit_time", "data"),
        State("spins", "value"),
        State("qpu_selection", "value"),
        State("embeddings_cached", "data"),
    ],
    prevent_initial_call=True,
)
def simulate(
    interval,
    job_id,
    job_submit_state,
    job_submit_time,
    spins,
    qpu_name,
    embeddings_cached,
) -> SimulateReturn:
    """Manage simulation: embedding, job submission."""

    if job_submit_state == "EMBEDDING":

        try:
            embedding = find_one_to_one_embedding(spins, qpus[qpu_name].edges)
            if embedding:
                embeddings_cached = json_to_dict(embeddings_cached)
                embeddings_cached.update({spins: embedding})

                return SimulateReturn(
                    wd_job_interval=200,
                    job_submit_state="SUBMITTED",
                    job_submit_time=datetime.datetime.now().strftime("%c"),
                    embeddings_cached=embeddings_cached,
                    embedding_is_cached=", ".join(str(em) for em in embeddings_cached.keys()),
                )

            return SimulateReturn(
                btn_simulate_disabled=False,
                wd_job_disabled=True,
                job_submit_state="FAILED",
            )
        except Exception:
            return SimulateReturn(
                btn_simulate_disabled=False,
                wd_job_disabled=True,
                job_submit_state="FAILED",
            )

    if job_submit_state in ["SUBMITTED", "PENDING", "IN_PROGRESS"]:
        job_submit_state = get_job_status(client, job_id, job_submit_time)
        wd_time = 1000

        if not job_submit_state:
            job_submit_state = "SUBMITTED"
            wd_time = 200

        return SimulateReturn(
            wd_job_interval=wd_time,
            wd_job_n_intervals=0,
            job_submit_state=job_submit_state,
        )

    return SimulateReturn(
        btn_simulate_disabled=False,
        wd_job_disabled=True,
        job_submit_state=(
            dash.no_update if job_submit_state in ["COMPLETED", "CANCELLED", "FAILED"] else "ERROR"
        ),
    )


@app.callback(
    Output("bar_job_status", "value"),
    Output("bar_job_status", "color"),
    Input("job_submit_state", "children"),
)
def set_progress_bar(job_submit_state):
    """Update progress bar for job submissions."""

    if ctx.triggered_id:
        return job_bar_display[job_submit_state][0], job_bar_display[job_submit_state][1]

    return job_bar_display["READY"][0], job_bar_display["READY"][1]


@app.callback(
    Output("error-modal", "is_open"),
    Input("modal_trigger", "data"),
    State("error-modal", "is_open"),
)
def toggle_modal(trigger, is_open):
    return True if trigger else is_open


if __name__ == "__main__":
    app.run_server(debug=True)
