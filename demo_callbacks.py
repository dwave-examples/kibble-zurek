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
from dash import MATCH
from demo_interface import CLIENT, SOLVERS, generate_tooltips, get_quench_duration_setting, get_slider_marks
import dimod
import numpy as np
from dash import ALL, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
from dwave.embedding import embed_bqm, is_valid_embedding
from dwave.system import DWaveSampler

from demo_configs import (
    DESCRIPTION,
    DESCRIPTION_NM,
    J_BASELINE,
    JOB_BAR_DISPLAY,
    MAIN_HEADER,
    MAIN_HEADER_NM,
    RING_LENGTHS,
)
from src.kz_calcs import *
from src.plots import *
from src.qa import *
from src.demo_enums import ProblemType


@dash.callback(
    Output({"type": "to-collapse-class", "index": MATCH}, "className"),
    Output({"type": "collapse-trigger", "index": MATCH}, "aria-expanded"),
    inputs=[
        Input({"type": "collapse-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "to-collapse-class", "index": MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> tuple[str, str]:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger (int): The (total) number of times a collapse button has been clicked.
        to_collapse_class (str): Current class name of the thing to collapse, 'collapsed' if not
            visible, empty string if visible.

    Returns:
        str: The new class name of the thing to collapse.
        str: The aria-expanded value.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes), "true"
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed", "false"


@dash.callback(
    Output("selected-problem", "data"),
    Output("tooltips", "children"),
    Output("quench-duration-setting", "children"),
    Output("coupling_strength", "value"),
    Output("coupling_strength", "marks"),
    Output("coupling_strength", "min"),
    Output("coupling_strength", "max"),
    Output("main-header", "children"),
    Output("main-description", "children"),
    inputs=[
        Input("tabs", "value"),
        State("selected-problem", "data"),
    ],
)
def update_selected_problem_type(
    tab_value: str,
    selected_problem: Union[ProblemType, int],
) -> tuple[str, int, list, float, dict[float, str], float, float, str, str]:
    """Updates the problem that is selected (KZ or KZ_NM), hides/shows settings accordingly,
        and updates the navigation options to indicate the currently active problem option.

    Args:
        tab_value: The value of the currently selected tab.
        selected_problem: The currently selected problem.

    Returns:
        selected-problem (int): Either KZ (``0`` or ``ProblemType.KZ``) or
            KZ_NM (``1`` or ``ProblemType.KZ_NM``).
        tooltips: The tooltips for the settings form.
        quench-duration-setting: The quench duration setting.
        coupling_strength-value: The value of the coupling strength slider setting.
        coupling_strength-marks: The marks of the coupling strength slider setting.
        coupling_strength-min: The minimum value of the coupling strength slider setting.
        coupling_strength-max: The maximum value of the coupling strength slider setting.
        main-header: The main header of the problem in the left column.
        main-description: The description of the problem in the left column.
    """
    selected_problem_value = int(tab_value.split("-")[-1])

    if selected_problem == selected_problem_value:
        raise PreventUpdate

    problem_type = ProblemType(selected_problem_value)
    isKZ = problem_type is ProblemType.KZ

    slider_value, slider_marks = get_slider_marks(problem_type)

    return (
        selected_problem_value,
        generate_tooltips(problem_type),
        get_quench_duration_setting(problem_type),
        slider_value,
        slider_marks,
        slider_marks[0]["value"],
        slider_marks[-1]["value"],
        MAIN_HEADER if isKZ else MAIN_HEADER_NM,
        DESCRIPTION if isKZ else DESCRIPTION_NM,
    )


@dash.callback(
    Output("anneal_duration", "className", allow_duplicate=True),
    Output("run-button", "disabled", allow_duplicate=True),
    Input("anneal_duration", "value"),
    prevent_initial_call=True,
)
def validate_quench_duration(ta: int | str):
    """Validate quench duration and prevent run if invalid."""
    if isinstance(ta, str):
        raise PreventUpdate

    invalid = not ta or ta < 5 or ta > 100

    return "error" if invalid else "", invalid


@dash.callback(
    Output("no-solver-modal", "opened"),
    Input("run-button", "n_clicks"),
)
def alert_no_solver(run_btn):
    """Notify if no quantum computer is accessible."""

    return not CLIENT


@dash.callback(
    Output("anneal_duration", "disabled"),
    Output("coupling_strength", "disabled"),
    Output({"type": "spins-option", "index": ALL}, "disabled"),
    Output("qpu_selection", "disabled"),
    inputs=[
        Input("job_submit_state", "children"),
    ],
    prevent_initial_call=True,
)
def disable_buttons(job_submit_state):
    """Disable user input during job submissions."""
    running_states = ["EMBEDDING", "SUBMITTED", "PENDING", "IN_PROGRESS"]
    done_states = ["COMPLETED", "CANCELLED", "FAILED"]
    is_running = job_submit_state in running_states

    if job_submit_state not in running_states + done_states:
        raise PreventUpdate

    return is_running, is_running, [is_running]*len(RING_LENGTHS), is_running


@dash.callback(
    Output("quench_schedule_filename", "children"),
    Output("quench_schedule_filename", "style"),
    Input("qpu_selection", "value"),
)
def set_schedule(qpu_name):
    """Set the schedule for the selected QPU."""

    schedule_filename = "FALLBACK_SCHEDULE.csv"
    schedule_filename_style = {"color": "#FFA143"}

    if qpu_name:
        for filename in [file for file in os.listdir("helpers") if "schedule.csv" in file.lower()]:

            if qpu_name.split(".")[0] in filename:  # Accepts & reddens older versions
                schedule_filename = filename

                if qpu_name in filename:
                    schedule_filename_style = {}

    return schedule_filename, schedule_filename_style


@dash.callback(
    Output("embeddings_cached", "data"),
    Output("embedding_is_cached", "children"),
    Input("qpu_selection", "value"),
)
def load_cached_embeddings(qpu_name):
    """Cache embeddings for the selected QPU."""

    embeddings_cached = {}  # Wipe out previous QPU's embeddings

    if qpu_name:
        for filename in [file for file in os.listdir("helpers") if ".json" in file and "emb_" in file]:
            if qpu_name.split(".")[0] in filename:
                with open(f"helpers/{filename}", "r") as fp:
                    embeddings_cached = json.load(fp)

                embeddings_cached = json_to_dict(embeddings_cached)

                # Validate that loaded embeddings' edges are still available on the selected QPU
                for length in list(embeddings_cached.keys()):
                    source_graph = dimod.to_networkx_graph(create_bqm(num_spins=length)).edges
                    target_graph = SOLVERS[qpu_name].edges
                    emb = embeddings_cached[length]

                    if not is_valid_embedding(emb, source_graph, target_graph):
                        del embeddings_cached[length]

    return embeddings_cached, ", ".join(str(embedding) for embedding in embeddings_cached.keys())


@dash.callback(
    Output("sample-v-theory-graph", "figure", allow_duplicate=True),
    Output("kz_data", "data", allow_duplicate=True),
    inputs=[
        Input("job_submit_state", "children"),
        State("graph-selection-radio", "value"),
        State("coupling_strength", "value"),  # previously input
        State("job_id", "data"),
        State("anneal_duration", "value"),
        State("spins", "value"),
        State("selected-problem", "data"),
        State("embeddings_cached", "data"),
        State("sample-v-theory-graph", "figure"),
        State("kz_data", "data"),  # get kibble zurek data point
    ],
    prevent_initial_call=True,
)
def add_graph_point_kz(
    job_submit_state,
    graph_selection,
    J,
    job_id,
    ta,
    spins,
    problem_type,
    embeddings_cached,
    figure,
    kz_data,
):
    """Add new point to kink density graph when KZ job finishes."""
    if job_submit_state != "COMPLETED" or problem_type is ProblemType.KZ_NM.value:
        raise PreventUpdate

    spins = int(spins)
    ta = int(ta.split(" ")[0]) if isinstance(ta, str) else ta
    embeddings_cached = json_to_dict(embeddings_cached)
    sampleset_unembedded = get_samples(CLIENT, job_id, spins, J, embeddings_cached[spins])
    _, kink_density = kink_stats(sampleset_unembedded, J)

    # Append the new data point
    kz_data.append((kink_density, ta))

    fig = dash.no_update if graph_selection == "schedule" else plot_kink_density(
        graph_selection, figure, kink_density, ta, J, problem_type=ProblemType.KZ
    )
    return fig, kz_data


@dash.callback(
    Output("kink-v-noise-graph", "figure", allow_duplicate=True),
    Output("kink-v-anneal-graph", "figure", allow_duplicate=True),
    Output("coupling_data", "data", allow_duplicate=True),  # store data using dcc
    Output("zne_estimates", "data", allow_duplicate=True),  # update zne_estimates
    Output("modal_trigger", "data"),
    inputs=[
        Input("job_submit_state", "children"),
        State("qpu_selection", "value"),
        State("coupling_strength", "value"),  # previously input
        State("quench_schedule_filename", "children"),
        State("job_id", "data"),
        State("anneal_duration", "value"),
        State("spins", "value"),
        State("selected-problem", "data"),
        State("embeddings_cached", "data"),
        State("kink-v-noise-graph", "figure"),
        State("kink-v-anneal-graph", "figure"),
        State("coupling_data", "data"),  # access previously stored data
        State("zne_estimates", "data"),  # Access ZNE estimates
    ],
    prevent_initial_call=True,
)
def add_graph_point_kz_nm(
    job_submit_state,
    qpu_name,
    J,
    schedule_filename,
    job_id,
    ta,
    spins,
    problem_type,
    embeddings_cached,
    figure_noise,
    figure_anneal,
    coupling_data,
    zne_estimates,
):
    """Add new point to Noise Ratio and Annealing Duration graphs when KZ Noise Mitigation job
    finishes."""
    if job_submit_state != "COMPLETED" or problem_type is ProblemType.KZ.value:
        raise PreventUpdate

    spins = int(spins)
    ta = int(ta.split(" ")[0]) if isinstance(ta, str) else ta
    embeddings_cached = json_to_dict(embeddings_cached)
    sampleset_unembedded = get_samples(CLIENT, job_id, spins, J, embeddings_cached[spins])
    _, kink_density = kink_stats(sampleset_unembedded, J)

    # Calculate lambda (previously kappa)
    # Added _ to avoid keyword restriction
    lambda_ = calclambda_(J=J, schedule_name=schedule_filename)

    fig_noise = plot_kink_density("coupling", figure_noise, kink_density, ta, J, lambda_)
    fig_anneal = plot_kink_density("kink_density", figure_anneal, kink_density, ta, J, lambda_)

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
        fig_noise, coupling_data, zne_estimates, ta_str
    )
    fig_anneal = plot_ze_estimates(fig_anneal, zne_estimates)

    return fig_noise, fig_anneal, coupling_data, zne_estimates, modal_trigger


@dash.callback(
    Output("job_submit_state", "children", allow_duplicate=True),
    inputs=[
        Input("selected-problem", "data"),
        Input("graph-selection-radio", "value"),
        Input("quench_schedule_filename", "children"),
        Input("coupling_strength", "value"),
        Input("spins", "value"),
        Input("anneal_duration", "value"),
        Input("qpu_selection", "value"),
    ],
    prevent_initial_call=True,
)
def reset_progress(
    problem_type,
    graph_selection,
    schedule_filename,
    J,
    spins,
    ta,
    qpu_selection,
):
    """Resets the progress on setting change."""
    return "READY"


@dash.callback(
    Output("sample-v-theory-graph", "figure"),
    Output("kz_data", "data"),
    inputs=[
        Input("selected-problem", "data"),
        Input("graph-selection-radio", "value"),
        Input("quench_schedule_filename", "children"),
        Input("coupling_strength", "value"),  # previously input
        Input("spins", "value"),
        Input("anneal_duration", "value"),
        State("kz_data", "data"),  # get kibble zurek data point
    ],
)
def load_new_graph_kz(
    problem_type,
    graph_selection,
    schedule_filename,
    J,
    spins,
    ta,
    kz_data,
):
    """Initiates graphics for kink density based on theory and QPU samples on page load and when
    when settings change."""
    if problem_type is ProblemType.KZ_NM.value:
        raise PreventUpdate

    if ctx.triggered_id in ["quench_schedule_filename", "spins", "coupling_strength"]:
        kz_data = []

    fig = plot_kink_densities_bg(
        graph_selection,
        [2, 350],
        J,
        schedule_filename,
        kz_data,
    )
    return fig, kz_data


@dash.callback(
    Output("kink-v-noise-graph", "figure"),
    Output("kink-v-anneal-graph", "figure"),
    Output("coupling_data", "data"),  # store data using dcc
    Output("zne_estimates", "data"),  # update zne_estimates
    inputs=[
        Input("quench_schedule_filename", "children"),
        Input("spins", "value"),
        Input("selected-problem", "data"),
    ],
)
def load_new_graphs_kz_nm(schedule_filename, spins, problem_type):
    """Initiates KZ Noise Mitigation graphs on page load and when settings change."""
    if problem_type is ProblemType.KZ.value:
        raise PreventUpdate

    time_range = [2, 1500]

    if not schedule_filename:
        schedule_filename = "FALLBACK_SCHEDULE.csv"

    n = theoretical_kink_density(time_range, J_BASELINE, schedule_filename)

    fig_noise = kink_v_noise_init_graph(n)
    fig_anneal = kink_v_anneal_init_graph(time_range, n)

    return fig_noise, fig_anneal, {}, {}


@dash.callback(
    Output("spin-orientation-graph", "figure"),
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
    spins = int(spins)

    if ctx.triggered_id == "job_submit_state":
        if job_submit_state != "COMPLETED":
            raise PreventUpdate

        embeddings_cached = json_to_dict(embeddings_cached)
        sampleset_unembedded = get_samples(CLIENT, job_id, spins, J, embeddings_cached[spins])
        kinks_per_sample, kink_density = kink_stats(sampleset_unembedded, J)
        best_indx = np.abs(kinks_per_sample - kink_density).argmin()
        best_sample = sampleset_unembedded.record.sample[best_indx]

    fig = plot_spin_orientation(num_spins=spins, sample=best_sample)
    return fig


class SubmitJobReturn(NamedTuple):
    """Return type for the ``submit_job`` callback function."""

    job_id: str = dash.no_update
    wd_job_n_intervals: int = 0


@dash.callback(
    Output("job_id", "data"),
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
) -> SubmitJobReturn:
    """Submit job and provide job ID."""

    solver = SOLVERS[qpu_name]
    spins = int(spins)
    ta_ns = int(ta_ns.split(" ")[0]) if isinstance(ta_ns, str) else ta_ns

    bqm = create_bqm(num_spins=spins, coupling_strength=J)

    embeddings_cached = json_to_dict(embeddings_cached)
    embedding = embeddings_cached[spins]
    annealing_time = ta_ns / 1000

    bqm_embedded = embed_bqm(bqm, embedding, DWaveSampler(solver=solver.name).adjacency)

    # ta_multiplier should be 1, unless (withNoiseMitigation and [J or schedule]) changes,
    # shouldn't change for MockSampler. In which case recalculate as
    # ta_multiplier=calclambda_(coupling_strength, schedule) as a function of the
    # correct schedule
    # State("ta_multiplier", "value") ? Should recalculate when J or schedule changes IFF noise mitigation tab?
    ta_multiplier = 1

    if problem_type is ProblemType.KZ_NM.value:
        ta_multiplier = calclambda_(J, schedule_name=filename)

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

    run_button_disabled: bool = True
    wd_job_disabled: bool = False
    wd_job_n_intervals: int = 0
    job_submit_state: str = dash.no_update
    job_submit_time: str = dash.no_update


@dash.callback(
    Output("run-button", "disabled"),
    Output("wd_job", "disabled"),
    Output("wd_job", "n_intervals", allow_duplicate=True),
    Output("job_submit_state", "children"),
    Output("job_submit_time", "data"),
    inputs=[
        Input("run-button", "n_clicks"),
        State("embedding_is_cached", "children"),
        State("spins", "value"),
    ],
    prevent_initial_call=True,
)
def run_button_click(
    run_btn_click,
    cached_embeddings,
    spins,
) -> RunButtonClickReturn:
    """Start simulation run when button is clicked."""
    if str(spins) in cached_embeddings.split(", "):  # If we have a cached embedding
        return RunButtonClickReturn(
            job_submit_state="SUBMITTED",
            job_submit_time=datetime.datetime.now().strftime("%c"),
        )

    return RunButtonClickReturn(job_submit_state="EMBEDDING")


class SimulateReturn(NamedTuple):
    """Return type for the ``simulate`` callback function."""

    run_button_disabled: bool = dash.no_update
    wd_job_disabled: bool = dash.no_update
    wd_job_interval: int = dash.no_update
    wd_job_n_intervals: int = dash.no_update
    job_submit_state: str = dash.no_update
    job_submit_time: str = dash.no_update
    embeddings_cached: dict = dash.no_update
    embedding_is_cached: str = dash.no_update


@dash.callback(
    Output("run-button", "disabled", allow_duplicate=True),
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
    spins = int(spins)

    if job_submit_state == "EMBEDDING":
        try:
            embedding = find_one_to_one_embedding(spins, SOLVERS[qpu_name].edges)
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
                run_button_disabled=False,
                wd_job_disabled=True,
                job_submit_state="FAILED",
            )
        except Exception:
            return SimulateReturn(
                run_button_disabled=False,
                wd_job_disabled=True,
                job_submit_state="FAILED",
            )

    if job_submit_state in ["SUBMITTED", "PENDING", "IN_PROGRESS"]:
        job_submit_state = get_job_status(CLIENT, job_id, job_submit_time)
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
        run_button_disabled=False,
        wd_job_disabled=True,
        job_submit_state=(
            dash.no_update if job_submit_state in ["COMPLETED", "CANCELLED", "FAILED"] else "ERROR"
        ),
    )


@dash.callback(
    Output("job-status-progress", "style"),
    Input("job_submit_state", "children"),
)
def set_progress_bar(job_submit_state):
    """Update progress bar for job submissions."""
    
    return {
        "width": f"{JOB_BAR_DISPLAY[job_submit_state if ctx.triggered_id else 'READY'][0]}%",
        "backgroundColor": JOB_BAR_DISPLAY[job_submit_state if ctx.triggered_id else 'READY'][1],
    }


@dash.callback(
    Output("error-modal", "opened"),
    Input("modal_trigger", "data"),
    State("error-modal", "opened"),
)
def toggle_modal(trigger, is_open):
    return True if trigger else is_open
