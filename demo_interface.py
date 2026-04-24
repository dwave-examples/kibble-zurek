# Copyright 2026 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file stores the Dash HTML layout for the app."""

from __future__ import annotations

from enum import EnumMeta
from itertools import chain

import dash_mantine_components as dmc
from dash import dcc, html
from dwave.cloud import Client

from demo_configs import (
    DEFAULT_QPU,
    DESCRIPTION,
    J_OPTIONS,
    MAIN_HEADER,
    RING_LENGTHS,
    SHOW_TOOLTIPS,
    THUMBNAIL,
    TOOL_TIPS_KZ,
    TOOL_TIPS_KZ_NM,
)
from src.demo_enums import ProblemType

THEME_COLOR = "#2d4376"

# Initialize: available QPUs, initial progress-bar status
try:
    CLIENT = Client.from_config(client="qpu")
    SOLVERS = {
        qpu.name: qpu for qpu in CLIENT.get_solvers(fast_anneal_time_range__covers=[0.005, 0.1])
    }

    if len(SOLVERS) < 1:
        raise Exception

    init_job_status = "READY"

except Exception:
    SOLVERS = {}
    CLIENT = None
    init_job_status = "NO SOLVER"


def slider(label: str, id: str, config: dict, marks: list | None = None) -> html.Div:
    """Slider element for value selection.

    Args:
        label: The title that goes above the slider.
        id: A unique selector for this element.
        config: A dictionary of slider configurations, see dmc.Slider Dash Mantine docs.
        marks: A list of values that should be marked on the slider.
    """
    return html.Div(
        className="slider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Slider(
                id=id,
                className="slider",
                **config,
                marks=(
                    marks
                    if marks
                    else [
                        {"value": config["min"], "label": f'{config["min"]}'},
                        {"value": config["max"], "label": f'{config["max"]}'},
                    ]
                ),
                labelAlwaysOn=True,
                thumbLabel=f"{label} slider",
                color=THEME_COLOR,
            ),
        ],
    )


def range_slider(label: str, id: str, config: dict) -> html.Div:
    """Range slider element for value selection.

    Args:
        label: The title that goes above the range slider.
        id: A unique selector for this element.
        config: A dictionary of range slider configurations, see dmc.RangeSlider Dash Mantine docs.
    """
    return html.Div(
        className="rangeslider-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.RangeSlider(
                id=id,
                className="slider",
                **config,
                marks=[
                    {"value": config["min"], "label": f'{config["min"]}'},
                    {"value": config["max"], "label": f'{config["max"]}'},
                ],
                labelAlwaysOn=True,
                thumbFromLabel=f"{label} slider start",
                thumbToLabel=f"{label} slider end",
                color=THEME_COLOR,
            ),
        ],
    )


def dropdown(label: str, id: str, options: list, value: str | None = None) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Select(
                id=id,
                data=options,
                value=value if value else options[0]["value"],
                allowDeselect=False,
            ),
        ],
    )


def checklist(label: str, id: str, options: list, values: list, inline: bool = True) -> html.Div:
    """Checklist element for option selection.

    Args:
        label: The title that goes above the checklist.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        values: A list of values that should be preselected in the checklist.
        inline: Whether the options of the checklist are displayed beside or below each other.
    """
    return html.Div(
        className="checklist-wrapper",
        children=[
            dmc.CheckboxGroup(
                id=id,
                className=f"checklist{' checklist--inline' if inline else ''}",
                label=label,
                value=values,
                children=dmc.Group(
                    [
                        dmc.Checkbox(
                            label=option["label"], value=option["value"], color=THEME_COLOR
                        )
                        for option in options
                    ],
                ),
            ),
        ],
    )


def checkbox(label: str, id: str, checked: bool) -> html.Div:
    """Checkbox element.

    Args:
        label: The title that goes above the checkbox.
        id: A unique selector for this element.
        checked: Whether the checkbox is checked or not.
    """
    return html.Div(
        className="checkbox-wrapper",
        children=[
            dmc.Checkbox(
                id=id,
                label=label,
                checked=checked,
                color=THEME_COLOR,
            )
        ],
    )


def radio(label: str, id: str, options: list, value: str, inline: bool = True) -> html.Div:
    """Radio element for option selection.

    Args:
        label: The title that goes above the radio.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        value: The value of the radio that should be preselected.
        inline: Whether the options are displayed beside or below each other.
    """
    return html.Div(
        className="radio-wrapper",
        children=[
            dmc.RadioGroup(
                id=id,
                className=f"radio{' radio--inline' if inline else ''}",
                label=label,
                value=value,
                children=dmc.Group(
                    [
                        dmc.Radio(
                            option["label"],
                            value=option["value"],
                            color=THEME_COLOR,
                            id={"type": f"{id}-option", "index": i},
                        )
                        for i, option in enumerate(options)
                    ]
                ),
            ),
        ],
    )


def get_slider_marks(problem_type: ProblemType) -> tuple[float, list[dict]]:
    """Get slider marks and default value based on problem type.

    Args:
        problem_type: Either ProblemType.KZ or ProblemType.KZ_NM.

    Returns:
        A tuple of the default slider value and a list of dictionaries of slider marks.
    """

    if problem_type is ProblemType.KZ_NM:
        marks = J_OPTIONS
        value = -1.8
    else:
        marks = [
            round(0.1 * val) if val % 10 == 0 else round(0.1 * val, 1)
            for val in chain(range(-20, 0, 2), range(2, 12, 2))
        ]
        value = -1.4

    return value, [{"value": mark, "label": f"{mark}"} for mark in marks]


def get_quench_duration_setting(problem_type: ProblemType) -> html.Div:
    """Get quench duration setting based on problem type.

    Args:
        problem_type: Either ProblemType.KZ or ProblemType.KZ_NM.

    Returns:
        A Div containing the quench duration setting.
    """

    options = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    dropdown_options = generate_options([f"{option} ns" for option in options])
    if problem_type is ProblemType.KZ_NM:
        return html.Div(
            [
                dropdown(
                    "Target Quench Duration [ns]",
                    "anneal-duration",
                    dropdown_options,
                    value="80 ns",
                )
            ],
            id="quench-duration-setting",
        )

    return html.Div(
        [
            html.Label("Quench Duration [ns]", htmlFor="anneal-duration"),
            dmc.NumberInput(
                id="anneal-duration",
                type="number",
                min=5,
                max=100,
                step=1,
                value=7,
            ),
        ],
        id="quench-duration-setting",
    )


def generate_options(options: list | EnumMeta) -> list[dict]:
    """Generates options for dropdowns, checklists, radios, etc."""
    if isinstance(options, EnumMeta):
        return [{"label": option.label, "value": f"{option.value}"} for option in options]

    return [{"label": f"{option}", "value": f"{option}"} for option in options]


def generate_settings_form() -> html.Div:
    """This function generates settings for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the settings for selecting the scenario, model, and solver.
    """
    spin_options = generate_options(RING_LENGTHS)
    qpu_options = generate_options(SOLVERS)
    slider_value, slider_marks = get_slider_marks(ProblemType.KZ)

    default_value = "No Leap Access"
    if DEFAULT_QPU in SOLVERS:
        default_value = DEFAULT_QPU
    elif len(SOLVERS):
        default_value = list(SOLVERS.keys())[0]

    return html.Div(
        className="settings",
        children=[
            radio(
                "Spins",
                "spins",
                spin_options,
                spin_options[0]["value"],
            ),
            html.Div(
                slider(
                    "Coupling Strength (J)",
                    "coupling-strength",
                    {
                        "value": slider_value,
                        "min": slider_marks[0]["value"],
                        "max": slider_marks[-1]["value"],
                        "step": 0.1,
                        "restrictToMarks": True,
                    },
                    marks=slider_marks,
                ),
                id="coupling-strength-wrapper",
            ),
            get_quench_duration_setting(ProblemType.KZ),
            dropdown(
                "QPU",
                "qpu-selection",
                sorted(qpu_options, key=lambda op: op["value"]),
                value=default_value,
            ),
            html.P(
                ["Quench Schedule: ", html.Span(id="schedule-filename")],
                className="caption",
            ),
            html.P(
                ["Cached Embeddings: ", html.Span(id="cached-embeddings")],
                className="caption",
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Run and cancel buttons to run the simulation."""
    return html.Div(
        id="button-group",
        children=[
            html.Button("Run Simulation", id="run-button", className="button"),
            html.Button(
                "Cancel Simulation",
                id="cancel-button",
                className="button",
                style={"display": "none"},
            ),
        ],
    )


def default_graph(title: str, id: str, load_radio: bool = False) -> html.Div:
    """Default graph element with a title and optional radio buttons for graph selection.

    Args:
        title: The title that goes above the graph.
        id: A unique selector for this graph element.
        load_radio: Whether to create radio buttons for graph selection or not.

    Returns:
        A Div containing the graph and optional radio buttons.
    """

    radio_options = [
        {"label": "Both", "value": "both"},
        {"label": "Kink Density", "value": "kink_density"},
        {"label": "Schedule", "value": "schedule"},
    ]
    return html.Div(
        [
            html.H3(title),
            (
                html.Div(
                    radio(
                        "",
                        "graph-selection-radio",
                        sorted(radio_options, key=lambda op: op["value"]),
                        radio_options[0]["value"],
                    ),
                    id="graph-radio-options",
                )
                if load_radio
                else ()
            ),
            dcc.Graph(
                id=f"{id}-graph",
                responsive=True,
                config={"displayModeBar": False},
            ),
        ]
    )


def show_progress() -> html.Div:
    """Show job submission and progress.

    Returns:
        A Div containing the job submission status and progress bar.
    """

    init_job_status = "READY"
    job_status_style = {"color": "#AA3A3C"} if init_job_status == "NO SOLVER" else {}

    return html.Div(
        [
            html.P(
                [
                    "Status: ",
                    html.Span(
                        id="job-submit-state",
                        children=f"{init_job_status}",
                        style=job_status_style,
                    ),
                ],
                className="caption",
            ),
            html.Div(
                className="progress-bar",
                title="Job Progress",
                children=[
                    html.Div(
                        id="job-status-progress",
                        className="progress-bar-fill",
                    )
                ],
            ),
        ],
        className="progress-wrapper",
    )


def no_solver_modal() -> dmc.Modal:
    """Modal to alert users that no solvers are available."""
    return dmc.Modal(
        title="Leap's Quantum Computers Inaccessible",
        id="no-solver-modal",
        centered=True,
        children=[
            html.Div(
                [
                    html.H2("Could not connect to a Leap quantum computer"),
                    html.P(
                        [
                            """
                            If you are running locally, set environment variables or a
                            dwave-cloud-client configuration file as described in the
                            """,
                            html.A(
                                " Ocean",
                                href="https://docs.dwavequantum.com/en/latest/ocean/sapi_access_basic.html",
                                target="_blank",
                                rel="noopener",
                            ),
                            " documentation.",
                        ],
                    ),
                    html.P(
                        [
                            "If you are running in an online IDE, see the ",
                            html.A(
                                "system documentation",
                                href="https://docs.dwavequantum.com/en/latest/leap_sapi/dev_env.html",
                                target="_blank",
                                rel="noopener",
                            ),
                            " on supported IDEs.",
                        ],
                    ),
                ]
            )
        ],
    )


def error_modal() -> dmc.Modal:
    """Modal to alert users that an error occurred creating a graph."""
    return dmc.Modal(
        title="Error",
        id="error-modal",
        centered=True,
        children=[
            html.P(
                "Fitting function failed likely due to ill conditioned data, please collect more."
            )
        ],
    )


def generate_tooltips(problem_type: ProblemType = ProblemType.KZ) -> list[dmc.Tooltip]:
    """Tooltip generator.

    Args:
        problem_type: Either ProblemType.KZ or ProblemType.KZ_NM.
    """
    if not SHOW_TOOLTIPS:
        return []

    tool_tips = TOOL_TIPS_KZ if problem_type is ProblemType.KZ else TOOL_TIPS_KZ_NM

    return [
        dmc.Tooltip(label=message, target=f"#{target}", multiline=True, w=300, color="#202239")
        for target, message in tool_tips.items()
    ]


def create_interface() -> html.Div:
    """Set the application HTML."""
    return html.Div(
        id="app-container",
        children=[
            html.A(  # Skip link for accessibility
                "Skip to main content",
                href="#main-content",
                id="skip-to-main",
                className="skip-link",
                tabIndex=1,
            ),
            dcc.Store(id="coupling-data", data={}),  # KZ NM plot points
            dcc.Store(id="zne-estimates", data={}),  # store zero noise extrapolation points
            dcc.Store(id="modal-trigger", data=False),
            dcc.Store(id="kz-data", data=[]),  # KZ plot point
            dcc.Store(id="selected-problem-type"),
            dcc.Store(id="job-submit-time"),
            dcc.Store(id="job-id"),
            dcc.Store(id="embeddings", data={}),
            dcc.Interval(
                id="wd-job",
                interval=500,
                n_intervals=0,
                disabled=True,
                max_intervals=1,
            ),
            no_solver_modal(),
            error_modal(),
            # Settings and results columns
            html.Main(
                className="columns-main",
                id="main-content",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                className="left-column-layer-1",  # Fixed width Div to collapse
                                children=[
                                    html.Div(
                                        className="left-column-layer-2",  # Padding and content wrapper
                                        children=[
                                            html.Div(
                                                [
                                                    html.H1(MAIN_HEADER, id="main-header"),
                                                    html.P(DESCRIPTION, id="main-description"),
                                                ],
                                                className="title-section",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                generate_settings_form(),
                                                                generate_run_buttons(),
                                                                show_progress(),
                                                                html.Div(
                                                                    children=generate_tooltips(),
                                                                    id="tooltips",
                                                                ),
                                                            ],
                                                            className="settings-and-buttons",
                                                        ),
                                                        className="settings-and-buttons-wrapper",
                                                    ),
                                                    # Left column collapse button
                                                    html.Div(
                                                        html.Button(
                                                            id={
                                                                "type": "collapse-trigger",
                                                                "index": 0,
                                                            },
                                                            className="left-column-collapse",
                                                            title="Collapse sidebar",
                                                            children=[
                                                                html.Div(className="collapse-arrow")
                                                            ],
                                                            **{"aria-expanded": "true"},
                                                        ),
                                                    ),
                                                ],
                                                className="form-section",
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dmc.Tabs(
                                id="tabs",
                                value=f"tab-{ProblemType.KZ.value}",
                                color="white",
                                children=[
                                    html.Header(
                                        className="banner",
                                        children=[
                                            html.Nav(
                                                [
                                                    dmc.TabsList(
                                                        [
                                                            dmc.TabsTab(
                                                                ProblemType.KZ.label,
                                                                id={
                                                                    "type": "problem-type",
                                                                    "index": 0,
                                                                },
                                                                value=f"tab-{ProblemType.KZ.value}",
                                                            ),
                                                            dmc.TabsTab(
                                                                ProblemType.KZ_NM.label,
                                                                id={
                                                                    "type": "problem-type",
                                                                    "index": 1,
                                                                },
                                                                value=f"tab-{ProblemType.KZ_NM.value}",
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Img(src=THUMBNAIL, alt="D-Wave logo"),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value=f"tab-{ProblemType.KZ.value}",
                                        tabIndex="12",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    default_graph(
                                                        "Spin States of Qubits in a 1D Ring",
                                                        "spin-orientation",
                                                    ),
                                                    default_graph(
                                                        "QPU Samples vs Kibble-Zurek Prediction",
                                                        "sample-v-theory",
                                                        True,
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value=f"tab-{ProblemType.KZ_NM.value}",
                                        tabIndex="13",
                                        children=[
                                            html.Div(
                                                className="tab-content-wrapper",
                                                children=[
                                                    default_graph(
                                                        "Zero-Noise Extrapolation of Kink Density",
                                                        "kink-v-noise",
                                                    ),
                                                    default_graph(
                                                        "Measured and Extrapolated Kink Densities",
                                                        "kink-v-anneal",
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
