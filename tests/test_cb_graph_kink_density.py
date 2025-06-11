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

from contextvars import copy_context

from src.demo_enums import ProblemType
import dimod
import numpy as np
import plotly
import pytest
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate

from app import add_graph_point_kz, add_graph_point_kz_nm, load_new_graph_kz, load_new_graphs_kz_nm

json_embeddings_file = {
    "512": {"1": [11], "0": [10], "2": [12]},
    "5": {"1": [11], "0": [10], "2": [12], "3": [13], "4": [14]},
}

sample_vs_theory = plotly.graph_objects.Figure(
    {
        "data": [
            {
                "type": "scatter",
                "x": np.array([1, 2, 3], dtype=np.int64),
                "xaxis": "x",
                "y": np.array([1, 2, 3], dtype=np.int64),
                "yaxis": "y",
            }
        ],
        "layout": {
            "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x"}},
            "xaxis3": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x"}},
            "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "y"}},
        },
    }
)

samples = dimod.as_samples(
    [
        [-1, -1, -1, +1, +1],
        [-1, -1, +1, +1, +1],
        [-1, -1, -1, +1, +1],
    ]
)
sampleset = dimod.SampleSet.from_samples(samples, "SPIN", 0)

parametrize_vals = [
    ("kz_graph_display", "both", "", ProblemType.KZ.value),
    ("kz_graph_display", "kink_density", "", ProblemType.KZ.value),
    ("kz_graph_display", "schedule", "", ProblemType.KZ.value),
    ("coupling_strength", "schedule", "", ProblemType.KZ.value),
    ("quench_schedule_filename", "schedule", "", ProblemType.KZ.value),
    ("job_submit_state", "", "SUBMITTED", ProblemType.KZ_NM.value),
    ("job_submit_state", "", "PENDING", ProblemType.KZ_NM.value),
    ("job_submit_state", "", "COMPLETED", ProblemType.KZ.value),
    ("job_submit_state", "", "COMPLETED", ProblemType.KZ_NM.value),
]


@pytest.mark.parametrize(
    "trigger_val, kz_graph_display_val, job_submit_state_val, problem_type", parametrize_vals
)
def test_add_graph_point_kz(mocker, trigger_val, kz_graph_display_val, job_submit_state_val, problem_type):
    """Test graph of kink density."""

    mocker.patch("app.get_samples", return_value=sampleset)

    def run_callback():
        context_value.set(
            AttributeDict(
                **{
                    "triggered_inputs": [
                        {"prop_id": trigger_val},
                    ]
                }
            )
        )

        return add_graph_point_kz(
            job_submit_state=job_submit_state_val,
            graph_selection=kz_graph_display_val,
            J=-1.4,
            job_id="1234",
            ta=10,
            spins=5,
            problem_type=problem_type,
            embeddings_cached=json_embeddings_file,
            figure=sample_vs_theory,
            kz_data=[],
        )

    ctx = copy_context()

    if job_submit_state_val == "COMPLETED" and problem_type == ProblemType.KZ.value:
        output = ctx.run(run_callback)

        assert type(output[0]) == plotly.graph_objects.Figure
        assert output[1][0][1] == 10
    else:
        with pytest.raises(PreventUpdate):
            ctx.run(run_callback)


@pytest.mark.parametrize(
    "trigger_val, kz_graph_display_val, job_submit_state_val, problem_type", parametrize_vals
)
def test_add_graph_point_kz_nm(mocker, trigger_val, kz_graph_display_val, job_submit_state_val, problem_type):
    """Test graph of kink density."""

    mocker.patch("app.get_samples", return_value=sampleset)

    def run_callback():
        context_value.set(
            AttributeDict(
                **{
                    "triggered_inputs": [
                        {"prop_id": trigger_val},
                    ]
                }
            )
        )

        return add_graph_point_kz_nm(
            job_submit_state=job_submit_state_val,
            qpu_name=None,
            J=-1.4,
            schedule_filename="FALLBACK_SCHEDULE.csv",
            job_id="1234",
            ta=10,
            spins=5,
            problem_type=problem_type,
            embeddings_cached=json_embeddings_file,
            figure_noise=sample_vs_theory,
            figure_anneal=sample_vs_theory,
            coupling_data={},
            zne_estimates={},
        )

    ctx = copy_context()

    if job_submit_state_val == "COMPLETED" and problem_type == ProblemType.KZ_NM.value:
        output = ctx.run(run_callback)

        assert type(output[0]) == plotly.graph_objects.Figure
        assert type(output[1]) == plotly.graph_objects.Figure
        assert "10" in output[2]
        assert output[3] == {}
        assert output[4] == False
    else:
        with pytest.raises(PreventUpdate):
            ctx.run(run_callback)


@pytest.mark.parametrize(
    "trigger_val, kz_graph_display_val, job_submit_state_val, problem_type", parametrize_vals
)
def test_load_new_graph_kz(mocker, trigger_val, kz_graph_display_val, job_submit_state_val, problem_type):
    """Test graph of kink density."""

    mocker.patch("app.get_samples", return_value=sampleset)

    def run_callback():
        context_value.set(
            AttributeDict(
                **{
                    "triggered_inputs": [
                        {"prop_id": trigger_val},
                    ]
                }
            )
        )

        return load_new_graph_kz(
            problem_type=problem_type,
            graph_selection=kz_graph_display_val,
            schedule_filename="FALLBACK_SCHEDULE.csv",
            J=-1.4,
            spins=5,
            ta=10,
            kz_data=[],
        )

    ctx = copy_context()

    if problem_type == ProblemType.KZ.value:
        output = ctx.run(run_callback)

        assert type(output[0]) == plotly.graph_objects.Figure
        assert output[1] == []
    else:
        with pytest.raises(PreventUpdate):
            ctx.run(run_callback)


@pytest.mark.parametrize(
    "trigger_val, kz_graph_display_val, job_submit_state_val, problem_type", parametrize_vals
)
def test_load_new_graphs_kz_nm(mocker, trigger_val, kz_graph_display_val, job_submit_state_val, problem_type):
    """Test graph of kink density."""

    mocker.patch("app.get_samples", return_value=sampleset)

    def run_callback():
        context_value.set(
            AttributeDict(
                **{
                    "triggered_inputs": [
                        {"prop_id": trigger_val},
                    ]
                }
            )
        )

        return load_new_graphs_kz_nm(
            schedule_filename="FALLBACK_SCHEDULE.csv",
            spins=5,
            problem_type=problem_type,
        )

    ctx = copy_context()

    if problem_type == ProblemType.KZ_NM.value:
        output = ctx.run(run_callback)

        assert type(output[0]) == plotly.graph_objects.Figure
        assert type(output[1]) == plotly.graph_objects.Figure
        assert output[2] == {}
        assert output[3] == {}
    else:
        with pytest.raises(PreventUpdate):
            ctx.run(run_callback)
