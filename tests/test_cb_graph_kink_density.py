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

import pytest

from contextvars import copy_context, ContextVar
from dash import no_update
from dash._callback_context import context_value
from dash._utils import AttributeDict
import numpy as np
import plotly

import dimod

from app import display_graphics_kink_density

kz_graph_display = ContextVar('kz_graph_display')
job_submit_state = ContextVar('job_submit_state')

json_embeddings_file = { \
    "512": {"1": [11], "0": [10], "2": [12]}, \
    "5": {"1": [11], "0": [10], "2": [12], "3": [13], "4": [14]} }

sample_vs_theory = plotly.graph_objects.Figure({
    'data': [{
              'type': 'scatter',
              'x': np.array([1, 2, 3], dtype=np.int64),
              'xaxis': 'x',
              'y': np.array([1, 2, 3], dtype=np.int64),
              'yaxis': 'y'}],
    'layout': {
               'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'x'}},
               'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'y'}}}
})

samples = dimod.as_samples([
    [-1, -1, -1, +1, +1], 
    [-1, -1, +1, +1, +1],
    [-1, -1, -1, +1, +1],])
sampleset = dimod.SampleSet.from_samples(samples, 'SPIN', 0)

parametrize_vals = [('kz_graph_display.value', 'both', 'dummy'), 
                    ('kz_graph_display.value', 'kink_density', 'dummy'),
                    ('kz_graph_display.value', 'schedule', 'dummy'),
                    ('coupling_strength.value', 'schedule', 'dummy'),
                    ('quench_schedule_filename.children', 'schedule', 'dummy'),
                    ('job_submit_state.children', 'dummy', 'SUBMITTED'),
                    ('job_submit_state.children', 'dummy', 'PENDING'),
                    ('job_submit_state.children', 'dummy', 'COMPLETED')]

@pytest.mark.parametrize('trigger_val, kz_graph_display_val, job_submit_state_val', 
                         parametrize_vals)
def test_graph_kink_density(mocker, trigger_val, kz_graph_display_val, job_submit_state_val):
    """Test graph of kink density."""

    mocker.patch('app.get_samples', return_value=sampleset)

    def run_callback():
        context_value.set(AttributeDict(**
            {'triggered_inputs': [{'prop_id': trigger_val},]}))

        return display_graphics_kink_density(
            kz_graph_display.get(), 2.5, "FALLBACK_SCHEDULE.csv", job_submit_state.get(), '1234', 
            5, 100, 7, 5, json_embeddings_file, sample_vs_theory)


    kz_graph_display.set(kz_graph_display_val)
    job_submit_state.set(job_submit_state_val)

    ctx = copy_context()

    output = ctx.run(run_callback)

    if trigger_val in [
        'kz_graph_display.value', 
        'coupling_strength.value', 
        'quench_schedule_filename.children']:
        assert type(output) == plotly.graph_objects.Figure
    elif job_submit_state_val == "COMPLETED":
        assert type(output) == plotly.graph_objects.Figure
    else:
        assert output == no_update