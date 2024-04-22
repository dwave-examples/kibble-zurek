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
import plotly

import dimod

from app import display_graphics_spin_ring

spins = ContextVar('spins')
job_submit_state = ContextVar('job_submit_state')
embeddings_cached = ContextVar('embeddings_cached')

json_embeddings_file = { \
    "512": {"1": [11], "0": [10], "2": [12]}, \
    "5": {"1": [11], "0": [10], "2": [12], "3": [13], "4": [14]} }

samples = dimod.as_samples([
    [-1, -1, -1, +1, +1], 
    [-1, -1, +1, +1, +1],
    [-1, -1, -1, +1, +1],])
sampleset = dimod.SampleSet.from_samples(samples, 'SPIN', 0)

parametrize_vals = [
(512, 'SUBMITTED', json_embeddings_file), (5, 'COMPLETED', json_embeddings_file)]

@pytest.mark.parametrize('spins_val, job_submit_state_val, embeddings_cached_val', parametrize_vals)
def test_graph_spins_spin_trigger(spins_val, job_submit_state_val, embeddings_cached_val):
    """Test graph of spin ring: spins trigger."""

    def run_callback():
        context_value.set(AttributeDict(**
            {'triggered_inputs': [{'prop_id': 'spins.value'},]}))

        return display_graphics_spin_ring(spins.get(), job_submit_state.get(), '1234', 2.5, embeddings_cached.get())

    spins.set(spins_val)
    job_submit_state.set(job_submit_state_val)
    embeddings_cached.set(embeddings_cached_val)

    ctx = copy_context()

    output = ctx.run(run_callback)
    assert type(output) == plotly.graph_objects.Figure

@pytest.mark.parametrize('spins_val, job_submit_state_val, embeddings_cached_val', parametrize_vals)
def test_graph_spins_job_trigger(mocker, spins_val, job_submit_state_val, embeddings_cached_val):
    """Test graph of spin ring: job-state trigger."""

    mocker.patch('app.get_samples', return_value=sampleset)

    def run_callback():
        context_value.set(AttributeDict(**
            {'triggered_inputs': [{'prop_id': 'job_submit_state.children'},]}))

        return display_graphics_spin_ring(spins.get(), job_submit_state.get(), '1234', 2.5, embeddings_cached.get())

    spins.set(spins_val)
    job_submit_state.set(job_submit_state_val)
    embeddings_cached.set(embeddings_cached_val)

    ctx = copy_context()

    output = ctx.run(run_callback)

    if job_submit_state_val == 'COMPLETED':
        assert type(output) == plotly.graph_objects.Figure
    else:
        output = no_update