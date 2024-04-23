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

from contextvars import copy_context
from dash import no_update
from dash._callback_context import context_value
from dash._utils import AttributeDict

import datetime

from app import simulate

before_test = datetime.datetime.now().strftime('%c')
parametrize_vals = [
    (512, [512, 1024], 'SUBMITTED', no_update),
    (2048, [512, 1024], 'EMBEDDING', 'needed')]

@pytest.mark.parametrize(
    'spins_val, cached_embedding_lengths_val, submit_state_out, embedding_found', 
    parametrize_vals)
def test_simulate_button_press(spins_val, cached_embedding_lengths_val, submit_state_out, 
                               embedding_found):
    """Test pressing Simulate button initiates submission."""

    def run_callback():
        context_value.set(AttributeDict(
            **{"triggered_inputs": [{"prop_id": "btn_simulate.n_clicks"}]}))

        return simulate(1, 2, 'dummy_job_id', 'READY', before_test,
            cached_embedding_lengths_val, spins_val, 
            'Advantage_system4.3', 'dummy')
    
    ctx = copy_context()

    output = ctx.run(run_callback)

    assert output[0:5] == (True, False, 0.5*1000, 0, submit_state_out) 
    assert output[6] == embedding_found

def mock_get_status(client, job_id, job_submit_time):

    if job_id == 'first few attempts':
        return None
    if job_id == 'first returned status':
        return 'PENDING'
    if job_id == 'early returning statuses':
        return 'PENDING'
    if job_id == 'impossible input status':
        return 'should make no difference'
    if job_id == '-1':
        return 'ERROR'
    if job_id == '0':
        return 'EMBEDDING'
    if job_id == '1':
        return 'IN_PROGRESS'
    if job_id == '2':
        return 'COMPLETED'
    if job_id == '3':
        return 'CANCELLED'
    if job_id == '4':
        return 'FAILED'

parametrize_names = 'job_id_val, job_submit_state_in, cached_embedding_lengths_val, ' +  \
    'spins_val, embeddings_found_in, btn_simulate_disabled_out, wd_job_disabled_out, ' + \
    'wd_job_intervals_out, wd_job_n_out, job_submit_state_out, job_submit_time_out, ' + \
    'embedding_found_out'

parametrize_vals = [
    (-1, 'READY', [512, 1024], 512, 'dummy embeddings found', False, True, 
     0, 0, 'ERROR', no_update, no_update),
    ('first few attempts', 'SUBMITTED', [512, 1024], 512, 'dummy embeddings found', True, False, 
     0.2*1000, 0, 'SUBMITTED', no_update, no_update)]

@pytest.mark.parametrize(parametrize_names, parametrize_vals)
def test_simulate_states(mocker, job_id_val, job_submit_state_in, cached_embedding_lengths_val, 
                         spins_val, embeddings_found_in, btn_simulate_disabled_out, 
                         wd_job_disabled_out, wd_job_intervals_out, wd_job_n_out, 
                         job_submit_state_out, job_submit_time_out, embedding_found_out):
    """Test transitions between states."""

    mocker.patch('app.get_job_status', new=mock_get_status)

    def run_callback():
        context_value.set(AttributeDict(
            **{"triggered_inputs": [{"prop_id": "wd_job.n_intervals"}]}))

        return simulate(1, 1, job_id_val, job_submit_state_in, before_test, 
                        cached_embedding_lengths_val, spins_val, 'Advantage_system4.3', 
                        embeddings_found_in)

    ctx = copy_context()

    output = ctx.run(run_callback)

    assert output[0:5] == (btn_simulate_disabled_out, wd_job_disabled_out, wd_job_intervals_out, 
                           wd_job_n_out, job_submit_state_out)