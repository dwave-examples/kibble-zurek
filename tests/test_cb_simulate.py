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

import datetime

from app import simulate

cached_embedding_lengths = ContextVar('cached_embedding_lengths')
spins = ContextVar('spins')

before_test = datetime.datetime.now().strftime('%c')
parametrize_vals = [
    (512, [512, 1024], 'SUBMITTED', no_update),
    (2048, [512, 1024], 'EMBEDDING', 'needed')]

@pytest.mark.parametrize('spins_val, cached_embedding_lengths_val, submit_state_out, embedding_found', 
                         parametrize_vals)
def test_simulate_button_press(spins_val, cached_embedding_lengths_val, submit_state_out, embedding_found):
    """Test pressing Simulate button initiates submission."""

    def run_callback():
        context_value.set(AttributeDict(
            **{"triggered_inputs": [{"prop_id": "btn_simulate.n_clicks"}]}))

        return simulate(1, 2, '1234', 'READY', before_test,
            cached_embedding_lengths.get(), spins.get(), 
            'Advantage_system4.3', 'needed')

    spins.set(spins_val)
    cached_embedding_lengths.set(cached_embedding_lengths_val)
    
    ctx = copy_context()

    output = ctx.run(run_callback)

    assert output[0:5] == (True, False, 0.5*1000, 0, submit_state_out) 
    assert output[6] == embedding_found
