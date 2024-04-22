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
from dash._callback_context import context_value
from dash._utils import AttributeDict

from app import submit_job

job_submit_time = ContextVar('job_submit_time')
qpu_name = ContextVar('qpu_name')
spins = ContextVar('spins')
J_offset = ContextVar('J_offset')
ta_ns = ContextVar('ta_ns')
embeddings_cached = ContextVar('embeddings_cached')

json_embeddings_file = { \
    "3": {"1": [11], "0": [10], "2": [12]}, \
    "5": {"1": [11], "0": [10], "2": [12], "3": [13], "4": [14]} }

class mock_computation():
    def wait_id(self): 
        return 1234

class mock_solver():
    def __init__(self):
        self.name = "dummy"
    def sample_bqm(self, **kwargs):
        return mock_computation()

class mock_qpus():
    def __init__(self):
        self.solvers = {'Advantage_system88.4': mock_solver()}
    def __getitem__(self, indx):
        return self.solvers[indx]

class dwave_sampler():
    def __init__(self, solver):
        self.adjacency = {
            10: {11, 12, 13, 14},
            11: {10, 12, 13, 14},
            12: {10, 11, 13, 14},}
    
def test_job_submission(mocker,):
    """Test job submission."""

    mocker.patch('app.qpus', new=mock_qpus())
    mocker.patch('app.DWaveSampler', new=dwave_sampler)

    def run_callback():
        context_value.set(AttributeDict(**
            {'triggered_inputs': [{'prop_id': 'job_submit_time.children'},]}))

        return submit_job(job_submit_time.get(), qpu_name.get(), spins.get(), 
                                J_offset.get(), ta_ns.get(), embeddings_cached.get())

    job_submit_time.set('11:45')
    qpu_name.set('Advantage_system88.4')
    spins.set(3)
    J_offset.set(2.3)
    ta_ns.set(7)
    embeddings_cached.set(json_embeddings_file)

    ctx = copy_context()
    output = ctx.run(run_callback)
        
    assert output == (1234)
    
