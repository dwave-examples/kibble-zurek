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
from dash._callback_context import context_value
from dash._utils import AttributeDict

from app import submit_job

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

        return submit_job(
            '11:45AM',
            'Advantage_system88.4',
            3,
            2.3,
            7,
            json_embeddings_file,
            0,
            "FALLBACK_SCHEDULE.csv",
            False,
        )

    ctx = copy_context()
    output = ctx.run(run_callback)
        
    assert output == (1234, False, False)
    

