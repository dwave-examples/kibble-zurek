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

from app import set_schedule

qpu_selection = ContextVar('qpu_selection')

all_schedules = ['09-1263A-B_Advantage_system4.1_fast_annealing_schedule.csv',
                 '09-1273A-E_Advantage_system6.3_fast_annealing_schedule.csv',
                 '09-1302A-D_Advantage2_prototype2.4_fast_annealing_schedule.csv',
                 'FALLBACK_SCHEDULE.csv',
                 ]

parametrize_vals = [
    ('Advantage_system4.1', 
     all_schedules,
     0, 
     {'color': 'white', 'fontSize': 12}), 
    ('Advantage_system6.4',
     all_schedules,
     1,
     {'color': 'red', 'fontSize': 12}), 
    ('Advantage2_prototype2.3',
     all_schedules,
     2,
     {'color': 'red', 'fontSize': 12}),
     ('Advantage25_system7.9',
     all_schedules,
     3,
     {'color': 'red', 'fontSize': 12}),
     ]

@pytest.mark.parametrize(['qpu_selection_val', 'schedule_name', 'indx', 'style'], parametrize_vals)
def test_schedule_selection(mocker, qpu_selection_val, schedule_name, indx, style):
    """Test schedule selection."""

    mocker.patch('app.os.listdir', return_value=schedule_name)

    def run_callback():
        context_value.set(AttributeDict(**
            {'triggered_inputs': [{'prop_id': 'qpu_selection.value'}]}))

        return set_schedule(qpu_selection.get())

    qpu_selection.set(qpu_selection_val)

    ctx = copy_context()

    output = ctx.run(run_callback)
    assert output == (schedule_name[indx], style)
