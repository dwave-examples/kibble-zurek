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

import pytest
from dash._callback_context import context_value
from dash._utils import AttributeDict

from app import set_progress_bar
from helpers.layouts_components import job_bar_display

parametrize_vals = [
    (f"{status}", job_bar_display[status][0], job_bar_display[status][1])
    for status in job_bar_display.keys()
]
parametrize_vals.extend([tuple(["BREAK FUNCTION", "exception", "exception"])])


@pytest.mark.parametrize(
    "job_submit_state_val, bar_job_status_value, bar_job_status_color", parametrize_vals
)
def test_set_progress_bar(job_submit_state_val, bar_job_status_value, bar_job_status_color):
    """Test job-submission progress bar."""

    def run_callback():
        context_value.set(
            AttributeDict(**{"triggered_inputs": [{"prop_id": "job_submit_state.children"}]})
        )

        return set_progress_bar(job_submit_state_val)

    ctx = copy_context()

    try:
        output = ctx.run(run_callback)
        assert output == (bar_job_status_value, bar_job_status_color)
    except KeyError:
        assert job_submit_state_val == "BREAK FUNCTION"
