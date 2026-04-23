# Copyright 2024 D-Wave Systems Inc.
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

from demo_configs import RING_LENGTHS
import pytest
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate

from demo_callbacks import disable_buttons

parametrize_names = [
    "job_submit_state_val",
    "anneal_duration_val",
    "coupling_strength_val",
    "spins_val_out",
    "qpu_selection_val",
]

spins_disabled = [True]*len(RING_LENGTHS)
spins_enabled = [False]*len(RING_LENGTHS)
parametrize_vals = [
    ("EMBEDDING", True, True, spins_disabled, True),
    ("SUBMITTED", True, True, spins_disabled, True),
    ("PENDING", True, True, spins_disabled, True),
    ("IN_PROGRESS", True, True, spins_disabled, True),
    ("COMPLETED", False, False, spins_enabled, False),
    ("CANCELLED", False, False, spins_enabled, False),
    ("FAILED", False, False, spins_enabled, False),
    ("FAKE", False, False, spins_enabled, False),
]


@pytest.mark.parametrize(parametrize_names, parametrize_vals)
def test_disable_buttons(
    job_submit_state_val,
    anneal_duration_val,
    coupling_strength_val,
    spins_val_out,
    qpu_selection_val,
):
    """Test disabling buttons used during job submission."""

    def run_callback():
        context_value.set(
            AttributeDict(
                **{
                    "triggered_inputs": [{"prop_id": "job_submit_state.children"}],
                    "state_values": [{"prop_id": "spins.options"}],
                }
            )
        )

        return disable_buttons(job_submit_state_val)

    ctx = copy_context()

    if job_submit_state_val == "FAKE":
        with pytest.raises(PreventUpdate):
            ctx.run(run_callback)
    else:
        output = ctx.run(run_callback)
        assert output == (
            anneal_duration_val,
            coupling_strength_val,
            spins_val_out,
            qpu_selection_val,
        )
