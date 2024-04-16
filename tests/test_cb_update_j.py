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

import pytest

from contextvars import copy_context, ContextVar
from dash._callback_context import context_value
from dash._utils import AttributeDict

from app import update_j_output

coupling_strength = ContextVar('coupling_strength')

@pytest.mark.parametrize("input_val, output_val",
    [(0, "J=-2.0"), (0.5, "J=-1.5"), (1.1234, "J=-0.9"), (2.99, "J=1.0")])
def test_update_j_output(input_val, output_val):
    """Test that J is correctly updated from knob input."""

    def run_callback():
        context_value.set(AttributeDict(**{"triggered_inputs":
            [{"prop_id": "coupling_strength.value"}]}))

        return update_j_output(coupling_strength.get())

    coupling_strength.set(input_val)

    ctx = copy_context()

    output = ctx.run(run_callback)
    assert output == output_val