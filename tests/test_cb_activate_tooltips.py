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

from app import activate_tooltips
from helpers.tooltips import tool_tips

tooltips_show = ContextVar('tooltips_show')

turn_off = [dict(display='none') for _ in tool_tips.keys()]
turn_on = [dict() for _ in tool_tips.keys()]

@pytest.mark.parametrize("input_val, output_vals",
    [('off', turn_off), ('on', turn_on)])
def test_activate_tooltips(input_val, output_vals):
    """Test tooltips are shown or not."""

    def run_callback():
        context_value.set(AttributeDict(**{"triggered_inputs":
            [{"prop_id": "tooltips_show.value"}]}))

        return activate_tooltips(tooltips_show.get())

    tooltips_show.set(input_val)

    ctx = copy_context()

    output = ctx.run(run_callback)
    assert list(output) == output_vals