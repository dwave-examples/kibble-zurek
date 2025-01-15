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

import dimod
import pandas as pd
import pytest

from helpers.qa import *


def test_create_bqm():
    """Test BQM creation."""

    output = create_bqm(num_spins=3, coupling_strength=-1.0)

    assert output.linear == {0: 0.0, 1: 0.0, 2: 0.0}
    assert output.quadratic == {(1, 0): -1.0, (2, 0): -1.0, (2, 1): -1.0}

    output = create_bqm(num_spins=2, coupling_strength=0.5)

    assert output.linear == {0: 0.0, 1: 0.0}
    assert output.quadratic == {(1, 0): 1.0}


def test_embedding():
    """Test embedder."""

    edges = [(0, 1), (1, 2), (0, 2)]

    output = find_one_to_one_embedding(spins=2, sampler_edgelist=edges)
    assert len(output) == 2

    output = find_one_to_one_embedding(spins=3, sampler_edgelist=edges)
    assert len(output) == 3


def test_format_converter():
    """Test embedder."""

    json_embedding = {"512": {"1": [11], "0": [10], "2": [12]}, "5": {"1": [11], "0": [10]}}

    output = json_to_dict(json_embedding)

    assert output == {5: {0: [10], 1: [11]}, 512: {0: [10], 1: [11], 2: [12]}}
