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
import sys
import os

import dimod
from dimod.testing import *
from unittest.mock import patch
from dimod import SampleSet

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler
from MockKibbleZurekSampler import MockKibbleZurekSampler

@pytest.fixture
def default_sampler():
    nodelist = ['a', 'b']
    edgelist = [('a', 'b')]
    return MockKibbleZurekSampler(nodelist=nodelist, edgelist=edgelist)

@pytest.fixture
def custom_sampler():
    custom_nodelist = [0, 1, 2]
    custom_edgelist = [(0, 1), (1, 2)]
    substitute_sampler = SimulatedAnnealingSampler()
    substitute_kwargs = {
        'beta_range': [1, 2],
        'num_sweeps': 200
    }
    return MockKibbleZurekSampler(
        nodelist=custom_nodelist,
        edgelist=custom_edgelist,
        topology_type='chimera',
        topology_shape=[4, 4, 4],
        substitute_sampler=substitute_sampler,
        substitute_kwargs=substitute_kwargs
    )

@pytest.fixture
def sample_bqm():
    return BinaryQuadraticModel({'a': 1.0, 'b': -1.0}, {('a', 'b'): 0.5}, 0.0, 'BINARY')



def test_initialization(default_sampler, custom_sampler):
    # #assert default_sampler.topology_type == 'pegasus'
    # #assert default_sampler.topology_shape == [16]
    # assert isinstance(default_sampler.substitute_sampler, SimulatedAnnealingSampler)
    # #assert default_sampler.substitute_kwargs['beta_range'] == [0, 3]
    # assert default_sampler.substitute_kwargs['beta_schedule_type'] == 'linear'
    # assert default_sampler.substitute_kwargs['num_sweeps'] == 100
    # assert default_sampler.substitute_kwargs['randomize_order'] is True
    # assert default_sampler.substitute_kwargs['proposal_acceptance_criteria'] == 'Gibbs'
    # assert default_sampler.sampler_type == 'mock'

    assert custom_sampler.topology_type == 'chimera'
    assert custom_sampler.topology_shape == [4, 4, 4]
    assert custom_sampler.nodelist == [0, 1, 2]
    assert custom_sampler.edgelist == [(0, 1), (1, 2)]

    assert isinstance(custom_sampler.substitute_sampler, SimulatedAnnealingSampler)
    assert custom_sampler.substitute_kwargs['beta_range'] == [1, 2]
    assert custom_sampler.substitute_kwargs['num_sweeps'] == 200

   

# def test_sample_with_default_annealing_time(default_sampler, sample_bqm):
#     sampleset = default_sampler.sample(sample_bqm)

#     # default anneal _time should be 20
#     expected_num_sweeps = int(20 * 1000)
#     assert default_sampler.kwargs['num_sweeps']== expected_num_sweeps

def test_sample_with_custom_annealing_time(default_sampler, sample_bqm):
    pass


def test_sample_preserves_vartype(default_sampler, sample_bqm):
    pass


def test_bqm_vartype_conversion(default_sampler, sample_bqm):
    pass

def test_substitute_sampler_call_parameters(default_sampler, sample_bqm):
    pass


