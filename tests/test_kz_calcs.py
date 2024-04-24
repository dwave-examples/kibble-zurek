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
import pandas as pd

from helpers.kz_calcs import *

schedule = pd.read_csv('helpers/FALLBACK_SCHEDULE.csv')

@pytest.mark.parametrize('t_a_ns, J1, J2',
    [([7, 25], -1.0, -0.3), ([10, 30], 1.0, 0.3), ])
def test_theory(t_a_ns, J1, J2):
    """Test predicted kink density."""

    output1 = theoretical_kink_density(
        annealing_times_ns=t_a_ns, 
        J=J1, 
        schedule=schedule, 
        schedule_name='FALLBACK_SCHEDULE.csv')
    
    output2 = theoretical_kink_density(
        annealing_times_ns=t_a_ns, 
        J=J2, 
        schedule=schedule, 
        schedule_name='FALLBACK_SCHEDULE.csv')

    assert output1[0] > output1[1]
    assert output1[0] < output2[0]
    assert output1[1] < output2[1]