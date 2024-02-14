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

def create_bqm(num_spins=500, coupling_strength=-1):
    """
    Create a BQM representing a FM chain. 

    Args:
        num_spins: Length of chain

        coupling_strength: value of J

    Returns:
        dimod BQM  
    """
    bqm = dimod.BinaryQuadraticModel(vartype='SPIN')
    for spin in range(num_spins):
        bqm.add_quadratic(spin, (spin + 1) % num_spins, coupling_strength)
    return bqm
