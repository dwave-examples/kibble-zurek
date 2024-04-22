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

import numpy as np

__all__ = ['kink_stats', 'theoretical_kink_density']

def theoretical_kink_density(annealing_times_ns, J, schedule, schedule_name):
    """
    Calculate the kink density predicted for given the coupling strength and annealing times. 

    Args:
        annealing_times_ns: Iterable of annealing times, in nanoseconds.

        J: Coupling strength between the spins of the ring.

        schedule: Anneal schedule for the selected QPU. 

        schedule_name: Filename of anneal schedule. Used to compensate for schedule energy 
        overestimate. 

    Returns:
        Kink density per anneal time, as a NumPy array.  
    """

    # See the Code section of the README.md file for an explanation of the
    # following code. 

    COMPENSATION_SCHEDULE_ENERGY = 0.8 if 'Advantage_system' in schedule_name else 1.0

    A = COMPENSATION_SCHEDULE_ENERGY * schedule['A(s) (GHz)']
    B = COMPENSATION_SCHEDULE_ENERGY * schedule['B(s) (GHz)']         
    s = schedule['s']

    A_tag = A.diff()/s.diff()       # Derivatives of the energies for fast anneal 
    B_tag = B.diff()/s.diff()

    sc_indx = abs(A - B*abs(J)).idxmin()    # Anneal fraction, s, at the critical point

    
    b_numerator  = 1e9 * np.pi * A[sc_indx] # D-Wave's schedules are in GHz
    b_denominator  = B_tag[sc_indx]/B[sc_indx] - A_tag[sc_indx]/A[sc_indx]
    b = b_numerator / b_denominator 

    return np.power([1e-9 * t for t in annealing_times_ns], -0.5) / (2 * np.pi * np.sqrt(2 * b))

def kink_stats(sampleset, J):
    """
    Calculate kink density for the sample set. 

    Calculation is the number of sign switches per sample divided by the length
    of the ring for ferromagnetic coupling. For anti-ferromagnetic coupling,
    kinks are any pairs of identically-oriented spins. 

    Args:
        sampleset: dimod sample set.

        J: Coupling strength between the spins of the ring.

    Returns:
        Switches/non-switches per sample and the average kink density across 
        all samples.  
    """
    samples_array = sampleset.record.sample
    sign_switches = np.diff(samples_array, 
                            prepend=samples_array[:,-1].reshape(len(samples_array), 1))
    
    if J < 0:

        switches_per_sample = np.count_nonzero(sign_switches, 1)
        kink_density = np.mean(switches_per_sample) / sampleset.record.sample.shape[1]

        return switches_per_sample, kink_density
    
    else:

        non_switches_per_sample = np.count_nonzero(sign_switches==0, 1)
        kink_density = np.mean(non_switches_per_sample) / sampleset.record.sample.shape[1]
    
        return non_switches_per_sample, kink_density
    