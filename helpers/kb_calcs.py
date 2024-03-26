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

import numpy as np
import pandas as pd

__all__ = ["kink_stats", "theoretical_kink_density"]

def theoretical_kink_density(annealing_times_ns, J, schedule_name):
    """
    Calculate the coherent-theory kink density for the coupling strength & annealing times. 

    Args:
        annealing_times_ns: iterable of annealing times in nanoseconds

        J: Coupling strength

        schedule_name: Filename of anneal schedule

    Returns:
        n(ta).  
    """

    if schedule_name:
        schedule = pd.read_csv(f'helpers/{schedule_name}')
    else:
        schedule = pd.read_csv('helpers/09-1302A-B_Advantage2_prototype2.2_annealing_schedule.csv')

    A = schedule['A(s) (GHz)']
    B = schedule['B(s) (GHz)']         
    C = schedule['C (normalized)']

    A_tag = A.diff()/C.diff()
    B_tag = B.diff()/C.diff()

    sc_indx = abs(A - B*abs(J)).idxmin()

    b_top = (A[sc_indx]*0.5*1e9)*2*np.pi
    b_bottom = (B_tag[sc_indx]/B[sc_indx]) - (A_tag[sc_indx]/A[sc_indx])
    b = b_top/b_bottom

    return np.power([t*1e-9 for t in annealing_times_ns], -0.5)/(2*np.pi*np.sqrt(2*b))

def kink_stats(sampleset, J):
    """
    Calculate the average kink density for the sampleset. 

    Calculation is the number of sign switches per sample in the ring/boundary-pinned chain 
    divided by the length of the chain. 

    Args:
        sampleset: dimod sampleset

        J: Coupling strength

    Returns:
        Scalar average.  
    """
    samples_array = sampleset.record.sample
    sign_switches = np.diff(samples_array, prepend=samples_array[:,-1].reshape(len(samples_array),1))
    
    if J < 0:
        switches_per_sample = np.count_nonzero(sign_switches, 1)
        kink_density = np.mean(switches_per_sample)/sampleset.record.sample.shape[1]

        return switches_per_sample, kink_density
    else:
        non_switches_per_sample = np.count_nonzero(sign_switches==0, 1)
        kink_density = np.mean(non_switches_per_sample)/sampleset.record.sample.shape[1]
    
        return non_switches_per_sample, kink_density
    