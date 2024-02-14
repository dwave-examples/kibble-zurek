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
import numpy as np

def avg_kink_density(sampleset, j_sign="+"):
    """
    Calculate the average kink density for the sampleset. 

    Calculation is the number of sign switches per sample in the ring/boundary-pinned chain 
    divided by the length of the chain. 

    Args:
        sampleset: dimod sampleset

    Returns:
        Scalar average.  
    """
    samples_array = sampleset.record.sample
    sign_switches = np.diff(samples_array, prepend=samples_array[:,-1].reshape(len(samples_array),1))
    switches_per_sample = np.count_nonzero(sign_switches, 1)
    kink_density = np.mean(switches_per_sample)/sampleset.record.sample.shape[1]
    if j_sign == "-":
        return 1 - kink_density
    else: 
        return kink_density 
    
    raise ValueError('J_sign must be "+" or "-" only')