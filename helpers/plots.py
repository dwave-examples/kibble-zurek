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

import matplotlib.pyplot as plt
from helpers.kb_calcs import theoretical_kink_density

def plot_kink_densities(kink_densities, coupling_strengths):
    """
    Plot found densities versus theory. 

    Args:
        densities: Length of chain

        coupling_strengths: value of J

    """

    cmap = plt.get_cmap('rainbow', len(coupling_strengths))
    i = 0
    for J in sorted(set(j for (j,ta) in kink_densities.keys())):
        kink_densities_j = [(ta, val) for (j, ta), val in kink_densities.items() if j==J]
        plt.plot([ta for (ta, kink) in kink_densities_j], [kink for (ta, kink) in kink_densities_j], "*", color=cmap(i), label=J)
        n = theoretical_kink_density([ta for (ta, kink) in kink_densities_j], J)
        plt.plot([ta for (ta, kink) in kink_densities_j], n, "-", color=cmap(i), label=f"Theory {J}")
        i += 1

    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.xlabel('Anneal Time [ns]')
    plt.ylabel('Kink Density')