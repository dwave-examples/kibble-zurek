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

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from helpers.kb_calcs import theoretical_kink_density

__all__ = ["plot_kink_densities"]

def plot_kink_densities(kink_densities, coupling_strengths):
    """
    Plot found densities versus theory. 

    Args:
        densities: Length of chain

        coupling_strengths: value of J

    """
    fig = go.Figure()
    for J in sorted(set(j for (j,ta) in kink_densities.keys())):
        kink_densities_j = [(ta, val) for (j, ta), val in kink_densities.items() if j==J]
        n = theoretical_kink_density([ta for (ta, kink) in kink_densities_j], J)
        fig.add_trace(
            go.Scatter(
                x=np.asarray([ta for (ta, kink) in kink_densities_j]), 
                y=np.asarray(n),
                mode='lines',
                name='lines'
            )
        )

    fig.update_layout(
        title='Kink Density: Theory Vs. QPU Samples',
        xaxis_title='Anneal Time [ns]',
        yaxis_title='Kink Density',
        xaxis_type = "log", 
        yaxis_type = "log"
    )

    return fig