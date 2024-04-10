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


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dimod

from helpers.kz_calcs import theoretical_kink_density
from helpers.qa import create_bqm

__all__ = ["plot_kink_densities_bg", "plot_kink_density", "plot_spin_orientation"]

def plot_kink_densities_bg(time_range, coupling_strength, schedule_name):
    """
    Plot background of KZ-theory density and (lightly) energy scales. 

    Args:
        time_range: Max and min quench times, as a list.

        coupling_strength: Value of J.

        schedule_name: Filename of anneal schedule.
    
    Returns:
        Plotly figure of predicted kink densities (main) and energy scales (background).
    """
    if schedule_name:
        schedule = pd.read_csv(f'helpers/{schedule_name}')
    else:
        schedule = pd.read_csv('helpers/FALLBACK_SCHEDULE.csv')

    A = schedule['A(s) (GHz)']
    B = schedule['B(s) (GHz)']         
    C = schedule['C (normalized)']

    # Display in Joule
    a = A/1.5092E24     
    b = B/1.5092E24

    n = theoretical_kink_density(time_range, coupling_strength, schedule_name)
    
    predicted_plus = go.Scatter(
        x=np.asarray(time_range), 
        y=np.asarray(1.1 * n),
        mode='lines',
        name='<b>Predicted (&plusmn;10%)',
        xaxis="x1",
        yaxis="y1",
        line_color='black', 
        line_width=1,
    )
    
    predicted_minus = go.Scatter(
        x=np.asarray(time_range), 
        y=np.asarray(0.90 * n),
        mode='lines',
        xaxis="x1",
        yaxis="y1",
        line_color='black', 
        line_width=1,
        fill='tonexty',
        fillcolor="white",
        showlegend=False,
    )
    
    energy_transverse = go.Scatter(
        x=C, # to get time_range[1]*C where C=1 equals max(t_a); also for problem plot     
        y=a, 
        mode='lines',
        name="A(C(s))", 
        xaxis="x2",
        yaxis='y2',
        line_color='blue',
        opacity=0.15,
    )

    energy_problem = go.Scatter(
        x=C, # see above comment     
        y=abs(coupling_strength)*b, 
        mode='lines',
        name="B(C(s))", 
        xaxis="x2",
        yaxis='y2',
        line_color='red',
        opacity=0.15,
    )

    layout = go.Layout(
        title='QPU Simulation Vs. Kibble-Zurek Prediction',
        title_font_color="rgb(243, 120, 32)",
        xaxis=dict(
            title='<b>Quench Duration [ns]<b>', 
            type="log", 
            range=[np.log10(time_range[0] - 1), np.log10(time_range[1] + 10)],  
        ),
        yaxis=dict(
            title='<b>Kink Density<b>', 
            type="log",
        ),
        xaxis2=dict(
            title={
                'text': 'C(s)', 
                'standoff': 0,
            }, 
            overlaying='x1', 
            side="top", 
            type="log", 
            range=[-1, 0],  # Minimal C=0.1 seems reasonable 
        ),
        yaxis2=dict(
            title='Energy [Joule]',  
            overlaying='y1', 
            side='right', 
            type="linear", 
            range=[0, np.max(b)],
        ),
        legend=dict(x=0.6, y=0.9),
        margin=dict(b=5,l=5,r=20,t=80),
        #plot_bgcolor='white',  # Kept for reference
    )

    fig=go.Figure(
        data=[predicted_plus, predicted_minus, energy_transverse, energy_problem], 
        layout=layout
    )

    fig.add_annotation(
        xref="x",
        yref="y",
        x=np.log10(0.25*(time_range[1])),
        y=np.log10(1.0*n.min()),
        text="Coherent",
        axref="x",
        ayref="y",
        ax=np.log10(0.50*(time_range[1])),
        ay=np.log10(1.0*n.min()),
        arrowhead=5,
    )
 
    fig.add_annotation(
        xref="x",
        yref="y",
        x=np.log10(0.5*(time_range[1])),
        y=np.log10(1.2*n.min()),
        text="Thermalized",
        axref="x",
        ayref="y",
        ax=np.log10(0.3*(time_range[1])),
        ay=np.log10(1.2*n.min()),
        arrowhead=5,
    )

    return fig

def plot_kink_density(fig_dict, kink_density, anneal_time):
    """Add QPU-based kink density to kink-density plot.

    Args:
        fig_dict: Existing background Plotly figure, as a dict.

        kink_density: Calculated kink density derived from last QPU sampleset.

        anneal_time: Anneal time used for input kink density.
    
    Returns:
        Updated Plotly figure with a marker at (anneal time, kink-density).
    """

    fig=go.Figure(
        fig_dict
    )

    return fig.add_trace(
        go.Scatter(
            x=[anneal_time], 
            y=[kink_density], 
            xaxis="x1",
            yaxis="y1",
            showlegend=False,
            marker=dict(size=10, 
                        color="black",
                        symbol="x",
            )
        )
    )


def plot_spin_orientation(num_spins=512, sample=None):
    """Plot the ring of spins. 

    Args:
        num_spins: Number of spins.

        sample: Single sample from the QPU's sampleset.

    Returns:
        Plotly figure of orientation for all spins in the ring.
    """
    
    cone_size = num_spins/20    # Based on how it looks

    z = np.linspace(0, 10, num_spins)
    x, y = z * np.cos(5 * z), z * np.sin(5 * z)

    if sample is None:

        cones_red = cones_blue = np.ones(num_spins, dtype=bool)
        num_cones_red = num_cones_blue = num_spins

    else:

        cones_red = ~np.isnan(np.where(sample == 1, z, np.nan))
        cones_blue = ~cones_red
        num_cones_red = np.count_nonzero(cones_red)
        num_cones_blue = num_spins - num_cones_red
     
    spins_up = go.Cone(
        x = x[cones_red],
        y = y[cones_red],
        z = z[cones_red],
        u=num_cones_red*[0],
        v=num_cones_red*[0],
        w=num_cones_red*[1],
        showlegend=False,
        showscale=False,
        colorscale=[[0, 'red'], [1, 'red']],
        hoverinfo=None,
        sizemode="absolute",
        sizeref=cone_size
    )

    spins_down = go.Cone(
        x=x[cones_blue],
        y=y[cones_blue],
        z=z[cones_blue],
        u=num_cones_blue*[0],
        v=num_cones_blue*[0],
        w=num_cones_blue*[-1],
        showlegend=False,
        showscale=False,
        colorscale=[[0, 'blue'], [1, 'blue']],
        hoverinfo=None,
        sizemode="absolute",
        sizeref=cone_size
    )

    fig = go.Figure(
        data=[spins_up, spins_down],
        layout=go.Layout(
            title=f'Spin States of {num_spins} Qubits in a 1D Ring',
            title_font_color="rgb(243, 120, 32)",
            showlegend=False,
            margin=dict(b=0,l=0,r=0,t=40),
            scene=dict(
                xaxis=dict(
                    showticklabels=False, 
                    visible=False,
                ),
                yaxis=dict(
                    showticklabels=False, 
                    visible=False,
                ),
                zaxis=dict(
                    showticklabels=False, 
                    visible=False,
                ),
                camera_eye=dict(x=0.15, y=1.25, z=0.15)
            )
        )
    )

    fig.add_layout_image(
        dict(
            source="assets/spin_states.png",
            xref="paper", 
            yref="paper",
            x=0.95, 
            y=0.05,
            sizex=0.4, 
            sizey=0.4,
            xanchor="right", 
            yanchor="bottom",
        )
    )

    return fig
