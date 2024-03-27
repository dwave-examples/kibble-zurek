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
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dimod

from helpers.kb_calcs import theoretical_kink_density
from helpers.qa import create_bqm

__all__ = ["plot_kink_densities_bg", "plot_kink_density", "plot_spin_orientation"]

def plot_kink_densities_bg(time_range, coupling_strength, schedule_name):
    """
    Plot density based on theory and energy scales. 

    Args:
        time_range: max and min quench times

        coupling_strength: value of J

        schedule_name: Filename of anneal schedule

    """
    if schedule_name:
        schedule = pd.read_csv(f'helpers/{schedule_name}')
    else:
        schedule = pd.read_csv('helpers/09-1302A-B_Advantage2_prototype2.2_annealing_schedule.csv')

    A = schedule['A(s) (GHz)']
    B = schedule['B(s) (GHz)']         
    C = schedule['C (normalized)']

    # Display in Joule
    a = A/1.5092E24     
    b = B/1.5092E24

    n = theoretical_kink_density(time_range, coupling_strength, schedule_name)
    
    trace1_p = go.Scatter(
            x=np.asarray(time_range), 
            y=np.asarray(1.1 * n),
            mode='lines',
            name='Predicted (&plusmn;10%)',
            xaxis="x1",
            yaxis="y1",
            line_color='lightgrey', 
            line_width=1,
            )
    
    trace1_m = go.Scatter(
            x=np.asarray(time_range), 
            y=np.asarray(0.90 * n),
            mode='lines',
            xaxis="x1",
            yaxis="y1",
            line_color='lightgrey', 
            line_width=1,
            fill='tonexty',
            fillcolor="white",
            showlegend=False,)
    
    trace2 = go.Scatter(
        x=C, #time_range[1]*C,   # C=1 --> MAX(t_a)     
        y=a, 
        mode='lines',
        name="A(C(s))", 
        xaxis="x2",
        yaxis='y2',
        line_color='blue',
        opacity=0.4)

    trace3 = go.Scatter(
        x=C, #time_range[1]*C,    # C=1 --> MAX(t_a)     
        y=abs(coupling_strength)*b, 
        mode='lines',
        name="B(C(s))", 
        xaxis="x2",
        yaxis='y2',
        line_color='red',
        opacity=0.4)

    layout = go.Layout(
        title='QPU Simulation Vs. Kibble-Zurek Prediction',
        title_font_color="rgb(243, 120, 32)",
        xaxis=dict(
            title='Quench Time [ns]', 
            type="log", range=[np.log10(time_range[0] - 1), np.log10(time_range[1] + 10)]),     # exponents for log
        yaxis=dict(
            title='Kink Density', 
            type="log"),
        xaxis2=dict(
            title={
                'text': 'C(s)', 
                'standoff':0}, 
            overlaying='x1', 
            side="top", 
            type="log", 
            range=[-1, 0]),
        yaxis2=dict(
            title='Energy [Joule]',  
            overlaying='y1', 
            side='right', 
            type="linear", 
            range=[0, np.max(b)]),
        legend=dict(x=0.6, y=0.9)
    )

    fig=go.Figure(data=[trace1_p, trace1_m, trace2, trace3], layout=layout)

    print(f"1.5*(time_range[0] - 1) {1.5*(time_range[0] - 1)} 0.7*(time_range[1] + 1) {0.7*(time_range[1] + 1)}")

    fig.add_annotation(
        xref="x",
        yref="y",
        x=np.log10(1.5*(time_range[0] - 1)),
        y=np.log10(1.2*n.min()),
        text="Coherent",
        axref="x",
        ayref="y",
        ax=np.log10(3*(time_range[0] - 1)),
        ay=np.log10(1.2*n.min()),
        arrowhead=5,
    )
 
    fig.add_annotation(
        xref="x",
        yref="y",
        x=np.log10(0.8*(time_range[1] + 1)),
        y=np.log10(1.2*n.min()),
        text="Adiabatic",
        axref="x",
        ayref="y",
        ax=np.log10(0.4*(time_range[1] + 1)),
        ay=np.log10(1.2*n.min()),
        arrowhead=5,
    )

    return fig

def plot_kink_density(fig_dict, kink_density, anneal_time):
    """"""

    fig=go.Figure(fig_dict)

    return fig.add_trace(
        go.Scatter(
            x=[anneal_time], 
            y=[kink_density], 
            xaxis="x1",
            yaxis="y1",
            showlegend=False,
            marker=dict(size=10, 
                        color="black",
                        symbol="x")
        )
    )


def plot_spin_orientation(num_spins=512, sample=None):
    """"""

    fig = go.Figure()       # Erase previous plot

    conesize = num_spins/20

    #sample = np.random.choice([-1, 1], size=num_spins)

    z = np.linspace(0, 10, num_spins)
    x, y = z*np.cos(5*z), z*np.sin(5*z)

    if sample is None:

        cones_red = cones_blue = np.ones(num_spins, dtype=bool)
        num_cones_red = num_cones_blue = num_spins

    else:

        cones_red = ~np.isnan(np.where(sample==1, z, np.nan))
        cones_blue = ~cones_red
        num_cones_red = np.count_nonzero(cones_red)
        num_cones_blue = num_spins - num_cones_red
     
    trace_red = go.Cone(
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
        sizeref=conesize
    )

    trace_blue = go.Cone(
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
        sizeref=conesize
    )

    fig = go.Figure(
        data=[trace_red, trace_blue],
        layout=go.Layout(
            title=f'Spin States of {num_spins} Qubits in a 1D Ring',
            title_font_color="rgb(243, 120, 32)",
            showlegend=False,
            margin=dict(b=0,l=0,r=0,t=60),
            scene=dict(
                xaxis=dict(showticklabels=False, visible=False),
                yaxis=dict(showticklabels=False, visible=False),
                zaxis=dict(showticklabels=False, visible=False),
            camera_eye=dict(x=1, y=1, z=0.5)
            )
        )
    )

    fig.add_layout_image(
    dict(
        source="assets/spin_states.png",
        xref="paper", yref="paper",
        x=0.05, y=0.05,
        sizex=0.3, sizey=0.3,
        xanchor="left", yanchor="bottom"
    )
)

    return fig
