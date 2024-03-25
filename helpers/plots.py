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
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from helpers.kb_calcs import avg_kink_density, theoretical_kink_density

__all__ = ["plot_kink_densities_bg", "plot_kink_density"]

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

    fig = go.Figure()

    n = theoretical_kink_density(time_range, coupling_strength, schedule_name)
    
    trace1_p = go.Scatter(
            x=np.asarray(time_range), 
            y=np.asarray(1.1 * n),
            mode='lines',
            name='Theory &plusmn;10%',
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
        title='Kink Density: Theory Vs. QPU Simulation',
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
            #range=[-26, -22]),
            range=[0, np.max(b)]),
        legend=dict(x=0.7, y=0.9)
    )

    fig=go.Figure(data=[trace1_p, trace1_m, trace2, trace3], layout=layout)
 
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
