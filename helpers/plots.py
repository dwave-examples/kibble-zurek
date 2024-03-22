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

from helpers.kb_calcs import theoretical_kink_density

# Temporarily using the standard schedule
schedule = pd.read_csv('helpers/09-1302A-B_Advantage2_prototype2.2_annealing_schedule.csv')

A = schedule['A(s) (GHz)']
B = schedule['B(s) (GHz)']         
C = schedule['C (normalized)']

# Display in Joule
a = A/1.5092E24     
b = B/1.5092E24

__all__ = ["plot_kink_densities_bg"]

def plot_kink_densities_bg(time_range, coupling_strength):
    """
    Plot density based on theory and energy scales. 

    Args:
        time_range: max and min quench times

        coupling_strength: value of J

    """
    fig = go.Figure()

    n = theoretical_kink_density(time_range, coupling_strength)
    
    trace1 = go.Scatter(
            x=np.asarray(time_range), 
            y=np.asarray(n),
            mode='lines',
            name='Theory',
            yaxis="y1",
            line_color='lightgrey', 
            line_width=10)
    
    trace2 = go.Scatter(
        x=time_range[1]*C,   # C=1 --> MAX(t_a)     
        y=a, 
        mode='lines',
        name="A(C(s))", 
        yaxis='y2',
        line_color='blue',
        opacity=0.4)

    trace3 = go.Scatter(
        x=100*C,        
        y=abs(coupling_strength)*b, 
        mode='lines',
        name="B(C(s))", 
        yaxis='y2',
        line_color='red',
        opacity=0.4)

    layout = go.Layout(
        title='Kink Density: Theory Vs. QPU Simulation',
        xaxis=dict(title='Quench Time [ns]', type="log", range=[0, 2]),     # exponents for log
        yaxis=dict(title='Kink Density', type="log"),
        yaxis2=dict(title='Energy [Joule]',  overlaying='y', side='right', type="log", range=[-23, -25]),
        legend=dict(x=0, y=1)
    )

    fig=go.Figure(data=[trace1, trace2, trace3], layout=layout)
 
    return fig