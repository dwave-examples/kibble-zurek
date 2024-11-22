# Copyright 2024 D-Wave
#
#    Licensed under the A_ghzpache License, Version 2.0 (the "License");
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

from dash import no_update
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from helpers.kz_calcs import theoretical_kink_density

__all__ = ['plot_kink_densities_bg', 'plot_kink_density', 'plot_spin_orientation', ]

def plot_kink_densities_bg(display, time_range, coupling_strength, schedule_name):
    """
    Plot background of theoretical kink-density and QPU energy scales. 

    Args:

        display: Displays plots of type "both", "kink_density", or "schedule". 

        time_range: Maximum and minimum quench times, as a list.

        coupling_strength: Coupling strength between spins in ring.

        schedule_name: Filename of anneal schedule.
    
    Returns:
        Plotly figure of predicted kink densities and/or QPU energy scales.
    """
    if schedule_name:
        schedule = pd.read_csv(f'helpers/{schedule_name}')
    else:
        schedule = pd.read_csv('helpers/FALLBACK_SCHEDULE.csv')

    A_ghz = schedule['A(s) (GHz)']
    B_ghz = schedule['B(s) (GHz)']    
    s = schedule['s']     

    # Display in Joule
    A_joule = A_ghz/1.5092E24     
    B_joule = B_ghz/1.5092E24

    n = theoretical_kink_density(time_range, coupling_strength, schedule, schedule_name)
    
    predicted_plus = go.Scatter(
        x=np.asarray(time_range), 
        y=np.asarray(1.1 * n),
        mode='lines',
        name='<b>Predicted (&plusmn;10%)',
        xaxis='x1',
        yaxis='y1',
        line_color='black', 
        line_width=1,
    )
    
    predicted_minus = go.Scatter(
        x=np.asarray(time_range), 
        y=np.asarray(0.90 * n),
        mode='lines',
        xaxis='x1',
        yaxis='y1',
        line_color='black', 
        line_width=1,
        fill='tonexty',
        fillcolor='white',
        showlegend=False,
    )
    
    x_axis = 'x2'
    y_axis = 'y2'
    opacity = 0.15
    if display == 'schedule':
        x_axis = 'x1'
        y_axis = 'y1'
        opacity = 1

    energy_transverse = go.Scatter(
        x=s,     
        y=A_joule, 
        mode='lines',
        name='A(s)', 
        xaxis=x_axis,
        yaxis=y_axis,
        line_color='blue',
        opacity=opacity,
    )

    energy_problem = go.Scatter(
        x=s,      
        y=abs(coupling_strength) * B_joule, 
        mode='lines',
        name='B(s)', 
        xaxis=x_axis,
        yaxis=y_axis,
        line_color='red',
        opacity=opacity,
    )

    x_axis1 = dict(
        title='<b>Quench Duration [ns]<b>', 
        type='log', 
        range=[np.log10(time_range[0] - 1), np.log10(time_range[1] + 10)],  
    )

    y_axis1 = dict(
        title='<b>Kink Density<b>', 
        type='log',
    )

    x_axis2 = dict(
        title={
            'text': 'Normalized Fast-Anneal Fraction, s', 
            'standoff': 0,
        }, 
        side='top' if display != 'schedule' else 'bottom', 
        type='log' if display != 'schedule' else 'linear', 
        range=[-1, 0] if display != 'schedule' else [0, 1],  # Minimal s=0.1 for log seems reasonable 
    )
    
    y_axis2 = dict(
        title='Energy [Joule]',  
        side='right' if display != 'schedule' else 'left', 
        type='linear', 
    )

    x_axis3 = dict(
        title='<b>Coupling Strength<b>'
    )
    if display == 'kink_density':

        fig_layout = go.Layout(
            xaxis=x_axis1,
            yaxis=y_axis1,
        )

        fig_data = [predicted_plus, predicted_minus]

    elif display == 'schedule':

        fig_layout = go.Layout(
            xaxis=x_axis2,
            yaxis=y_axis2,
        )

        fig_data = [energy_transverse, energy_problem]
    elif display == 'coupling':

        fig_layout = go.Layout(
            xaxis=x_axis3,
            yaxis=y_axis1,
        )

        fig_data = [predicted_plus, predicted_minus]
    else:   # Display both plots together

        x_axis2.update({'overlaying': 'x1'})
        y_axis2.update({'overlaying': 'y1'})

        fig_layout = go.Layout(
            xaxis=x_axis1,
            yaxis=y_axis1,
            xaxis2=x_axis2,
            yaxis2=y_axis2,
        )

        fig_data = [predicted_plus, predicted_minus, energy_transverse, energy_problem]

    fig=go.Figure(
        data=fig_data, 
        layout=fig_layout
    )

    fig.update_layout(
        legend=dict(x=0.7, y=0.9),  
        margin=dict(b=5,l=5,r=20,t=10)  
    )

    if display != 'schedule':

        fig.add_annotation(
            xref='x',
            yref='y',
            x=np.log10(0.25*(time_range[1])),
            y=np.log10(1.0*n.min()),
            text='Coherent',
            axref='x',
            ayref='y',
            ax=np.log10(0.50*(time_range[1])),
            ay=np.log10(1.0*n.min()),
            arrowhead=5,
        )
    
        fig.add_annotation(
            xref='x',
            yref='y',
            x=np.log10(0.5*(time_range[1])),
            y=np.log10(1.2*n.min()),
            text='Thermalized',
            axref='x',
            ayref='y',
            ax=np.log10(0.3*(time_range[1])),
            ay=np.log10(1.2*n.min()),
            arrowhead=5,
        )

    return fig

def plot_kink_density(display, fig_dict, kink_density, anneal_time):
    """Add kink density from QPU samples to plot.

    Args:

        display: Displays plots of type "both", "kink_density", or "schedule". 

        fig_dict: Existing background Plotly figure, as a dict.

        kink_density: Calculated kink density derived from QPU sample set.

        anneal_time: Anneal time used for the kink density.
    
    Returns:
        Updated Plotly figure with a marker at (anneal time, kink-density).
    """
    
    if display == 'schedule':
        return no_update
    
    fig=go.Figure(
        fig_dict
    )

    fig.add_trace(
        go.Scatter(
            x=[anneal_time], 
            y=[kink_density], 
            xaxis='x1',
            yaxis='y1',
            showlegend=False,
            marker=dict(size=10, 
                        color='black',
                        symbol='x',
            )
        )
    )

    return fig

def plot_spin_orientation(num_spins=512, sample=None):
    """Plot the ring of spins. 

    Args:
        num_spins: Number of spins in the ring.

        sample: Single sample from a sample set.

    Returns:
        Plotly figure of orientation for all spins in the ring.
    """
    
    cone_size = 0.5 # Based on how it looks

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
        sizemode='raw',
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
        sizemode='raw',
        sizeref=cone_size
    )

    fig = go.Figure(
        data=[spins_up, spins_down],
        layout=go.Layout(
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
            source='assets/spin_states.png',
            xref='paper', 
            yref='paper',
            x=0.95, 
            y=0.05,
            sizex=0.4, 
            sizey=0.4,
            xanchor='right', 
            yanchor='bottom',
        )
    )

    return fig
