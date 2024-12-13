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
from helpers.qa import fitted_function

from helpers.kz_calcs import theoretical_kink_density

__all__ = [
    "plot_kink_densities_bg",
    "plot_kink_density",
    "plot_spin_orientation",
    "plot_zne_fitted_line",
]

ta_color_theme = {
    5: "#1F77B4",  # Dark Blue
    10: "#FF7F0E",  # Dark Orange
    20: "#2CA02C",  # Dark Green
    40: "#D62728",  # Dark Red
    80: "#9467BD",  # Dark Purple
    160: "#8C564B",  # Brown
    320: "#E377C2",  # Dark Pink
    640: "#17BECF",  # Teal
    1280: "#BCBD22",  # Olive Green
}
coupling_color_theme = {
    -1.8: "#1F77B4",  # Dark Blue
    -1.6: "#FF7F0E",  # Dark Orange
    -1.4: "#E377C2",  # Dark Pink
    -1.2: "#2CA02C",  # Dark Green
    -1: "#D62728",  # Dark Red
    -0.8: "#9467BD",  # Dark Purple
    -0.6: "#8C564B",  # Brown
}
coupling_label = {
    -1.8: False,
    -1.6: False,
    -1.4: False,
    -1.2: False,
    -1: False,
    -0.8: False,
    -0.6: False,
}


def plot_kink_densities_bg(
    display, time_range, J_base, schedule_name, coupling_data, zne_estimates, kz_data=None, url=None
):
    """
    Plot the background of theoretical kink density and QPU energy scales.

    This function generates a Plotly figure that displays the theoretical
    predictions for kink densities along with QPU energy scales based on
    the provided anneal schedule. It supports different display modes
    such as "both", "kink_density", "schedule", and "coupling".

    Args:
        display (str): The type of plot to display. Options are:
            - "both": Display both kink density and schedule.
            - "kink_density": Display only the kink density plot.
            - "schedule": Display only the anneal schedule.
            - "coupling": Display coupling-related plots.
        time_range (list of float): A list containing the minimum and maximum
            quench times [min_quench_time, max_quench_time] in nanoseconds.
        J_base (float): The base coupling strength between spins in the ring.
        schedule_name (str): The filename of the anneal schedule CSV file.
            If not provided, a fallback schedule is used.
        coupling_data (dict): A dictionary containing coupling-related data
            structured as {ta_str: [data_points]}, where each data point
            includes "coupling_strength" and "kink_density".
        zne_estimates (dict): A dictionary to store Zero-Noise Extrapolation
            (ZNE) estimates structured as {ta_str: estimate}.

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object containing the
        predicted kink densities and/or QPU energy scales based on the
        specified display mode.
    """
    if schedule_name:
        schedule = pd.read_csv(f"helpers/{schedule_name}")
    else:
        schedule = pd.read_csv("helpers/FALLBACK_SCHEDULE.csv")

    A_ghz = schedule["A(s) (GHz)"]
    B_ghz = schedule["B(s) (GHz)"]
    s = schedule["s"]

    # Display in Joule
    A_joule = A_ghz / 1.5092e24
    B_joule = B_ghz / 1.5092e24

    n = theoretical_kink_density(time_range, J_base, schedule, schedule_name)

    predicted_plus = go.Scatter(
        x=np.asarray(time_range),
        y=np.asarray(1.1 * n),
        mode="lines",
        name="<b>Predicted (&plusmn;10%)",
        xaxis="x1",
        yaxis="y1",
        line_color="black",
        line_width=1,
    )

    predicted_minus = go.Scatter(
        x=np.asarray(time_range),
        y=np.asarray(0.90 * n),
        mode="lines",
        xaxis="x1",
        yaxis="y1",
        line_color="black",
        line_width=1,
        fill="tonexty",
        fillcolor="white",
        showlegend=False,
    )

    x_axis = "x2"
    y_axis = "y2"
    opacity = 0.15
    if display == "schedule":
        x_axis = "x1"
        y_axis = "y1"
        opacity = 1

    energy_transverse = go.Scatter(
        x=s,
        y=A_joule,
        mode="lines",
        name="A(s)",
        xaxis=x_axis,
        yaxis=y_axis,
        line_color="blue",
        opacity=opacity,
    )

    energy_problem = go.Scatter(
        x=s,
        y=abs(J_base) * B_joule,
        mode="lines",
        name="B(s)",
        xaxis=x_axis,
        yaxis=y_axis,
        line_color="red",
        opacity=opacity,
    )

    x_axis1 = dict(
        title="<b>Quench Duration [ns]<b>",
        type="log",
        range=[np.log10(time_range[0] - 1), np.log10(time_range[1] + 10)],
    )

    y_min = (0.9 * n).min()
    y_max = (1.1 * n).max()

    y_axis1 = dict(
        title="<b>Kink Density<b>",
        type="log",
        range=[np.log10(y_min), np.log10(y_max)],
    )

    x_axis2 = dict(
        title={
            "text": "Normalized Fast-Anneal Fraction, s",
            "standoff": 0,
        },
        side="top" if display != "schedule" else "bottom",
        type="log" if display != "schedule" else "linear",
        range=(
            [-1, 0] if display != "schedule" else [0, 1]
        ),  # Minimal s=0.1 for log seems reasonable
    )

    y_axis2 = dict(
        title="Energy [Joule]",
        side="right" if display != "schedule" else "left",
        type="linear",
    )

    x_axis3 = dict(title="<b>Noise Level (-1.8/J)<b>", type="linear", range=[0, 3])

    if display == "kink_density":
        fig_layout = go.Layout(
            xaxis=x_axis1,
            yaxis=y_axis1,
        )
        if url == 'Demo2':
            _coupling_label = {
                -1.8: False,
                -1.6: False,
                -1.4: False,
                -1.2: False,
                -1: False,
                -0.8: False,
                -0.6: False,
            }
            fig_data = [predicted_plus, predicted_minus]
            for ta_str, data_points in coupling_data.items():
                for point in data_points:
                    _J = point["coupling_strength"]
                    color = coupling_color_theme[_J]

                    if not _coupling_label[_J]:
                        legend = True
                        _coupling_label[_J] = True
                    else:
                        legend = False

                    kink_density = point["kink_density"]

                    fig_data.append(
                        go.Scatter(
                            x=[ta_str],
                            y=[kink_density],
                            xaxis="x1",
                            yaxis="y1",
                            mode="markers",
                            name=f"Coupling Strength: {_J}",
                            showlegend=legend,
                            marker=dict(size=10, color=color, symbol="x"),
                        )
                    )
                # Plot ZNE estimates
                for ta_str, a in zne_estimates.items():
                    fig_data.append(
                        go.Scatter(
                            x=[ta_str],
                            y=[a],
                            mode="markers",
                            name="ZNE Estimate",
                            marker=dict(size=12, color="purple", symbol="diamond"),
                            showlegend=False,
                            xaxis="x1",
                            yaxis="y1",
                        )
                    )
        else:
            fig_data = [predicted_plus, predicted_minus]
            
    elif display == "schedule":

        fig_layout = go.Layout(
            xaxis=x_axis2,
            yaxis=y_axis2,
        )

        fig_data = [energy_transverse, energy_problem]
    elif display == "coupling":

        fig_layout = go.Layout(
            xaxis3=x_axis3,
            yaxis1=y_axis1,
        )

        fig_data = []

        # Plot data points from 'coupling_data'
        for ta_str, data_points in coupling_data.items():
            label = False
            ta_value = float(ta_str)
            color = ta_color_theme[ta_value]
            for point in data_points:
                kappa = point["kappa"]
                kink_density = point["kink_density"]
                if not label:
                    fig_data.append(
                        go.Scatter(
                            x=[kappa],
                            y=[kink_density],
                            xaxis="x3",
                            yaxis="y1",
                            mode="markers",
                            name=f"Anneal Time: {ta_value} ns",
                            showlegend=True,
                            marker=dict(size=10, color=color, symbol="x"),
                        )
                    )
                    label = True
                else:
                    fig_data.append(
                        go.Scatter(
                            x=[kappa],
                            y=[kink_density],
                            xaxis="x3",
                            yaxis="y1",
                            showlegend=False,
                            marker=dict(size=10, color=color, symbol="x"),
                        )
                    )

    else:  # Display both plots together

        x_axis2.update({"overlaying": "x1"})
        y_axis2.update({"overlaying": "y1"})

        fig_layout = go.Layout(
            xaxis=x_axis1,
            yaxis=y_axis1,
            xaxis2=x_axis2,
            yaxis2=y_axis2,
        )

        fig_data = [predicted_plus, predicted_minus, energy_transverse, energy_problem]
        # for ta_str, data_points in coupling_data.items():
        #     for point in data_points:
        #         color = "black"
        #         kink_density = point["kink_density"]
        #         fig_data.append(
        #             go.Scatter(
        #                 x=[ta_str],
        #                 y=[kink_density],
        #                 xaxis="x1",
        #                 yaxis="y1",
        #                 showlegend=False,
        #                 marker=dict(size=10, color=color, symbol="x"),
        #             )
        #         )
            # Plot ZNE estimates
            # for ta_str, a in zne_estimates.items():
            #     fig_data.append(
            #         go.Scatter(
            #             x=[ta_str],
            #             y=[a],
            #             mode="markers",
            #             name="ZNE Estimate",
            #             marker=dict(size=12, color="purple", symbol="diamond"),
            #             showlegend=False,
            #             xaxis="x1",
            #             yaxis="y1",
            #         )
            #     )
        # Add previously computed kz_data points
        if "k" in kz_data:
            for pair in kz_data["k"]:
                fig_data.append(
                    go.Scatter(
                        x=[pair[1]],       
                        y=[pair[0]],
                        mode="markers",
                        marker=dict(size=10, color="black", symbol="x"),
                        xaxis="x1",
                        yaxis="y1",
                        showlegend=False
                    )
                )
    fig = go.Figure(data=fig_data, layout=fig_layout)

    fig.update_layout(legend=dict(x=0.1, y=0.1), margin=dict(b=5, l=5, r=20, t=10))

    if display != "schedule" and display != "coupling":

        fig.add_annotation(
            xref="x",
            yref="y",
            x=np.log10(0.25 * (time_range[1])),
            y=np.log10(1.0 * n.min()),
            text="Coherent",
            axref="x",
            ayref="y",
            ax=np.log10(0.50 * (time_range[1])),
            ay=np.log10(1.0 * n.min()),
            arrowhead=5,
        )

        fig.add_annotation(
            xref="x",
            yref="y",
            x=np.log10(0.5 * (time_range[1])),
            y=np.log10(1.2 * n.min()),
            text="Thermalized",
            axref="x",
            ayref="y",
            ax=np.log10(0.3 * (time_range[1])),
            ay=np.log10(1.2 * n.min()),
            arrowhead=5,
        )

    return fig


def plot_kink_density(display, fig_dict, kink_density, anneal_time, J, url=None):
    """
    Add a kink density marker from QPU samples to an existing plot.

    Depending on the display mode, this function updates the provided
    Plotly figure with a new marker representing the calculated kink
    density at a specific anneal time and coupling strength.

    Args:
        display (str): The type of plot to display. Options are:
            - "both": Display both kink density and schedule.
            - "kink_density": Display only the kink density plot.
            - "schedule": Display only the anneal schedule.
            - "coupling": Display coupling-related plots.
        fig_dict (dict): The existing background Plotly figure as a dictionary.
        kink_density (float): The calculated kink density derived from QPU samples.
        anneal_time (str or float): The anneal time corresponding to the kink density.
            It can be a string identifier or a numerical value in nanoseconds.
        J (float): The coupling strength associated with the kink density.

    Returns:
        plotly.graph_objs.Figure or dash.no_update:
            - If display is "schedule", returns `no_update` indicating no changes.
            - Otherwise, returns the updated Plotly figure with the new kink density marker.
    """
    if display == "schedule":
        return no_update

    fig = go.Figure(fig_dict)

    if url == "Demo1":
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
    
    ta_value = float(anneal_time)

    if display == "coupling":
        color = ta_color_theme[ta_value]
        kappa = -1.8 / J
        fig.add_trace(
            go.Scatter(
                x=[kappa],
                y=[kink_density],
                xaxis="x3",
                yaxis="y1",
                showlegend=False,
                marker=dict(
                    size=10,
                    color=color,
                    symbol="x",
                ),
            )
        )
        fig.update_layout(
            xaxis3=fig.layout.xaxis3,
            yaxis1=fig.layout.yaxis1,
        )
        return fig

    if display == "kink_density":
        color = coupling_color_theme[J]
    else:
        color = "black"

    if not coupling_label[J]:
        legend = True
        coupling_label[J] = True
    else:
        legend = False
    
    fig.add_trace(
        go.Scatter(
            x=[anneal_time],
            y=[kink_density],
            xaxis="x1",
            yaxis="y1",
            mode="markers",
            name=f"Coupling Strength: {J}",
            showlegend=legend,
            marker=dict(
                size=10,
                color=color,
                symbol="x",
            ),
        )
    )

    return fig


def plot_spin_orientation(num_spins=512, sample=None):
    """
    Visualize the orientation of spins in a ring using 3D cones.

    This function generates a 3D Plotly figure representing the orientation
    of spins arranged in a ring. Each spin is depicted as a cone pointing
    upwards or downwards based on the provided sample. If no sample is
    provided, all spins are shown pointing upwards by default.

    Args:
        num_spins (int, optional): The total number of spins in the ring.
            Defaults to 512.
        sample (array-like, optional): A single sample from a sample set
            indicating spin orientations. Each element should be:
            - `1` for spin up.
            - `-1` or any other value for spin down.
            If `None`, all spins are assumed to be up.

    Returns:
        plotly.graph_objs.Figure: A 3D Plotly figure displaying the spin orientations.
    """

    cone_size = 0.5  # Based on how it looks

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
        x=x[cones_red],
        y=y[cones_red],
        z=z[cones_red],
        u=num_cones_red * [0],
        v=num_cones_red * [0],
        w=num_cones_red * [1],
        showlegend=False,
        showscale=False,
        colorscale=[[0, "red"], [1, "red"]],
        hoverinfo=None,
        sizemode="raw",
        sizeref=cone_size,
    )

    spins_down = go.Cone(
        x=x[cones_blue],
        y=y[cones_blue],
        z=z[cones_blue],
        u=num_cones_blue * [0],
        v=num_cones_blue * [0],
        w=num_cones_blue * [-1],
        showlegend=False,
        showscale=False,
        colorscale=[[0, "blue"], [1, "blue"]],
        hoverinfo=None,
        sizemode="raw",
        sizeref=cone_size,
    )

    fig = go.Figure(
        data=[spins_up, spins_down],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=40),
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
                camera_eye=dict(x=0.15, y=1.25, z=0.15),
            ),
        ),
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


def plot_zne_fitted_line(
    fig, coupling_data, qpu_name, zne_estimates, zne_graph_display, ta_str
):
    """
    Fit a curve to the coupling data and plot the Zero-Noise Extrapolation (ZNE) estimate.

    This function performs curve fitting on the provided coupling data for a
    specific anneal time and adds the fitted curve along with the ZNE estimate
    to the existing Plotly figure. The fitting method varies based on the QPU
    used. It also handles the removal of any existing fitting curves to avoid
    duplication.

    Args:
        fig (plotly.graph_objs.Figure): The existing Plotly figure to update.
        coupling_data (dict): A dictionary containing coupling-related data
            structured as {ta_str: [data_points]}, where each data point
            includes "kappa" and "kink_density".
        qpu_name (str): The name of the Quantum Processing Unit (QPU) used.
            Determines the fitting method.
        zne_estimates (dict): A dictionary to store Zero-Noise Extrapolation
            (ZNE) estimates structured as {ta_str: estimate}.
        kz_graph_display (str): The type of graph display. Typically aligns with
            the `display` parameter in other functions, such as "coupling".
        ta_str (str): The anneal time identifier as a string.

    Returns:
        tuple:
            - zne_estimates (dict): Updated dictionary with new ZNE estimates.
            - modal_trigger (bool): A flag indicating whether a modal was triggered
              due to ill conditioned data for fitting.
    """
    modal_trigger = False
    if len(coupling_data[ta_str]) > 2:

        data_points = coupling_data[ta_str]
        x = np.array([point["kappa"] for point in data_points])
        y = np.array([point["kink_density"] for point in data_points])

        # Ensure there are enough unique x values for fitting
        if len(np.unique(x)) > 1:
            # Fit a 1st degree polynomial (linear fit)
            if qpu_name == "Diffusion [Classical]":
                # Fancy non-linear function
                y_func_x = fitted_function(x, y, method="mixture_of_exponentials")
            else:
                # Pure quadratic (see paper) # y = a + b x^2
                y_func_x = fitted_function(x, y, method="pure_quadratic")

            if y_func_x is not None:
                zne_estimates[ta_str] = y_func_x(0)
                x_fit = np.linspace(0, max(x), 100)
                y_fit = y_func_x(x_fit)
            else:
                modal_trigger = True
            # Remove existing fitting curve traces to prevent duplication
            fig.data = [
                trace
                for trace in fig.data
                if not (
                    trace.name == "Fitting Curve"
                    and trace.legendgroup == f"ta_{ta_str}"
                )
            ]
            # Remove existing ZNE Estimate traces to prevent duplication
            fig.data = [
                trace
                for trace in fig.data
                if not (
                    trace.name == "ZNE Estimate" and trace.legendgroup == f"ta_{ta_str}"
                )
            ]

            if zne_graph_display == "coupling" and y_func_x is not None:
                x_axis = "x3"
                y_axis = "y1"
                x_zne = 0
                # Add the new fitting curve
                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        name="Fitting Curve",
                        legendgroup=f"ta_{ta_str}",
                        line=dict(color="green", dash="dash"),
                        showlegend=True,
                        xaxis=x_axis,
                        yaxis=y_axis,
                    )
                )
            else:
                x_axis = "x1"
                y_axis = "y1"
                x_zne = float(ta_str)
            # for ta_str, a in zne_estimates.items():
            if y_func_x is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[x_zne],
                        y=[zne_estimates[ta_str]],
                        mode="markers",
                        name="ZNE Estimate",
                        legendgroup=f"ta_{ta_str}",
                        marker=dict(size=12, color="purple", symbol="diamond"),
                        showlegend=False,
                        xaxis=x_axis,
                        yaxis=y_axis,
                    )
                )
    return zne_estimates, modal_trigger
