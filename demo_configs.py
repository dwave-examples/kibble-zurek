# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file stores input parameters for the app."""

from dash import html

THUMBNAIL = "assets/dwave_logo.png"

DEBUG = False

APP_TITLE = "Coherent Annealing"
MAIN_HEADER = "Kibble-Zurek Simulation"
DESCRIPTION = """\
Use a quantum computer to simulate the formation of topological defects in a 1D ring 
of spins undergoing a phase transition, described by the Kibble-Zurek mechanism.
"""

# config settings for ZNE tab
MAIN_HEADER_NM = "Zero-Noise Extrapolation"
DESCRIPTION_NM = [
    "Statistics of a (target) J=-1.8 chain at quench duration t",
    html.Sub("target"),
    ", can be inferred by running at weaker coupling and longer quench duration (t",
    html.Sub("programmed"),
    """). Longer programmed times (at weaker
    coupling) are subject to more noise. When collecting data at several noise levels,
    an extrapolation to a denoised result is possible. At short target quench durations,
    there is weak environmental coupling and denoising has little impact. At long target
    quench durations, there is strong environmental coupling and denoising improves
    agreement with Kibble-Zurek theory.
    """
]

J_BASELINE = -1.8
J_OPTIONS = [-1.8, -1.6, -1.4, -1.2, -1, -0.9, -0.8, -0.7]

DEFAULT_QPU = "Advantage2_system1.3"  # If not available, the first returned will be default

SHOW_TOOLTIPS = False  # Determines whether tooltips are on or off
