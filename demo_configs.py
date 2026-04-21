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

THUMBNAIL = "static/dwave_logo.svg"

APP_TITLE = "Coherent Annealing"
MAIN_HEADER = "Kibble-Zurek Simulation"
DESCRIPTION = """\
Use a quantum computer to simulate the formation of topological defects in a 1D ring 
of spins undergoing a phase transition, described by the Kibble-Zurek mechanism.
"""

# config settings for ZNE tab
MAIN_HEADER_NM = "Zero-Noise Extrapolation"
DESCRIPTION_NM = """
    Statistics of a target J=-1.8 chain can be inferred by running with weaker
    coupling and longer quench duration, however, these runs are subject to more noise.
    Extrapolation to a denoised result is possible by collecting data at several noise levels.
    """

J_BASELINE = -1.8
J_OPTIONS = [-1.8, -1.6, -1.4, -1.2, -1, -0.9, -0.8, -0.7]

DEFAULT_QPU = "Advantage2_system1"  # If not available, the first returned will be default

SHOW_TOOLTIPS = True  # Determines whether tooltips are on or off

RING_LENGTHS = [512, 1024, 2048]

JOB_BAR_DISPLAY = {
    "READY": [0, "#737373"],
    "EMBEDDING": [20, "#FF7006"],
    "NO SOLVER": [100, "#AA3A3C"],
    "SUBMITTED": [40, "#03B8FF"],
    "PENDING": [60, "#2A7DE1"],
    "IN_PROGRESS": [85, "#2A7DE1"],
    "COMPLETED": [100, "#17BEBB"],
    "CANCELLED": [100, "#737373"],
    "FAILED": [100, "#AA3A3C"],
}

TOOL_TIPS_KZ_NM = {
    "coupling-strength-wrapper": "Coupling strength between spins in the ferromagnetic ring.",
    "qpu_selection": "Quantum computers available to your account/project token.",
    "quench_schedule_filename": """The fast-anneal schedule for the selected quantum computer.
        If none exists, one from a different quantum computer is used (expect inaccuracies).""",
}

TOOL_TIPS_KZ = {
    "coupling-strength-wrapper": """Coupling strength between spins in the ring.
        Range of -2 (ferromagnetic) to +1 (anti-ferromagnetic).""",
    "qpu_selection": "Quantum computers available to your account/project token.",
    "quench_schedule_filename": """The fast-anneal schedule for the selected quantum computer.
        If none exists, one from a different quantum computer is used (expect inaccuracies).""",
}
