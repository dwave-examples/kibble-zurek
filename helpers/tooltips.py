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

tool_tips = {
    "anneal_duration":
f"""Duration of the quantum anneal. Range of 5 to 320 nanoseconds.""",
    "kz_graph_display":
f"""Plot selection: Defects vs anneal duration or defects vs noise level""",
    "spins":
f"""Number of spins in the 1D ring.""",
    "coupling_strength":
f"""Coupling strength, J, between spins in the ferromagnetic ring. 
Range of -1.8 to -0.6.
""",
    "qpu_selection":
f"""Selection from quantum computers available to your account/project token.""",
    "embedding_is_cached":
f"""Whether or not a minor-embedding is cached for the selected QPU, for each 
of the available number of spins. If not available, an attempt is made to find
an embedding the first time you submit a problem. 
""",
    "btn_simulate":
f"""Click to (minor-embed if a cached embedding is unavailable) and 
submit the problem to your selected QPU.
""",
    "quench_schedule_filename":
f"""CSV file with the fast-anneal schedule for the selected quantum computer.
If none exists, uses one from a different quantum computer (expect inaccuracies).
You can download schedules from
https://docs.dwavesys.com/docs/latest/doc_physical_properties.html
""",
    "job_submit_state":
f"""Status of the last submission to the quantum computer (or initial state).""",
    "btn_reset":
f"""Clear all existing data stored for the current run and reset all plots.""",
}