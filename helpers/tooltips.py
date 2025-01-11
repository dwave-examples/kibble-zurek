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

tool_tips_demo2 = {
    "anneal_duration": f"""Duration of the quantum anneal. Range of 5 to 320 nanoseconds.""",
    "spins": f"""Number of spins in the 1D ring.""",
    "coupling_strength": f"""Coupling strength between spins in the ferromagnetic ring. 
Range of -1.8 to -0.6.
""",
    "qpu_selection": f"""Selection from quantum computers available to your account/project token.""",
    "quench_schedule_filename": f"""The fast-anneal schedule for the selected quantum computer.
If none exists, one from a different quantum computer is used (expect inaccuracies).
""",
}

tool_tips_demo1 = {
    "anneal_duration": f"""Duration of the quantum anneal. Range of 5 to 100 nanoseconds.""",
    "spins": f"""Number of spins in the 1D ring.""",
    "coupling_strength": f"""Coupling strength between spins in the ring. 
Range of -2 (ferromagnetic) to +1 (anti-ferromagnetic).
""",
    "qpu_selection": f"""Selection from quantum computers available to your account/project token.""",
    "quench_schedule_filename": f"""The fast-anneal schedule for the selected quantum computer.
If none exists, one from a different quantum computer is used (expect inaccuracies).
""",
}
