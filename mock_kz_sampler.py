# Copyright 2025 D-Wave
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

import numpy as np

from dimod import SampleSet
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler


class MockKibbleZurekSampler(MockDWaveSampler):
    """Perform a quench (fixed beta = 1/temperature) evolution.

    The MockSampler is configured to use standard Markov Chain Monte Carlo
    with Gibbs acceptance criteria from a random initial condition.
    Defects diffuse (power law 1/2) and eliminate, but are also
    created by thermal excitations. We will seek to take a limit of high
    coupling strength where thermal excitations are removed, leaving only the
    diffusion.
    """

    def __init__(
        self,
        topology_type="pegasus",
        topology_shape=[16],
        kink_density_limit_absJ1=0.04,
    ):
        substitute_sampler = SimulatedAnnealingSampler()
        # At equilibrium <xi xj> = (t^{L-1} + t)/(1 + t^L), t = -tanh(beta J)
        # At large time (equilibrium) for long chains
        # <x_i x_{i+1}> lessthansimilarto t,
        # At J=-1 we want a kink density to bottom out. Therefore:
        beta = np.arctanh(1 - 2 * kink_density_limit_absJ1)
        substitute_kwargs = {
            "beta_range": [beta, beta],  # Quench
            "randomize_order": True,
            "num_reads": 1000,
            "proposal_acceptance_criteria": "Gibbs",
        }
        super().__init__(
            topology_type=topology_type,
            topology_shape=topology_shape,
            substitute_sampler=substitute_sampler,
            substitute_kwargs=substitute_kwargs,
        )
        self.sampler_type = "mock"
        self.mocked_parameters.add("annealing_time")
        self.mocked_parameters.add("num_sweeps")
        self.parameters.update({"num_sweeps": []})

    def sample(self, bqm, **kwargs):
        # TODO: corrupt bqsm with noise proportional to annealing_time
        _bqm = bqm.change_vartype("SPIN", inplace=False)

        # Extract annealing_time from kwargs (if provided)
        annealing_time = kwargs.pop("annealing_time", 20)  # 20us default.
        num_sweeps = int(annealing_time * 1000)  # 1000 sweeps per microsecond

        ss = super().sample(bqm=_bqm, num_sweeps=num_sweeps, **kwargs)

        ss.change_vartype(bqm.vartype)  # Not required but safe

        ss = SampleSet.from_samples_bqm(ss, bqm)

        return ss

    def get_sampler(self):
        """
        Return the sampler instance.
        """
        return self
