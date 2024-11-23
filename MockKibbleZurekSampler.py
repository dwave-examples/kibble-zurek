from dimod import SampleSet
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.temperatures import fluxbias_to_h
from dwave.system.testing import MockDWaveSampler


class MockKibbleZurekSampler(MockDWaveSampler):
    def __init__(
        self,
        nodelist=None,
        edgelist=None,
        properties=None,
        broken_nodes=None,
        broken_edges=None,
        topology_type='pegasus',
        topology_shape=[16],
        parameter_warnings=True,
        substitute_sampler=None,
        substitute_kwargs=None,
        exact_solver_cutoff=0,
    ):
        if substitute_sampler is None:
            self.substitute_sampler = SimulatedAnnealingSampler()
        if substitute_kwargs is None:
            self.substitute_kwargs = {'beta_range': [0.01, 100],
                                      'beta_schedule_type': 'geometric',
                                      'num_sweeps': 1,
                                      'randomize_order': True,
                                      'proposal_acceptance_criteria': 'Gibbs'}
        super().__init__(
            nodelist=nodelist,
            edgelist=edgelist,
            properties=properties,
            broken_nodes=broken_nodes,
            broken_edges=broken_edges,
            topology_type=topology_type,
            topology_shape=topology_shape,
            parameter_warnings=parameter_warnings,
            substitute_sampler=substitute_sampler,
            substitute_kwargs=substitute_kwargs,
            exact_solver_cutoff=exact_solver_cutoff,
        )
        self.sampler_type = 'mock'
        self.mocked_parameters.add('annealing_time')
        self.parameters.update({'num_sweeps': []})
        
    def sample(self, bqm, **kwargs):
        _bqm = bqm.change_vartype('SPIN', inplace=False)
        
        # Extract annealing_time from kwargs (if provided)
        annealing_time = kwargs.pop('annealing_time', 20)  # 20us default.
        num_sweeps = int(annealing_time * 1000//5)  # 1000 sweeps per microsecond
        # Extract flux biases from kwargs (if provided)
        # flux_biases = kwargs.pop('flux_biases', {})
        # flux_to_h_factor = fluxbias_to_h()
        # for v in _bqm.variables:
        #     bias = _bqm.get_linear(v)
        #     _bqm.set_linear(v, bias + flux_to_h_factor * flux_biases[v])

        # TO DO: corrupt bqm with noise proportional to annealing_time
        
        ss = super().sample(bqm=_bqm, num_sweeps=num_sweeps, **kwargs)

        ss.change_vartype(bqm.vartype)

        ss = SampleSet.from_samples_bqm(ss, bqm)

        return ss

    def get_sampler(self):
        """
        Return the sampler instance.
        """
        return self
