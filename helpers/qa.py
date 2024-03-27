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

import dimod
from dwave.cloud.api import exceptions, Problems
from dwave.embedding import unembed_sampleset
import minorminer

from helpers.cached_embeddings import cached_embeddings

__all__ = ["create_bqm", "find_one_to_one_embedding", "get_job_status", "get_samples"]

def create_bqm(num_spins=512, coupling_strength=-1.4):
    """
    Create a binary quadratic model (BQM) representing a ferromagnetic 1D ring. 

    Args:
        num_spins: Number of spins, which is the length of the ring.

        coupling_strength: Value of J.

    Returns:
        dimod BQM.  
    """
    bqm = dimod.BinaryQuadraticModel(vartype='SPIN')

    for spin in range(num_spins):
        bqm.add_quadratic(spin, (spin + 1) % num_spins, coupling_strength)
    
    return bqm

def find_one_to_one_embedding(ising_chain_length, sampler_edgelist):
    """
    Find an embedding with chains of length one for the spin ring. 

    Args:
        ising_chain_length: Length of ring, which is the number of spins. 

        sampler_edgelist: Edges of the QPU. 

    Returns:
        Embedding, as a dict of format ``{spin: [qubit]}``.  
    """
    bqm = create_bqm(ising_chain_length)

    for tries in range(3):

        print(f"Attempt {tries + 1} to find an embedding...")   # TODO: move this

        embedding = minorminer.find_embedding(bqm.quadratic, sampler_edgelist) 

        if max(len(val) for val in embedding.values()) == 1:
            return embedding
        
    raise ValueError("Failed to find a good embedding in 3 tries")  # TODO: terminate gracefully

def get_job_status(client, job_id, job_submit_time):
    """Return status of a submitted job.

    Args:
        client: dwave-cloud-client Client instance. 

        job_id: Identification string of the job. 

        job_submit_time: Clock time of submission for identification.

    Returns:
        Embedding, as a dict of format ``{spin: [qubit]}``.
    """

    p = Problems.from_config(client.config)

    try:

        status = p.get_problem_status(job_id)
        label_time = dict(status)["label"].split("submitted: ")[1]

        if label_time == job_submit_time:

            return status.status.value
        
        else:

            return None
    
    except exceptions.ResourceNotFoundError as err:

        return None

def get_samples(client, job_id, num_spins, J, qpu_name):
    """Retrieve an unembedded sampleset for a given job ID. 

    Args:
        client: dwave-cloud-client Client instance. 

        job_id: Identification string of the job. 

        num_spins: Number of spins, which is the length of the ring.

        coupling_strength: Value of J.

        qpu_name: Name of the quantum computer the job was submitted to. 

    Returns:
        Unembedded dimod sampleset. .
    """
    
    sampleset = client.retrieve_answer(job_id).sampleset
            
    bqm = create_bqm(num_spins=num_spins, coupling_strength=J)
    embedding = cached_embeddings[qpu_name][num_spins]
    
    return  unembed_sampleset(sampleset, embedding, bqm)
   