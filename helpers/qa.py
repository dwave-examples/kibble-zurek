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

import json

import dimod
import numpy as np
import scipy
from dwave.cloud.api import Problems, exceptions
from dwave.embedding import unembed_sampleset
from numpy.polynomial.polynomial import Polynomial
from minorminer.subgraph import find_subgraph

__all__ = [
    "create_bqm",
    "find_one_to_one_embedding",
    "get_job_status",
    "get_samples",
    "json_to_dict",
    "fitted_function",
]


def create_bqm(num_spins=512, coupling_strength=-1.4):
    """
    Create a binary quadratic model (BQM) representing a magnetic 1D ring.

    Args:
        num_spins: Number of spins in the ring.
        coupling_strength: Coupling strength between spins in the ring.

    Returns:
        dimod BQM.
    """
    bqm = dimod.BinaryQuadraticModel(vartype="SPIN")

    for spin in range(num_spins):
        bqm.add_quadratic(spin, (spin + 1) % num_spins, coupling_strength)

    return bqm


def find_one_to_one_embedding(spins, sampler_edgelist, timeout=60):
    """
    Find an embedding with chains of length one for the ring of spins.

    Args:
        spins: Number of spins.
        sampler_edgelist: Edges (couplers) of the QPU.
        timeout: Maximum time allowed for search.

    Returns:
        Embedding, as a dict of format {spin: [qubit]}.
    """
    ring_edges = {(i, (i+1) % spins) for i in range(spins)}
    emb_1to1 = find_subgraph(ring_edges, sampler_edgelist, timeout=timeout)

    return {k: (v,) for k, v in emb_1to1.items()}


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

        return None

    except exceptions.ResourceNotFoundError:
        return None


def get_samples(client, job_id, num_spins, J, embedding):
    """Retrieve an unembedded sample set for a given job ID.

    Args:
        client: dwave-cloud-client Client instance.
        job_id: Identification string of the job.
        num_spins: Number of spins in the ring.
        coupling_strength: Coupling strength between spins in the ring.
        qpu_name: Name of the quantum computer the job was submitted to.
        embedding: Embedding used for the job.

    Returns:
        Unembedded dimod sample set.
    """

    bqm = create_bqm(num_spins=num_spins, coupling_strength=J)

    if '"type": "SampleSet"' in job_id:  # See modifications to submit_job
        sampleset = dimod.SampleSet.from_serializable(json.loads(job_id))
    else:
        sampleset = client.retrieve_answer(job_id).sampleset

    return unembed_sampleset(sampleset, embedding, bqm)


def json_to_dict(emb_json):
    """Retrieve an unembedded sampleset for a given job ID.

    Args:
        emb_json: JSON-formatted dict of embeddings, as
            {'spins': {'node1': [qubit1], 'node2': [qubit2], ...}, ...}.

    Returns:
        Embedding in standard dict format.

    """

    return {
        int(key): {int(node): qubits for node, qubits in emb.items()}
        for key, emb in emb_json.items()
    }


def fitted_function(xdata, ydata, method="polynomial", degree=1):
    """
    Generate a fitting function based on the provided data points and method.

    Args:
        xdata: Array-like, independent variable data points.
        ydata: Array-like, dependent variable data points.
        method: A string specifying the fitting method. Options include:
            - "polynomial": Fits a polynomial of degree `degree`.
            - "pure_quadratic": Fits a pure quadratic model, y = a + b*x^2.
            - "mixture_of_exponentials": Fits a mixture of exponential functions.
            - "sigmoidal_crossover": Fits a sigmoidal crossover model.
        degree: The degree of the polynomial.

    Returns:
        Callable function that takes a single argument `x` and returns the fitted value.
        Returns `None` if the fitting process fails.

    Raises:
        ValueError: If the specified method is unknown.
    """
    if method == "polynomial":
        coeffs = Polynomial.fit(xdata, ydata, deg=degree).convert().coef

        def y_func_x(x):
            return np.polynomial.polynomial.polyval(x, coeffs)

    elif method == "pure_quadratic":
        # y = a + b x**2
        coeffs = Polynomial.fit(xdata**2, ydata, deg=1).convert().coef

        def y_func_x(x):
            return np.polynomial.polynomial.polyval(x**2, coeffs)

    elif method == "mixture_of_exponentials":
        # The no thermal noise case has two sources.
        # Kink-probability(T=0, t) ~ A t^{-1/2} ~ (1 - tanh(beta_eff))/2
        # Kink-probability(T, Inf) ~ (1 - tanh(beta J))/2
        # Kink-probability(T, t) ~ ? mixture of exponents
        # Two independent sources: Const1 +  Const2 exp(Const3*x)
        # This type of function is quite difficult to fit.
        def mixture_of_exponentials(x, p_0, p_1, p_2):
            # Strictly positive form.
            # TODO: Change to force saturation. Large x should go sigmoidally towards 0.5
            return np.exp(p_2) / 2 * (1 + np.exp(p_1 + np.exp(p_0) * x))

        # Take p_1 = 1; p_2 = min(x); take max(y) occurs at max(x)
        maxy = np.max(ydata)
        maxx = np.max(xdata)
        miny = np.min(ydata)
        p0 = [np.log(np.log(2 * maxy / miny - 1) / (maxx - 1)), 0, np.log(miny)]

        try:
            p, _ = scipy.optimize.curve_fit(
                f=mixture_of_exponentials, xdata=xdata, ydata=ydata, p0=p0
            )
        except:
            return None

        def y_func_x(x):
            return mixture_of_exponentials(x, *p)

    elif method == "sigmoidal_crossover":
        # Kink-probability(T, t) ~ sigmoidal crossover.
        # Better? Requires atleast 4 points! Not tested.
        # Sigmoidal cross-over between two positive limits.
        # This type of function is quite difficult to fit.
        def sigmoidal_crossover(x, p_0, p_1, p_2, p_3):
            # Strictly positive form.
            # TODO: Change to force saturation. Large x should go sigmoidally towards 0.5
            return np.exp(p_3) * (1 + np.exp(p_2) * np.tanh(np.exp(p_1) * (x - np.exp(p_0))))

        # Small lp1 << lp0, and lp0= (maxx-minxx)/2; We can linearize:
        # lp3*(1 + lp2( lp1 x - lp0)) = lp0*lp2*lp3 + lp1*lp2*lp3 x # WIP
        # lp2 = lp3: equal parts constant and crossover
        # x=0 -> miny therefore lp0*lp2*lp3 = miny
        # x=maxx -> maxy therefore (maxy - miny)/maxx = lp1*lp2*lp3
        maxy = np.max(ydata)
        maxx = np.max(xdata)
        miny = np.min(ydata)
        lp0 = (maxx + 1) / 2
        lp1 = lp0 / 10  # Should really choose rate 1/10 to satisfy final condition.
        lp2lp3 = miny / lp0
        p0 = (
            np.log(lp0),
            np.log(lp1),
            np.log(np.sqrt(lp2lp3)),
            np.log(np.sqrt(lp2lp3)),
        )
        try:
            p, _ = scipy.optimize.curve_fit(f=sigmoidal_crossover, xdata=xdata, ydata=ydata, p0=p0)
        except:
            return None

        def y_func_x(x):
            return sigmoidal_crossover(x, *p)

    else:
        raise ValueError("Unknown method")

    return y_func_x
