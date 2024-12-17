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

import numpy as np
import pandas as pd

__all__ = ["kink_stats", "theoretical_kink_density_prefactor", "theoretical_kink_density", "calc_kappa", "calc_lambda"]


def theoretical_kink_density_prefactor(J, schedule_name=None):
    """Time rescaling factor

    Calculate the rescaling of time necessary to replicate
    the behaviour of a linearized schedule at coupling strength 1.
    See: "Error Mitigation in Quantum Annealing"

    Args:
        J: Coupling strength between the spins of the ring.

        schedule_name: Filename of anneal schedule. Used to compensate for
            schedule energy overestimate.

    Returns:
        Kink density per anneal time, as a NumPy array.
    """

    # See the Code section of the README.md file for an explanation of the
    # following code.
    if schedule_name is None:
        schedule = pd.read_csv("helpers/FALLBACK_SCHEDULE.csv")
    else:
        schedule = pd.read_csv(f"helpers/{schedule_name}")

    COMPENSATION_SCHEDULE_ENERGY = 0.8 if (schedule_name is not None and "Advantage_system" in schedule_name) else 1.0

    A = COMPENSATION_SCHEDULE_ENERGY * schedule["A(s) (GHz)"]
    B = COMPENSATION_SCHEDULE_ENERGY * schedule["B(s) (GHz)"]
    s = schedule["s"]

    A_tag = A.diff() / s.diff()  # Derivatives of the energies for fast anneal
    B_tag = B.diff() / s.diff()

    sc_indx = abs(A - B * abs(J)).idxmin()  # Anneal fraction, s, at the critical point

    b_numerator = 1e9 * np.pi * A[sc_indx]  # D-Wave's schedules are in GHz
    b_denominator = B_tag[sc_indx] / B[sc_indx] - A_tag[sc_indx] / A[sc_indx]
    b = b_numerator / b_denominator

    return b

def theoretical_kink_density(annealing_times_ns, J=None, schedule=None, schedule_name=None, b=None):
    """
    Calculate the kink density as a function of anneal time

    Args:
        annealing_times_ns: Iterable of annealing times, in nanoseconds.

        b: A timescale based on linearization of the schedule at criticality

        J: Coupling strength between the spins of the ring.

        schedule_name: Filename of anneal schedule. Used to compensate for 
            schedule energy overestimate.

    Returns:
        Kink density per anneal time, as a NumPy array.
    """
    if b is None:
        b = theoretical_kink_density_prefactor(J, schedule_name)
    return np.power([1e-9 * t * b for t in annealing_times_ns], -0.5) / (
        2 * np.pi * np.sqrt(2)
    )

def calc_kappa(J, J_baseline=-1.8):
    """Coupling ratio

    See "Quantum error mitigation in quantum annealing" usage."""
    return abs(J_baseline / J)

def calc_lambda(J, *, schedule_name=None, J_baseline=-1.8):
    """Time rescaling factor (relative to J_baseline)

    Rate through the transition is modified non-linearly by the
    rescaling of J. If |J| is smaller than |J_baseline| we effectively move
    more slowly through the critical region, the ratio of timescales is > 1.
    See "Quantum error mitigation in quantum annealing" usage.
    """
    if schedule_name is None:
        # Fallback, assume ideal linear schedule
        kappa = calc_kappa(J, J_baseline)
        return kappa
    else:
        b_ref = theoretical_kink_density_prefactor(J_baseline, schedule_name)
        b = theoretical_kink_density_prefactor(J, schedule_name)

        return b/b_ref

def kink_stats(sampleset, J):
    """
    Calculate kink density for the sample set.

    Calculation is the number of sign switches per sample divided by the length
    of the ring for ferromagnetic coupling. For anti-ferromagnetic coupling,
    kinks are any pairs of identically-oriented spins.

    Args:
        sampleset: dimod sample set.

        J: Coupling strength between the spins of the ring.

    Returns:
        Switches/non-switches per sample and the average kink density across
        all samples.
    """
    samples_array = sampleset.record.sample
    sign_switches = np.diff(
        samples_array, prepend=samples_array[:, -1].reshape(len(samples_array), 1)
    )

    if J < 0:

        switches_per_sample = np.count_nonzero(sign_switches, 1)
        kink_density = np.mean(switches_per_sample) / sampleset.record.sample.shape[1]

        return switches_per_sample, kink_density

    non_switches_per_sample = np.count_nonzero(sign_switches == 0, 1)
    kink_density = np.mean(non_switches_per_sample) / sampleset.record.sample.shape[1]

    return non_switches_per_sample, kink_density
