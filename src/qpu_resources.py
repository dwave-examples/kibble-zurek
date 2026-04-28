# Copyright 2026 D-Wave
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

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

from dwave.cloud import Client


class QPUResources(NamedTuple):
    client: Client | None
    solvers: dict
    init_job_status: str


@lru_cache(maxsize=1)
def get_qpus() -> QPUResources:
    """Get QPU resources, including client and solvers. Results are cached."""
    try:
        client = Client.from_config(client="qpu")
        solvers = {
            qpu.name: qpu for qpu in client.get_solvers(fast_anneal_time_range__covers=[0.005, 0.1])
        }

        if not solvers:
            raise Exception

        return QPUResources(client=client, solvers=solvers, init_job_status="READY")

    except Exception:
        return QPUResources(client=None, solvers={}, init_job_status="NO SOLVER")


def get_client() -> Client | None:
    """Get the client, if available. Returns None if no client is available."""
    return get_qpus().client


def get_solvers() -> dict:
    """Get available solvers, if any. Returns an empty dictionary if no solvers are available."""
    return get_qpus().solvers


def get_init_job_status() -> str:
    """Get initial job status. "READY" if solvers are available, "NO SOLVER" otherwise."""
    return get_qpus().init_job_status
