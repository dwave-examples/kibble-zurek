from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple

from dwave.cloud import Client


class QPUResources(NamedTuple):
    client: Client | None
    solvers: dict
    init_job_status: str


@lru_cache(maxsize=1)
def get_qpus() -> QPUResources:
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
    return get_qpus().client


def get_solvers() -> dict:
    return get_qpus().solvers


def get_init_job_status() -> str:
    return get_qpus().init_job_status
