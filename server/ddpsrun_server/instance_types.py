"""A job that names its machine: which AWS instance types a pin can use (HYPERUN-INSTANCE-PIN).

WHAT A PIN IS. HyperunJob's spec.resources.instanceType names the exact machine
("t3.xlarge"). PACSrun then hands its solver a candidate list of that one name
(internal/pacs/filter.go:215-224) instead of letting it rank its whole pool, so the job
runs on that machine or not at all. The other two ways to ask, by GPU name and by GPU
memory, describe a need; a pin is the experiment condition itself.

WHY THE GATEWAY CHECKS IT. A pin also skips the refusals every unpinned CPU ask
carries (filter.go:99-103). PACSrun refuses nothing about a pinned type up front: a
type its routes cannot place is accepted and the job stays Pending. So this module
answers, before the job exists, what PACSrun would answer only by not placing it:

  * is the type one PACSrun knows (aws_types.csv, from PACSrun's own catalogue);
  * can either of PACSrun's routes place it (the `refused` column; the generator,
    tools/gen_aws_types.py, names the code behind each value);
  * does one pod fit on it, by the solver's own arithmetic (`pods_per_machine`).

It also prices the type (aws_type_prices.csv). prices.csv cannot: it keeps only the
cheapest machine per GPU model and count, so most types never appear in it.
"""
from __future__ import annotations

import csv
import difflib
import pathlib
import re
from dataclasses import dataclass
from functools import lru_cache

AWS_TYPES_FILE = "aws_types.csv"
AWS_TYPE_PRICES_FILE = "aws_type_prices.csv"

# HYPERUN-DAEMONSET-RESERVE, as PACSrun measured it on this cluster. PACSrun sums the
# DaemonSets' requests before every solve and sends the three numbers to its solver
# (internal/pacs/client.go:237-241); its log line "placement: daemonset reserve measured
# from the cluster" read cpuMilli 210, memoryMiB 444, podSlots 7 on all 15 solves of
# 2026-10-07. The gateway cannot measure them, so it uses that reading. A new DaemonSet
# makes them larger, and then a pod this module passes at the very edge of a machine may
# not fit.
DAEMONSET_CPU_MILLI = 210.0
DAEMONSET_MEMORY_MIB = 444.0
DAEMONSET_POD_SLOTS = 7
DAEMONSET_MEASURED_ON = "2026-10-07"

# The solver's node model, copied from solver/kubepacs_cli.py:355-358.
VM_MEMORY_OVERHEAD = 0.075
KUBE_RESERVED_BASE_MIB = 255.0
KUBE_RESERVED_PER_POD_MIB = 11.0
EVICTION_THRESHOLD_MIB = 100.0

# What PACSrun assumes for a pod that states no size (internal/pacs/client.go:166-170).
DEFAULT_POD_CPUS = 1.0
DEFAULT_POD_MEMORY_GIB = 1.0

# What each `refused` value means, worded for the person whose job it is.
REFUSALS = {
    "arm64": (
        "{name} is an arm64 machine. PACSrun's solver drops every arm64 row and its GPU "
        "route offers x86_64 only, because a job's image is built for amd64 "
        "(solver/kubepacs_cli.py:180-181), so nothing would place it."),
    "xen": (
        "{name} is a Xen-based machine. PACSrun's AWS driver reads every stage of a job off "
        "the machine's latest serial console, which a Xen machine does not serve, so the "
        "driver refuses it (internal/pacs/policy.go, HYPERUN-XEN-FAMILY)."),
    "no-route": (
        "{name} has no row in PACSrun's CoreMark table, and the solver drops a type it "
        "cannot rank (solver/kubepacs_cli.py:104-105). A CPU machine has no other route, "
        "and for a GPU machine the GPU route has no NVIDIA row for it either."),
}

_QUANTITY = re.compile(r"^\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*([a-zA-Z]*)\s*$")
_QUANTITY_SUFFIX = {
    "": 1.0, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18,
    "Ki": 2.0**10, "Mi": 2.0**20, "Gi": 2.0**30, "Ti": 2.0**40, "Pi": 2.0**50, "Ei": 2.0**60,
}
_SKYPILOT_DATE = re.compile(r"file dated (\d{4}-\d{2}-\d{2})")
_CATALOGUE_DATE = re.compile(r"fetched (\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class InstanceType:
    """One row of aws_types.csv.

    Attributes:
        name: the EC2 instance type, "t3.xlarge".
        vcpus: vCPUs on the whole machine.
        memory_gib: memory on the whole machine, GiB.
        max_pods: kubelet's pod ceiling on it; 0 when the catalogue had none.
        gpu: the GPU model it carries, "" for none.
        gpus: how many of them, 0 for none.
        refused: "" when PACSrun can place it; otherwise a key of REFUSALS.
    """

    name: str
    vcpus: float
    memory_gib: float
    max_pods: int
    gpu: str
    gpus: int
    refused: str


@dataclass(frozen=True)
class TypePrice:
    """One row of aws_type_prices.csv: one type in one region.

    Attributes:
        usd_per_hour: the on-demand list price, or None when the catalogue has none.
        spot_low: the cheapest zone's spot price, or None when no zone sells spot.
        spot_high: the dearest zone's.
        zones: how many zones the spot range is taken over.
    """

    instance: str
    region: str
    usd_per_hour: float | None
    spot_low: float | None
    spot_high: float | None
    zones: int


@dataclass(frozen=True)
class Room:
    """What one machine leaves for pods, in the solver's arithmetic.

    Attributes:
        cpus: vCPUs left after Kubernetes' and the DaemonSets' share.
        memory_gib: memory left after the same, GiB.
        pod_slots: pods kubelet still admits; None when the catalogue has no ceiling.
    """

    cpus: float
    memory_gib: float
    pod_slots: int | None


def _lines(filename: str) -> list[str]:
    """The lines of a table beside this module. A missing table raises: one that read
    as empty would refuse every pin as unknown."""
    path = pathlib.Path(__file__).with_name(filename)
    with path.open(newline="") as handle:
        return handle.read().splitlines()


def _number(text: str) -> float | None:
    return float(text) if text else None


@lru_cache(maxsize=1)
def types() -> dict[str, InstanceType]:
    """Every instance type PACSrun's catalogue lists, by name."""
    rows = csv.DictReader(line for line in _lines(AWS_TYPES_FILE) if not line.startswith("#"))
    return {
        row["instance"]: InstanceType(
            name=row["instance"],
            vcpus=float(row["vcpus"]),
            memory_gib=float(row["memory_gib"]),
            max_pods=int(row["max_pods"] or 0),
            gpu=row["gpu"],
            gpus=int(row["gpus"] or 0),
            refused=row["refused"],
        )
        for row in rows
    }


@lru_cache(maxsize=1)
def _prices() -> dict[str, list[TypePrice]]:
    out: dict[str, list[TypePrice]] = {}
    rows = csv.DictReader(
        line for line in _lines(AWS_TYPE_PRICES_FILE) if not line.startswith("#"))
    for row in rows:
        out.setdefault(row["instance"], []).append(TypePrice(
            instance=row["instance"],
            region=row["region"],
            usd_per_hour=_number(row["usd_per_hour"]),
            spot_low=_number(row["spot_low"]),
            spot_high=_number(row["spot_high"]),
            zones=int(row["zones"] or 0),
        ))
    return out


@lru_cache(maxsize=1)
def _header() -> str:
    return _lines(AWS_TYPE_PRICES_FILE)[0]


def priced_on() -> str:
    """The date of the SkyPilot catalogue file the prices were read from."""
    found = _SKYPILOT_DATE.search(_header())
    return found.group(1) if found else "an unrecorded date"


def catalogue_on() -> str:
    """The date PACSrun's own catalogue was fetched."""
    found = _CATALOGUE_DATE.search(_header())
    return found.group(1) if found else "an unrecorded date"


def lookup(name: str) -> InstanceType | None:
    """The type by its EC2 name, or None when PACSrun's catalogue does not list it."""
    return types().get(name.strip())


def suggestions(name: str) -> list[str]:
    """Up to three listed names close to a misspelt one."""
    return difflib.get_close_matches(name.strip(), list(types()), n=3, cutoff=0.8)


def quantity(text: str, unit: float = 1.0) -> float | None:
    """A Kubernetes quantity ("500m", "4Gi", "2") in `unit`s, or None if unreadable.

    The same reading PACSrun gives cpus and memory before it calls its solver
    (internal/pacs/client.go:166-170): cpus in cores, memory in GiB when unit is 2**30.
    """
    found = _QUANTITY.match(text or "")
    if not found or found.group(2) not in _QUANTITY_SUFFIX:
        return None
    return float(found.group(1)) * _QUANTITY_SUFFIX[found.group(2)] / unit


def _kube_reserved_cpu(vcpus: float) -> float:
    """EKS's CPU reserve in cores, in tiers (solver/kubepacs_cli.py:380-392)."""
    milli = vcpus * 1000
    reserved = min(milli, 1000) * 0.06
    if milli > 1000:
        reserved += min(milli - 1000, 1000) * 0.01
    if milli > 2000:
        reserved += min(milli - 2000, 2000) * 0.005
    if milli > 4000:
        reserved += (milli - 4000) * 0.0025
    return reserved / 1000


def room(machine: InstanceType) -> Room:
    """What the solver believes one machine leaves for pods (kubepacs_cli.py:394-405)."""
    reserved_mib = (KUBE_RESERVED_BASE_MIB + KUBE_RESERVED_PER_POD_MIB * machine.max_pods
                    + EVICTION_THRESHOLD_MIB) if machine.max_pods > 0 else 0.0
    return Room(
        cpus=(machine.vcpus - _kube_reserved_cpu(machine.vcpus)
              - DAEMONSET_CPU_MILLI / 1000),
        memory_gib=(machine.memory_gib * (1 - VM_MEMORY_OVERHEAD) - reserved_mib / 1024
                    - DAEMONSET_MEMORY_MIB / 1024),
        pod_slots=(machine.max_pods - DAEMONSET_POD_SLOTS) if machine.max_pods > 0 else None,
    )


def pod_size(cpus: str | None, memory: str | None) -> tuple[float, float] | None:
    """One pod's (vCPUs, GiB) as PACSrun reads them, or None if either is unreadable."""
    pod_cpus = quantity(cpus) if cpus else DEFAULT_POD_CPUS
    pod_memory = quantity(memory, 2.0**30) if memory else DEFAULT_POD_MEMORY_GIB
    if not pod_cpus or not pod_memory:
        return None
    return pod_cpus, pod_memory


def pods_per_machine(machine: InstanceType, cpus: str | None,
                     memory: str | None) -> int | None:
    """How many pods of this size the solver puts on one machine of this type.

    The solver's fit, line for line (solver/kubepacs_cli.py:407-432): what is left of
    the machine divided by one pod, the smaller of CPU and memory, capped by kubelet's
    pod ceiling. A GPU machine that is pinned is claimed whole (PACSrun's
    placement.go:1357-1360), so it holds one pod at most.

    Returns:
        The count, 0 when one pod does not fit, None when cpus or memory is unreadable.
    """
    size = pod_size(cpus, memory)
    if size is None:
        return None
    left = room(machine)
    fit = min(left.cpus // size[0], left.memory_gib // size[1])
    if machine.gpus > 0:
        fit = min(fit, 1)
    if left.pod_slots is not None:
        fit = min(fit, left.pod_slots)
    return max(0, int(fit))


def aws_regions(regions: list[str] | None) -> list[str]:
    """The AWS regions a placement names, in order: "aws/us-east-1" and "us-east-1".

    A bare vendor word names no region and is left out, as is anything written
    `<vendor>/<place>` for another vendor.
    """
    found: list[str] = []
    for entry in regions or []:
        head, _, tail = entry.strip().partition("/")
        region = tail if head == "aws" else ("" if tail or not head[-1:].isdigit() else head)
        if region and region not in found:
            found.append(region)
    return found


def price_in(name: str, regions: list[str], capacity: str,
             first_that_sells: bool) -> TypePrice | None:
    """The price row a job pinned to this type would pay, or None if there is none.

    Args:
        regions: where the job may run, in the order the walk asks them.
        capacity: "spot" or "on-demand"; a row without that price is skipped.
        first_that_sells: True for the `ordered` walk, which buys from the first region
            that answers; False for `cheapest` and `compare`, which rank them all.
    """
    def price(row: TypePrice) -> float | None:
        return row.spot_low if capacity == "spot" else row.usd_per_hour

    by_region = {row.region: row for row in _prices().get(name, [])}
    sold = [by_region[r] for r in regions if r in by_region and price(by_region[r]) is not None]
    if not sold:
        return None
    if first_that_sells:
        return sold[0]
    return min(sold, key=lambda row: price(row) or 0.0)
