"""Generate the two tables a pinned instance type is judged and priced from.

  aws_types.csv        one row per instance type in PACSrun's own catalogue: its size,
                       and whether PACSrun can place it when a job pins it (`refused`).
  aws_type_prices.csv  one row per (placeable type, region) the SkyPilot catalogue prices.

WHY A PIN NEEDS ITS OWN TABLES (HYPERUN-INSTANCE-PIN). A job may name its machine,
spec.resources.instanceType, and PACSrun then hands its solver exactly that one name
(internal/pacs/filter.go:215-224). A pin also SKIPS the refusals every other CPU ask
carries (filter.go:99-103). So a pinned machine that PACSrun's routes cannot place is
not refused anywhere: the job is accepted and stays Pending. These tables let the
gateway refuse it at submit instead, for the reason PACSrun would have hit:

  arm64     the solver drops every arm64 row (solver/kubepacs_cli.py:180-181) and the
            GPU decider offers x86_64 only (pkg/decider/skycatalog/aws.go, awsArch),
            because the job's image is amd64.
  xen       the AWS driver cannot read a Xen machine's console and refuses it before
            renting (internal/pacs/policy.go, driverUnsupportedFamily).
  no-route  nothing can rank it. A CPU machine reaches the built-in solver only, and
            the solver drops a type with no CoreMark row (kubepacs_cli.py:104-105). A
            GPU machine is asked of the SkyPilot-catalogue decider first, which answers
            only NVIDIA rows it has (skycatalog/aws.go, nvidiaOnAWS), and falls back to
            the solver; it is refused only when both would fail.

WHAT IS READ:
  <pacsrun>/data/catalog/instance_types.csv     the types PACSrun knows (filter.go:216)
  <pacsrun>/data/catalog/catalog_meta.json      when that catalogue was fetched
  <pacsrun>/solver/aws_coremark_singlecore.csv  the solver's CoreMark rows
  <pacsrun>/internal/pacs/policy.go             the Xen families, read out of the map
  ~/.sky/catalogs/v8/aws/vms.csv                prices, one row per zone

Point --pacsrun at a checkout of PACSrun's main, and regenerate whenever one of those
files changes there: a stale table refuses and accepts by the old rules.

Run: python tools/gen_aws_types.py --pacsrun <a PACSrun checkout>
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import os
import pathlib
import re
import sys

SKYPILOT = pathlib.Path(os.path.expanduser("~/.sky/catalogs/v8/aws/vms.csv"))

# The map literal in policy.go, and one "family": true entry inside it.
_XEN_MAP = re.compile(r"driverUnsupportedFamily\s*=\s*map\[string\]bool\{(.*?)\}", re.S)
_XEN_ENTRY = re.compile(r'"([a-z0-9]+)"\s*:\s*true')
# skycatalog/aws.go: awsManufacturer and awsPlainAZ, copied so the two read a row alike.
_MANUFACTURER = re.compile(r"'Manufacturer'\s*:\s*'([^']*)'")
_PLAIN_ZONE = re.compile(r"^[a-z0-9]+-az[0-9]+$")


def package_dir() -> pathlib.Path:
    """The server package: the directory beside tools/ that holds measurements.py."""
    server = pathlib.Path(__file__).resolve().parents[1]
    found = [p for p in server.iterdir() if (p / "measurements.py").is_file()]
    if len(found) != 1:
        raise SystemExit(f"expected one package directory under {server}, found {found}")
    return found[0]


def xen_families(policy_go: pathlib.Path) -> set[str]:
    """The families the AWS driver cannot run, read out of PACSrun's policy.go."""
    block = _XEN_MAP.search(policy_go.read_text())
    families = set(_XEN_ENTRY.findall(block.group(1))) if block else set()
    if not families:
        raise SystemExit(f"{policy_go}: driverUnsupportedFamily not found or empty")
    return families


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verdict(row: dict[str, str], xen: set[str], coremark: set[str],
            nvidia_rows: set[str]) -> str:
    """Why PACSrun could not place this pinned type, or "" when it can."""
    name = row["InstanceType"]
    if "x86_64" not in row["Architecture"]:
        return "arm64"
    if name.split(".", 1)[0] in xen:
        return "xen"
    if row["GpuCount"]:
        return "" if name in nvidia_rows or name in coremark else "no-route"
    return "" if name in coremark else "no-route"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pacsrun", required=True, type=pathlib.Path,
                        help="a checkout of PACSrun's main")
    parser.add_argument("--skypilot", type=pathlib.Path, default=SKYPILOT)
    args = parser.parse_args()

    catalogue = read_csv(args.pacsrun / "data/catalog/instance_types.csv")
    fetched = json.loads((args.pacsrun / "data/catalog/catalog_meta.json").read_text())
    coremark = {r["InstanceType"] for r in read_csv(
        args.pacsrun / "solver/aws_coremark_singlecore.csv")}
    xen = xen_families(args.pacsrun / "internal/pacs/policy.go")

    zones: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    nvidia_rows: set[str] = set()
    for row in read_csv(args.skypilot):
        zones[row["InstanceType"]].append(row)
        maker = _MANUFACTURER.search(row.get("GpuInfo") or "")
        if (maker and maker.group(1).strip().upper() == "NVIDIA"
                and row.get("Arch") == "x86_64"):
            nvidia_rows.add(row["InstanceType"])

    package = package_dir()
    sky_on = datetime.date.fromtimestamp(args.skypilot.stat().st_mtime).isoformat()
    source = (f"PACSrun data/catalog (fetched {fetched.get('generated_at', '?')[:10]}), "
              f"its CoreMark table and policy.go, and ~/.sky/catalogs/v8/aws/vms.csv "
              f"(file dated {sky_on})")

    refused = collections.Counter()
    placeable: list[str] = []
    with (package / "aws_types.csv").open("w", newline="") as handle:
        handle.write(f"# generated by tools/gen_aws_types.py from {source}. Do not hand-edit.\n")
        out = csv.writer(handle, lineterminator="\n")
        out.writerow(["instance", "vcpus", "memory_gib", "max_pods", "gpu", "gpus", "refused"])
        for row in sorted(catalogue, key=lambda r: r["InstanceType"]):
            why = verdict(row, xen, coremark, nvidia_rows)
            refused[why or "placeable"] += 1
            if not why:
                placeable.append(row["InstanceType"])
            out.writerow([row["InstanceType"], row["VCpus"], row["MemoryGiB"],
                          row["MaxPods"], row["GpuName"], row["GpuCount"], why])

    priced = 0
    mixed: list[str] = []
    with (package / "aws_type_prices.csv").open("w", newline="") as handle:
        handle.write(f"# generated by tools/gen_aws_types.py from {source}. Do not hand-edit.\n")
        out = csv.writer(handle, lineterminator="\n")
        out.writerow(["instance", "region", "usd_per_hour", "spot_low", "spot_high", "zones"])
        for name in placeable:
            by_region: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
            for row in zones.get(name, []):
                by_region[row["Region"]].append(row)
            for region in sorted(by_region):
                rows = by_region[region]
                on_demand = {float(r["Price"]) for r in rows if r["Price"]}
                # Spot is per zone, and only the plain zones are ones the decider buys in.
                spot = [float(r["SpotPrice"]) for r in rows
                        if r["SpotPrice"] and _PLAIN_ZONE.match(r["AvailabilityZone"])]
                if not on_demand and not spot:
                    continue
                if len(on_demand) > 1:
                    mixed.append(f"{name}/{region}")
                out.writerow([name, region,
                              f"{max(on_demand):.4f}" if on_demand else "",
                              f"{min(spot):.4f}" if spot else "",
                              f"{max(spot):.4f}" if spot else "", len(spot)])
                priced += 1

    print(f"{len(catalogue)} types: {dict(refused)}; {priced} (type, region) price rows",
          file=sys.stderr)
    if mixed:
        print(f"on-demand price differs between zones, the dearest was kept: {mixed[:5]}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
