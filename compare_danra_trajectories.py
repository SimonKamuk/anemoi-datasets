#!/usr/bin/env python
# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare two trajectories datasets, e.g. the one anemoi-datasets built from
danra_anemoi_trajectories.yaml and the one danra_trajectories_xarray.py built.

    python compare_danra_trajectories.py REFERENCE.zarr OTHER.zarr

Reports the differences in coordinates and metadata, then for each variable:
values that are NaN in one dataset only, the fraction that are bit-identical,
and the largest difference, absolute and relative to the variable's stdev.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr

# Expected to differ between two builds.
VOLATILE = {"uuid", "latest_write_timestamp", "provenance_load", "total_size", "total_number_of_files"}


def compare_attributes(a, b):
    print("== Metadata")
    for key in sorted(set(a) | set(b)):
        if key in VOLATILE:
            continue
        if key not in a or key not in b:
            print(f"  {key}: only in {'reference' if key in a else 'other'}")
        elif a[key] != b[key]:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                keys = sorted(k for k in set(a[key]) | set(b[key]) if a[key].get(k) != b[key].get(k))
                print(f"  {key}: differs for {keys}")
                for k in keys[:10]:
                    print(f"      {k}: {json.dumps(a[key].get(k))[:150]} != {json.dumps(b[key].get(k))[:150]}")
            else:
                print(f"  {key}: {json.dumps(a[key])[:150]} != {json.dumps(b[key])[:150]}")


def compare_chunk(a, b, i, s):
    x = a[i, :, 0, s, :]
    y = b[i, :, 0, s, :]
    nan_x, nan_y = np.isnan(x), np.isnan(y)
    both = ~nan_x & ~nan_y
    diff = np.where(both, np.abs(x.astype(np.float64) - y), 0.0)
    return dict(
        only_reference=(nan_x & ~nan_y).sum(axis=1),
        only_other=(nan_y & ~nan_x).sum(axis=1),
        valid=both.sum(axis=1),
        equal=(both & (x == y)).sum(axis=1),
        max=diff.max(axis=1),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("other")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--skip-missing", action="store_true", help="leave out the base dates the reference has as missing"
    )
    args = parser.parse_args()

    a = zarr.open_group(args.reference, mode="r")
    b = zarr.open_group(args.other, mode="r")
    compare_attributes(dict(a.attrs), dict(b.attrs))

    print("== Arrays")
    for name in ("base_dates", "steps", "latitudes", "longitudes"):
        same = a[name].shape == b[name].shape and np.array_equal(a[name][:], b[name][:])
        print(f"  {name}: {'identical' if same else 'DIFFERENT'}")
    if a["data"].shape != b["data"].shape:
        raise SystemExit(f"  data shapes differ: {a['data'].shape} != {b['data'].shape}")

    variables = a.attrs["variables"]
    stdev = a["stdev"][:]
    print("  statistics, largest relative difference:")
    for name in ("mean", "stdev", "minimum", "maximum"):
        x, y = a[name][:], b[name][:]
        scale = np.maximum(stdev if name == "mean" else np.abs(x), 1e-30)
        worst = np.argmax(np.abs(x - y) / scale)
        print(
            f"    {name:8s} {np.abs(x - y)[worst] / scale[worst]:.2e} ({variables[worst]}: {x[worst]:.6g} vs {y[worst]:.6g})"
        )

    data_a, data_b = a["data"], b["data"]
    n_dates, _, _, n_steps, _ = data_a.shape
    base_dates = a["base_dates"][:]
    dates = range(n_dates)
    if args.skip_missing:
        missing = np.array(a.attrs.get("missing_dates", []), dtype="datetime64[s]")
        dates = [i for i in dates if base_dates[i] not in missing]
        print(f"  leaving out the {n_dates - len(dates)} base dates the reference has as missing")
    keys = [(i, s) for i in dates for s in range(n_steps)]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda k: compare_chunk(data_a, data_b, *k), keys))

    total = {key: sum(r[key] for r in results) for key in ("only_reference", "only_other", "valid", "equal")}
    maxima = np.stack([r["max"] for r in results])  # (chunks, variables)
    worst = maxima.argmax(axis=0)
    largest = maxima.max(axis=0)

    print("== Data, worst first (relative = largest difference / stdev)")
    print(
        f"  {'variable':16s} {'NaN ref only':>12s} {'NaN other only':>14s} {'identical':>9s} {'largest':>10s} {'relative':>9s}  where"
    )
    order = np.argsort(
        -(largest / np.where(stdev > 0, stdev, 1)) - 1e30 * (total["only_reference"] + total["only_other"])
    )
    for v in order:
        i, s = keys[worst[v]]
        print(
            f"  {variables[v]:16s} {total['only_reference'][v]:12d} {total['only_other'][v]:14d}"
            f" {total['equal'][v] / max(total['valid'][v], 1):9.2%} {largest[v]:10.3g}"
            f" {largest[v] / stdev[v] if stdev[v] > 0 else 0:9.2e}  {base_dates[i]} step {s}"
        )


if __name__ == "__main__":
    main()
