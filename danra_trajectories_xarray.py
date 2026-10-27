#!/usr/bin/env python
# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Build the dataset of danra_anemoi_trajectories.yaml with xarray and dask.

The same dataset as ``anemoi-datasets create`` writes for that recipe -- the
array is ``(base_dates, variables, ensembles, steps, cells)`` with step 0 from
the DANRA analysis and steps 1-3h from the forecast started at the base date --
but read in blocks of base dates rather than field by field.

The DANRA stores are chunked by 256 time steps, so a block of 32 base dates
needs 1/8 of each chunk. ``read_rows`` reads only the blosc blocks of a chunk
that a block needs, which cuts the data read from disk by about that much
(zarr-python 2 can do this in principle, but its partial reads fail on these
stores).

Three stages, as with anemoi-datasets:

    python danra_trajectories_xarray.py init     OUT.zarr --template ANEMOI.zarr
    python danra_trajectories_xarray.py load     OUT.zarr --part 1/4   # parts in parallel
    python danra_trajectories_xarray.py finalise OUT.zarr

``init`` copies the metadata that does not depend on the dates (variable
metadata, data request, recipe, ...) from a dataset anemoi-datasets has built
from the recipe, so the two describe their variables identically. ``load``
writes each block once and keeps the partial statistics of each of its base
dates in ``OUT.zarr.stats/``, which is also what makes a restarted load skip
the blocks already done. ``finalise`` merges those of the statistics period
into the statistics arrays; as the partial statistics cover every base date,
it can be rerun with another period (``--statistics-start/--statistics-end``)
without reading the data again.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import logging
import mmap
import os
import struct
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from earthkit.meteo.solar import cos_solar_zenith_angle
from earthkit.meteo.thermo import array as thermo
from numcodecs import blosc

LOG = logging.getLogger("danra_trajectories")


ROOT = "/dmidata/projects/cloudphysics/danra/data/v0.6.1"

START = "1990-09-01T00:00"
END = "2023-12-31T21:00"
FREQUENCY = np.timedelta64(3, "h")
# The stores start at 1990-09-01T00, so every base date has its analysis and
# forecast; only the fields by validity time whose window reaches back before
# the forecasts start (1990-09-01T01) are NaN.
MISSING = []
STEPS = np.arange(4) * np.timedelta64(1, "h")
HOUR = np.timedelta64(1, "h")

LEVELS = [100, 200, 250, 300, 400, 500, 600, 700, 800, 850, 925, 1000]

# Output name: name in the store.
PRESSURE = {"t": "t", "u": "u", "v": "v", "w": "tw", "z": "z", "clwc": "cwat", "ciwc": "ciwc"}
SINGLE = {
    "10u": "u10m",
    "10v": "v10m",
    "2t": "t2m",
    "skt": "t0m",
    "msl": "pres_seasurface",
    "sp": "pres0m",
    "tcw": "pwat_column",
    "z0": "roughness_length",
    "st": "soil_temperature",
    "tcc": "tcc",
    "lcc": "lcc",
    "mcc": "mcc",
    "hcc": "hcc",
    "sd": "snow_water_equivalent",
    "cape": "cape_column",
}
# Hourly accumulations, and the hourly gust maximum, given over each of the
# windows as {name}_{hours}hr.
ACCUMULATIONS = {
    "tp": "tp_accum_past_hr",
    "ssrd": "grad_accum_past_hr",
    "strd": "lwavr_accum_past_hr",
    "ssr": "nswrs_accum_past_hr",
    "str": "nlwrs_accum_past_hr",
    "tirf": "rain_accum_past_hr",
    "tsnowp": "snow_accum_past_hr",
    "tgrp": "grpl_accum_past_hr",
}
FORCINGS = [
    "cos_julian_day",
    "cos_latitude",
    "cos_local_time",
    "cos_longitude",
    "insolation",
    "sin_julian_day",
    "sin_latitude",
    "sin_local_time",
    "sin_longitude",
]
WINDOWS = (1, 3, 6)
VARIABLES = sorted(
    [f"{p}_{level}" for p in PRESSURE for level in LEVELS]
    + [f"q_{level}" for level in LEVELS]
    + list(SINGLE)
    + ["2d", "swv", "100u", "100v"]
    + [f"{name}_{hours}hr" for name in list(ACCUMULATIONS) + ["sf", "10fg"] for hours in WINDOWS]
    + ["al", "lsm", "z"]
    + FORCINGS
)

# The valid-time fields look back over up to 6 hours (24 for al), so a block
# reads the forecasts started up to 24 hours before its first base date.
LOOKBACK = np.timedelta64(24, "h")
SOURCE_CHUNK = 256  # time chunk of the DANRA stores


def as_datetime64(value):
    return np.datetime64(value, "s")


def iso(value):
    return pd.Timestamp(value).isoformat()


# ----------------------------------------------------------------------------
# Reading and deriving the fields of one block of base dates
# ----------------------------------------------------------------------------


def read_rows(path, start, stop, chunks, dtype):
    """Rows ``start:stop`` of the leading axis of a blosc-compressed zarr
    chunk, reading only the blosc blocks they lie in.
    """
    dtype = np.dtype(dtype)
    row = int(np.prod(chunks[1:])) * dtype.itemsize
    with open(path, "rb") as f:
        header = f.read(16)
        nbytes, blocksize, cbytes = struct.unpack("<III", header[4:16])
        assert nbytes == chunks[0] * row, f"{path}: unexpected chunk size"
        if header[2] & 0x2:  # stored uncompressed
            f.seek(16 + start * row)
            return np.frombuffer(f.read((stop - start) * row), dtype).reshape((stop - start,) + tuple(chunks[1:]))

        # blosc needs the header and block offsets in place, and the blocks it
        # decompresses at their offsets; the rest of the buffer is never read
        # (an anonymous mmap, so its pages are not even allocated).
        nblocks = -(-nbytes // blocksize)
        starts = f.read(4 * nblocks)
        offsets = np.append(np.frombuffer(starts, "<i4"), cbytes)
        first, last = start * row // blocksize, (stop * row - 1) // blocksize
        buffer = mmap.mmap(-1, cbytes)
        buffer[: 16 + len(starts)] = header + starts
        f.seek(offsets[first])
        f.readinto(memoryview(buffer)[offsets[first] : offsets[last + 1]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        raw = blosc.decompress_partial(buffer, start * row // dtype.itemsize, (stop - start) * row // dtype.itemsize)
    return np.frombuffer(raw, dtype).reshape((stop - start,) + tuple(chunks[1:]))


class Sources:
    """The DANRA stores: xarray for their coordinates, ``read_rows`` for their
    values.
    """

    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(workers)
        self._datasets = {}
        self._arrays = {}
        an = self.dataset("single_levels_analysis")
        self.lat = an.lat.values.ravel()
        self.lon = an.lon.values.ravel()
        self.lsm = an.lsm.values.ravel()
        self.z = an.surface_geopotential.values.ravel()

    def dataset(self, store):
        if store not in self._datasets:
            self._datasets[store] = xr.open_zarr(f"{ROOT}/{store}.zarr", chunks=None, drop_variables=["valid_time"])
        return self._datasets[store]

    def array(self, store, name):
        if (store, name) not in self._arrays:
            array = zarr.open_array(f"{ROOT}/{store}.zarr/{name}", mode="r")
            assert array.compressor.codec_id == "blosc" and not array.filters and array.order == "C", array.info
            self._arrays[store, name] = array
        return self._arrays[store, name]

    def read(self, store, names, start, stop, **selection):
        """Values at time indices ``start:stop`` of each of ``names``, at the
        values ``selection`` of dimensions chunked by 1 (e.g. the pressure
        levels), and all of the other dimensions.

        Returns ``{name: array}``, with the dimensions of the store.
        """
        ds = self.dataset(store)
        result, tasks = {}, []
        for name in names:
            array, dims = self.array(store, name), ds[name].dims
            assert dims[-2:] == ("y", "x"), dims
            # Decode as xarray does; the stores only use _FillValue (NaN, or
            # 9999 for the SURFEX fields).
            encoding = ds[name].encoding
            assert not {"scale_factor", "add_offset", "missing_value"} & set(encoding), (name, encoding)
            fill = encoding.get("_FillValue", np.nan)
            chunks, shape = array.chunks, array.shape

            # Chunk indices to read along each dimension, and where they go.
            wanted = [range(start // chunks[0], (stop - 1) // chunks[0] + 1)]
            for axis in range(1, len(shape)):
                if dims[axis] in selection:
                    assert chunks[axis] == 1, (name, dims[axis])
                    wanted.append([ds[dims[axis]].values.tolist().index(v) for v in selection[dims[axis]]])
                else:
                    wanted.append(range(-(-shape[axis] // chunks[axis])))
            out = np.empty(
                [stop - start] + [len(w) if dims[a] in selection else shape[a] for a, w in enumerate(wanted) if a],
                array.dtype,
            )
            result[name] = out

            for key in np.ndindex(*map(len, wanted)):
                index = [w[k] for w, k in zip(wanted, key)]
                first, last = max(start, index[0] * chunks[0]), min(stop, (index[0] + 1) * chunks[0])
                target = [slice(first - start, last - start)]
                for axis in range(1, len(shape)):
                    if dims[axis] in selection:
                        target.append(slice(key[axis], key[axis] + 1))
                    else:
                        lo = index[axis] * chunks[axis]
                        target.append(slice(lo, min(lo + chunks[axis], shape[axis])))
                path = f"{ROOT}/{store}.zarr/{name}/{'.'.join(map(str, index))}"
                rows = (first - index[0] * chunks[0], last - index[0] * chunks[0])
                tasks.append((out, tuple(target), path, rows, chunks, array.dtype, fill))

        def run(task):
            out, target, path, rows, chunks, dtype, fill = task
            values = read_rows(path, *rows, chunks, dtype)
            out[target] = values[(slice(None),) + tuple(slice(0, t.stop - t.start) for t in target[1:])]
            if not np.isnan(fill):
                region = out[target]
                region[region == fill] = np.nan

        list(self.pool.map(run, tasks))
        return result

    def time_index(self, store, dim, base_dates):
        times = self.dataset(store)[dim].values
        start = int(np.searchsorted(times, base_dates[0]))
        assert np.array_equal(
            times[start : start + len(base_dates)], base_dates
        ), f"{store} does not have the base dates"
        return start

    def instantaneous(self, kind, names, base_dates, **selection):
        """Fields at the 4 steps of each base date: step 0 is the analysis,
        steps 1-3h the forecast started at the base date.

        Returns ``{name: array (base_dates, steps, [level,] cells)}``.
        """
        an, fc = f"{kind}_analysis", f"{kind}_forecast_PT3H"
        assert np.array_equal(self.dataset(fc).forecast_period.values, STEPS[1:])
        start = self.time_index(an, "time", base_dates)
        analysis = self.read(an, names, start, start + len(base_dates), **selection)
        start = self.time_index(fc, "analysis_time", base_dates)
        forecast = self.read(fc, names, start, start + len(base_dates), **selection)

        result = {}
        for name in names:
            values = np.concatenate([analysis.pop(name)[:, None], forecast.pop(name)], axis=1)
            result[name] = values.reshape(values.shape[:-2] + (-1,))
        return result

    def by_valid_time(self, names, base_dates):
        """The forecasts as one hourly series by validity time.

        The 3-hourly runs with lead times 1-3h cover every hour, so flattening
        (run, lead time) gives a contiguous hourly series. It starts
        ``LOOKBACK`` before the first base date (or where the store starts)
        and ends 3h after the last.

        Returns ``({name: array (times, cells)}, times, index)``, where
        ``index`` (base_dates, steps) gives the position in the series of each
        step.
        """
        store = "single_levels_forecast_PT3H"
        runs = self.dataset(store).analysis_time.values
        assert np.array_equal(self.dataset(store).forecast_period.values, STEPS[1:])
        start = int(np.searchsorted(runs, base_dates[0] - LOOKBACK))
        stop = self.time_index(store, "analysis_time", base_dates) + len(base_dates)
        values = self.read(store, names, start, stop)

        valid = (runs[start:stop, None] + STEPS[None, 1:]).ravel()
        assert np.all(np.diff(valid) == HOUR), "the forecasts do not cover every hour"

        # Hours before the store starts (step 0 of its first base date) are missing.
        pad = max(0, int((valid[0] - base_dates[0]) // HOUR))
        valid = np.concatenate([valid[0] - np.arange(pad, 0, -1) * HOUR, valid])
        index = (base_dates[:, None] + STEPS[None, :] - valid[0]) // HOUR
        assert index.max() < len(valid), "the forecasts do not cover the base dates"

        series = {}
        for name in names:
            flat = values.pop(name).reshape(len(valid) - pad, -1)
            series[name] = np.concatenate([np.full((pad, flat.shape[1]), np.nan), flat])
        return series, valid, index


def rolling(values, steps, operation):
    """xarray's rolling operation over a full window, as in the recipe's
    ``rolling_operation`` patch.
    """
    rolled = xr.DataArray(values, dims=("time", "cell")).rolling(time=steps, min_periods=steps)
    return getattr(rolled, operation)().values


def rolling_last(values, steps):
    """The latest value that is not missing in each window of ``steps``; the
    first windows look back as far as the series goes.
    """
    result = values.copy()
    for shift in range(1, min(steps, len(values))):
        missing = np.isnan(result[shift:])
        result[shift:][missing] = values[:-shift][missing]
    return result


def julian_day(date):
    delta = date - datetime.datetime(date.year, 1, 1)
    return delta.days + delta.seconds / 86400.0


def local_time(date, lon):
    delta = date - datetime.datetime(date.year, date.month, date.day)
    hours = (delta.days + delta.seconds / 86400.0) * 24
    return (lon / 360.0 * 24.0 + hours) % 24


def build_block(sources, base_dates, put):
    """Compute every variable for ``base_dates`` and pass each to
    ``put(name, array (base_dates, steps, cells))``.
    """
    n = len(base_dates)

    # Pressure levels. r (a fraction) is converted to % and then to q.
    for name, source in PRESSURE.items():
        values = sources.instantaneous("pressure_levels", [source], base_dates, pressure=LEVELS)[source]
        for i, level in enumerate(LEVELS):
            put(f"{name}_{level}", values[:, :, i])
        del values

    fields = sources.instantaneous("pressure_levels", ["r", "t"], base_dates, pressure=LEVELS)
    for i, level in enumerate(LEVELS):
        r = fields["r"][:, :, i] * 100
        put(f"q_{level}", thermo.specific_humidity_from_relative_humidity(fields["t"][:, :, i], r, 100.0 * level))
    del fields

    # Single levels.
    fields = sources.instantaneous(
        "single_levels", list(SINGLE.values()) + ["r2m", "soil_moisture", "soil_ice"], base_dates
    )
    for name, source in SINGLE.items():
        put(name, fields[source] * 0.001 if name == "sd" else fields[source])
    r2m = fields["r2m"] * 100
    r2m[r2m == 0] = 1.0e-4  # as anemoi-transform's r_to_d
    put("2d", thermo.dewpoint_from_relative_humidity(t=fields["t2m"], r=r2m))
    put("swv", fields["soil_moisture"] + fields["soil_ice"])
    del fields, r2m

    # 100 m winds.
    fields = sources.instantaneous("height_levels", ["u", "v"], base_dates, height=[100])
    put("100u", fields["u"][:, :, 0])
    put("100v", fields["v"][:, :, 0])
    del fields

    # By validity time: sums and maximum gust over the past 1, 3 and 6 hours,
    # and the latest daylight albedo.
    names = list(ACCUMULATIONS.values()) + ["ugst10m_past_hr", "vgst10m_past_hr", "albedo"]
    series, valid, index = sources.by_valid_time(names, base_dates)
    gust = (series["ugst10m_past_hr"] ** 2 + series["vgst10m_past_hr"] ** 2) ** 0.5

    def window(values, hours, operation):
        # The hourly values are the store's own; longer windows roll them.
        return (values if hours == 1 else rolling(values, hours, operation))[index]

    # Sums that cannot be negative are clipped at 0: the hourly values have
    # float noise below it. str, net longwave, can be negative.
    for hours in WINDOWS:
        for name, source in ACCUMULATIONS.items():
            summed = window(series[source], hours, "sum")
            if name == "tp":
                summed = summed * 0.001
            put(f"{name}_{hours}hr", summed if name == "str" else np.clip(summed, 0, None))
        snow = window(series["snow_accum_past_hr"], hours, "sum")
        graupel = window(series["grpl_accum_past_hr"], hours, "sum")
        put(f"sf_{hours}hr", np.clip((snow + graupel) * 0.001, 0, None))
        put(f"10fg_{hours}hr", window(gust, hours, "max"))

    # Albedo where it is not negative and the sun is more than 1.15 degrees
    # up; the latest such value in the past 24 hours.
    albedo = series["albedo"]
    albedo[albedo < 0] = np.nan
    for t, date in enumerate(pd.to_datetime(valid).to_pydatetime()):
        cos = cos_solar_zenith_angle(date, sources.lat, sources.lon).astype(np.float32)
        albedo[t][~(cos > 0.02)] = np.nan
    put("al", rolling_last(albedo, 24)[index])
    del series, albedo, gust

    # Constants.
    put("lsm", sources.lsm)
    put("z", sources.z)

    # Forcings, at each step's validity time.
    lat, lon = np.deg2rad(sources.lat), np.deg2rad(sources.lon)
    put("cos_latitude", np.cos(lat))
    put("sin_latitude", np.sin(lat))
    put("cos_longitude", np.cos(lon))
    put("sin_longitude", np.sin(lon))

    dates = pd.to_datetime((base_dates[:, None] + STEPS[None, :]).ravel()).to_pydatetime()
    shape = (n, len(STEPS), -1)

    day = np.array([julian_day(d) for d in dates]) / 365.25 * np.pi * 2
    put("cos_julian_day", np.cos(day)[:, None].repeat(len(lat), axis=1).reshape(shape))
    put("sin_julian_day", np.sin(day)[:, None].repeat(len(lat), axis=1).reshape(shape))

    hours = np.stack([local_time(d, sources.lon) for d in dates]) / 24 * np.pi * 2
    put("cos_local_time", np.cos(hours).reshape(shape))
    put("sin_local_time", np.sin(hours).reshape(shape))
    del hours

    insolation = np.stack([cos_solar_zenith_angle(d, sources.lat, sources.lon) for d in dates])
    put("insolation", insolation.reshape(shape))


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------


def statistics_dates(base_dates):
    """anemoi-datasets' default statistics period for a trajectories dataset:
    the first 80% of the validity times when the dataset spans less than 10
    years, else all but the last year (all but the last 3 from 20 years).
    """
    valid = np.unique((base_dates[:, None] + STEPS[None, :]).ravel())
    first, last = pd.Timestamp(valid[0]), pd.Timestamp(valid[-1])
    years = round((last - first).total_seconds() / (365.25 * 24 * 60 * 60))
    if len(valid) < 10:
        end = valid[max(1, int(len(valid) * 0.8)) - 1]
    elif years < 10:
        end = valid[int(len(valid) * 0.8) - 1]
    else:
        end_year = last.year - (3 if years >= 20 else 1)
        end = valid[pd.DatetimeIndex(valid).year == end_year][-1]
    return valid[0], end


STATISTICS = ("count", "mean", "m2", "minimum", "maximum")


def date_statistics(values):
    """Count, mean, M2 (sum of squared deviations from the mean), minimum and
    maximum of the values that are not NaN, for each base date: ``values`` is
    (base_dates, ...).
    """
    values = values.reshape(len(values), -1).astype(np.float64)
    valid = ~np.isnan(values)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, np.nansum(values, axis=1) / count, 0.0)
    m2 = np.nansum((values - mean[:, None]) ** 2, axis=1)
    minimum = np.where(count > 0, np.min(np.where(valid, values, np.inf), axis=1), np.inf)
    maximum = np.where(count > 0, np.max(np.where(valid, values, -np.inf), axis=1), -np.inf)
    return count, mean, m2, minimum, maximum


def merge_statistics(stats):
    """Merge per-date statistics, arrays (base_dates, variables), into
    (variables,): the parallel algorithm of Chan et al., which is exact.
    """
    count = stats["count"].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (stats["count"] * stats["mean"]).sum(axis=0) / count
    m2 = (stats["m2"] + stats["count"] * (stats["mean"] - mean) ** 2).sum(axis=0)
    return dict(
        count=count, mean=mean, m2=m2, minimum=stats["minimum"].min(axis=0), maximum=stats["maximum"].max(axis=0)
    )


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------


def init(args):
    base_dates = np.arange(as_datetime64(args.start), as_datetime64(args.end) + FREQUENCY, FREQUENCY)
    missing = [as_datetime64(m) for m in args.missing]
    stats_start, stats_end = statistics_dates(base_dates)

    template = zarr.open_group(args.template, mode="r")
    attrs = dict(template.attrs)
    if attrs["variables"] != VARIABLES:
        raise ValueError(f"The template's variables differ: {set(attrs['variables']) ^ set(VARIABLES)}")

    sources = Sources(1)
    if not np.array_equal(template["latitudes"][:], sources.lat):
        raise ValueError("The template's grid differs")

    shape = (len(base_dates), len(VARIABLES), 1, len(STEPS), len(sources.lat))
    LOG.info("Creating %s with shape %s (%.1f TB uncompressed)", args.output, shape, np.prod(shape) * 4 / 1e12)

    root = zarr.open_group(args.output, mode="w" if args.overwrite else "w-")
    data = root.create_dataset(
        "data", shape=shape, chunks=(1, len(VARIABLES), 1, 1, len(sources.lat)), dtype="float32", fill_value=np.nan
    )
    data.attrs["_ARRAY_DIMENSIONS"] = ["time", "variable", "ensemble", "step", "cell"]
    for name, values, dimension in (
        ("base_dates", base_dates.astype("datetime64[s]"), "time"),
        ("steps", STEPS.astype("timedelta64[us]"), "step"),
        ("latitudes", sources.lat, "cell"),
        ("longitudes", sources.lon, "cell"),
    ):
        root.array(name, values).attrs["_ARRAY_DIMENSIONS"] = [dimension]

    recipe = dict(attrs["recipe"])
    recipe["base_dates"] = dict(
        recipe["base_dates"], start=iso(base_dates[0]), end=iso(base_dates[-1]), missing=[iso(m) for m in missing]
    )
    attrs.update(
        shape=list(shape),
        start_base_date=iso(base_dates[0]),
        end_base_date=iso(base_dates[-1]),
        start_date=iso(base_dates[0] + STEPS[0]),
        end_date=iso(base_dates[-1] + STEPS[-1]),
        last_date=str(pd.Timestamp(base_dates[-1] + STEPS[-1])),
        missing_dates=[iso(m) for m in missing],
        statistics_start_date=iso(stats_start),
        statistics_end_date=iso(stats_end),
        recipe=recipe,
        uuid=str(uuid.uuid4()),
        constant_fields=[],
        provenance_load=dict(
            script=os.path.abspath(__file__),
            module_versions={
                m: importlib.metadata.version(m)
                for m in ("numpy", "pandas", "xarray", "dask", "zarr", "earthkit-meteo")
            },
        ),
        latest_write_timestamp=datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(),
    )
    for key in ("total_size", "total_number_of_files"):
        attrs.pop(key, None)
    root.attrs.update(attrs)
    LOG.info("Statistics from %s to %s", stats_start, stats_end)


def blocks(base_dates, block):
    """Ranges of base dates, aligned with the chunks of the DANRA stores so
    that a block never reads two of them.
    """
    offset = (base_dates[0] - as_datetime64(START)) // FREQUENCY
    edges = [i for i in range(1, len(base_dates)) if (offset + i) % block == 0]
    edges = [0] + edges + [len(base_dates)]
    return list(zip(edges[:-1], edges[1:]))


def block_summary(buffer, present, pool):
    """Partial statistics of each base date of a block, and whether each
    variable is the same at every step of the ``present`` ones.
    """

    def statistics(v):
        return date_statistics(buffer[:, v])

    def constant(v):
        if not present.any():
            return True, ""
        fields = buffer[present, v, 0].reshape(-1, buffer.shape[-1])
        same = all(np.array_equal(f, fields[0], equal_nan=True) for f in fields[1:])
        return same, hashlib.sha256(fields[0].tobytes()).hexdigest()

    stats = list(pool.map(statistics, range(buffer.shape[1])))
    constants = list(pool.map(constant, range(buffer.shape[1])))
    # Each statistic is (base_dates, variables).
    summary = {key: np.stack([s[i] for s in stats], axis=1) for i, key in enumerate(STATISTICS)}
    summary.update(
        constant=np.array([c[0] for c in constants]),
        hash=np.array([c[1] for c in constants]),
        present=present.any(),
    )
    return summary


def load(args):
    root = zarr.open_group(args.output, mode="r+")
    data = root["data"]
    base_dates = root["base_dates"][:]
    missing = np.isin(base_dates, np.array(root.attrs["missing_dates"], dtype="datetime64[s]"))

    todo = blocks(base_dates, args.block)
    part, parts = map(int, args.part.split("/"))
    todo = todo[(part - 1) * len(todo) // parts : part * len(todo) // parts]

    stats_dir = args.output.rstrip("/") + ".stats"
    os.makedirs(stats_dir, exist_ok=True)

    sources = Sources(args.workers)
    index = {name: i for i, name in enumerate(VARIABLES)}
    # Each statistics thread holds a float64 copy of a variable's block.
    pool = ThreadPoolExecutor(min(args.workers, 16))

    for number, (start, end) in enumerate(todo):
        marker = os.path.join(stats_dir, f"{start:06d}-{end:06d}.npz")
        if os.path.exists(marker):
            continue
        began = time.time()
        buffer = np.full((end - start,) + data.shape[1:], np.nan, dtype=np.float32)
        done = set()

        def put(name, values, buffer=buffer, done=done):
            if values.ndim > 1:
                values = values.reshape((-1, len(STEPS), buffer.shape[-1]))
            buffer[:, index[name], 0] = values
            done.add(name)

        build_block(sources, base_dates[start:end], put)
        assert done == set(VARIABLES), set(VARIABLES) - done
        buffer[missing[start:end]] = np.nan
        read = time.time()

        da.store(
            da.from_array(buffer, chunks=data.chunks),
            data,
            regions=(slice(start, end),) + (slice(None),) * 4,
            lock=False,
        )
        written = time.time()

        summary = block_summary(buffer, ~missing[start:end], pool)
        np.savez(marker + ".tmp.npz", **summary)
        os.replace(marker + ".tmp.npz", marker)

        LOG.info(
            "Block %d/%d, %s to %s: read and computed in %.0fs, written in %.0fs, statistics in %.0fs",
            number + 1,
            len(todo),
            base_dates[start],
            base_dates[end - 1],
            read - began,
            written - read,
            time.time() - written,
        )


def finalise(args):
    root = zarr.open_group(args.output, mode="r+")
    base_dates = root["base_dates"][:]
    stats_dir = args.output.rstrip("/") + ".stats"

    stats = {key: np.empty((len(base_dates), len(VARIABLES))) for key in STATISTICS}
    constant, hashes = np.ones(len(VARIABLES), dtype=bool), None
    for start, end in blocks(base_dates, args.block):
        path = os.path.join(stats_dir, f"{start:06d}-{end:06d}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Block {start}-{end} is not loaded ({path}); is --block the one used to load?")
        with np.load(path) as block:
            for key in STATISTICS:
                stats[key][start:end] = block[key]
            if block["present"]:
                constant &= block["constant"]
                if hashes is None:
                    hashes = block["hash"]
                constant &= block["hash"] == hashes

    # As anemoi-datasets: the period is of validity times, and a trajectory
    # counts only if all its steps are in it.
    default_start, default_end = statistics_dates(base_dates)
    stats_start = as_datetime64(args.statistics_start or default_start)
    stats_end = as_datetime64(args.statistics_end or default_end)
    kept = (base_dates >= stats_start - STEPS[0]) & (base_dates <= stats_end - STEPS[-1])
    if not kept.any():
        raise ValueError(f"No trajectory lies within {stats_start} to {stats_end}")
    merged = merge_statistics({key: values[kept] for key, values in stats.items()})

    if np.any(merged["count"] == 0):
        raise ValueError(f"No statistics for {[v for v, c in zip(VARIABLES, merged['count']) if c == 0]}")
    variance = np.maximum(merged["m2"] / merged["count"], 0.0)
    for name, values in (
        ("mean", merged["mean"]),
        ("minimum", merged["minimum"]),
        ("maximum", merged["maximum"]),
        ("stdev", np.sqrt(variance)),
    ):
        root.array(name, values.astype(np.float64), overwrite=True).attrs["_ARRAY_DIMENSIONS"] = ["variable"]

    constants = sorted(v for v, c in zip(VARIABLES, constant) if c)
    metadata = dict(root.attrs["variables_metadata"])
    for name in constants:
        metadata[name] = dict(metadata[name], constant_in_time=True)

    files, size = 0, 0
    for directory, _, names in os.walk(args.output):
        files += len(names)
        size += sum(os.path.getsize(os.path.join(directory, n)) for n in names)

    root.attrs.update(
        constant_fields=constants,
        variables_metadata=metadata,
        statistics_start_date=iso(stats_start),
        statistics_end_date=iso(stats_end),
        total_number_of_files=files,
        total_size=size,
        latest_write_timestamp=datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(),
    )
    LOG.info(
        "Statistics of %d trajectories, from %s to %s (validity times)", kept.sum(), iso(stats_start), iso(stats_end)
    )
    LOG.info("Constant fields: %s", constants)
    LOG.info("Statistics written; %s files, %.1f GB", files, size / 1e9)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("init", help="create the dataset and its metadata")
    command.add_argument("output")
    command.add_argument("--template", required=True, help="a dataset anemoi-datasets built from the recipe")
    command.add_argument("--start", default=START, help="first base date (default %(default)s)")
    command.add_argument("--end", default=END, help="last base date (default %(default)s)")
    command.add_argument("--missing", nargs="*", default=MISSING, help="base dates left missing (NaN)")
    command.add_argument("--overwrite", action="store_true")

    command = commands.add_parser("load", help="fill (part of) the dataset; restartable")
    command.add_argument("output")
    command.add_argument("--part", default="1/1", help="the i-th of n parts of the blocks, as i/n")
    command.add_argument("--block", type=int, default=32, help="base dates per block; a divisor of 256")
    command.add_argument("--workers", type=int, default=32, help="threads")
    # parts:
    # wrk1:
    # hertz:
    # volta:
    # ampere:
    # gauss:

    command = commands.add_parser("finalise", help="write the statistics; can be rerun with another period")
    command.add_argument("output")
    command.add_argument("--block", type=int, default=32, help="as given to load")
    command.add_argument(
        "--statistics-start",
        help="first validity time of the statistics, e.g. 1990-09-01T00:00 (default: anemoi's)",
    )
    command.add_argument(
        "--statistics-end",
        help="last validity time of the statistics, e.g. 2020-12-31T23:00 (default: anemoi's)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if SOURCE_CHUNK % getattr(args, "block", SOURCE_CHUNK):
        parser.error(f"--block must divide {SOURCE_CHUNK}")
    if hasattr(args, "workers"):
        dask.config.set(scheduler="threads", num_workers=args.workers)

    {"init": init, "load": load, "finalise": finalise}[args.command](args)


if __name__ == "__main__":
    main()
