# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for preloading a window of an Xarray dataset's time axis.

Fields are otherwise read one at a time, each triggering its own read of the
underlying store.  When the store is chunked with many time steps per chunk, a
whole chunk has to be decompressed to yield a single time slice, so the same
chunk is decompressed once per field.  Preloading the window removes that
repetition; these tests pin down that it changes performance only, never
values.
"""

import datetime

import numpy as np
import pytest
import xarray as xr

from anemoi.datasets.create.sources.xarray_support import DEFAULT_PRELOAD_BUDGET
from anemoi.datasets.create.sources.xarray_support import XarrayFieldList
from anemoi.datasets.create.sources.xarray_support import _preload_fieldlist
from anemoi.datasets.create.sources.xarray_support.time import Analysis
from anemoi.datasets.create.sources.xarray_support.time import Constant
from anemoi.datasets.create.sources.xarray_support.time import ForecastFromBaseTimeAndDate
from anemoi.datasets.create.sources.xarray_support.time import ForecastFromValidTimeAndBaseTime
from anemoi.datasets.create.sources.xarray_support.time import ForecastFromValidTimeAndStep

N_LAT, N_LON = 3, 4
BASE = datetime.datetime(2021, 1, 1, 0)
N_RUNS = 8
RUN_FREQUENCY_H = 6
STEPS_H = (0, 3, 6)

GRID_RULES = {
    "latitude": {"name": "latitude"},
    "longitude": {"name": "longitude"},
}
FORECAST_FLAVOUR = {
    "rules": {
        **GRID_RULES,
        "date": {"name": "analysis_time"},
        "step": {"name": "forecast_duration"},
    }
}
ANALYSIS_FLAVOUR = {"rules": {**GRID_RULES, "time": {"name": "time"}}}


def _grid() -> dict:
    return {
        "latitude": np.linspace(-60.0, 60.0, N_LAT),
        "longitude": np.linspace(0.0, 270.0, N_LON),
    }


def make_forecast_dataset() -> xr.Dataset:
    """Build a dask-backed forecast store indexed by run time and lead time.

    ``t`` holds ``100 * run_index + step_hours``, so every ``(run, step)`` slot
    is unique and a misplaced field would be detectable.

    Returns
    -------
    xr.Dataset
        A chunked dataset with ``analysis_time`` and ``forecast_duration``.
    """
    runs = np.array([np.datetime64(BASE + datetime.timedelta(hours=RUN_FREQUENCY_H * i), "ns") for i in range(N_RUNS)])
    steps = np.array([np.timedelta64(h, "h").astype("timedelta64[ns]") for h in STEPS_H])

    values = np.empty((N_RUNS, len(STEPS_H), N_LAT, N_LON), dtype="float64")
    for i in range(N_RUNS):
        for j, h in enumerate(STEPS_H):
            values[i, j] = 100 * i + h

    ds = xr.Dataset(
        {"t": (("analysis_time", "forecast_duration", "latitude", "longitude"), values)},
        coords={
            "analysis_time": ("analysis_time", runs, {"standard_name": "forecast_reference_time"}),
            "forecast_duration": ("forecast_duration", steps, {"standard_name": "forecast_period"}),
            **_grid(),
        },
    )
    # Chunk over the whole time axis, the layout that makes per-field reads
    # expensive in the first place.
    return ds.chunk({"analysis_time": N_RUNS})


def make_analysis_dataset() -> xr.Dataset:
    """Build a dask-backed analysis store with a constant field alongside.

    Returns
    -------
    xr.Dataset
        A chunked dataset with a time-varying ``t`` and a constant ``lsm``.
    """
    times = np.array([np.datetime64(BASE + datetime.timedelta(hours=3 * i), "ns") for i in range(N_RUNS)])
    values = np.empty((N_RUNS, N_LAT, N_LON), dtype="float64")
    for i in range(N_RUNS):
        values[i] = 1000 + i

    ds = xr.Dataset(
        {
            "t": (("time", "latitude", "longitude"), values),
            "lsm": (("latitude", "longitude"), np.full((N_LAT, N_LON), 7.0)),
        },
        coords={"time": ("time", times, {"standard_name": "time"}), **_grid()},
    )
    return ds.chunk({"time": N_RUNS})


def make_constant_dataset() -> xr.Dataset:
    """Build a dataset with no time axis at all.

    Returns
    -------
    xr.Dataset
        A dataset holding only a constant field.
    """
    return xr.Dataset(
        {"lsm": (("latitude", "longitude"), np.full((N_LAT, N_LON), 7.0))},
        coords=_grid(),
    )


def forecast_fieldlist() -> XarrayFieldList:
    """Return the forecast store as a field list."""
    return XarrayFieldList.from_xarray(make_forecast_dataset(), flavour=FORECAST_FLAVOUR)


def analysis_fieldlist() -> XarrayFieldList:
    """Return the analysis store as a field list."""
    return XarrayFieldList.from_xarray(make_analysis_dataset(), flavour=ANALYSIS_FLAVOUR)


# ---------------------------------------------------------------------------
# Time.preload_window
# ---------------------------------------------------------------------------


def test_preload_window_on_run_time_axis() -> None:
    """A run/lead-time store is windowed on its run-time coordinate."""
    time = ForecastFromBaseTimeAndDate.__new__(ForecastFromBaseTimeAndDate)
    time.date_coordinate_name = "analysis_time"
    time.step_coordinate_name = "forecast_duration"

    last = BASE + datetime.timedelta(hours=12)

    assert time.preload_window(BASE, last) == {"analysis_time": slice(BASE, last)}


@pytest.mark.parametrize(
    "cls,attrs,dim",
    [
        (Analysis, {"time_coordinate_name": "time"}, "time"),
        (
            ForecastFromValidTimeAndStep,
            {"time_coordinate_name": "time", "step_coordinate_name": "step", "date_coordinate_name": None},
            "time",
        ),
        (
            ForecastFromValidTimeAndBaseTime,
            {"date_coordinate_name": "reference_time", "time_coordinate_name": "time"},
            "time",
        ),
    ],
)
def test_preload_window_on_validity_time_axis(cls: type, attrs: dict, dim: str) -> None:
    """Validity-time-indexed stores are windowed on their time coordinate."""
    time = cls.__new__(cls)
    for name, value in attrs.items():
        setattr(time, name, value)

    last = BASE + datetime.timedelta(hours=12)

    assert time.preload_window(BASE, last) == {dim: slice(BASE, last)}


def test_preload_window_empty_for_a_constant() -> None:
    """A constant field has no time axis, so there is nothing to window."""
    assert Constant().preload_window(BASE, BASE + datetime.timedelta(hours=12)) == {}


# ---------------------------------------------------------------------------
# _preload_fieldlist
# ---------------------------------------------------------------------------


def test_preload_off_by_default_returns_the_same_fieldlist() -> None:
    """``preload=False`` leaves the field list untouched."""
    fs = forecast_fieldlist()

    assert _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=12), FORECAST_FLAVOUR, False) is fs


def test_preload_gives_a_memory_backed_fieldlist() -> None:
    """After preloading, the dataset is no longer lazy."""
    fs = forecast_fieldlist()
    assert fs.ds["t"].chunks is not None

    loaded = _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=12), FORECAST_FLAVOUR, True)

    assert loaded is not fs
    assert loaded.ds["t"].chunks is None


def test_preload_narrows_the_dataset_to_the_window() -> None:
    """Only the requested window is held in memory."""
    fs = forecast_fieldlist()
    last = BASE + datetime.timedelta(hours=RUN_FREQUENCY_H)  # two runs

    loaded = _preload_fieldlist(fs, BASE, last, FORECAST_FLAVOUR, True)

    assert fs.ds.sizes["analysis_time"] == N_RUNS
    assert loaded.ds.sizes["analysis_time"] == 2
    # The step axis is left whole.
    assert loaded.ds.sizes["forecast_duration"] == len(STEPS_H)


@pytest.mark.parametrize("step_hours", STEPS_H)
def test_preload_preserves_forecast_values_and_metadata(step_hours: int) -> None:
    """A preloaded field is the same field, with the same run and step."""
    fs = forecast_fieldlist()
    basetime = BASE + datetime.timedelta(hours=RUN_FREQUENCY_H)  # run index 1
    valid_time = basetime + datetime.timedelta(hours=step_hours)

    lazy = fs.sel_forecast(valid_time, basetime)[0]
    loaded = _preload_fieldlist(
        fs, BASE, BASE + datetime.timedelta(hours=RUN_FREQUENCY_H * N_RUNS), FORECAST_FLAVOUR, True
    ).sel_forecast(valid_time, basetime)[0]

    assert np.array_equal(loaded.to_numpy(), lazy.to_numpy())
    # 100 * run_index + step_hours
    assert loaded.to_numpy().flatten()[0] == pytest.approx(100 + step_hours)
    for key in ("date", "time", "step", "valid_datetime", "param"):
        assert loaded.metadata(key) == lazy.metadata(key)


def test_preload_preserves_every_field_of_the_window() -> None:
    """Every ``(run, step)`` pair of the window survives preloading intact."""
    fs = forecast_fieldlist()
    last_run = BASE + datetime.timedelta(hours=RUN_FREQUENCY_H * (N_RUNS - 1))
    loaded = _preload_fieldlist(fs, BASE, last_run, FORECAST_FLAVOUR, True)

    checked = 0
    for i in range(N_RUNS):
        basetime = BASE + datetime.timedelta(hours=RUN_FREQUENCY_H * i)
        for step_hours in STEPS_H:
            valid_time = basetime + datetime.timedelta(hours=step_hours)
            before = fs.sel_forecast(valid_time, basetime)
            after = loaded.sel_forecast(valid_time, basetime)
            assert len(before) == len(after) == 1
            assert np.array_equal(after[0].to_numpy(), before[0].to_numpy())
            checked += 1

    assert checked == N_RUNS * len(STEPS_H)


def test_preload_preserves_analysis_values_and_constants() -> None:
    """A store mixing a time-varying and a constant field preloads correctly."""
    fs = analysis_fieldlist()
    loaded = _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=21), ANALYSIS_FLAVOUR, True)

    for param in ("t", "lsm"):
        before = fs.sel(valid_datetime=BASE, param=param)
        after = loaded.sel(valid_datetime=BASE, param=param)
        # A constant has no time coordinate, so it does not answer a
        # validity-time selection in either field list.
        assert len(before) == len(after)
        if len(before):
            assert np.array_equal(after[0].to_numpy(), before[0].to_numpy())

    assert loaded.ds["lsm"].shape == (N_LAT, N_LON)


def test_preload_skips_a_dataset_with_no_time_axis() -> None:
    """A constant-only dataset has nothing to window, so it is left alone."""
    fs = XarrayFieldList.from_xarray(make_constant_dataset(), flavour={"rules": GRID_RULES})

    assert _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=12), {"rules": GRID_RULES}, True) is fs


def test_preload_skips_an_empty_fieldlist() -> None:
    """An empty field list is returned unchanged."""
    fs = XarrayFieldList(make_forecast_dataset(), [])

    assert _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=12), FORECAST_FLAVOUR, True) is fs


# ---------------------------------------------------------------------------
# The memory budget
# ---------------------------------------------------------------------------


def test_preload_declines_to_exceed_the_budget(caplog: pytest.LogCaptureFixture) -> None:
    """Over budget, reading falls back to a field at a time rather than failing."""
    fs = forecast_fieldlist()

    with caplog.at_level("WARNING"):
        result = _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=42), FORECAST_FLAVOUR, 1)

    assert result is fs
    assert "over the" in caplog.text


def test_preload_accepts_a_human_readable_budget() -> None:
    """The budget may be given as a size string as well as a byte count."""
    fs = forecast_fieldlist()
    window = (BASE, BASE + datetime.timedelta(hours=42))

    assert _preload_fieldlist(fs, *window, FORECAST_FLAVOUR, "1GiB").ds["t"].chunks is None
    assert _preload_fieldlist(fs, *window, FORECAST_FLAVOUR, "1K") is fs


def test_default_budget_is_a_valid_size() -> None:
    """``preload: true`` resolves to a usable byte budget."""
    from anemoi.utils.humanize import human_to_bytes

    assert human_to_bytes(DEFAULT_PRELOAD_BUDGET) > 0


# ---------------------------------------------------------------------------
# Narrowing the window to what is being asked for
# ---------------------------------------------------------------------------


def make_multi_variable_dataset() -> xr.Dataset:
    """Build a store holding more variables and levels than a caller wants.

    Returns
    -------
    xr.Dataset
        A chunked dataset with four variables on four levels.
    """
    times = np.array([np.datetime64(BASE + datetime.timedelta(hours=3 * i), "ns") for i in range(N_RUNS)])
    levels = np.array([250, 500, 850, 1000])
    shape = (N_RUNS, len(levels), N_LAT, N_LON)

    ds = xr.Dataset(
        {name: (("time", "level", "latitude", "longitude"), np.full(shape, i + 1.0)) for i, name in enumerate("qrtu")},
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "level": ("level", levels, {"units": "hPa"}),
            **_grid(),
        },
    )
    return ds.chunk({"time": N_RUNS})


def multi_variable_fieldlist() -> XarrayFieldList:
    """Return the four-variable store as a field list."""
    flavour = {"rules": {**GRID_RULES, "time": {"name": "time"}, "level": {"name": "level"}}, "levtype": "pl"}
    return XarrayFieldList.from_xarray(make_multi_variable_dataset(), flavour=flavour), flavour


def test_preload_drops_unrequested_variables() -> None:
    """Only the requested parameters are read."""
    fs, flavour = multi_variable_fieldlist()
    assert set(fs.ds.data_vars) == set("qrtu")

    loaded = _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=21), flavour, True, param=["t", "r"])

    assert set(loaded.ds.data_vars) == {"t", "r"}


def test_preload_drops_unrequested_levels() -> None:
    """A criterion naming a dimension narrows that dimension."""
    fs, flavour = multi_variable_fieldlist()
    assert fs.ds.sizes["level"] == 4

    loaded = _preload_fieldlist(
        fs, BASE, BASE + datetime.timedelta(hours=21), flavour, True, param=["t"], level=[500, 850]
    )

    assert list(loaded.ds["level"].values) == [500, 850]


def test_preload_keeps_a_dimension_it_would_reduce_to_one_value() -> None:
    """Narrowing to a single value is skipped, to keep field shapes intact.

    A length-one coordinate is treated as scalar, and ``Variable.sel`` then
    keeps the dimension rather than indexing it away, which would give the
    fields an extra leading dimension.
    """
    fs, flavour = multi_variable_fieldlist()

    loaded = _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=21), flavour, True, param=["t"], level=[500])

    assert loaded.ds.sizes["level"] == 4


@pytest.mark.parametrize("levels", [[500], [500, 850], [250, 500, 850, 1000]])
def test_preload_narrowing_preserves_values_and_shape(levels: list) -> None:
    """Narrowing changes what is read, never a field's contents or shape."""
    fs, flavour = multi_variable_fieldlist()
    loaded = _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=21), flavour, True, param=["t"], level=levels)

    before = fs.sel(valid_datetime=BASE, param="t", level=levels[0])
    after = loaded.sel(valid_datetime=BASE, param="t", level=levels[0])

    assert len(before) == len(after) == 1
    assert after[0].to_numpy().shape == before[0].to_numpy().shape
    assert np.array_equal(after[0].to_numpy(), before[0].to_numpy())
    assert after[0].shape == before[0].shape
    assert after[0].metadata("levelist") == levels[0]


def test_preload_skips_when_the_request_selects_nothing() -> None:
    """A request naming none of this dataset's variables is left alone."""
    fs, flavour = multi_variable_fieldlist()

    assert _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=21), flavour, True, param=["nope"]) is fs


def test_preload_falls_back_when_narrowing_fails() -> None:
    """An unusable criterion falls back to reading a field at a time."""
    fs, flavour = multi_variable_fieldlist()

    result = _preload_fieldlist(
        fs, BASE, BASE + datetime.timedelta(hours=21), flavour, True, param=["t"], level=[12345, 67890]
    )

    assert result is fs


def test_preload_skips_a_single_step_window() -> None:
    """A one-element window is left alone.

    It would replace exactly one read with one read, and a length-one
    coordinate is treated as scalar, so ``Variable.sel`` would keep the
    dimension instead of indexing it away and the fields would come out with an
    extra dimension.
    """
    fs = forecast_fieldlist()

    assert _preload_fieldlist(fs, BASE, BASE, FORECAST_FLAVOUR, True) is fs


def test_preload_keeps_field_shape_for_a_time_last_dataset() -> None:
    """A dataset whose time axis is its *last* dimension still round-trips.

    Flattening a forecast archive onto a validity-time axis (the
    ``analysis_lead_to_valid_time`` patch) appends the new axis, leaving dims
    ordered ``(y, x, time)``.  ``XArrayField`` reads a field's shape from the
    trailing two dimensions, so a retained length-one time axis there would be
    mistaken for part of the grid.
    """
    times = np.array([np.datetime64(BASE + datetime.timedelta(hours=i), "ns") for i in range(4)])
    ds = xr.Dataset(
        {"t": (("latitude", "longitude", "time"), np.zeros((N_LAT, N_LON, 4)))},
        coords={"time": ("time", times, {"standard_name": "time"}), **_grid()},
    ).chunk({"time": 4})
    fs = XarrayFieldList.from_xarray(ds, flavour=ANALYSIS_FLAVOUR)

    lazy = fs.sel(valid_datetime=BASE, param="t")[0]
    loaded = _preload_fieldlist(fs, BASE, BASE + datetime.timedelta(hours=3), ANALYSIS_FLAVOUR, True)
    after = loaded.sel(valid_datetime=BASE, param="t")[0]

    assert after.shape == lazy.shape == (N_LAT, N_LON)
    assert after.to_numpy().shape == lazy.to_numpy().shape


def test_preload_skips_constants_selected_from_a_mixed_store(caplog: pytest.LogCaptureFixture) -> None:
    """Selecting only the constants of a mixed store is not an error.

    The window has to come from the variables actually wanted: taking it from
    all of them would name a time dimension the narrowed dataset does not have,
    and the selection would raise.
    """
    fs = analysis_fieldlist()
    assert "time" in fs.ds.dims

    with caplog.at_level("WARNING"):
        result = _preload_fieldlist(
            fs, BASE, BASE + datetime.timedelta(hours=21), ANALYSIS_FLAVOUR, True, param=["lsm"]
        )

    assert result is fs
    assert caplog.text == ""
