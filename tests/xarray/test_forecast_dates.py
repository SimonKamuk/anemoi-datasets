# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for reading Xarray datasets by ``(valid_time, basetime)`` pair.

This is what lets an Xarray-backed source feed the ``trajectories`` layout,
where a field is identified by the run it came from and its lead time rather
than by validity time alone.
"""

import datetime

import numpy as np
import pytest
import xarray as xr

from anemoi.datasets.create.sources.xarray_support import XarrayFieldList
from anemoi.datasets.create.sources.xarray_support import _stamp_forecast_pair
from anemoi.datasets.create.sources.xarray_support.time import Analysis
from anemoi.datasets.create.sources.xarray_support.time import Constant
from anemoi.datasets.create.sources.xarray_support.time import ForecastFromBaseTimeAndDate
from anemoi.datasets.create.sources.xarray_support.time import ForecastFromValidTimeAndBaseTime
from anemoi.datasets.create.sources.xarray_support.time import ForecastFromValidTimeAndStep

N_LAT, N_LON = 3, 4
BASE = datetime.datetime(2021, 1, 1, 12)

GRID_RULES = {
    "latitude": {"name": "latitude"},
    "longitude": {"name": "longitude"},
}


def _grid() -> dict:
    return {
        "latitude": np.linspace(-60.0, 60.0, N_LAT),
        "longitude": np.linspace(0.0, 270.0, N_LON),
    }


def make_forecast_dataset(
    n_runs: int = 4,
    run_frequency_h: int = 6,
    steps_h: tuple[int, ...] = (0, 3, 6, 9),
) -> xr.Dataset:
    """Build a forecast store indexed by run time and lead time.

    ``t`` is filled analytically as ``100 * run_index + step_hours`` so that
    every ``(run, step)`` slot holds a value unique to that pair, and a
    misplaced field is therefore detectable.

    Parameters
    ----------
    n_runs : int
        Number of model runs (base times).
    run_frequency_h : int
        Spacing between runs, in hours.
    steps_h : tuple of int
        The forecast lead times, in hours.

    Returns
    -------
    xr.Dataset
        A dataset with ``analysis_time`` and ``forecast_duration`` dimensions.
    """
    runs = np.array([np.datetime64(BASE + datetime.timedelta(hours=run_frequency_h * i), "ns") for i in range(n_runs)])
    steps = np.array([np.timedelta64(h, "h").astype("timedelta64[ns]") for h in steps_h])

    values = np.empty((n_runs, len(steps_h), N_LAT, N_LON), dtype="float64")
    for i in range(n_runs):
        for j, h in enumerate(steps_h):
            values[i, j] = 100 * i + h

    return xr.Dataset(
        {"t": (("analysis_time", "forecast_duration", "latitude", "longitude"), values)},
        coords={
            "analysis_time": ("analysis_time", runs, {"standard_name": "forecast_reference_time"}),
            "forecast_duration": ("forecast_duration", steps, {"standard_name": "forecast_period"}),
            **_grid(),
        },
    )


def make_analysis_dataset(n_times: int = 8, frequency_h: int = 3) -> xr.Dataset:
    """Build an analysis store indexed by validity time, plus a constant field.

    Parameters
    ----------
    n_times : int
        Number of analysis times.
    frequency_h : int
        Spacing between analysis times, in hours.

    Returns
    -------
    xr.Dataset
        A dataset with a time-varying ``t`` and a time-invariant ``lsm``.
    """
    times = np.array([np.datetime64(BASE + datetime.timedelta(hours=frequency_h * i), "ns") for i in range(n_times)])
    values = np.empty((n_times, N_LAT, N_LON), dtype="float64")
    for i in range(n_times):
        values[i] = 1000 + i

    return xr.Dataset(
        {
            "t": (("time", "latitude", "longitude"), values),
            "lsm": (("latitude", "longitude"), np.full((N_LAT, N_LON), 7.0)),
        },
        coords={"time": ("time", times, {"standard_name": "time"}), **_grid()},
    )


def forecast_fieldlist() -> XarrayFieldList:
    """Return the forecast store as a field list keyed by run and step."""
    flavour = {
        "rules": {
            **GRID_RULES,
            "date": {"name": "analysis_time"},
            "step": {"name": "forecast_duration"},
        }
    }
    return XarrayFieldList.from_xarray(make_forecast_dataset(), flavour=flavour)


def analysis_fieldlist() -> XarrayFieldList:
    """Return the analysis store as a field list keyed by validity time."""
    flavour = {"rules": {**GRID_RULES, "time": {"name": "time"}}}
    return XarrayFieldList.from_xarray(make_analysis_dataset(), flavour=flavour)


# ---------------------------------------------------------------------------
# Time.forecast_selection
# ---------------------------------------------------------------------------


def test_time_class_detected_from_flavour() -> None:
    """A run/lead-time store is recognised as a forecast, not as an analysis."""
    assert isinstance(forecast_fieldlist().variables[0].time, ForecastFromBaseTimeAndDate)
    assert isinstance(analysis_fieldlist().variables[0].time, Analysis)


def test_forecast_selection_uses_coordinate_names() -> None:
    """The selection is keyed by coordinate name, so ``sel`` can slice directly.

    Selecting on the MARS-style ``date``/``step`` aliases would fall through to
    a metadata scan over every field of the variable.
    """
    time = ForecastFromBaseTimeAndDate.__new__(ForecastFromBaseTimeAndDate)
    time.date_coordinate_name = "analysis_time"
    time.step_coordinate_name = "forecast_duration"

    selection = time.forecast_selection(BASE + datetime.timedelta(hours=6), BASE)

    assert selection == {
        "analysis_time": BASE,
        "forecast_duration": np.timedelta64(6 * 3600, "s"),
    }


def test_forecast_selection_step_is_numpy_timedelta() -> None:
    """The step must be a ``numpy.timedelta64``.

    Step coordinates hold ``timedelta64`` values: an ``int`` of hours does not
    compare equal to one, and a :class:`datetime.timedelta` raises a
    ``TypeError`` inside ``numpy.searchsorted``.
    """
    time = ForecastFromBaseTimeAndDate.__new__(ForecastFromBaseTimeAndDate)
    time.date_coordinate_name = "analysis_time"
    time.step_coordinate_name = "forecast_duration"

    step = time.forecast_selection(BASE + datetime.timedelta(hours=3), BASE)["forecast_duration"]

    assert isinstance(step, np.timedelta64)


def test_forecast_selection_analysis_ignores_basetime() -> None:
    """An analysis is identified by its validity time alone."""
    time = Analysis.__new__(Analysis)
    time.time_coordinate_name = "time"

    valid_time = BASE + datetime.timedelta(hours=6)

    assert time.forecast_selection(valid_time, BASE) == {"time": valid_time}


def test_forecast_selection_constant_serves_every_pair() -> None:
    """A constant field needs no selection: it is valid for every pair."""
    assert Constant().forecast_selection(BASE + datetime.timedelta(hours=6), BASE) == {}


def test_forecast_selection_valid_time_and_step() -> None:
    """A validity-time/step store selects on both."""
    time = ForecastFromValidTimeAndStep.__new__(ForecastFromValidTimeAndStep)
    time.time_coordinate_name = "time"
    time.step_coordinate_name = "step"
    time.date_coordinate_name = None

    valid_time = BASE + datetime.timedelta(hours=6)

    assert time.forecast_selection(valid_time, BASE) == {
        "time": valid_time,
        "step": np.timedelta64(6 * 3600, "s"),
    }


def test_forecast_selection_valid_time_and_base_time() -> None:
    """A validity-time/base-time store selects on both."""
    time = ForecastFromValidTimeAndBaseTime.__new__(ForecastFromValidTimeAndBaseTime)
    time.date_coordinate_name = "reference_time"
    time.time_coordinate_name = "time"

    valid_time = BASE + datetime.timedelta(hours=6)

    assert time.forecast_selection(valid_time, BASE) == {
        "reference_time": BASE,
        "time": valid_time,
    }


# ---------------------------------------------------------------------------
# XarrayFieldList.sel_forecast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step_hours", [0, 3, 6, 9])
def test_sel_forecast_picks_the_requested_step(step_hours: int) -> None:
    """Each step of a run is retrieved, and carries that run and step."""
    fs = forecast_fieldlist()
    valid_time = BASE + datetime.timedelta(hours=step_hours)

    selected = fs.sel_forecast(valid_time, BASE)

    assert len(selected) == 1
    field = selected[0]
    assert field.metadata("date") == 20210101
    assert field.metadata("time") == 1200
    assert field.metadata("step") == step_hours
    assert field.metadata("valid_datetime") == valid_time.isoformat()
    # 100 * run_index + step_hours, with run_index == 0 for BASE.
    assert field.to_numpy().flatten()[0] == pytest.approx(step_hours)


def test_sel_forecast_disambiguates_overlapping_runs() -> None:
    """The same validity time is served from whichever run is asked for.

    ``BASE + 6h`` is reachable as ``BASE + step 6h`` and as the second run
    (``BASE + 6h``) at ``step 0``.  Selecting by validity time alone could not
    tell the two apart; the pair does.
    """
    fs = forecast_fieldlist()
    valid_time = BASE + datetime.timedelta(hours=6)

    from_first_run = fs.sel_forecast(valid_time, BASE)
    from_second_run = fs.sel_forecast(valid_time, valid_time)

    assert len(from_first_run) == 1
    assert len(from_second_run) == 1

    assert from_first_run[0].metadata("step") == 6
    assert from_second_run[0].metadata("step") == 0

    # run 0 step 6 -> 6 ; run 1 step 0 -> 100
    assert from_first_run[0].to_numpy().flatten()[0] == pytest.approx(6)
    assert from_second_run[0].to_numpy().flatten()[0] == pytest.approx(100)


def test_sel_forecast_returns_nothing_for_an_absent_step() -> None:
    """A step the store does not hold yields no field rather than a wrong one."""
    fs = forecast_fieldlist()

    assert len(fs.sel_forecast(BASE + datetime.timedelta(hours=1), BASE)) == 0


def test_sel_forecast_mixes_analysis_and_constant_variables() -> None:
    """One store may hold both a time-varying and a time-invariant variable.

    Each variable is asked separately how to resolve the pair, so the constant
    is returned alongside the analysis instead of being dropped.
    """
    fs = analysis_fieldlist()

    params = sorted(f.metadata("param") for f in fs.sel_forecast(BASE, BASE))

    assert params == ["lsm", "t"]


def test_sel_forecast_keeps_constants_when_the_analysis_has_no_such_time() -> None:
    """A constant is still served at a step the analysis does not cover."""
    fs = analysis_fieldlist()

    # The analysis is 3-hourly, so there is nothing valid at BASE + 1h.
    params = sorted(f.metadata("param") for f in fs.sel_forecast(BASE + datetime.timedelta(hours=1), BASE))

    assert params == ["lsm"]


# ---------------------------------------------------------------------------
# _stamp_forecast_pair
# ---------------------------------------------------------------------------


def test_stamp_labels_constants_with_the_requested_pair() -> None:
    """A constant field, which carries no time metadata, becomes placeable."""
    fs = analysis_fieldlist()
    valid_time = BASE + datetime.timedelta(hours=2)

    selected = fs.sel_forecast(valid_time, BASE, param="lsm")
    assert selected[0].metadata("date", default=None) is None

    stamped = _stamp_forecast_pair(selected, valid_time, BASE)

    assert stamped[0].metadata("date") == 20210101
    assert stamped[0].metadata("time") == 1200
    assert stamped[0].metadata("step") == 2
    assert stamped[0].metadata("valid_datetime") == valid_time.isoformat()


def test_stamp_relabels_an_analysis_served_at_a_later_step() -> None:
    """An analysis-like store reports the validity time as its run time.

    Without re-labelling, a field served for step 3 would be filed under a base
    time three hours late.  This is the case for a forecast archive flattened
    onto a validity-time axis by ``analysis_lead_to_valid_time``.
    """
    fs = analysis_fieldlist()
    valid_time = BASE + datetime.timedelta(hours=3)

    selected = fs.sel_forecast(valid_time, BASE, param="t")
    # The store's own view: run time == validity time, step 0.
    assert selected[0].metadata("time") == 1500
    assert selected[0].metadata("step") == 0

    stamped = _stamp_forecast_pair(selected, valid_time, BASE)

    assert stamped[0].metadata("time") == 1200
    assert stamped[0].metadata("step") == 3
    assert stamped[0].metadata("valid_datetime") == valid_time.isoformat()


def test_stamp_is_a_no_op_for_a_native_forecast_store() -> None:
    """A store already keyed by run and step keeps the values it derived."""
    fs = forecast_fieldlist()
    valid_time = BASE + datetime.timedelta(hours=6)

    selected = fs.sel_forecast(valid_time, BASE)
    before = {k: selected[0].metadata(k) for k in ("date", "time", "step", "valid_datetime")}

    stamped = _stamp_forecast_pair(selected, valid_time, BASE)

    assert {k: stamped[0].metadata(k) for k in before} == before


def test_stamp_rejects_a_sub_hourly_step() -> None:
    """Whole-hour steps are assumed throughout the trajectories pipeline."""
    fs = forecast_fieldlist()
    valid_time = BASE + datetime.timedelta(minutes=90)

    with pytest.raises(AssertionError):
        _stamp_forecast_pair(fs.sel_forecast(BASE, BASE), valid_time, BASE)
