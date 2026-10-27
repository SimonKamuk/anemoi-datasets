# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the patches that derive or mask variables of an Xarray dataset."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from earthkit.meteo.solar import cos_solar_zenith_angle

from anemoi.datasets.create.sources.xarray_support.patch import patch_dataset
from anemoi.datasets.create.sources.xarray_support.patch import patch_mask_night
from anemoi.datasets.create.sources.xarray_support.patch import patch_rolling_operation
from anemoi.datasets.create.sources.xarray_support.patch import patch_valid_range
from anemoi.datasets.create.sources.xarray_support.patch import patch_vector_magnitude

HOURS = pd.date_range("2021-03-20T00:00", periods=48, freq="1h")


def _series(values: list[float]) -> xr.Dataset:
    """A one-point dataset holding ``values`` along an hourly time axis."""
    time = HOURS[: len(values)]
    return xr.Dataset({"a": ("time", np.array(values, dtype=np.float32))}, coords={"time": time})


def _grid(chunks: dict | None = None) -> xr.Dataset:
    """Two days of hourly ones on a small curvilinear grid spanning longitudes."""
    lat, lon = np.meshgrid([10.0, 55.0], [-120.0, 0.0, 120.0], indexing="ij")
    ds = xr.Dataset(
        {"albedo": (("time", "y", "x"), np.ones((len(HOURS), 2, 3), dtype=np.float32), {"units": "1"})},
        coords={"time": HOURS, "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )
    return ds.chunk(chunks) if chunks else ds


def test_vector_magnitude_keeps_units_and_components() -> None:
    ds = xr.Dataset(
        {
            "u": ("time", np.array([3.0, 0.0]), {"units": "m s-1", "standard_name": "eastward_wind"}),
            "v": ("time", np.array([4.0, -2.0]), {"units": "m s-1"}),
        }
    )
    ds = patch_vector_magnitude(ds, {"speed": ["u", "v"]})

    np.testing.assert_allclose(ds["speed"].values, [5.0, 2.0])
    assert ds["speed"].attrs == {"units": "m s-1"}
    assert {"u", "v"} <= set(ds.data_vars)


def test_rolling_last_takes_the_latest_value_that_is_not_missing() -> None:
    nan = np.nan
    ds = patch_rolling_operation(
        _series([1, nan, nan, 4, nan, nan, nan]),
        {"dim": "time", "steps": 3, "vars": ["a"], "operation": "last"},
    )
    # Windows of three: the value carries over two missing steps, not three.
    np.testing.assert_array_equal(ds["a"].values, [1, 1, 1, 4, 4, 4, nan])


def test_rolling_last_is_the_same_across_chunks() -> None:
    """Chunk boundaries are bridged by the overlap, so chunking changes nothing."""
    rng = np.random.default_rng(0)
    values = rng.random(40).astype(np.float32)
    values[rng.random(40) < 0.6] = np.nan
    config = {"dim": "time", "steps": 5, "vars": ["a"], "operation": "last"}

    eager = patch_rolling_operation(_series(list(values)), config)["a"]
    lazy = patch_rolling_operation(_series(list(values)).chunk({"time": 3}), config)["a"]

    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, eager.values)


def test_rolling_last_keeps_attributes() -> None:
    ds = _series([1.0, np.nan])
    ds["a"].attrs["units"] = "1"
    ds = patch_rolling_operation(ds, {"dim": "time", "steps": 2, "vars": ["a"], "operation": "last"})
    assert ds["a"].attrs == {"units": "1"}


def test_mask_night_follows_the_cos_solar_zenith_angle_forcing() -> None:
    ds = patch_mask_night(_grid(), {"vars": ["albedo"], "latitude": "lat", "longitude": "lon"})
    lat, lon = ds["lat"].values, ds["lon"].values

    for i, time in enumerate(HOURS):
        date = time.to_pydatetime()
        daylight = cos_solar_zenith_angle(date, lat, lon) > 0
        np.testing.assert_array_equal(~np.isnan(ds["albedo"].values[i]), daylight, err_msg=str(date))

    # The grid spans the day line, so every hour has both day and night.
    assert np.isnan(ds["albedo"].values).any(axis=(1, 2)).all()
    assert (~np.isnan(ds["albedo"].values)).any(axis=(1, 2)).all()
    assert ds["albedo"].attrs == {"units": "1"}


def test_mask_night_threshold_masks_low_sun() -> None:
    config = {"vars": ["albedo"], "latitude": "lat", "longitude": "lon"}
    horizon = patch_mask_night(_grid(), config)["albedo"].values
    low_sun = patch_mask_night(_grid(), {**config, "min_cos_solar_zenith_angle": 0.5})["albedo"].values
    assert np.isnan(low_sun).sum() > np.isnan(horizon).sum()
    assert np.isnan(low_sun[np.isnan(horizon)]).all()


def test_mask_night_is_lazy_and_chunked_like_the_variable() -> None:
    ds = patch_mask_night(_grid({"time": 12, "x": 2}), {"vars": ["albedo"], "latitude": "lat", "longitude": "lon"})
    albedo = ds["albedo"]
    assert albedo.chunks is not None
    assert albedo.chunksizes["time"] == (12, 12, 12, 12)

    eager = patch_mask_night(_grid(), {"vars": ["albedo"], "latitude": "lat", "longitude": "lon"})
    np.testing.assert_array_equal(albedo.values, eager["albedo"].values)


def test_mask_night_rejects_unknown_options() -> None:
    with pytest.raises(AssertionError, match="Unknown mask_night options"):
        patch_mask_night(_grid(), {"vars": ["albedo"], "latitude": "lat", "longitude": "lon", "treshold": 0.1})


def test_latest_daylight_value_through_patch_dataset() -> None:
    """Masking then taking the last value holds the day's value through the night.

    The patches are given in the wrong order on purpose: ``patch_dataset``
    applies them in its fixed order, masking before the rolling operation.
    """
    ds = _grid()
    ds["albedo"] = ds["albedo"] * xr.DataArray(np.arange(len(HOURS), dtype=np.float32), dims="time")
    patched = patch_dataset(
        ds.copy(),
        {
            "rolling_operation": {"dim": "time", "steps": 24, "vars": ["albedo"], "operation": "last"},
            "mask_night": {"vars": ["albedo"], "latitude": "lat", "longitude": "lon"},
        },
    )
    lat, lon = ds["lat"].values, ds["lon"].values
    day = np.array([cos_solar_zenith_angle(t.to_pydatetime(), lat, lon) > 0 for t in HOURS])
    # The value is the hour index of the latest daylight hour at or before each hour.
    last_day = np.where(day, np.arange(len(HOURS))[:, None, None], -1)
    last_day = np.maximum.accumulate(last_day, axis=0).astype(np.float32)
    expected = np.where(last_day < 0, np.nan, last_day)
    np.testing.assert_array_equal(patched["albedo"].values, expected)


def test_gust_maximum_is_the_maximum_of_the_speed() -> None:
    """The rolling maximum is taken of the speed, not of each component."""
    u = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 10.0, 0.0], dtype=np.float32)
    ds = xr.Dataset({"u": ("time", u), "v": ("time", v)}, coords={"time": HOURS[:3]})
    ds = patch_dataset(
        ds,
        {
            "rolling_operation": {"dim": "time", "steps": 2, "vars": ["gust"], "operation": "max"},
            "vector_magnitude": {"gust": ["u", "v"]},
        },
    )
    # Taking the maximum per component first would give hypot(10, 10) = 14.1.
    np.testing.assert_allclose(ds["gust"].values, [np.nan, 10.0, 10.0])


def test_valid_range_masks_values_outside_it() -> None:
    nan = np.nan
    ds = _series([0.2, -0.1, 0.0, 1.0, 1.5])
    ds["a"].attrs = {"units": "1"}

    np.testing.assert_allclose(
        patch_valid_range(ds.copy(), {"vars": ["a"], "minimum": 0})["a"].values, [0.2, nan, 0.0, 1.0, 1.5]
    )
    both = patch_valid_range(ds.copy(), {"vars": ["a"], "minimum": 0, "maximum": 1})
    np.testing.assert_allclose(both["a"].values, [0.2, nan, 0.0, 1.0, nan])
    assert both["a"].attrs == {"units": "1"}


def test_valid_range_needs_a_bound() -> None:
    with pytest.raises(AssertionError, match="needs a minimum or a maximum"):
        patch_valid_range(_series([1.0]), {"vars": ["a"]})


def test_valid_range_masks_before_the_latest_value_is_taken() -> None:
    """A value outside the range is replaced by the latest valid one, whatever
    order the patches are given in.
    """
    ds = patch_dataset(
        _series([0.2, 0.3, -0.5, 0.4]),
        {
            "rolling_operation": {"dim": "time", "steps": 3, "vars": ["a"], "operation": "last"},
            "valid_range": {"vars": ["a"], "minimum": 0},
        },
    )
    np.testing.assert_allclose(ds["a"].values, [0.2, 0.3, 0.3, 0.4])
