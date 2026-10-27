# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from earthkit.meteo.solar import cos_solar_zenith_angle

LOG = logging.getLogger(__name__)


def patch_attributes(ds: xr.Dataset, attributes: dict[str, dict[str, Any]]) -> xr.Dataset:
    """Patch the attributes of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    attributes : Dict[str, Dict[str, Any]]
        The attributes to patch.

    Returns
    -------
    Any
        The patched dataset.
    """
    for name, value in attributes.items():
        variable = ds[name]
        variable.attrs.update(value)

    return ds


def patch_coordinates(ds: xr.Dataset, coordinates: list[str]) -> xr.Dataset:
    """Patch the coordinates of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    coordinates : List[str]
        The coordinates to patch.

    Returns
    -------
    Any
        The patched dataset.
    """
    for name in coordinates:
        ds = ds.assign_coords({name: ds[name]})

    return ds


def patch_rename(ds: xr.Dataset, renames: dict[str, str]) -> xr.Dataset:
    """Rename variables in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    renames : dict[str, str]
        Mapping from old variable names to new variable names.

    Returns
    -------
    Any
        The patched dataset.
    """
    return ds.rename(renames)


def patch_sort_coordinate(ds: xr.Dataset, sort_coordinates: list[str]) -> xr.Dataset:
    """Sort the coordinates of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    sort_coordinates : List[str]
        The coordinates to sort.

    Returns
    -------
    Any
        The patched dataset.
    """

    for name in sort_coordinates:
        ds = ds.sortby(name)
    return ds


def patch_subset_dataset(ds: xr.Dataset, selection: dict[str, Any]) -> xr.Dataset:
    """Select a subset of the dataset using xarray's sel method.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    selection : dict[str, Any]
        Dictionary mapping dimension names to selection criteria.
        Keys must be existing dimension names in the dataset.
        Values can be any type accepted by xarray's sel method, including:
        - Single values (int, float, str, datetime)
        - Lists or arrays of values
        - Slices (using slice() objects)
        - Boolean arrays.

    Returns
    -------
    xr.Dataset
        The patched dataset containing only the selected subset.

    Examples
    --------
    >>> # Select specific time and pressure level
    >>> patch_subset_dataset(ds, {
    ...     'time': '2020-01-01',
    ...     'pressure': 500
    ... })

    >>> # Select a range using slice
    >>> patch_subset_dataset(ds, {
    ...     'lat': slice(-90, 90),
    ...     'lon': slice(0, 180)
    ... })
    """

    ds = ds.sel(selection)

    return ds


def patch_analysis_lead_to_valid_time(
    ds: xr.Dataset,
    time_coord_names: dict[Literal["analysis_time_coordinate", "lead_time_coordinate", "valid_time_coordinate"], str],
) -> xr.Dataset:
    """Convert analysis time and lead time coordinates to valid time.

    This function creates a new valid time coordinate by adding the analysis time
    and lead time coordinates, then stacks and reorganizes the dataset to use
    valid time as the primary time dimension.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    time_coord_names : dict[str, str]
        Dictionary mapping required keys to coordinate names in the dataset:

        - 'analysis_time_coordinate' : str
            Name of the analysis/initialization time coordinate.
        - 'lead_time_coordinate' : str
            Name of the forecast lead time coordinate.
        - 'valid_time_coordinate' : str
            Name for the new valid time coordinate to create.

    Returns
    -------
    xr.Dataset
        The patched dataset with valid time as the primary time coordinate.
        The analysis and lead time coordinates are removed.

    Examples
    --------
    >>> patch_analysis_lead_to_valid_time(ds, {
    ...     'analysis_time_coordinate': 'forecast_reference_time',
    ...     'lead_time_coordinate': 'step',
    ...     'valid_time_coordinate': 'time'
    ... })
    """

    assert time_coord_names.keys() == {
        "analysis_time_coordinate",
        "lead_time_coordinate",
        "valid_time_coordinate",
    }, "time_coord_names must contain exactly keys 'analysis_time_coordinate', 'lead_time_coordinate', and 'valid_time_coordinate'"

    analysis_time_coordinate = time_coord_names["analysis_time_coordinate"]
    lead_time_coordinate = time_coord_names["lead_time_coordinate"]
    valid_time_coordinate = time_coord_names["valid_time_coordinate"]

    valid_time = ds[analysis_time_coordinate] + ds[lead_time_coordinate]

    ds = (
        ds.assign_coords({valid_time_coordinate: valid_time})
        .stack(time_index=[analysis_time_coordinate, lead_time_coordinate])
        .set_index(time_index=valid_time_coordinate)
        .rename(time_index=valid_time_coordinate)
        .drop_vars([analysis_time_coordinate, lead_time_coordinate])
    )

    return ds


def patch_rolling_operation(
    ds: xr.Dataset, vars_operation_config: dict[Literal["dim", "steps", "vars", "operation"], str | int | list[str]]
) -> xr.Dataset:
    """Apply a rolling operation to specified variables in the dataset.

    This function calculates a rolling operation over a specified dimension for selected
    variables. The rolling window requires all periods to be present (min_periods=steps).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    vars_operation_config: dict
        Configuration for the rolling operation with the following keys:

        - 'dim' : str
            The dimension along which to apply the rolling operation (e.g., 'time').
        - 'steps' : int
            The number of steps in the rolling window.
        - 'vars' : list[str]
            List of variable names to apply the rolling operation to.
        - 'operation' : str
            The operation to apply ('sum', 'mean', 'min', 'max', 'std', etc.),
            or 'last' for the latest value in the window that is not missing.
            'last' does not need all periods: the first windows look back only
            as far as the data goes.

    Returns
    -------
    xr.Dataset
        The patched dataset with rolling operations applied to the specified variables.

    Examples
    --------
    >>> patch_rolling_operation(ds, {
    ...     'dim': 'time',
    ...     'steps': 3,
    ...     'vars': ['precipitation', 'radiation'],
    ...     'operation': 'sum'
    ... })
    """

    assert vars_operation_config.keys() == {
        "dim",
        "steps",
        "vars",
        "operation",
    }, "vars_operation_config must contain exactly keys 'dim', 'steps', 'vars', and 'operation'"

    dim = vars_operation_config["dim"]
    steps = vars_operation_config["steps"]
    vars = vars_operation_config["vars"]
    operation = vars_operation_config["operation"]

    for var in vars:
        if operation == "last":
            ds[var] = _rolling_last(ds[var], dim, steps)
            continue
        rolling = ds[var].rolling(dim={dim: steps}, min_periods=steps)
        ds[var] = getattr(rolling, operation)()

    return ds


def _rolling_last(variable: xr.DataArray, dim: str, steps: int) -> xr.DataArray:
    """The latest value that is not missing in each rolling window.

    Unlike the other rolling operations, a window does not need all its
    periods: steps before the start of the dimension count as missing, so the
    first windows look back as far as there is data.

    Each chunk is filled with ``steps - 1`` values of overlap from the chunk
    before, rather than with ``ffill``, so that it stays local along ``dim``:
    a lazily loaded window of the dataset then only reads the chunks it
    overlaps, instead of scanning from the start of the dimension.
    """
    axis = variable.get_axis_num(dim)

    def last(block: np.ndarray) -> np.ndarray:
        block = np.moveaxis(block, axis, 0)
        result = block.copy()
        for shift in range(1, min(steps, len(block))):
            missing = np.isnan(result[shift:])
            result[shift:][missing] = block[:-shift][missing]
        return np.moveaxis(result, 0, axis)

    data = variable.data
    if hasattr(data, "map_overlap"):
        data = data.map_overlap(last, depth={axis: (steps - 1, 0)}, boundary="none", dtype=data.dtype)
    else:
        data = last(np.asarray(data))
    return variable.copy(data=data)


def patch_vector_magnitude(ds: xr.Dataset, magnitudes: dict[str, list[str]]) -> xr.Dataset:
    """Add the magnitude of vectors given by their components.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    magnitudes : dict[str, list[str]]
        Mapping from the name of each new variable to the names of the
        components it is the magnitude of. The new variable takes the
        ``units`` and ``grid_mapping`` of the first component.

    Returns
    -------
    xr.Dataset
        The patched dataset, with the components kept.

    Examples
    --------
    >>> patch_vector_magnitude(ds, {"10fg": ["ugst10m", "vgst10m"]})
    """
    for name, components in magnitudes.items():
        first = ds[components[0]]
        magnitude = sum(ds[component] ** 2 for component in components) ** 0.5
        magnitude.attrs = {key: first.attrs[key] for key in ("units", "grid_mapping") if key in first.attrs}
        ds[name] = magnitude

    return ds


def patch_valid_range(
    ds: xr.Dataset, config: dict[Literal["vars", "minimum", "maximum"], float | list[str]]
) -> xr.Dataset:
    """Set values outside a valid range to missing (NaN).

    For fields with isolated values that cannot be right, such as a negative
    albedo. It runs before ``rolling_operation``, so a rolling ``last`` can
    then fill them with the latest valid value.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    config : dict
        The options:

        - ``vars``: the variables to mask.
        - ``minimum``, ``maximum``: the bounds of the valid range, inclusive.
          At least one is needed.

    Returns
    -------
    xr.Dataset
        The patched dataset.

    Examples
    --------
    >>> patch_valid_range(ds, {"vars": ["al"], "minimum": 0})
    """
    unknown = set(config) - {"vars", "minimum", "maximum"}
    assert not unknown, f"Unknown valid_range options {sorted(unknown)}"
    minimum, maximum = config.get("minimum"), config.get("maximum")
    assert minimum is not None or maximum is not None, "valid_range needs a minimum or a maximum"

    for name in config["vars"]:
        variable = ds[name]
        valid = True
        if minimum is not None:
            valid = valid & (variable >= minimum)
        if maximum is not None:
            valid = valid & (variable <= maximum)
        masked = variable.where(valid)
        masked.attrs = variable.attrs
        ds[name] = masked

    return ds


def patch_mask_night(
    ds: xr.Dataset,
    config: dict[
        Literal["vars", "time", "latitude", "longitude", "min_cos_solar_zenith_angle"], str | float | list[str]
    ],
) -> xr.Dataset:
    """Set variables to missing (NaN) where the sun is below the horizon.

    For fields that are only defined in daylight, such as an albedo. The cosine
    of the solar zenith angle is the one the ``cos_solar_zenith_angle`` forcing
    uses, evaluated at each value of the time coordinate. It is computed lazily,
    chunk by chunk, so only the times that are read are evaluated.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    config : dict
        The options:

        - ``vars``: the variables to mask.
        - ``time``: the one-dimensional time coordinate to evaluate the sun
          at, by default ``time``.
        - ``latitude``, ``longitude``: the latitude and longitude
          coordinates, by default ``latitude`` and ``longitude``.
        - ``min_cos_solar_zenith_angle``: values are kept where the cosine of
          the solar zenith angle exceeds this, by default 0 (the sun above the
          horizon).

    Returns
    -------
    xr.Dataset
        The patched dataset.

    Examples
    --------
    >>> patch_mask_night(ds, {
    ...     'vars': ['albedo'],
    ...     'latitude': 'lat',
    ...     'longitude': 'lon',
    ...     'min_cos_solar_zenith_angle': 0.02,
    ... })
    """
    unknown = set(config) - {"vars", "time", "latitude", "longitude", "min_cos_solar_zenith_angle"}
    assert not unknown, f"Unknown mask_night options {sorted(unknown)}"

    time = ds[config.get("time", "time")]
    latitude = ds[config.get("latitude", "latitude")]
    longitude = ds[config.get("longitude", "longitude")]
    threshold = config.get("min_cos_solar_zenith_angle", 0.0)

    assert time.ndim == 1, f"mask_night needs a one-dimensional time coordinate, got {time.dims}"

    def cos_sza(times, lat, lon):
        # A block is (time, *grid), time first as it is the first argument
        # below; evaluate each of its times on its part of the grid.
        shape = np.broadcast_shapes(times.shape, lat.shape, lon.shape)
        dates = pd.to_datetime(np.broadcast_to(times, shape)[(slice(None),) + (0,) * (len(shape) - 1)])
        lat, lon = np.broadcast_to(lat, shape)[0], np.broadcast_to(lon, shape)[0]
        result = np.empty(shape, dtype=np.float32)
        for i, date in enumerate(dates.to_pydatetime()):
            result[i] = cos_solar_zenith_angle(date, lat, lon)
        return result

    for name in config["vars"]:
        variable = ds[name]
        chunks = variable.chunksizes

        def chunked(coordinate: xr.DataArray) -> xr.DataArray:
            # A plain array, as a dimension coordinate cannot be chunked, and
            # chunked like the variable, so each block of the result lines up
            # with a block of the data it masks.
            array = xr.DataArray(coordinate.values, dims=coordinate.dims)
            return array.chunk({dim: chunks.get(dim, 1 if dim == time.dims[0] else -1) for dim in array.dims})

        cos = xr.apply_ufunc(
            cos_sza,
            chunked(time),
            chunked(latitude),
            chunked(longitude),
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        masked = variable.where(cos > threshold)
        masked.attrs = variable.attrs
        ds[name] = masked

    return ds


PATCHES = {
    "attributes": patch_attributes,
    "coordinates": patch_coordinates,
    "rename": patch_rename,
    "sort_coordinates": patch_sort_coordinate,
    "analysis_lead_to_valid_time": patch_analysis_lead_to_valid_time,
    "vector_magnitude": patch_vector_magnitude,
    "valid_range": patch_valid_range,
    "mask_night": patch_mask_night,
    "rolling_operation": patch_rolling_operation,
    "subset_dataset": patch_subset_dataset,
}


def patch_dataset(ds: xr.Dataset, patch: dict[str, dict[str, Any]]) -> Any:
    """Patch the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    patch : Dict[str, Dict[str, Any]]
        The patch to apply.

    Returns
    -------
    Any
        The patched dataset.
    """

    ORDER = [
        "coordinates",
        "attributes",
        "rename",
        "sort_coordinates",
        "subset_dataset",
        "analysis_lead_to_valid_time",
        "vector_magnitude",
        "valid_range",
        "mask_night",
        "rolling_operation",
    ]
    for what, values in sorted(patch.items(), key=lambda x: ORDER.index(x[0])):
        if what not in PATCHES:
            raise ValueError(f"Unknown patch type {what!r}")

        ds = PATCHES[what](ds, values)

    return ds
