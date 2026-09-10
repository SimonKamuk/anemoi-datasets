# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import TYPE_CHECKING
from typing import Any

import earthkit.data as ekd
from earthkit.data.core.fieldlist import MultiFieldList

from anemoi.datasets.create.sources.patterns import iterate_patterns

from .. import source_registry
from ..legacy import LegacySource
from .fieldlist import XarrayFieldList

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import xarray as xr


def check(what: str, ds: "xr.Dataset", paths: list[str], **kwargs: Any) -> None:
    """Checks if the dataset has the expected number of fields.

    Parameters
    ----------
    what : str
        Description of what is being checked.
    ds : xr.Dataset
        The dataset to check.
    paths : List[str]
        List of paths.
    **kwargs : Any
        Additional keyword arguments.
    """
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, {what}s={paths})")


def _open_fieldlist(
    emoji: str,
    context: Any,
    dataset: Any,
    options: dict[str, Any] | None,
    flavour: str | None,
    patch: Any | None,
    **kwargs: Any,
) -> XarrayFieldList:
    """Open *dataset* and wrap it as an :class:`XarrayFieldList`.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    dataset : Union[str, xr.Dataset]
        The dataset to open.
    options : Dict[str, Any], optional
        Additional options passed to ``xarray.open_dataset``.
    flavour : Optional[str], optional
        Flavour of the dataset.
    patch : Optional[Any], optional
        Patch for the dataset.
    **kwargs : Any
        Remaining selection criteria, for tracing only.

    Returns
    -------
    XarrayFieldList
        The dataset as a field list.
    """
    # Loading xarray may be long, so import it here to avoid slowing down module imports
    import xarray as xr

    if options is None:
        options = {}

    context.trace(emoji, dataset, options, kwargs)

    if isinstance(dataset, str) and (dataset.startswith("ec:") or dataset.startswith("ectmp:")):
        from anemoi.datasets.create.ecfs import get_ecfs_file

        dataset = get_ecfs_file(dataset)

    if isinstance(dataset, str) and dataset.endswith(".zarr"):
        # If the dataset is a zarr store, we need to use the zarr engine
        options["engine"] = "zarr"

    if isinstance(dataset, xr.Dataset):
        data = dataset
    else:
        print(f"Opening dataset {dataset} with options {options}")
        data = xr.open_dataset(dataset, **options)

    return XarrayFieldList.from_xarray(data, flavour=flavour, patch=patch)


def load_one(
    emoji: str,
    context: Any,
    dates: list[str],
    dataset: Any,
    *,
    options: dict[str, Any] | None = None,
    flavour: str | None = None,
    patch: Any | None = None,
    **kwargs: Any,
) -> ekd.FieldList:
    """Loads a single dataset.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    dates : List[str]
        List of dates.
    dataset : Union[str, xr.Dataset]
        The dataset to load.
    options : Dict[str, Any], optional
        Additional options for loading the dataset.
    flavour : Optional[str], optional
        Flavour of the dataset.
    patch : Optional[Any], optional
        Patch for the dataset.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    MultiFieldList
        The loaded dataset.
    """

    if options is None:
        options = {}

    fs = _open_fieldlist(emoji, context, dataset, options, flavour, patch, **kwargs)

    if len(dates) == 0:
        result = fs.sel(**kwargs)
    else:
        result = MultiFieldList([fs.sel(valid_datetime=date, **kwargs) for date in dates])

    if len(result) == 0:
        LOG.warning(f"No data found for {dataset} and dates {dates} and {kwargs}")
        LOG.warning(f"Options: {options}")

        for i, k in enumerate(fs):
            a = ["valid_datetime", k.metadata("valid_datetime", default=None)]
            for n in kwargs.keys():
                a.extend([n, k.metadata(n, default=None)])
            LOG.warning(f"{[str(x) for x in a]}")

            if i > 16:
                break

        # LOG.warning(data)

    return result


def load_many(emoji: str, context: Any, dates: list[datetime.datetime], pattern: str, **kwargs: Any) -> ekd.FieldList:
    """Loads multiple datasets.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    dates : List[str]
        List of dates.
    pattern : str
        Pattern for loading datasets.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    MultiFieldList
        The loaded datasets.
    """
    result = []

    for path, dates in iterate_patterns(pattern, dates, **kwargs):
        result.append(load_one(emoji, context, dates, path, **kwargs))

    return MultiFieldList(result)


def _stamp_forecast_pair(
    fieldlist: ekd.FieldList,
    valid_time: datetime.datetime,
    basetime: datetime.datetime,
) -> ekd.FieldList:
    """Label every field with the forecast pair it was requested for.

    Under the ``trajectories`` layout a field's position in the output array is
    decided by the ``(basetime, step)`` the pipeline *asked* for -- the writer
    reads ``date``, ``time`` and ``step`` from the field metadata to find its
    slot -- so those keys have to describe the request rather than whatever the
    source happens to call the field.  Re-labelling is needed because the two
    disagree in cases that are perfectly legitimate here:

    - a *constant* field (no time coordinate, e.g. a land-sea mask) carries no
      ``date``/``time``/``step`` at all, yet is valid for every pair;
    - an *analysis-like* dataset -- including a forecast archive flattened onto
      a validity-time axis by the ``analysis_lead_to_valid_time`` patch --
      reports ``date``/``time`` as the validity time and ``step`` as zero, so a
      field served for step 3 would otherwise be filed under a basetime three
      hours late.

    For a dataset that is already indexed by run and step the stamped values
    equal the ones derived from its coordinates, so this is a no-op.
    ``valid_datetime`` is set to *valid_time*, which is true in every case.

    Parameters
    ----------
    fieldlist : ekd.FieldList
        The fields selected for one forecast pair.
    valid_time : datetime.datetime
        The validity time the fields were selected for.
    basetime : datetime.datetime
        The model-run time the fields were selected for.

    Returns
    -------
    ekd.FieldList
        The fields, labelled with the requested pair.
    """
    from anemoi.transform.fields import new_field_with_metadata
    from anemoi.transform.fields import new_fieldlist_from_list

    hours = (valid_time - basetime).total_seconds() / 3600
    assert int(hours) == hours, f"Forecast step must be a whole number of hours, got {hours}"

    stamp = dict(
        date=int(basetime.strftime("%Y%m%d")),
        time=int(basetime.strftime("%H%M")),
        step=int(hours),
        valid_datetime=valid_time.isoformat(),
    )

    return new_fieldlist_from_list([new_field_with_metadata(field, **stamp) for field in fieldlist])


def load_one_forecast(
    emoji: str,
    context: Any,
    items: list[tuple[datetime.datetime, datetime.datetime]],
    dataset: Any,
    *,
    options: dict[str, Any] | None = None,
    flavour: str | None = None,
    patch: Any | None = None,
    **kwargs: Any,
) -> ekd.FieldList:
    """Loads a single dataset for a list of ``(valid_time, basetime)`` pairs.

    The forecast counterpart of :func:`load_one`, used by the ``trajectories``
    layout.  Each pair is resolved through
    :meth:`XarrayFieldList.sel_forecast`, so the field returned for a validity
    time is the one from the requested model run rather than an arbitrary run
    that happens to verify then.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    items : List[Tuple[datetime.datetime, datetime.datetime]]
        The ``(valid_time, basetime)`` pairs to load.
    dataset : Union[str, xr.Dataset]
        The dataset to load.
    options : Dict[str, Any], optional
        Additional options for loading the dataset.
    flavour : Optional[str], optional
        Flavour of the dataset.
    patch : Optional[Any], optional
        Patch for the dataset.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    MultiFieldList
        The loaded dataset.
    """

    if options is None:
        options = {}

    fs = _open_fieldlist(emoji, context, dataset, options, flavour, patch, **kwargs)

    if len(items) == 0:
        result = fs.sel(**kwargs)
    else:
        result = MultiFieldList([_stamp_forecast_pair(fs.sel_forecast(vt, bt, **kwargs), vt, bt) for vt, bt in items])

    if len(result) == 0:
        LOG.warning(f"No data found for {dataset} and forecast pairs {items} and {kwargs}")
        LOG.warning(f"Options: {options}")

        for i, k in enumerate(fs):
            a = ["valid_datetime", k.metadata("valid_datetime", default=None)]
            for n in ("date", "time", "step"):
                a.extend([n, k.metadata(n, default=None)])
            for n in kwargs.keys():
                a.extend([n, k.metadata(n, default=None)])
            LOG.warning(f"{[str(x) for x in a]}")

            if i > 16:
                break

    return result


def load_many_forecast(
    emoji: str,
    context: Any,
    items: list[tuple[datetime.datetime, datetime.datetime]],
    pattern: str,
    **kwargs: Any,
) -> ekd.FieldList:
    """Loads multiple datasets for a list of ``(valid_time, basetime)`` pairs.

    The forecast counterpart of :func:`load_many`.  Path patterns are expanded
    against the validity times, matching :func:`load_many`'s behaviour of
    handing every expanded path the full set of dates.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    items : List[Tuple[datetime.datetime, datetime.datetime]]
        The ``(valid_time, basetime)`` pairs to load.
    pattern : str
        Pattern for loading datasets.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    MultiFieldList
        The loaded datasets.
    """
    result = []

    valid_times = [vt for vt, _ in items]

    for path, _ in iterate_patterns(pattern, valid_times, **kwargs):
        result.append(load_one_forecast(emoji, context, items, path, **kwargs))

    return MultiFieldList(result)


@source_registry.register("xarray")
class LegacyXarraySource(LegacySource):
    name = "xarray"

    @staticmethod
    def _execute(context: Any, dates: list[str], url: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
        """Executes the loading of datasets.

        Parameters
        ----------
        context : Any
            Context object.
        dates : List[str]
            List of dates.
        url : str
            URL pattern for loading datasets.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ekd.FieldList
            The loaded datasets.
        """
        return load_many("🌐", context, dates, url, *args, **kwargs)

    @staticmethod
    def _execute_forecast(
        context: Any,
        items: list[tuple[datetime.datetime, datetime.datetime]],
        url: str,
        *args: Any,
        **kwargs: Any,
    ) -> ekd.FieldList:
        """Loads forecast ``(valid_time, basetime)`` pairs for the ``trajectories`` layout.

        Parameters
        ----------
        context : Any
            Context object.
        items : List[Tuple[datetime.datetime, datetime.datetime]]
            The ``(valid_time, basetime)`` pairs to load.
        url : str
            URL pattern for loading datasets.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        ekd.FieldList
            The loaded datasets.
        """
        return load_many_forecast("🌐", context, items, url, *args, **kwargs)
