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
from anemoi.utils.dates import as_datetime
from anemoi.utils.humanize import bytes_to_human
from anemoi.utils.humanize import human_to_bytes
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


DEFAULT_PRELOAD_BUDGET = "4GiB"
"""Memory allowed for one preloaded window when ``preload`` is given as ``True``."""


def _requested_names(fs: XarrayFieldList, **kwargs: Any) -> list[str]:
    """Return the names of the variables the caller's request can select.

    Parameters
    ----------
    fs : XarrayFieldList
        The field list to pick from.
    **kwargs : Any
        The caller's selection criteria; ``param``/``variable`` picks the
        variables, anything else is ignored here.

    Returns
    -------
    list of str
        The names of the matching variables, in the field list's own order.
    """
    names = [variable.name for variable in fs.variables]

    requested = kwargs.get("param", kwargs.get("variable"))
    if requested is None:
        return names

    if not isinstance(requested, (list, tuple)):
        requested = [requested]

    return [name for name in names if name in set(requested)]


def _narrow_to_request(
    fs: XarrayFieldList,
    first: datetime.datetime,
    last: datetime.datetime,
    **kwargs: Any,
) -> "tuple[xr.Dataset, dict[str, slice]] | None":
    """Cut the dataset down to the window, variables and levels being asked for.

    Preloading reads whatever it is handed, so the dataset is first narrowed to
    what the caller will actually select: without this a recipe asking for six
    parameters on eleven levels would still read every parameter on every level
    the store holds.

    Parameters
    ----------
    fs : XarrayFieldList
        The field list whose dataset is to be narrowed.
    first : datetime.datetime
        Earliest datetime the caller will ask for.
    last : datetime.datetime
        Latest datetime the caller will ask for.
    **kwargs : Any
        The caller's selection criteria.  ``param``/``variable`` picks the
        variables; any key that names a dimension of the dataset (``level``,
        ``number``, a bespoke ``pressure``, ...) narrows that dimension.

    Returns
    -------
    tuple of (xr.Dataset, dict of str to slice), or None
        The narrowed dataset and the window applied to it, or ``None`` when
        there is nothing worth preloading.
    """
    names = _requested_names(fs, **kwargs)
    if not names:
        LOG.debug("preload: the request selects no variable of this dataset")
        return None

    wanted = fs.ds[names]

    # The window is taken from the variables that are actually wanted: asking
    # the others could name a dimension the narrowed dataset does not have --
    # as happens for a source that selects only constant fields out of a store
    # whose other variables vary in time.
    window: dict[str, slice] = {}
    for variable in fs.variables:
        if variable.name in names:
            window.update(variable.time.preload_window(first, last))

    window = {dim: selection for dim, selection in window.items() if dim in wanted.dims}
    if not window:
        LOG.debug("preload: no time axis to preload")
        return None

    # Any remaining criterion that names a dimension can be applied up front.
    # `param` is not a dimension, and the time axis is already in `window`.
    #
    # A dimension is left alone when the criterion would leave a single value
    # along it: a length-one coordinate is treated as scalar, and `Variable.sel`
    # then keeps that dimension instead of indexing it away, so the fields would
    # come out with an extra dimension. Preloading must change what is read,
    # never the shape of a field.
    along_dims = {}
    for key, value in kwargs.items():
        if key not in wanted.dims or key in window:
            continue
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        along_dims[key] = list(value)

    return wanted.sel({**window, **along_dims}), window


def _preload_fieldlist(
    fs: XarrayFieldList,
    first: datetime.datetime,
    last: datetime.datetime,
    flavour: str | None,
    preload: bool | int | str,
    **kwargs: Any,
) -> XarrayFieldList:
    """Load the window of the time axis covering ``[first, last]`` into memory.

    Fields are otherwise read one at a time, each triggering its own read of the
    underlying store.  When the store is chunked with many time steps per chunk
    -- a common layout for long reanalysis archives -- a whole chunk has to be
    decompressed to yield a single time slice, so the same chunk is decompressed
    once per field.  Reading the window once and serving every field from memory
    removes that repetition; the effect grows with the number of dates in the
    window, so it is governed by ``build.group_by``.

    The returned field list is equivalent to *fs* over the window: the same
    fields with the same values, backed by memory rather than by the store.

    Parameters
    ----------
    fs : XarrayFieldList
        The (lazy) field list to preload from.  Any ``patch`` has already been
        applied to its dataset, so the window is taken on the patched axis --
        which is what keeps a patch such as ``rolling_operation`` correct, since
        it can still reach the values preceding the window.
    first : datetime.datetime
        Earliest datetime the caller will ask for.
    last : datetime.datetime
        Latest datetime the caller will ask for.
    flavour : Optional[str]
        Flavour of the dataset, needed to rebuild the field list.
    preload : Union[bool, int, str]
        ``False`` to read fields one at a time (the default).  ``True`` to
        preload within :data:`DEFAULT_PRELOAD_BUDGET`, or a size (bytes, or a
        string such as ``"8GiB"``) to set the budget explicitly.
    **kwargs : Any
        The selection criteria the caller will apply (``param``, ``level``,
        ...).  Used to narrow the window to the data actually wanted, so that
        unrequested variables and levels are not read.

    Returns
    -------
    XarrayFieldList
        A memory-backed field list, or *fs* unchanged when preloading is off,
        not applicable, or over budget.
    """
    if not preload or not fs.variables:
        return fs

    budget = human_to_bytes(DEFAULT_PRELOAD_BUDGET if preload is True else preload)

    try:
        narrowed = _narrow_to_request(fs, first, last, **kwargs)
    except Exception:
        LOG.warning("preload: could not narrow the window to %s; reading fields one at a time.", kwargs)
        LOG.debug("preload: narrowing failed", exc_info=True)
        return fs

    if narrowed is None:
        return fs

    subset, window = narrowed

    # Preloading a single time step gains nothing -- it is exactly the one read
    # it would replace -- and a length-one coordinate is treated as scalar, so
    # `Variable.sel` would keep that dimension rather than indexing it away and
    # the fields would come out misshapen. Leave such a window alone.
    short = {dim: int(subset.sizes[dim]) for dim in window if subset.sizes.get(dim, 0) < 2}
    if short:
        LOG.debug("preload: window too short to amortise anything (%s)", short)
        return fs

    nbytes = sum(variable.nbytes for variable in subset.data_vars.values())

    if nbytes > budget:
        LOG.warning(
            "preload: the window %s to %s needs %s, over the %s budget; reading fields one at a time instead. "
            "Lower 'build.group_by' or raise 'preload'.",
            first,
            last,
            bytes_to_human(nbytes),
            bytes_to_human(budget),
        )
        return fs

    LOG.info("preload: loading %s to %s (%s)", first, last, bytes_to_human(nbytes))

    # patch=None: the patches are already applied to fs.ds, and applying them
    # again to the window would be wrong (and would fail for the patches that
    # consume their input coordinates).
    return XarrayFieldList.from_xarray(subset.load(), flavour=flavour, patch=None)


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
    preload: bool | int | str = False,
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
    preload : Union[bool, int, str], optional
        Read the whole window of dates into memory in one go rather than a
        field at a time. See :func:`_preload_fieldlist`.
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
        if preload:
            moments = [as_datetime(date) for date in dates]
            fs = _preload_fieldlist(fs, min(moments), max(moments), flavour, preload, **kwargs)
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
    preload: bool | int | str = False,
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
    preload : Union[bool, int, str], optional
        Read the whole window of forecast pairs into memory in one go rather
        than a field at a time. See :func:`_preload_fieldlist`.
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
        if preload:
            # A store may be indexed by either axis, so the window has to span
            # the basetimes and the validity times alike.
            moments = [moment for pair in items for moment in pair]
            fs = _preload_fieldlist(fs, min(moments), max(moments), flavour, preload, **kwargs)
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
