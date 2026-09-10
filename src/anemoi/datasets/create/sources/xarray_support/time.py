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
from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
from anemoi.utils.dates import as_datetime

from .coordinates import Coordinate
from .variable import Variable

LOG = logging.getLogger(__name__)


def _step_selection(step: datetime.timedelta) -> np.timedelta64:
    """Convert a forecast step to a value that selects on a step coordinate.

    Step coordinates hold ``numpy.timedelta64`` values, which compare against
    neither ``int`` hours nor :class:`datetime.timedelta` (the latter raises a
    ``TypeError`` inside ``numpy.searchsorted``), so the step has to be handed
    over as a ``numpy.timedelta64``.

    Parameters
    ----------
    step : datetime.timedelta
        The forecast step.

    Returns
    -------
    numpy.timedelta64
        The step as a second-resolution ``numpy.timedelta64``.
    """
    return np.timedelta64(int(step.total_seconds()), "s")


class Time(ABC):
    """Base class for different time representations."""

    @classmethod
    def from_coordinates(cls, coordinates: list[Coordinate]) -> "Time":
        """Create a Time instance from a list of coordinates.

        Returns
        -------
        Union[ForecastFromValidTimeAndStep, Analysis, Constant, ForecastFromValidTimeAndBaseTime, ForecastFromBaseTimeAndDate]
            An instance of a subclass of Time.

        Args
        ----
        coordinates : List[Coordinate]
            List of coordinate objects.
        """
        time_coordinate = [c for c in coordinates if c.is_time]
        step_coordinate = [c for c in coordinates if c.is_step]
        date_coordinate = [c for c in coordinates if c.is_date]

        if len(date_coordinate) == 0 and len(time_coordinate) == 1 and len(step_coordinate) == 1:
            return ForecastFromValidTimeAndStep(time_coordinate[0], step_coordinate[0])

        if len(date_coordinate) == 0 and len(time_coordinate) == 1 and len(step_coordinate) == 0:
            return Analysis(time_coordinate[0])

        if len(date_coordinate) == 0 and len(time_coordinate) == 0 and len(step_coordinate) == 0:
            return Constant()

        if len(date_coordinate) == 1 and len(time_coordinate) == 1 and len(step_coordinate) == 0:
            return ForecastFromValidTimeAndBaseTime(date_coordinate[0], time_coordinate[0])

        if len(date_coordinate) == 1 and len(time_coordinate) == 0 and len(step_coordinate) == 1:
            return ForecastFromBaseTimeAndDate(date_coordinate[0], step_coordinate[0])

        if len(date_coordinate) == 1 and len(time_coordinate) == 1 and len(step_coordinate) == 1:
            return ForecastFromValidTimeAndStep(time_coordinate[0], step_coordinate[0], date_coordinate[0])

        LOG.error("")
        LOG.error(f"{len(date_coordinate)} date_coordinate")
        for c in date_coordinate:
            LOG.error("    %s %s %s %s", c, c.is_date, c.is_time, c.is_step)
            # LOG.error('    %s', c.variable)

        LOG.error("")
        LOG.error(f"{len(time_coordinate)} time_coordinate")
        for c in time_coordinate:
            LOG.error("    %s %s %s %s", c, c.is_date, c.is_time, c.is_step)
            # LOG.error('    %s', c.variable)

        LOG.error("")
        LOG.error(f"{len(step_coordinate)} step_coordinate")
        for c in step_coordinate:
            LOG.error("    %s %s %s %s", c, c.is_date, c.is_time, c.is_step)
            # LOG.error('    %s', c.variable)

        raise NotImplementedError(f"{len(date_coordinate)=} {len(time_coordinate)=} {len(step_coordinate)=}")

    @abstractmethod
    def select_valid_datetime(self, variable: Variable) -> str | None:
        """Select the valid datetime for a given variable.

        Parameters
        ----------
        variable : Variable
            The variable to select the datetime for.

        Returns
        -------
        Optional[str]
            The name of the time coordinate.
        """
        pass

    @abstractmethod
    def fill_time_metadata(self, coords_values: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Fill metadata with time information.

        Args
        ----
        coords_values : Dict[str, Any]
            Coordinate values.
        metadata : Dict[str, Any]
            Metadata dictionary.
        """

        pass

    def forecast_selection(self, valid_time: datetime.datetime, basetime: datetime.datetime) -> dict[str, Any] | None:
        """Return the coordinate selection isolating one forecast field.

        Maps a ``(valid_time, basetime)`` pair -- as requested by the
        ``trajectories`` layout -- onto a selection keyed by this dataset's own
        coordinate *names*, so that
        :meth:`~anemoi.datasets.create.sources.xarray_support.variable.Variable.sel`
        can slice the coordinates directly.  Selecting on the MARS-style
        ``date``/``step`` aliases instead would fall through to
        :class:`~anemoi.datasets.create.sources.xarray_support.variable.FilteredVariable`,
        which scans every field of the variable.

        A ``(valid_time, basetime)`` pair is used rather than a bare validity
        time because the two do not determine each other: in a forecast archive
        the same validity time is reached from several runs (e.g. ``12Z`` is
        both ``00Z + 12h`` and ``06Z + 6h``), so a validity time alone selects
        more than one field.

        Parameters
        ----------
        valid_time : datetime.datetime
            The validity time of the wanted field.
        basetime : datetime.datetime
            The model-run time the wanted field must come from.

        Returns
        -------
        dict of str to Any, or None
            Keyword arguments for ``sel``, or ``None`` if this time
            representation cannot serve the requested pair.
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' does not implement forecast_selection; "
            "this dataset cannot be used with the 'trajectories' layout."
        )

    def preload_window(self, first: datetime.datetime, last: datetime.datetime) -> dict[str, slice]:
        """Return the slice of the datetime axis covering ``[first, last]``.

        Used to load a whole window of the time axis into memory in one read,
        instead of reading each field on its own.  A store chunked with many
        time steps per chunk has to decompress a whole chunk to yield a single
        time slice, so reading fields one at a time re-reads the same chunk
        once per field.

        The keys are the dataset's own coordinate *names*, so the result can be
        passed straight to ``xarray.Dataset.sel``.  An empty dict means there
        is nothing to preload (a constant field has no time axis).

        Parameters
        ----------
        first : datetime.datetime
            Earliest datetime the caller will ask for.
        last : datetime.datetime
            Latest datetime the caller will ask for.

        Returns
        -------
        dict of str to slice
            The window to select, keyed by coordinate name.
        """
        return {}


class Constant(Time):
    """Represents a constant time."""

    def fill_time_metadata(self, coords_values: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Fill metadata with time information.

        Parameters
        ----------
        coords_values : Dict[str, Any]
            Coordinate values.
        metadata : Dict[str, Any]
            Metadata dictionary.
        """
        return None

    def select_valid_datetime(self, variable: Variable) -> None:
        """Select the valid datetime for a given variable.

        Parameters
        ----------
        variable : Variable
            The variable to select the datetime for.
        """
        return None

    def forecast_selection(self, valid_time: datetime.datetime, basetime: datetime.datetime) -> dict[str, Any] | None:
        """Return an empty selection: a constant field serves every forecast pair.

        Parameters
        ----------
        valid_time : datetime.datetime
            The validity time of the wanted field (unused).
        basetime : datetime.datetime
            The model-run time (unused).

        Returns
        -------
        dict of str to Any
            An empty dict -- the single field is valid for any pair.
        """
        return {}


class Analysis(Time):
    """Represents an analysis time."""

    def __init__(self, time_coordinate: Coordinate) -> None:
        """Initialize Analysis with a time coordinate.

        Parameters
        ----------
        time_coordinate : Coordinate
            The time coordinate.
        """
        self.time_coordinate_name = time_coordinate.variable.name

    def fill_time_metadata(self, coords_values: dict[str, Any], metadata: dict[str, Any]) -> Any:
        """Fill metadata with time information.

        Parameters
        ----------
        coords_values : Dict[str, Any]
            Coordinate values.
        metadata : Dict[str, Any]
            Metadata dictionary.

        Returns
        -------
        Any
            The valid datetime.
        """
        valid_datetime = coords_values[self.time_coordinate_name]

        metadata["date"] = int(as_datetime(valid_datetime).strftime("%Y%m%d"))
        metadata["time"] = int(as_datetime(valid_datetime).strftime("%H%M"))
        metadata["step"] = 0

        return valid_datetime

    def select_valid_datetime(self, variable: Variable) -> str:
        """Select the valid datetime for a given variable.

        Parameters
        ----------
        variable : Variable
            The variable to select the datetime for.

        Returns
        -------
        str
            The name of the time coordinate.
        """
        return self.time_coordinate_name

    def forecast_selection(self, valid_time: datetime.datetime, basetime: datetime.datetime) -> dict[str, Any] | None:
        """Select the analysis valid at *valid_time*.

        An analysis carries no step axis, so the run it belongs to is its own
        validity time and *basetime* does not narrow the selection further.

        Parameters
        ----------
        valid_time : datetime.datetime
            The validity time of the wanted field.
        basetime : datetime.datetime
            The model-run time (unused).

        Returns
        -------
        dict of str to Any
            Selection on the time coordinate.
        """
        return {self.time_coordinate_name: valid_time}

    def preload_window(self, first: datetime.datetime, last: datetime.datetime) -> dict[str, slice]:
        """Return the validity-time window to preload.

        Parameters
        ----------
        first : datetime.datetime
            Earliest datetime the caller will ask for.
        last : datetime.datetime
            Latest datetime the caller will ask for.

        Returns
        -------
        dict of str to slice
            The window on the time coordinate.
        """
        return {self.time_coordinate_name: slice(first, last)}


class ForecastFromValidTimeAndStep(Time):
    """Represents a forecast time derived from valid time and step."""

    def __init__(
        self, time_coordinate: Coordinate, step_coordinate: Coordinate, date_coordinate: Coordinate | None = None
    ) -> None:
        """Initialize ForecastFromValidTimeAndStep with time, step, and optional date coordinates.

        Args
        ----
        time_coordinate : Coordinate
            The time coordinate.
        step_coordinate : Coordinate
            The step coordinate.
        date_coordinate : Optional[Coordinate]
            The date coordinate.
        """
        self.time_coordinate_name = time_coordinate.variable.name
        self.step_coordinate_name = step_coordinate.variable.name
        self.date_coordinate_name = date_coordinate.variable.name if date_coordinate else None

    def fill_time_metadata(self, coords_values: dict[str, Any], metadata: dict[str, Any]) -> Any:
        """Fill metadata with time information.

        Returns
        -------
        Any
            The valid datetime.

        Args
        ----
        coords_values : Dict[str, Any]
            Coordinate values.
        metadata : Dict[str, Any]
            Metadata dictionary.
        """
        valid_datetime = coords_values[self.time_coordinate_name]
        step = coords_values[self.step_coordinate_name]

        assert isinstance(step, datetime.timedelta)
        base_datetime = valid_datetime - step

        hours = step.total_seconds() / 3600
        assert int(hours) == hours

        metadata["date"] = int(as_datetime(base_datetime).strftime("%Y%m%d"))
        metadata["time"] = int(as_datetime(base_datetime).strftime("%H%M"))
        metadata["step"] = int(hours)

        # When date is present, it should be compatible with time and step

        if self.date_coordinate_name is not None:
            # Not sure that this is the correct assumption
            assert coords_values[self.date_coordinate_name] == base_datetime, (
                coords_values[self.date_coordinate_name],
                base_datetime,
            )

        return valid_datetime

    def select_valid_datetime(self, variable: Variable) -> str:
        """Select the valid datetime for a given variable.

        Parameters
        ----------
        variable : Variable
            The variable to select the datetime for.

        Returns
        -------
        str
            The name of the time coordinate.
        """
        return self.time_coordinate_name

    def forecast_selection(self, valid_time: datetime.datetime, basetime: datetime.datetime) -> dict[str, Any] | None:
        """Select on validity time and step.

        Parameters
        ----------
        valid_time : datetime.datetime
            The validity time of the wanted field.
        basetime : datetime.datetime
            The model-run time the wanted field must come from.

        Returns
        -------
        dict of str to Any
            Selection on the time and step coordinates.
        """
        return {
            self.time_coordinate_name: valid_time,
            self.step_coordinate_name: _step_selection(valid_time - basetime),
        }

    def preload_window(self, first: datetime.datetime, last: datetime.datetime) -> dict[str, slice]:
        """Return the validity-time window to preload.

        The step axis is left whole: it is short, and every step of a window is
        wanted anyway.

        Parameters
        ----------
        first : datetime.datetime
            Earliest datetime the caller will ask for.
        last : datetime.datetime
            Latest datetime the caller will ask for.

        Returns
        -------
        dict of str to slice
            The window on the time coordinate.
        """
        return {self.time_coordinate_name: slice(first, last)}


class ForecastFromValidTimeAndBaseTime(Time):
    """Represents a forecast time derived from valid time and base time."""

    def __init__(self, date_coordinate: Coordinate, time_coordinate: Coordinate) -> None:
        """Initialize ForecastFromValidTimeAndBaseTime with date and time coordinates.

        Args
        ----
        date_coordinate : Coordinate
            The date coordinate.
        time_coordinate : Coordinate
            The time coordinate.
        """
        self.date_coordinate_name = date_coordinate.name
        self.time_coordinate_name = time_coordinate.name

    def fill_time_metadata(self, coords_values: dict[str, Any], metadata: dict[str, Any]) -> Any:
        """Fill metadata with time information.

        Returns
        -------
        Any
            The valid datetime.

        Args
        ----
        coords_values : Dict[str, Any]
            Coordinate values.
        metadata : Dict[str, Any]
            Metadata dictionary.
        """
        valid_datetime = coords_values[self.time_coordinate_name]
        base_datetime = coords_values[self.date_coordinate_name]

        step = valid_datetime - base_datetime

        hours = step.total_seconds() / 3600
        assert int(hours) == hours

        metadata["date"] = int(as_datetime(base_datetime).strftime("%Y%m%d"))
        metadata["time"] = int(as_datetime(base_datetime).strftime("%H%M"))
        metadata["step"] = int(hours)

        return valid_datetime

    def select_valid_datetime(self, variable: Variable) -> str:
        """Select the valid datetime for a given variable.

        Parameters
        ----------
        variable : Variable
            The variable to select the datetime for.

        Returns
        -------
        str
            The name of the time coordinate.
        """
        return self.time_coordinate_name

    def forecast_selection(self, valid_time: datetime.datetime, basetime: datetime.datetime) -> dict[str, Any] | None:
        """Select on base time and validity time.

        Parameters
        ----------
        valid_time : datetime.datetime
            The validity time of the wanted field.
        basetime : datetime.datetime
            The model-run time the wanted field must come from.

        Returns
        -------
        dict of str to Any
            Selection on the date and time coordinates.
        """
        return {
            self.date_coordinate_name: basetime,
            self.time_coordinate_name: valid_time,
        }

    def preload_window(self, first: datetime.datetime, last: datetime.datetime) -> dict[str, slice]:
        """Return the validity-time window to preload.

        Parameters
        ----------
        first : datetime.datetime
            Earliest datetime the caller will ask for.
        last : datetime.datetime
            Latest datetime the caller will ask for.

        Returns
        -------
        dict of str to slice
            The window on the time coordinate.
        """
        return {self.time_coordinate_name: slice(first, last)}


class ForecastFromBaseTimeAndDate(Time):
    """Represents a forecast time derived from base time and date."""

    def __init__(self, date_coordinate: Coordinate, step_coordinate: Coordinate) -> None:
        """Initialize ForecastFromBaseTimeAndDate with date and step coordinates.

        Args
        ----
        date_coordinate : Coordinate
            The date coordinate.
        step_coordinate : Coordinate
            The step coordinate.
        """
        self.date_coordinate_name = date_coordinate.name
        self.step_coordinate_name = step_coordinate.name

    def fill_time_metadata(self, coords_values: dict[str, Any], metadata: dict[str, Any]) -> Any:
        """Fill metadata with time information.

        Returns
        -------
        Any
            The valid datetime.

        Args
        ----
        coords_values : Dict[str, Any]
            Coordinate values.
        metadata : Dict[str, Any]
            Metadata dictionary.
        """
        date = coords_values[self.date_coordinate_name]
        step = coords_values[self.step_coordinate_name]
        assert isinstance(step, datetime.timedelta)

        metadata["date"] = int(as_datetime(date).strftime("%Y%m%d"))
        metadata["time"] = int(as_datetime(date).strftime("%H%M"))

        hours = step.total_seconds() / 3600

        assert int(hours) == hours
        metadata["step"] = int(hours)

        return date + step

    def select_valid_datetime(self, variable: Variable) -> str | None:
        """Select the valid datetime for a given variable.

        Parameters
        ----------
        variable : Variable
            The variable to select the datetime for.

        Returns
        -------
        Optional[str]
            The name of the time coordinate.
        """
        raise NotImplementedError("ForecastFromBaseTimeAndDate.select_valid_datetime")

    def forecast_selection(self, valid_time: datetime.datetime, basetime: datetime.datetime) -> dict[str, Any] | None:
        """Select on base time and step.

        Parameters
        ----------
        valid_time : datetime.datetime
            The validity time of the wanted field.
        basetime : datetime.datetime
            The model-run time the wanted field must come from.

        Returns
        -------
        dict of str to Any
            Selection on the date and step coordinates.
        """
        return {
            self.date_coordinate_name: basetime,
            self.step_coordinate_name: _step_selection(valid_time - basetime),
        }

    def preload_window(self, first: datetime.datetime, last: datetime.datetime) -> dict[str, slice]:
        """Return the run-time window to preload.

        This store is indexed by run time, not validity time, so the window is
        taken on the date coordinate.  Callers pass a range that spans the
        basetimes they will ask for.  The step axis is left whole.

        Parameters
        ----------
        first : datetime.datetime
            Earliest datetime the caller will ask for.
        last : datetime.datetime
            Latest datetime the caller will ask for.

        Returns
        -------
        dict of str to slice
            The window on the date coordinate.
        """
        return {self.date_coordinate_name: slice(first, last)}
