# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod
from typing import Any

from anemoi.datasets.create.input.context import Context

from ..source import Source

LOG = logging.getLogger(__name__)


class LegacySource(Source):
    """A legacy source class.

    Parameters
    ----------
    context : Context
        The context in which the source is created.
    *args : tuple
        Positional arguments.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(self, context: Context, *args: Any, **kwargs: Any) -> None:
        super().__init__(context, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    @abstractmethod
    def _execute(context, *args, **kwargs):
        pass

    def execute_valid_dates(self, dates: Any) -> Any:
        return self._execute(self.context, dates, *self.args, **self.kwargs)

    def execute_forecast_dates(self, dates: Any) -> Any:
        """Handle a ``ForecastDates`` request via the optional ``_execute_forecast`` hook.

        Legacy sources declare trajectory support by defining a static
        ``_execute_forecast(context, items, ...)``, mirroring ``_execute`` but
        receiving ``(valid_time, basetime)`` pairs instead of dates.  Sources
        that do not define it fall back to the base class, which raises the
        standard "does not support the trajectory layout" error.

        Parameters
        ----------
        dates : ForecastDates
            The forecast-date argument from the pipeline.

        Returns
        -------
        Any
            The loaded data.
        """
        execute_forecast = getattr(self, "_execute_forecast", None)
        if execute_forecast is None:
            return super().execute_forecast_dates(dates)
        return execute_forecast(self.context, dates.items, *self.args, **self.kwargs)
