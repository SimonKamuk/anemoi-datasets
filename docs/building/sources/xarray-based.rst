######################
 xarray-based Sources
######################

More generally, you can specify any valid xarray.open_dataset_ arguments
as the source, and anemoi-dataset will try to build a dataset from it.
Examples of valid xarray.open_dataset_ arguments include: netCDF, Zarr,
OpenDAP, etc.

.. literalinclude:: yaml/xarray-based.yaml
   :language: yaml

See :ref:`create-cf-data` for more information.

.. _xarray-preload:

*******************************
 Reading a window into memory
*******************************

By default each field is read on its own. That is fine when a field maps
onto its own chunk of the store, but many long archives are chunked with
a large number of time steps per chunk -- so a whole chunk has to be
decompressed to yield a single time slice, and the same chunk is
decompressed again for every field that lands in it. On a store holding
256 time steps per chunk this makes a field read roughly two orders of
magnitude more expensive than it needs to be.

Set ``preload`` to read the whole window of dates a build group needs in
one go and serve every field from memory instead:

.. code:: yaml

   input:
     xarray-zarr:
       url: /path/to/store.zarr
       preload: 8GiB
       param: [2t, 10u, 10v]

``preload`` accepts:

-  ``false`` (the default) -- read one field at a time;
-  ``true`` -- preload within a default budget of 4 GiB;
-  a size, as a byte count or a string such as ``"8GiB"``.

The window covers only the variables, levels and dates the source is
actually asked for. When it would exceed the budget, the build logs a
warning and falls back to reading one field at a time, so a value that is
set too low costs speed but never correctness. The output is unchanged
either way.

.. note::

   The saving grows with the number of dates in a window, so it is
   governed by ``build.group_by``: a group of one date has nothing to
   amortise. Memory grows with the window in the same proportion, so
   ``build.group_by`` and ``preload`` are two ends of one trade-off --
   raise ``group_by`` for speed, and set ``preload`` to the memory you
   can spare.

.. _cf conventions: http://cfconventions.org/

.. _wildcards: https://en.wikipedia.org/wiki/Glob_(programming)

.. _xarray: https://docs.xarray.dev/en/stable/index.html

.. _xarray.open_dataset: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
