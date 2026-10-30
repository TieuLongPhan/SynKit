Implementation and source layout
================================

The public API is exported from ``synkit.Chem.Mapper``. Solver implementations
live under ``solvers/``. Reaction preparation and result handling remain in
Python.

::

   Mapper/
     mapper.py            Method selection and common results
     reaction_search.py   Reaction preparation and output
     _compat.py           Historical imports and build commands
     chem/                Chemistry conversion
     graph/               Graph models and symmetry helpers
     io/                  Mapping serialization
     solvers/
       slap/              Python SLAP
       exact/             Python exact methods and C++ graph bindings
       native/            Shared C++ bindings and explicit builds
         cpp/             C++ sources, headers, CMake and examples
     reporting/           Enumeration analysis and spectra
     validation/          Mapping checks and reference comparison
     adapters/            Prediction seeds and executable templates

Entry points
------------

``map_reaction_implicit`` uses the C++ core to return heavy-atom classes with
implicit-H scoring. ``Mapper("native")`` selects the C++ solver through the
common interface and uses explicit-H reaction semantics by default.
``Mapper()`` retains the Python ``propagate`` default. See :doc:`/mapper` for
query options, result fields and hydrogen modes.

Native builds
-------------

Platform wheels bundle the compiled library. A checkout or editable install
can build it explicitly:

.. code-block:: python

   from synkit.Chem.Mapper import Mapper
   from synkit.Chem.Mapper.solvers.native.build import build_mapper_native

   library = build_mapper_native("build/mapper-core")
   result = Mapper("native", solver="pabs", library_path=library).map("CO>>CO")

The builder records compiler, source and library checksums. An explicit
``library_path`` overrides the bundled library. Importing the package and
calling a mapper do not compile code automatically.

The standard build provides ten native configurations. OR-Tools C++ support
adds CP-SAT, SCIP/MILP and KMILP adapters, plus optional PABS LP support.
MaxSAT requires a qualified native executable. Available configurations can
be inspected with ``Mapper.native_methods(available_only=True)``.

Compatibility
-------------

The former ``exact``, ``slap`` and ``_native`` paths remain aliases to the
same modules. The ``synkit.Chem.Reaction.Mapper`` namespace and historical
builder commands are also supported. New implementation code uses the
``synkit.Chem.Mapper.solvers`` paths.

Testing
-------

.. code-block:: console

   conda run -n synkit bash scripts/pytest.sh Test/Chem/Mapper/

The API, chemistry preparation, native solvers and class comparisons have
separate tests under ``Test/Chem/Mapper/``.
