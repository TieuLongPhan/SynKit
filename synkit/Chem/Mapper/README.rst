Mapper source layout
====================

Public entry points are exported from ``synkit.Chem.Mapper``. The selected
native mapper uses PABS with SLAP starting mappings. Use the main documentation for the
`user guide <../../../doc/mapper.rst>`_ and
`implementation notes <../../../doc/mapper/implementation.rst>`_.

::

   Mapper/
     __init__.py          Public exports
     mapper.py            Method selection and common results
     reaction_search.py   Reaction preparation and output
     _compat.py           Historical import and command aliases
     chem/                Chemistry conversion
     graph/               Graph models and symmetry helpers
     io/                  Mapping serialization
     solvers/
       slap/              Python heuristic
       exact/             Python exact methods and C++ graph bindings
       native/            Shared C++ bindings and builds
         cpp/             C++ sources, headers, CMake and examples
     reporting/           Enumeration analysis and spectra
     validation/          Reference and mapping checks
     adapters/            Prediction seeds and executable templates

All solver implementations live under ``solvers/``. Native source and bindings
stay together in ``solvers/native/``. The user guide and implementation notes live in ``doc/mapper/``.

Platform wheels bundle the shared C++ library. Source checkouts use an explicit
build::

   from synkit.Chem.Mapper import Mapper, map_reaction_implicit
   from synkit.Chem.Mapper.solvers.native.build import build_mapper_native

   library = build_mapper_native("build/mapper-core")
   result = map_reaction_implicit("CCO>>COC", library_path=library)
   mapper = Mapper("native", solver="pabs", library_path=library)

The old ``exact``, ``slap`` and ``_native`` import paths remain aliases to the
same implementations. New code uses the ``solvers`` paths. The historical
``synkit.Chem.Reaction.Mapper`` namespace is also supported.
