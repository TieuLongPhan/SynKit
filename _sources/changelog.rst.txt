Changelog
=========

Version 0.0.7
-------------

**Highlights**

- Refactored source-code structure into six primary submodules at the root level:  
  `IO`, `Chem`, `Graph`, `Rule`, `Synthesis`, and `Vis`.  
- Added MØD‐free operation mode: functions requiring MØD now raise clear errors but fall back to pure‐Python implementations where available.  

IO Module
^^^^^^^^^

- Exposed core I/O utilities directly in `synkit.IO`:  
  `chemical_converter.py`, `data_io.py`, and `debug.py`.  

Chem Module
^^^^^^^^^^^

- Introduced **`CanonRSMI`** for atom–atom mapping (AAM) canonicalization.  
- Moved **`AAMValidator`** into `synkit.Chem.Reaction` for consistency.  

Graph Module
^^^^^^^^^^^^

- Added **`SynGraph`** wrapper for reaction and molecule graphs.  
- New canonicalisation backends:  
  - **node‐type sort**  
  - **Morgan‐prime hashing**  
  - **Weisfeiler–Lehman refinement**  
- Renamed “Cluster” to **Matcher**; enhanced **`GraphMatcher`** and **`SubgraphMatch`**.  
- Added **`SubgraphSearchEngine`** with three strategies:  
  - `component‐aware`  
  - `arbitrary`  
  - `backtracking`  
- Introduced **`SING`** and **`TURBOIS`** for mapping multiple patterns in a single host graph.  
- Extended **`GraphCluster`** and **`BatchClustering`** to support both `nx` and `mod` backends.  
- Enhanced **`WLHash`** to hash lists of node/edge attributes.  
- Added **`MTG`** submodule for Mechanistic Transition Graphs (direct rule composition).  
- New **`Hydrogen`** submodule for reaction-center H-completion and **`Context`** for radius-based expansion.  

Rule Module
^^^^^^^^^^^

- Introduced **`SynRule`** wrapper supporting NetworkX graphs and GML.  
- Reorganized into three packages:  
  - **Apply** (retro-prediction via partial composition)  
  - **Compose** (rule composition)  
  - **Modify** (rule editing and H-handling)  
- Provided non‐MØD fallbacks where possible.  

Synthesis Module
^^^^^^^^^^^^^^^^

- Divided into three submodules:  
  - **Reactor** (`nx` via `SynReactor`; `mod` via `MODReactor`/`MODAAM`)  
  - **CRN** (Chemical Reaction Network builder via `MODCRN`)  
  - **MSR** (multi-step reaction pathfinder)  
- **`SynReactor`** now supports implicit‐H templates.  
- **`MODCRN`** wraps MØD for CRN generation; requires manual re-run for PDF summaries.  

Vis Module
^^^^^^^^^^

- Visualization tools organized under **`synkit.Vis`**:  
  - **`RXNVis`** (reaction visualisation)  
  - **`RuleVis`** (template/rule visualisation)  
  - **`GraphVisualizer`** (generic graph editing & display)  

Documentation
^^^^^^^^^^^^^

- Added comprehensive examples for each submodule.  
- Scaffolding for an API Reference page.  
