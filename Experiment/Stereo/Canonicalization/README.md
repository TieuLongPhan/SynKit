# Canonicalization

This track has four maintained exact protocols:

1. `configuration_free_local.py`: exhaustive raw local representations after
   supplied configuration is erased;
2. `configuration_free_global.py`: simultaneous mixed-family formal products
   for A/B/C, complete raw-A/raw-B quotients, and the raw-C/VS146 stress
   quotient;
3. `atom_relabel.py global-local`: selective global-by-local certificate
   invariance, backed by `global_local.py`;
4. `multi_element.py`: composition of multiple configured stereo elements.

Run the small maintained gates with:

```bash
python Experiment/Stereo/Canonicalization/configuration_free_local.py internal \
  --family tetrahedral --output /tmp/tetrahedral.json \
  --table /tmp/tetrahedral.csv

python Experiment/Stereo/Canonicalization/atom_relabel.py global-local \
  --family tetrahedral --permutations 2 --class-relabelings 1 \
  --exhaustive-max-atoms 0
```

`run.py` remains a configured-input/source-adapter diagnostic and a regression
surface, but its former retained reports are superseded by the stricter
configuration-free local matrix. `rota_local.py` is likewise superseded by the
RotA task in `configuration_free_local.py`.

The local runner defaults to `direct_single_carrier_transport`: it exhausts
the independently derived proper local frames and asks exact rooted
constitutional isomorphism queries. Every accepted merge has a replayable map;
negative decisions exhaust the admissible bindings. Original typed node and
edge colors decide map validity. It does not enumerate or report the order
of the whole constitutional symmetry group. Ten-second, 1 GiB and 50,000-count
limits produce explicit oracle refusals. The count limit applies separately
to frame bindings, rooted queries and proposed maps. Select the earlier
complete-group method with
`--reference-method exhaustive_constitutional_automorphisms`; no automatic
fallback hides a cap refusal. Joint-carrier validation still uses the
separately bounded complete-group oracle.

Use `python -m Experiment.Stereo.reproduce --output-dir /tmp/stereo-bounded`
for the bounded maintained checks. The `--tier full` sequence requires pinned
CIP SMILES and coordinates and includes formal A/B/C without implicit raw
stress. `Experiment/Stereo/benchmark.sh` is the historical expensive sequence;
its full raw B and raw C/VS146 tail is not the bounded default.

Full protocol details and retained reports are in
`Experiment/Stereo/Data/Canonicalization/`.
