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

Run the complete maintained matrix with
`Experiment/Stereo/benchmark.sh`. It includes raw A, formal A/B/C, full raw B,
and raw C/VS146 as the final long-running stress stages.

Full protocol details and retained reports are in
`Experiment/Stereo/Data/Canonicalization/`.
