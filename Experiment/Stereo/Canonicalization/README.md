# Canonicalization

This track has three distinct protocols:

1. `run.py`: exhaustive local-reference permutations on a fixed graph;
2. `atom_relabel.py global-local`: selective global-by-local certificate
   invariance, backed by `global_local.py`;
3. `multi_element.py`: composition of multiple configured stereo elements.

Run the small maintained gates with:

```bash
python Experiment/Stereo/Canonicalization/run.py internal \
  --family tetrahedral

python Experiment/Stereo/Canonicalization/atom_relabel.py global-local \
  --family tetrahedral --permutations 2 --class-relabelings 1 \
  --exhaustive-max-atoms 0
```

Full protocol details and retained reports are in
`Experiment/Stereo/Data/Canonicalization/`.
