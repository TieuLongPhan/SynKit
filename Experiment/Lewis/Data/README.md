# Lewis benchmark input registry

These files are the immutable inputs shared by the Lewis experiments.

| File | Records | Purpose | SHA-256 | Terms |
| --- | ---: | --- | --- | --- |
| `all.csv` | 5,426 | Original RMechDB radical corpus | `602caa0680832d8f8e4fc8e49b8391cab70f3c3339f29dba28f4a8ab13b3c832` | CC BY-NC-ND 4.0; see `LICENSES/RMechDB-DATA-NOTICE.md` |
| `combinatorial_all.csv` | 95,888 | Original PMechDB polar corpus | `33a4c90486f4291d6e525a55605d17fcbc46bc900bdaa81801e282a0b5607383` | CC BY-NC-ND 4.0; see `LICENSES/PMechDB-DATA-NOTICE.md` |
| `benchmark.json.gz` | 39,732 | Project-owned partial-AAM and rule-replay corpus | `57bcd1910e24e11d88d31983ae07080585afebae7738dd529a9abae53807b1b4` | SynKit project data |
| `hydrogen.pkl.gz` | 109 | PartialAAMs real-reaction hydrogen-extension corpus | `d0b64765d9da17f34b983b4293a5bdb02f8928cf91c16a129485996c5dc35cbb` | PartialAAMs project data |

Despite their historical filenames, `benchmark.json.gz` is currently plain
JSON and `hydrogen.pkl.gz` is currently plain pickle. Their experiment loaders
detect compression by file magic and handle either form.

The RMechDB and PMechDB files are redistributed as unmodified original
datasets. They are third-party data and are not covered by SynKit's MIT
license. Do not modify them in place; derived reports belong in the owning
experiment's output directory.
