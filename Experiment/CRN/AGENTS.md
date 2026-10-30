# CRN experiment scope

- Keep experiment drivers, benchmark inputs, timing harnesses, and result/report
  tools here. Put regression test cases and assertions in `Test/CRN/`; do not
  create a second test runner inside `Experiment/CRN`.
- Reuse the public `synkit.CRN` APIs and shared benchmark inputs. Extend existing
  studies and helpers before adding overlapping scripts or copied fixtures.
- Keep inputs self-contained. Do not depend on manuscript directories, LaTeX
  sources, or paper-specific helpers and filenames. Document rendering belongs
  with the document that consumes the experiment reports.
- Scientific outcome checks and report-completeness checks belong in study
  reports. Keep generated outputs, native build artifacts, and downloaded XML
  caches out of version control and package distributions.
