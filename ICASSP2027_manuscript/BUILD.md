# ICASSP 2027 manuscript build

This directory is a self-contained snapshot of the ICASSP 2027 manuscript:
`neural_pf_tracking_calibration_error_geometry.tex`, its bibliography, style
files, and the three PDF figures it includes.

## Contents

- `neural_pf_tracking_calibration_error_geometry.tex`: main manuscript
- `neural_pf_tracking_calibration_error_geometry.bib`: bibliography used by the manuscript
- `spconf.sty`, `IEEEbib.bst`: ICASSP/IEEE style files
- `csp_diagram_finished_x.pdf`, `figure1_controlled_error_geometry_exact.pdf`,
  `gridsfm_raw_vs_csp16_discrete3.pdf`: figures included by the manuscript
- `build/latex/`: most recent LaTeX auxiliary files and build log
- `output/pdf/`: compiled manuscript PDF

## Reproduce the PDF

Docker Desktop must be installed and running. From this directory, run:

```sh
./scripts/compile-overleaf.sh
```

The script uses the pinned TeX Live 2025 image:

`texlive/texlive@sha256:ccf0168bb3dc1e5ba18094131ebb57177f90eca37ab2727bc2d2afb54ad60a51`

It runs pdfLaTeX, BibTeX, and three final pdfLaTeX passes. The outputs are:

- `build/latex/neural_pf_tracking_calibration_error_geometry.log`
- `output/pdf/neural_pf_tracking_calibration_error_geometry-tl2025.pdf`

To compile a different top-level manuscript placed in this directory:

```sh
./scripts/compile-overleaf.sh AnotherManuscript.tex
```

## Verify a build

```sh
grep -E '(^!|Undefined control sequence|Emergency stop|Fatal error|LaTeX Warning: (Citation|Reference)|There were undefined)' \
  build/latex/neural_pf_tracking_calibration_error_geometry.log
```

The tracked PDF was rebuilt successfully with TeX Live 2025 on 2026-09-09.
It contains five US Letter pages. The build has no LaTeX errors, undefined
citations, or undefined references. Two pre-existing layout warnings remain:
an 18.82 pt overfull box in the author block and `\small` being used in math
mode in the controlled-error equation. Neither is introduced by this packaging
step.
