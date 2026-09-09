#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly IMAGE="texlive/texlive@sha256:ccf0168bb3dc1e5ba18094131ebb57177f90eca37ab2727bc2d2afb54ad60a51"
readonly MAIN_ARG="${1:-neural_pf_tracking_calibration_error_geometry.tex}"
readonly MAIN="${MAIN_ARG%.tex}"
readonly BUILD_DIR="build/latex"
readonly OUTPUT_DIR="output/pdf"

if [[ -z "${MAIN}" ]]; then
  echo "Usage: $0 [main-file.tex]" >&2
  exit 2
fi

if [[ "${MAIN_ARG}" == */* || ! -f "${PROJECT_ROOT}/${MAIN}.tex" ]]; then
  echo "Main file must be a top-level .tex file in ${PROJECT_ROOT}: ${MAIN_ARG}" >&2
  exit 2
fi

mkdir -p "${PROJECT_ROOT}/${BUILD_DIR}" "${PROJECT_ROOT}/${OUTPUT_DIR}"

docker run --rm \
  --platform linux/amd64 \
  --volume "${PROJECT_ROOT}:/work" \
  --workdir /work \
  "${IMAGE}" \
  bash -c "
    rm -f '${BUILD_DIR}/${MAIN}'.*
    pdflatex -interaction=nonstopmode -file-line-error -synctex=1 \
      -output-directory='${BUILD_DIR}' '${MAIN}.tex'
    if grep -q '\\\\bibdata' '${BUILD_DIR}/${MAIN}.aux'; then
      (
        cd '${BUILD_DIR}'
        BIBINPUTS='/work:' BSTINPUTS='/work:' bibtex '${MAIN}' || true
      )
    fi
    pdflatex -interaction=nonstopmode -file-line-error -synctex=1 \
      -output-directory='${BUILD_DIR}' '${MAIN}.tex'
    pdflatex -interaction=nonstopmode -file-line-error -synctex=1 \
      -output-directory='${BUILD_DIR}' '${MAIN}.tex'
    pdflatex -interaction=nonstopmode -file-line-error -synctex=1 \
      -output-directory='${BUILD_DIR}' '${MAIN}.tex'
    cp '${BUILD_DIR}/${MAIN}.pdf' '${OUTPUT_DIR}/${MAIN}-tl2025.pdf'
  "

echo "Built ${PROJECT_ROOT}/${OUTPUT_DIR}/${MAIN}-tl2025.pdf"
