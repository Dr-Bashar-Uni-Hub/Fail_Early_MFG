# Execution notes

This repository curates previously executed, independently versioned research stages. Scripts have not been refactored into a unified pipeline, because doing so could change the frozen experiment. Some stage scripts use local file names or require the original `framework.py` import name. For F5.2 and F6.1, place `f52_f61_framework.py` beside the runner under the name `framework.py` if the original import requires it. Read each source before running and ensure the original UCI SECOM files are available locally.

Primary verification command: `python tests/verify_archive.py`. This checks derived outputs, not scientific generalizability or full simulator optimality.

For raw trajectory replication, the F7.1/F8 runner is `src/f71_f8_simulation.py` and its archived raw CSV is `data/derived/f71_f8_raw.csv`. Reproducing other stages requires inspecting their frozen configurations.

Figures are derived from archived CSVs; they are not additional experiments.
