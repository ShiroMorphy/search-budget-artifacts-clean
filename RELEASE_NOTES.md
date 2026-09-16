# Release notes

## v1.0.1 (2026-09-16)

This traceable editorial-metadata release aligns the public reproducibility
record with the revised submission title and positioning:

- updated the package title and citation metadata to identify the Gaussian
  continuous-causal-emergence setting explicitly;
- recorded that the manuscript foregrounds the coupled-network result: a
  fixed search budget can create an apparent threshold that is not robust to
  adaptive step sizing;
- made no changes to numerical code, data-retrieval workflow, archived
  outputs, figures, or formalization relative to v1.0.0.

The prior v1.0.0 release remains immutable and citable.

## v1.0.0 (revised 2026-09-16)

This release replaces the earlier `v1.0.0` snapshot while preserving a fully
traceable Git history: the earlier snapshot remains the direct parent of this
release commit. The replacement aligns the public reproducibility package with
the final submission package.

Changes in this revision:

- regenerated all four publication figures with the final, legible layout and
  unambiguous `\hat q` labels;
- archived the per-window empirical witness table used to summarize the
  18-window feasible-witness diagnostic;
- archived the corresponding feasible projections and covariance inputs;
- replaced the diagnostic runner so that a clean rerun regenerates the
  per-window table, aggregate Figure 2 input, and witness archive together.

The manuscript, cover letter, and journal highlights are intentionally absent
from this repository because they are submission materials rather than
reproducibility dependencies.
