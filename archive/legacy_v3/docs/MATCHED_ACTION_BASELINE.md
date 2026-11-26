# Reduced matched-action full-horizon comparator

**Purpose:** address the previously acknowledged comparator-action asymmetry without claiming to have solved the legacy 12-period simulator.

The benchmark has eight discrete periods, two latent manufacturing defect regimes (3.5%, 24%), prior weights (0.7, 0.3), pilot and production batches of three, a two-period report delay, at most two pilot batches, and at most one pending report. Each policy can choose the same feasible actions: pilot, deploy, abandon, wait, produce, and rollback. Observations arrive only at the due period. Common random numbers index potential observations by time, so policy comparisons are paired.

The exact baseline uses a Bellman recursion over time, mode, observed good/bad counts, pending report due time, and pilot count. It evaluates future report outcomes using posterior-predictive binomial mixtures. A separate exhaustive recursion checks five representative states. The greedy and fixed policies receive exactly the same observations and action opportunities but use different action-selection rules.

**Caveats:** The exact planner optimizes expected value under its two-point prior, not each fixed true-defect scenario. It selects immediate deployment at the root in the current parameterization. The baseline is mathematically exact *for this reduced model only*. It does not prove a proposed queue-aware heuristic is superior, and it is not a matched-action comparison within the legacy F7.1/F8 12-period simulator. The physical observation stream and economics are synthetic. The model restricts the system to one outstanding report and does not permit continued production while a report is pending.

**Reproduction:** `python run_pipeline.py` or `python src/matched_action_baseline.py --output data/derived --seeds 250`.
