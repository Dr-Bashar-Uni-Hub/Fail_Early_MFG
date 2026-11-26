# Comparator fairness audit

All three policies in the sensitivity experiment observe the same released reports, share physical random streams and have the same pilot, deployment and production cost accounting. **Their action constraints differ**: fixed does not rollback; reactive and queue_guard can rollback; fixed stops pilot at a different threshold. These are matched-information but **not matched-action** policies. The sensitivity results are diagnostic, not algorithmic superiority evidence. A fully matched-action, full-horizon optimizer has not been run.

The original F7.1/F8 experiments use scenario-specific rollback cost 18 for the expensive scenario; the new sweep overrides that with an explicit parameter. Results are a new controlled simulation, not reanalysis of plant outcomes.
