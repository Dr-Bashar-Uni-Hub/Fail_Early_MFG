# Mathematical and algorithmic screening (QM-01 to QM-08) — outcome and decisions

**Scope.** Screening of the seven candidate directions (A–G) proposed for raising the mathematical and algorithmic
novelty of the V5.0 manuscript. V5.0 is preserved unchanged as the baseline (`archive/` history and the V5.0 results remain
reproducible); everything added in V6.0 is listed in `CHANGELOG.md`.

**Bottom line.** No direction yields a strong original mathematical contribution that survives a prior-art check. Two
directions yield *useful applied methodological extensions* with correct, verified propositions whose closest
predecessors are cited explicitly; they were added to the manuscript. The manuscript keeps its V5.0 title and positioning
as an applied *Quality Engineering* paper. Raising "mathematical novelty" to 7.5–8.5/10 is not achievable without
overstating established results, and was not attempted.

## Novelty assessment matrix

| Direction | Closest prior art (verified) | Unresolved gap found | Proof status | Experimental status | Verdict |
|---|---|---|---|---|---|
| A. Decision-time information loss | Blackwell information ordering; value of information (DeGroot 1970); POMDP theory | None beyond applying nested-information monotonicity | Established | — | Insufficient novelty (not pursued) |
| B. Delay-aware exposure feasibility | Detection delay + reporting delay is additive under unchanged observation sequences; economic design of control charts | A computable latency requirement L_max(B) for an exposure limit, validated in simulation, with counterexamples | Proposition 2: elementary, proved | Prediction error ≤ 0.04 periods for all policies; L_max identical to simulation for every limit tested; per-run identity fails in up to 14% of EWMA runs | Established result with industrial application → **added** (Section 4.5, 5.6) |
| C. Model-ageing theory / ADIE | Lipschitz bound |Δ(L)| ≤ KL is elementary; temporal generalization and concept-drift literature | Whether ageing curves predict delay loss | Not provable without strong assumptions | Only two datasets; cannot be validated out of sample | Experimentally unsupported (kept as stated hypothesis) |
| D. Decomposition of evaluation optimism | Telescoping identity; grouped/blocked cross-validation; leakage literature | Statistical identification beyond controlled contrasts | Identity is trivial | C/M/W contrasts repeated with three independent seeds | Useful applied methodology (already in V5.0, strengthened) |
| E. Drift-aware risk calibration | Adaptive conformal inference (Gibbs and Candès 2021); conformal PID / quantile tracking (Angelopoulos, Candès and Tibshirani 2023); delayed-feedback quantile tracking with bound (b+|q₀|+ηh)/(ηn) (Wang and Hyndman, arXiv:2410.13115, Corollary 1) | Feedback that is **action-dependent** (inspected defects confirmed at once, released defects only after delay) and slip defined over defects | Proposition 1: proved; bound (b+|θ₀|+γ(M+1))/(γK) with M = maximum released defects awaiting feedback | Bound verified on 200 adversarial sequences (max ratio 0.78); applied to the Siemens AOI replay | Straightforward extension of established results with industrial application → **added** (Section 3.8, 5.3) |
| F. Selective feedback | Selective labels, inverse-propensity weighting, partial monitoring | Identifiability requires that released defects are eventually observed | Remark (trivial non-identifiability without feedback) | Not run | Stated as an assumption and limitation |
| G. Delay-aware economic SPC optimization | Economic and economic–statistical design of control charts (Duncan 1956 onward) | None established | — | — | Insufficient novelty (not pursued) |

## Proposition 1 (Direction E) — statement and proof sketch

Scores s_i ∈ (−b, b); release if s_i < θ. A defect that is inspected is confirmed immediately (e = 0); a released defect
is confirmed after a delay (e = 1). On each confirmation, θ ← θ + γ(α − e). Let M be the maximum number of released defects
awaiting confirmation at any time and K the number of confirmed defects.

*Claim.* |K⁻¹ Σ_k e_k − α| ≤ (b + |θ₀| + γ(M + 1)) / (γK) for every score sequence.

*Proof.* Summing the updates gives θ_K − θ₀ = γ Σ_k (α − e_k), so it suffices to bound |θ_K|. If θ > b, every new alarm is
released, so no new inspected defect (the only source of upward steps) can occur; θ can exceed b by at most one step γα.
If θ < −b, no alarm is released, so downward steps can come only from released defects already awaiting confirmation, at
most M, plus the step that crossed −b; hence θ ≥ −b − γ(1−α)(M+1). Therefore |θ_K| ≤ b + γ(M+1) and
|Σ_k(α − e_k)| ≤ (|θ_K| + |θ₀|)/γ, which gives the claim. ∎

*Relation to prior art.* For a fixed delay h the same argument gives the Wang–Hyndman bound with h in place of M + 1; the
extension here is that delays depend on the action (only releases are delayed) and the in-flight count M replaces the
fixed horizon. This is a modest extension, not a new theory.

*Limits (impossibility).* The guarantee concerns the long-run average slip over all defects, not any week or any finite
window; it says nothing about volume reduction; and it requires that every released defect is eventually discovered. If
released defects are never observed, slip is not identifiable from feedback and no feedback rule can control it.

## Proposition 2 (Direction B) — statement

If a policy is deployed before the shift and its monitoring statistic observes the same post-shift outcomes for every
delay, a report delay L postpones rollback by exactly L periods, so per-run exposure is min(H_post, Z + L) with Z the
exposure under immediate results. Hence E(L) = E[D·min(H_post, Z+L)], E(L+1) − E(L) = P(D = 1, Z + L < H_post) ≤ 1, and
L_max(B) = max{L : E(L) ≤ B}. Elementary; its value is as a computable latency requirement. Counterexamples: the per-run
identity fails when the monitoring state at the shift depends on L (EWMA in up to 14% of runs; CUSUM in 1.3%); a policy that
never rolls back (FX) has exposure independent of L.

## Rejected or deferred
* A unified named framework (DQIF) — not justified: the surviving pieces have different assumptions and objectives.
* Theorems E1–E6 as stated in the plan — finite-sample *future* risk control under arbitrary drift is impossible without
  assumptions; only the long-run deterministic average guarantee of Proposition 1 survives.
* ADIE — requires more datasets than are available.

## Repeated decomposition seeds (QE-F03)
The C/M/W experiment was repeated with three independent random half-week and training samples (`results/decomposition/`).
