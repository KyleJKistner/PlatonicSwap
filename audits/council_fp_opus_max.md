ℓ` small, so unedited copies at earlier positions persist and get re-read downstream.

**Procedure.**
1. **Fix the echo first, without masks.** Define positives as all positions whose A-side causal effect is ≥ ε (the §1.2 data rule).
   - An organ is a per-token MLP, so it then fires on every entity token, not only the last one.
   - This is self-consistent. The positions that carry the old attribute (and so cause echo) are exactly those whose states encode the entity, which makes them detectable.
   - Target: `C(ℓ, S_org)` near your knockout ceiling. Predict it task-free before running the task.
2. **Placement.** Choose `ℓ* = argmax_ℓ C(ℓ, S_org)·κ̂_B(ℓ)`, where `κ̂_B` is the predicted normalized margin from §2.2.
   - This trades write-commit (earlier layers are better) against read portability (mid-depth is better).
   - `C` is fully task-free. `κ̂` uses only A-side labels plus unlabeled pairs; no B-side task outcome is consulted.
   - Choose `ℓ_A` from the same `C_A(ℓ)` curve computed on A.
3. **Use a band only if `C(ℓ*, S_org) < 1 − ε`.**
   - Take the smallest contiguous band `L` that contains `ℓ*`.
   - Fit per-layer fractions `f_ℓ ≥ 0` (at most 3 parameters) by least squares, so that adding `Σ f_ℓ Δh_ℓ` reproduces `L_s′` on the minimal pairs.
   - Each band layer gets its own socket, `c`, `u`, and write `f_ℓ·d_B(ℓ)`.
   - These are still pure native SwiGLU rows, and unlearning is still a per-layer slice.
4. **70B/80L models.** Measure the curves directly rather than extrapolating from Granite. The cost is O(pairs × layers) forward passes, with no task involved.

### 3.2 Single best sealed Capability 2: BIO-RIPPLE (taxonomic class swap → property ripple)

**Why (A):**
- Base 1.5–3B models are highly competent on animal class and property facts.
- Ripple is scored as forced-choice log-odds, which is unambiguous and cheap.
- There is no geographic vocabulary and no template overlap with Cap-1.

**Why not (B) or (C):**
- **(B)** Base models don't follow instructions. Language or format switching needs generative scoring (language ID or a judge). SmolLM2 is English-centric, so baseline competence would differ across recipients.
- **(C)** Unit or currency ripple runs through arithmetic. At 1.5–3B that would measure arithmetic failure, not mid-layer integration.

**Spec:**
- **Rules.** R1: felids → birds. R2: rodents → fish. These give maximally contrasting property bundles, and two rules mirror Cap-1's two-twin footprint.
- **Construction data.** Class-membership statements only, across at least 8 counterbalanced templates. `Y` comes from the identical frozen Cap-1 procedure.
- **Split.** Split by species, 50/50, with at least 10 test species per rule that are never touched during construction. Held-out templates must have disjoint n-grams.
- **Ripple battery.** None of these appear in construction; each is a two-way forced choice:
  - covering: feathers vs fur, scales vs fur
  - reproduction: lays eggs vs gives birth
  - mouth: beak vs teeth
  - breathing: gills vs lungs
  - limbs: two vs four legs, fins vs legs
  - habitat: water vs land
  - offspring term (e.g., chicks vs cubs), as the composed hop
- **Eligibility, pre-registered.** Keep only items that unedited A and every B already answer correctly.
  - Report the in-context ceiling (prepend "{x} is a bird.") as the ripple normalizer.
  - Publish both filtered and unfiltered numbers.
- **Locality.** Canids, bovids and primates; real birds and fish, which must stay unchanged; unrelated facts; KL/ΔPPL on held-out text.
- **Controls.**
  - shuffled-R (expected ≈ 0)
  - random `c` with matched norms and thresholds (expected ≈ 0)
  - gate-only and up-only ablations
  - native-B oracle
  - in-context ceiling
- **Headline metric.** Fraction of the native-B oracle recovered. This normalizes for task difficulty, so Cap-1 and Cap-2 are directly comparable.
- **Power.** At least 100 items per (model, metric) cell. Bootstrap CIs over *species*, which is the unit of generalization, not over items.
- **Budget.** Sockets and `C(ℓ)` are task-free and precomputed once. The sealed run is closed-form construction (seconds) plus a few thousand short fp16 forward passes per model, so it fits in 10 minutes.

---

## 4. Strict Anti-Overfitting Verification Checklist

Both capabilities must pass with identical frozen code.

**Sealing**
1. Freeze the code hash, ε, corpus hash and depth law before generating any Cap-2 data. Hash the sealed test set. Do one run and report everything, counting abstentions as failures.
2. Audit constants: grep the organ builder for float literals. Only ε and mathematical constants (0.2785, Φ⁻¹) are allowed.
3. Re-run Cap-1 on smollm2 with counterbalanced templates. If the 46–69% does not survive, retract it as the baseline.

**Validity of the law**
4. Retro-test (§2.2): using unlabeled data only, every task-1822 zero is either flagged or attributed to a measured term.
5. Predicted margin vs realized accuracy is monotone across all (model, rule) cells. Report Spearman ρ.
6. Pushforward closure: the pushforward of A's general-text moments matches native B moments on held-out text, within CI.
7. For each organ, report tail coverage (N_tail) and tail-vs-bulk channel agreement.
8. Rank-1 portability: report the R² of `y_B ≈ g·T_out y_A` on positives.

**Efficacy** (species-bootstrap CIs, at least 100 items per cell)
9. Direct accuracy on unseen entities × held-out templates.
10. Ripple normalized by the in-context ceiling, plus the fraction of the native-B oracle recovered.

**Controls**
11. Shuffled-R ≈ 0 and random `c` ≈ 0. Include gate-only and up-only ablations, and report chance level for every metric.
12. Dose–response is monotone, with s = 1 at or past the knee.

**Locality**
13. On text disjoint from G, measure:
    - KL, ΔPPL, and a per-token false-fire rate ≤ ε
    - protected-neighbor accuracy
    - a lobe-zone audit of tokens in `[t1 − 5/β, t1]`
14. Composability: with the Cap-1 and Cap-2 organs installed together, neither loses more than its CI, and locality is unchanged.

**Unlearning**
15. After slicing the added rows, check state-dict hash equality and `torch.equal` on logits (same dtype, device and batch shape), with `d_ff` restored. This is guaranteed by construction, so report it as a sanity check, not a finding.

**Scope**
16. The layer is chosen by the task-free law. Any layer sweep is shown only as post-hoc analysis.
17. State the README claim precisely:
    - Passing both capabilities supports "domain-general entity-rule transplant."
    - "Arbitrary capability" (code, refusal, style) still needs a sealed non-entity task with an input-dependent write (§1.2 step 7).
