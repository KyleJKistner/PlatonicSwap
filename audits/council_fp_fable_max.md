# Panelist 2 review: PlatonicSwap generality audit

**Bottom line: do not push yet.** The twin-step mechanism is sound, but the generality evidence is one recipient, two facts, constants tuned on that recipient, and a known template confound. Sample sizes are tiny: 37.5% is 3/8 or 6/16, and even 6/16 has a 95% Wilson interval of 18–61%.

One change removes nearly all the hand-tuning: **stop transporting donor weights. Push donor data through an empirically calibrated channel and fit the organ in recipient coordinates, with thresholds set on real recipient statistics.** The formulas below are derived, not run; the depth/band law in §3 is a hypothesis with a stated falsification test.

Notation, from unlabeled pairs at char-aligned word-final positions, in RMS-normalized pre-γ coordinates:

```
G = Σ_BA Σ_AA⁻¹              A→B generative regression (D_B × D_A)
N = Σ_BB − G Σ_AA Gᵀ          B variance not explained by A
channel:  x_B | x_A ≈ μ_B + G(x_A − μ_A) + e,   Cov(e) = N
ρ_i = canonical correlations of (x_A, x_B)
```

## 1. Verdict and task-agnostic organ law (Q1)

**Why P-ORGAN-0 collapsed, beyond 0.40²**

1. **Pullback is a low-pass filter in ρ-space.** `w_B = M_inᵀ w_A` reads, in donor terms, the functional `Kᵀ w_A` with `K = M_in G`. For a least-squares socket, `eig(K) = {ρ_i²}`, so each component of `w_A` on mode i is scaled by ρ_i² and the direction rotates toward shared modes. `cos ≈ 0.40` is that rotation, not noise. A Stiefel socket does the same and additionally assumes ρ_i is constant.
2. **Donor-optimal discriminants anti-select for transportability.** Per canonical mode with class gap Δ_i and donor within-class variance s_i²:
   ```
   A-optimal weight:  Δ_i / s_i²
   B-optimal weight:  ρ_i Δ_i / (ρ_i² s_i² + 1 − ρ_i²)
   ```
   The recipient noise floor `1 − ρ_i²` does not shrink with s_i, and SGD piles weight on small-s_i modes. Example: mode 1 (Δ=1, s²=1, ρ=0.95) plus mode 2 (Δ=0.3, s²=0.01, ρ=0.3) gives donor d′ = 3.16, pulled-back d′ = 0.20, channel-matched d′ = 0.95.
3. **Ramp SwiGLU is degree 2.** `(ρ_g g + b_g + ε_g)(ρ_u u + b_u + ε_u)` has signal ρ_g ρ_u, leak terms linear in g and u, and a DC term `E[ε_g ε_u] ≠ 0` because both errors come from the same x_B. That is your zero-point drift.
4. **Joint 32-neuron solutions are cancellation codes.** Background silence is `Σ_j d_j a_j ≈ 0` with individual `a_j ≠ 0`. Transport errors are incoherent, so the residue is amplified by `κ = Σ‖d_j a_j‖ / ‖Σ d_j a_j‖`.
5. **The solution is unidentified.** With n ≪ D, components of w outside the training span are never constrained by the loss, and `M x_B` puts energy there. The 62–92% native-B oracle only shows the function class is adequate.

**Why twin-step survived.** `∂s/∂z = 0` on both class supports, so transport error enters as a tail probability rather than an amplitude. Each unit is individually silent, and the mean-difference direction sits in high-variance shared modes. But 1683 is only half-digital: `up` is still an analog path, and `u_ent` falling from 1.234 to 0.381 on Granite is that path collapsing.

**The law.** Refuse to fit if frame ID alone separates X₊ from X₋ (AUROC ≠ 0.5); compute contrasts within frame.

```
f(x) = Σ_c s_c(x) · v_c
s_c  = [SiLU(β(z − t1)) − SiLU(β(z − t2))] / (β(t2 − t1)),   z = w_cᵀx
up ≡ constant pin (BOS-nulled);  2 rows per cell
```

1. **Cells and payload (donor only).** `v_c` is the mean target displacement of cell c. Cells are clusters of Y (one if Y is constant), with the count chosen by donor-side cross-validation of predicted hit rate.
2. **Push-forward.** For each donor activation, `x̃_B = μ_B + G(x_A − μ_A) + e`, with e resampled from empirical calibration residuals. This is the only augmentation; the α-range and `0.5·Σ_ε` go away.
3. **Direction.** `w_c = S⁻¹(m₊ − m_bg)`, using simulated positives against real B calibration tokens, simulated in-domain negatives and other cells, with Ledoit–Wolf shrinkage on S. The Gaussian closed form is `w = (G S_A Gᵀ + N)⁻¹ G Δ_A`.
4. **Thresholds in B units.** `t1` is the (1−α) order statistic of z over real B calibration tokens and simulated negatives. `t2` is the α′ order statistic of simulated positives. Let `Γ = t2 − t1`.
5. **Sharpness.** `β = 0.2785 / (δΓ)`, because `sup_{z≤t1}|s|` and `sup_{z≥t2}|s−1|` are both at most `0.2785 / (β(t2−t1))`, where 0.2785 is the maximum of |SiLU| on its negative lobe.
6. **Rows.** `gate = β(w − t·pin)`, `up = pin`, `down = ± Δh_B / (rm_B·βΓ)`.

The only free numbers are three tolerances in operational units: false-fire α, miss α′, and plateau ripple δ. Derive α from the KL budget as `α = ε_KL / KL_fire` and declare all three once, before capability 2 exists.

**Scope.** This is channel-aware ROME with a saturating read. It is general over capabilities that are, at one layer, a linearly triggerable low-rank residual update: facts, concept rules, topic-gated steering. It is not general over capabilities needing new attention routing. "Works for ANY capability" is indefensible; "the same frozen code either transplants or refuses, and the refusal is calibrated" is defensible.

**(a) vs (b).** Choose (b), relocated to recipient coordinates: it is convex, has a unique optimum, needs no seeds, and yields the abstention statistic for free. Option (a) with `αx + (1−α)μ + ε` has the wrong noise geometry (scalar where the channel is per-mode), and its 0.3–1.0 range is a testbed constant in disguise. Allow SGD later only for multi-cell organs, with step 2 as its sole noise source.

## 2. Cross-architecture invariance law and abstention (Q2)

**(i) Socket distribution**

- **Prose-only is not yet shown to be the cause.** Socket type is perfectly collinear with recipient, layer and smollm2-tuned constants. `AU_unrel = 0.307` is a sign flip, which is a confound signature rather than attenuation. Run the missing 2×2: smollm2 with prose-only, and the other three with the dual socket.
- **Law.** Transport error on a target population is variance plus bias. Variance scales with the target states' leverage `xᵀ(Σ_π + λI)⁻¹x` under the calibration design; bias comes from local nonlinearity. This fits `AU_up_gen = 1.000` (coarse, high-variance) surviving while the fine within-entity directions die. Your 350 anchors are a hand-made importance weight toward the test distribution.
- **Universal distribution.** Use a G-optimal design over a large diverse pool (prose, code, math, dialogue, QA, tables, lists), with task entities and templates blacklisted by n-gram. Choose weights π to maximize `log det Σ_π`, which minimizes worst-case prediction variance over the pool's support (Kiefer–Wolfowitz); leverage-score sampling is the one-pass approximation. Compute it from donor activations only and freeze it.
- **Coverage certificate (donor-side, label-free).** If organ-input leverage exceeds the pool maximum, extend the pool (never with task entities) or abstain. If coverage passes but predictions miss, refit with donor-side density-ratio weights `ω = p_task/p_cal`. Abstain when held-out weighted R² fails a permutation null.

**(ii) Read**

- **Width.** G is rectangular. Modes lost when D_B < D_A have ρ = 0 and get zero weight automatically; when D_B > D_A, B-private dimensions enter only through N. No width constants are needed.
- **Slope and intercept.** Do not calibrate thresholds across models at all. `r·σ_A/σ_B` shrinks by r², and `σ_A/(r·σ_B)` amplifies noise by 1/r. Use only the generative relation on standardized calibration scores, `z_B = r·z_A + η` with `Var η = 1 − r²`, to predict where positives land:
  ```
  z_B⁺ ~ ( r·D ,  r²ŝ² + 1 − r² )
  proceed iff   r·D − κ·sqrt(1 − r²(1 − ŝ²))  ≥  Q_α
  ```
  Here D is the donor class gap in calibration σ, ŝ the donor positive spread, `κ = sqrt((1−α′)/α′)` (Cantelli), and Q_α the real B background quantile. This is equivalent to Γ > 0.
- **r_min is derived, not constant.** With κ=2 and Q=3, a D=8 feature tolerates r ≈ 0.58, while D=5 needs r ≈ 0.83.
- **No noise magnification (proof).**
  1. `1[z_B > t]` is invariant to affine rescaling of z_B, and no 1/r appears anywhere.
  2. `t1` is an order statistic of real B scores, so by exchangeability `P(fire on a fresh general token) ≤ (N−k+1)/(N+1)` for any w_B, including a pure-noise direction.
  3. Fired amplitude lies in `[1−δ, 1+δ]·v` regardless of the scale of z.
  
  So as r → 0 only the hit rate degrades: the organ fails silent, never loud. Hard-negative false fires are model-based and carry no such guarantee.
- **One untestable assumption.** The channel fit on calibration text must hold on the task population. Report predicted versus realized d′ for every cell.

**(iii) Write**

- **Fit on post-layer residual states, not `mlp_out` or the MLP delta.** States are what downstream layers read; MLP deltas sample only the directions that layer natively writes. Fitting in RMS-normalized coordinates bounds every token at norm √D, which replaces the `2.5×` outlier filter.
- **σ-isometry on above-null shared modes** (OLS shrinks by ρ_i, which is what `gain=1.15` was patching):
  ```
  δ_A   = Δh_A / rms_A
  M_iso = Σ_BB^{1/2} V_k U_kᵀ Σ_AA^{-1/2}     k = held-out ρ_i above the permutation-null edge
  Δh_B  = rms_B · M_iso δ_A
  ```
- **Scale factors.** `rm` enters exactly once, as `1/rm_B` in `down`. `emb_mult` and stream scale enter only through `rms_B`, because the read side is scale-free under RMSNorm.
- **Abstain** if the retained fraction `‖U_kᵀ Σ_AA^{-1/2} δ_A‖² / ‖Σ_AA^{-1/2} δ_A‖²` is not above the random-direction value k/D_A.
- **Amplitude check without a gain.** Inject `s·Δh_B` ungated on unlabeled contexts (build-time only) and solve `KL_B(s*) = KL_A(1)`. If the confidence interval for s* excludes 1, fix the socket. This check is necessary because write amplitude is not identifiable from second-order unlabeled statistics alone: weights anywhere between ρ_i and 1 are consistent.

## 3. Depth/band law and sealed second capability (Q3)

**Depth and band (hypothesis).** Compute two task-free curves on B by resample-patching the last token of multi-token words in unlabeled text. Build-time hooks do not violate your inference constraints.

```
C_B(ℓ)    = E KL(clean ‖ patch last token at ℓ) / E KL(clean ‖ patch whole word at ℓ=0)
H_B(ℓ→ℓ′) = 1 − ‖h_ℓ′^patch − h_ℓ′^clean‖_Σ / ‖h_ℓ^patch − h_ℓ^clean‖_Σ     (same position)
```

- **Where.** `ℓ* = argmax_ℓ P̂_hit(ℓ)·C_B(ℓ)`, with P̂_hit from §2(ii). No B-side task data is used.
- **How strongly.** If self-healing H_B is non-negligible (expected for Granite), add band layers where incremental healing is largest. Each re-asserts only what the model undid: `amp_k = [H(ℓ*→ℓ_k) − H(ℓ*→ℓ_{k−1})]·Δh_B`, with its own gate and socket, until retention reaches 1 − α′. Do not over-drive one layer.
- **Falsification.** From unlabeled data alone, C_B and H_B must reproduce Granite's 53.6% versus 93.5% gap, and ℓ* must match the task-grid optimum on capability 1. `‖Δh_ℓ‖/‖h_ℓ‖` alone measures step size, not routing. Five architectures is thin; the 80-layer extrapolation is unsupported until tested on a deeper model.

**Sealed capability: BIRD → FISH class rewrite at the species token (option A).** Base 1.5–3B models are highly competent on closed-set animal attributes and it has no geographic overlap. I reject (C) because arithmetic is too weak to give measurable ripple, and (B) because base models respond unreliably and ripple is ill-defined.

- **Donor supervision.** 24 land birds × 4 frames. Negatives are mammals, reptiles and insects, plus flying non-birds (bat, butterfly, airplane) and aquatic mammals (whale, dolphin).
- **Sealed test.** 24 unseen, non-polysemous land birds, scored by forced-choice first-token logit. Pre-filter items by unedited B competence, fixed before the implant.
  - **Direct:** "A {e} is a kind of" (fish vs bird).
  - **Ripple ×4:** covering (scales/feathers), breathing (gills/lungs), limbs (fins/wings), medium (water/air).
  - **Compositional ×2, read at a different position:** "Mia bought a {e} as a pet. At home she keeps it in a" (tank/cage); "Can a {e} breathe underwater? Answer:" (Yes/No).
  - **Invariant:** "A baby {e} hatches from an" (egg) must not change.
  - **Locality:** bat fur, whale lungs, other animals, all capability-1 prompts, and general KL/PPL.
- **Controls.** Shuffled-R; the France organ on bird prompts and the reverse (both ≈ 0); a norm-matched random payload; the native-B oracle.
- **Pass.** With 96 ripple and 48 compositional trials per recipient, the Wilson lower bound must exceed the shuffled upper bound and reach at least half the oracle. A correct ABSTAIN also passes.
- **Caveat.** This is the same kind of capability as capability 1, so it tests domain generality, not capability-type generality. Keep a reserve (e.g. metal → gas), because any post-hoc fix burns this one.

## 4. Anti-overfitting checklist (both capabilities, same frozen commit)

1. **Freeze first.** Hash the code, tolerances, pool and sockets before capability-2 data enters the repo. Run once on all four recipients and report every cell.
2. **Remove every tuned constant.** Derive or delete `t1_scale`, `u_ent_frac`, `β`, `width`, `gain`, `0.5Σ_ε`, the `2.5×` filter, the 80/20 mix, the 350 anchors, and the hand-picked layers L8/L9/L13.
3. **Confound certificates.** Frame-only AUROC is 0.5, and the pool has zero n-gram overlap with task entities or templates.
4. **Run the 2×2** (socket type × recipient), and rerun smollm2 capability 1 under the frozen law with no regression.
5. **Check prediction calibration.** Predicted and realized hit rates agree across organ × recipient × layer. Abstentions are correct: forcing an abstained cell fails.
6. **Statistics.** At least 96 trials per cell with Wilson intervals, at least 3 noise seeds, and thresholds relative to the native-B oracle. Explain why direct efficacy is below ripple.
7. **Controls.** Shuffled-R on read only and on write only; a wrong-organ swap; a norm-matched random payload; and a late-layer flashcard organ that passes direct but must fail ripple, to show the metric discriminates.
8. **Locality.** False-fire rate on fresh text stays within the conformal bound, and KL/PPL stay within budget. With both organs co-installed neither affects the other, and slicing one leaves the other bit-identical.
9. **Unlearning.** Report the `0.000000` result, but label it as true by construction, not as evidence.
10. **Sensitivity.** Halving or doubling each tolerance, or resampling the pool, changes no pass/fail outcome.
11. **Depth.** The task-free ℓ* matches the grid optimum on capability 1 and is used blind on capability 2.
12. **Width and Granite.** 2048→1536, 2048→3072 and Granite each either pass or correctly abstain; a silent 0% is a fail.
