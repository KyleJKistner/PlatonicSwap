# PlatonicSwap — Round 2 Council Verdict: 85–95% Ceiling Analysis & v1 Closeout Plan

> **Date:** 2026-11-26 | **Audited State:** Commit `7745b81` (`/tmp/PlatonicSwap/`) + `/tmp/platos_key_lab/` (including completed Cap-1 `results_cap1_channel_pushforward.json` and Sealed Cap-2 `results_bio_ripple_sealed.json`) + all 24 full-text papers in `docs/papers/` + all 6 Round 1 reports.
> **Round 2 Roster (5/5 unanimous, 100% read-only audit):**
> 1. **Opus 5.5 Max #1** — Ceiling & Bottleneck Autopsy (Gate FNR vs. Write Distortion vs. EM Ceiling)
> 2. **Opus 5.5 Max #2** — Literature-to-Algorithm Upgrade Matrix & Cap-2 Seal Integrity Audit
> 3. **Opus 5.5 Max #3** — Repo Closeout Engineer & Publication Auditor (Defects B1–B11, R1–R6, README audit)
> 4. **Fable 5.2 Max** — Quantitative Mathematical Ceiling Bounds (exact CPU recomputation on frozen caches)
> 5. **Astra 6.2 Ultra** — Lead Architect Executive Verdict (1,403 serialized `k/n` ratios & 192 McNemar p-values verified)

---

## 1. Direct Verdict: Can the Literature Findings Push v1 to 85–95% Production-Grade?

**Unanimous 5/5 Verdict: NO for single-layer, zero-recipient-SGD v1 — YES as a realistic v2 roadmap target (FC first; EM requires structural relaxation + metric reform).**

### Why Single-Layer Zero-SGD v1 Cannot Hit 85–95% (at `<1%` Locality & `0%` HardNeg)
1. **The Native Oracle (`84.9–94.3%` Ripple FC) Is NOT a Production Ceiling — It Sacrifices All Selectivity:**
   - Across the 4 recipients on Cap-2 `BIO-RIPPLE`, `NATIVE_B_ORACLE` reaches **92.2% pooled Direct FC** (`177/192`) and **84.9–94.3% Ripple FC** ONLY by flipping **66/96 (`68.8%`) of in-domain hard negatives under FC** (including `6/6` aquatic mammals on all 4 recipients!) and breaking **157/722 (`21.7%`)** base-correct unrelated facts (`42/181`, `43/192`, `48/186`, `24/163`).
   - **Exact algebraic root cause (Fable 5.2 Max + Opus #1):** Although `platonic_organ_engine.py:32-34` documents `t1 = max(general, in-domain)` and `t2 = Q_{alpha_miss}`, lines `963–967` actually set `t1` from general text only and `t2 = t1 + std` (`alpha_neg` never enters `t1`). Under the KKT solve, other animals land at `34.65%` of the target margin (`5.1–6.7σ`), *above* full write (`4.4–5.2σ`). When re-solved as a class-selective single-row gate (`0–2%` other animals passing), **even the read-side TPR ceiling caps at `83–91%`**.
2. **The Pushforward → Oracle FC Gap Is Quantitatively Explained by the Write Bridge (Zero Free Parameters):**
   - The active concept-write bridge is fitted on only 1,200 sentences (300 concepts) in a 2,048-D space, yielding held-out cosines of **0.381 / 0.414 / 0.398 / 0.500** (`R² = -0.131 / -0.060 / -0.114 / +0.103`).
   - Multiplying held-out write cosine × push/native norm ratio predicts log-odds transfer efficiency **$\eta_{\text{pred}} = 0.35 / 0.34 / 0.34 / 0.44$**, matching **observed $\eta = 0.47 / 0.35 / 0.34 / 0.53$** with zero free parameters and the exact same recipient ranking (Fable §1.3).
   - Across all 24 papers in `docs/papers/`, **no method achieves $\eta \ge 0.90$ cross-family transfer at low FPR** (`bakker_2026` best calibrated low-FPR $\eta$ is `0.80` at **10% FPR**, 50× looser than our `0.2%`).
   - Realistic **v1.1** single-layer zero-SGD ceiling (bilinear twin-AND + implemented two-sided `t1/t2` + 10× larger concept-write corpus, validated on a fresh **Sealed Cap-3**): **~55–80% cross-family Ripple FC** at `≤1%` far-locality and near-zero HardNeg.
3. **85–95% Free-Generation Exact Match (EM) Is Blocked Under the Tested Single-Layer `v1` Construction and Current Scorer:**
   - Even **unedited base models** score only **72.9–85.4% EM** on invariant facts they know (`100%` FC) and **35.3–47.1% EM** naming a hard negative's *own* class under the `startswith` alias scorer.
   - Even `NATIVE_B_ORACLE` achieves **24/24 Mammal Direct FC → 0/24 Mammal Direct EM** on both SmolLM2 and Qwen-1.5B (`33.9%` pooled Direct EM), and single-site edits leave the original subject readable in other tokens/layers (`Plato's Key` 55% self-swap; our Table 1C required attention knockout to cross `86–92%`).

### Minimum Architectural Relaxations That Actually Cross 85–95% (v2 Roadmap)
| Path | Relaxation | Evidence | Crosses ≥85% FC? | Crosses ≥85% EM? |
|---|---|---|---|---|
| **v1.1 (Closed-Form, Single-Layer, Cap-3)** | Bilinear twin-AND (`--ablate-domain-up`, `87.5/85.4% EM, 0/175 broken` natively) + implemented two-sided `t1/t2` (general + in-domain conformal thresholds; *note: not Kramár et al. Eq. 12, which optimizes a single weighted-error threshold*) + 10× domain-covering write corpus + donor gain law | Fable §1.3 probit, Opus #2 §3.3 | Partial (**~60–80%** cross-family; 85%+ only on strongest recipient or native-installed G1) | No (`~34–50%` cross-family) |
| **v2-A (Frozen-Base Appended-Write Optimization)** | ≤25-step recipient-side gradient optimization of the appended write vector `v*` on frozen recipient (`ROME`/`MEMIT` potency, still append-only & bit-exact removable; not zero-training) | `meng_2022_rome` (100/96.4), `meng_2023_memit` (98.9/88.6) | **Yes** | **Yes** (direct & paraphrase, after EM scorer reform) |
| **v2-B (Pure Zero-SGD)** | Compiled subject-route attention-head organ + RoPE intertwiner $\mathcal J$ (reproducing our own Pillar 1 `86–92%` knockout jump in native weights, tested first on RoPE-nested `Qwen2.5-3B → Falcon3-3B`) + multi-site depth cocycle | Table 1C (`76.5→85.9`, `43.8→91.5`), Fable O1/O2, Opus #3 Arch A | **Likely** | **2/3–3/3 pairs** (untested as compiled weights) |

---

## 2. Executive Closeout Verdict for v1: `GO — FREEZE ENGINE & SHIP v1 NOW`

**Unanimous 5/5 Verdict:** **Freeze the v1 engine (`SHA-256: 22ef57c9d196e12f...`) and sealed results (`cd6c3c02a0881e04...`) immediately. Do NOT modify a single compiled float or re-run Cap-2 under a "sealed" label.**

Why:
- Sealed Cap-2 `BIO-RIPPLE` **genuinely transforms v1** from a single-capability demo into a strong, reproducible existence proof:
  - **8/8 compiled rule-organs** (2 rules × 4 recipients) beat both BASE and forced shuffled-read controls on unseen downstream attributes (`SmolLM2: 111/192 = 57.8% [49.5, 65.6]` vs BASE `12.5% [7.8, 17.7]`, species-level exact sign-test `p <= 4.8e-7`).
  - **0/722 base-correct unrelated facts broken** across all 4 solo recipients (`0/181, 0/192, 0/186, 0/163`) — already recorded inside `results_bio_ripple_sealed.json` (`loc=—/—` in Table 4B is purely a `build_repo_tables.py` key-lookup bug!).
  - **8/8 automatic abstentions** on shuffled-read controls across Cap-1 + Cap-2; `CROSS_ORGAN_SWAP = 0.0%` direct on 4/4.
  - **Bit-exact modular removal** (`max |Δlogit| = 0.0`) of *two working co-implanted capabilities* (`Cap-1 + Cap-2` on solo SmolLM2: `8192 → 8200 → 8196 → 8192`).
- Tuning the engine after inspecting Cap-2 would destroy v1's single-execution provenance without reaching 85–95% anyway. Any accuracy upgrade belongs in **v1.1 / v2** validated on a fresh, externally pre-registered **Sealed Cap-3**.

---

## 3. Executable v1 Closeout Checklist

### P0 — Must Do Before Tagging/Pushing `v1.0` (Zero GPU Compute)

1. **Provenance Lock (Touch NONE of These Files):**
   - Keep `experiments/platonic_organ_engine.py` & `platonicswap/platonic_organ_engine.py` byte-identical (`22ef57c9d196e12f2cd3c9ecc79ec0813803df43d5a6009f5422904948802306`), `experiments/bio_ripple_sealed.py` (`001d296b4c278ab8...`), `results/results_cap1_channel_pushforward.json` (`bcf2dc70997e22c8...`), and `results/results_bio_ripple_sealed.json` (`cd6c3c02a0881e04...`).
   - Record all pinned SHA-256 digests + socket cache manifest in a README **Provenance** box.

2. **Repair `experiments/build_repo_tables.py` (11 Concrete Defects B1–B11) & Regenerate `results/master_audited_results.md`:**
   - **B1:** Point default input/output at `results/` (not `experiments/`), removing hard-coded `/tmp/platos_key_lab/` paths.
   - **B2:** Fix the 16 `loc=—/—` cells in Table 4B by falling back to `newly_broken_all / n_orig_correct_all` (`0/181, 0/192, 0/186, 0/163` vs Oracle `42/181, 43/192, 48/186, 24/163`).
   - **B3:** Remove bare `|` inside cell strings in Table 4B-ii (currently 9 cells under 6 headers, breaking GitHub rendering!) and L23 foil; assert `len(row) == len(headers)` in `md_table`.
   - **B4:** Fix Stage A/B keys (`stage_A_multifamily`, `stage_B_multidonor_composition`) so Table 2B and Table 3A render cleanly instead of `inspect custom keys` / `[PENDING]`.
   - **B5–B6:** Match Granite's native ceiling per layer/mask/config and include `ridge_clamped_1_5` in FreeGeom.
   - **B7–B11:** Print **BOTH FC and EM** for Cap-2 HardNegs (`8.3–33.3%` FC vs `0–4.2%` EM; note Granite unaligned `ff = 0.641%`); mark `IN_CONTEXT` HardNeg as `n/a (unprefixed)` and rename to *in-context premise baseline*; note Cap-1 `Kyoto`/`Nagoya` 12.5% BASE floor; relabel L23 `W_down` foil (`not measured` ripple, non-canonical `λ=10`); add per-rule Cap-2, species CIs, eligible-subset hops (flagging `hop1_heat_lamp` base floor), and per-attribute Cap-1 scorecards.

3. **Fix Fresh-Clone Reproducibility (R1–R6) Without Changing Engine SHA:**
   - Fix `platonicswap/extract_and_baseline.py` so `DATASET*_PATH` resolves to `<repo>/data/` and `LAB_DIR` to `<repo>/experiments/` (currently breaks `bio_ripple_sealed.py --unit-test` and makes `platonic_organ_engine.py --unit-test` silently skip overlap checks!).
   - Copy 3 supporting upstream JSONs (`results_baseline.json`, `results_referee_controls.json`, `results_systems.json`) + `run_referee_controls.py` into the repo so `pilot0`–`pilot3` run cleanly.
   - Update `REPRODUCE.md` with explicit `huggingface-cli download` commands (all loaders use `local_files_only=True`), full per-recipient Pilot 3 & `--stages` commands, and an exact Python/PyTorch lockfile.

4. **Honesty Errata & Top-Level `README.md` Rewrite:**
   - Update docstring only on `experiments/pilot_organs_p0.py:900–921` (`target_bos_gate=-20` is unapplied in the affine arm; bias-free headline is unaffected).
   - Document `platonic_organ_engine.py` docstring discrepancies (`alpha_neg=0.02` vs code `0.03`; `t1/t2` general-text-only vs documented two-sided) in a README **Errata** section rather than editing the hashed engine file.
   - Retire overstated claims in `README.md`: replace `+24.49 pp vs MEMIT`, `41% → 83–92% over Plato's Key`, `knowledge-gap transplant`, `beats in-context ceiling`, `0.0% HardNeg` (EM-only), `p = 1e-26` item-level headline, `4 architecture families` (`4 pretraining families, 3 HF classes`), `zero forgetting` (`0/722 on an out-of-domain probe, 95% bound ≤1.8%/recipient`), `strictly pre-registered` (`specified in advance, single recorded execution`), and `selective exact unlearning` (`bit-exact modular removal`). Also fix README line 77 (`SLICE_OUT_CAP2` returned biology direct to `0%`, ripple to `12.0%/5.2%`) and note Granite's first-sub-token FC tokenization caveat (`" re"` vs `" m"`).

5. **Clean-Checkout CPU Verification Gate:**
   - Clone repo to a fresh directory with `/tmp/platos_key_lab` absent, verify both engine SHAs (`22ef57c9...`), run `--unit-test` on `bio_ripple_sealed.py` and `platonic_organ_engine.py`, run `python experiments/build_repo_tables.py`, confirm `0` matches for `loc=—/—|inspect custom keys|\[PENDING\]`, and verify every README number against the regenerated master.

### P1 — Optional Pre-Close Diagnostic (≤20 min MPS, Strictly Read-Only on Frozen Organs)
- **Per-item gate firing & 2×2 read/write hybrid decomposition** (driver script importing the *unmodified* frozen engine, writing a separate diagnostic JSON): log actual `z = w_c · x_B` on the 48 Cap-2 species & 24 HardNegs and cross native vs. transferred readers/writers. Closes the only remaining attribution unknown without touching v1's seal. *(Optional — can also be the very first step of v2 if you prefer a zero-GPU v1 closeout.)*

### P2 — Strictly Defer to `V2_ROADMAP.md` (Fresh Sealed Cap-3)
1. **v1.1 single-layer upgrades (validated on pre-registered Cap-3 with full-continuation FC, fixed EM parser, no `hop1_heat_lamp`, ≥48 entities/rule):** Bilinear twin-AND (`--ablate-domain-up`), implemented two-sided `t1/t2` conformal thresholds (*note: distinct from Kramár et al. Eq. 12's single weighted-error threshold*), 10× domain-covering write corpus, donor gain law, saturating BOS gate (`-20`), `ggml` 256-multiple padding, and Organ Manifest + Re-Conjugate-on-Update (`ROU`).
2. **v2 multi-layer / attention hidden-rule falsification:** RoPE torus-dissociation test (`Qwen2.5-3B → Falcon3-3B` vs `SmolLM2-1.7B`), organic arithmetic-carry neuron transplant, multi-layer cyclic digit rotation (Arch A + B), and few-shot frozen-recipient write-vector refinement (`ROME`/`MEMIT` hybrid) to cross 85–95% on both FC and EM.
