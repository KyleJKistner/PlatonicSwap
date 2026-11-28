# Pythia-160m Matched-Width v1: Zero-SGD, Zero-Hook Standalone Circuit Transplant

> **Scope & Honest Claim:** Using task-guided circuit selection (strictly on discovery splits) and **100% closed-form, zero-SGD, zero-Adam interface fits on generic code** (zero task examples, zero shared files between fit and retention, zero tokenized task-ID occurrences), we compiled 24 frozen donor attention heads from `EleutherAI/pythia-160m` into a standalone 15-layer `GPTNeoXForCausalLM` checkpoint of `EleutherAI/pythia-160m-seed2@step2000` while keeping all **148/148** original recipient parameter tensors byte-identical. Across three synthetic `A B ... A -> B` binding formats (`n = 256` each on strictly untouched vocabulary `pool[1000:1500]`), the offline-reloaded, zero-hook checkpoint improved full-vocabulary top-1 accuracy and within-table value selection over `STOCK`, 5 independent disjoint `RAND` head grafts, KL-matched `RAND`, and edge-only ablation (all 12 Holm-Bonferroni contrasts `p <= 3.82e-4`). Edge-only cuts and two-foil key-rerouting swaps proved that the donated `Block_Prev -> Block_Ind` connection contributed causally to binding. **Explicit costs & limitations:** +21.26M stored parameters (+13.10%; +4.73M / +2.91% non-zero), held-out code perplexity increased +30.7% (`KL = 0.2414`), and this augments an early-emerging induction circuit on one matched-width cross-seed pair (not yet a zero-induction checkpoint or cross-width pair).

---

## 1. Exact Provenance & SHA-Pinned Artifacts

| Field | Value |
|---|---|
| **Donor** | `EleutherAI/pythia-160m` @ commit `50f5173d932e8e61f858120bcb800b97af589f46` (`12L, d=768, 12H`) |
| **Recipient** | `EleutherAI/pythia-160m-seed2` (`step2000`) @ commit `b9e11cd228f272561b1bdbda208934ce19f0a3ed` (`12L, d=768, 12H`) |
| **Compiled Architecture** | Vanilla 15-layer `GPTNeoXForCausalLM` (`L0..5` frozen + 3 inserted standard `GPTNeoXLayer` blocks with zeroed MLPs + `L6..11` frozen) |
| **Donor Circuit (Nominated on `disc_rank`)** | `Block_Prev` (`L6`, 8 heads `P8`): `(3,2),(3,0),(3,1),(1,2),(2,8),(3,4),(1,7),(1,0)`; `Block_Ind_Early` (`L7`, 7 heads): `(5,0),(4,6),(5,7),(4,11),(4,10),(5,10),(7,8)`; `Block_Ind_Late` (`L8`, 9 heads): `(8,2),(9,9),(8,11),(9,8),(10,1),(9,2),(9,6),(9,7),(8,10)` |
| **Closed-Form `Q_T = 0` Compilation** | Isotropic `LayerNorm` (`w=1, b=0`) + orthogonal Procrustes `R_1, R_h in O(768)` (`||R^T R - I||_max <= 1.19e-6`) with exact algebraic unscrambling (`M_mid = (1/s_h) R_h^T`, `M_in = diag(ln_A.w) R_h` so `R_h R_h^T = I`) + `Q5/Q8` ridge + generic-code mean-centering in `dense.bias`. Locked operating point `(ap=1.0, ae=0.8, al=0.7)` chosen on `disc_val`. |
| **Generic Corpus Hygiene** | 128 fit passages (33 stdlib `.py` files) vs 96 held-out KL/PPL passages (26 disjoint `.py` files): **0 shared files, 0 whole-word matches, 0 tokenized task-ID occurrences** across `pool[0:1500]`. |
| **Offline Reload & Integrity** | Loaded via `AutoModelForCausalLM.from_pretrained(..., local_files_only=True)` with **zero hooks**; **148/148** original tensors SHA-256 byte-identical (`0` mismatches); deleting layers `6..8` restores stock logits with `max_abs_diff = 0.0`. |
| **Script & Audited JSON** | [pythia_v1_standalone_transplant.py](../experiments/pythia_v1_standalone_transplant.py) · [results_pythia_v1_standalone.json](../results/results_pythia_v1_standalone.json) |

---

## 2. Entry Gates (Discovery Split `disc_val`, `n = 256`)

- **Gate 1 (Donor Lead):** On `disc_val rs6` (`n = 256`), Donor `145/256` (`56.6%`) vs `STOCK` `89/256` (`34.8%`) = **`+21.9 pp`** (`>= +20 pp`).
- **Gate 2 (Strict Family-Matched Key-Swap Isolation):** Keeping `L0` clean and corrupting **all non-circuit `L1..L11` attention heads + all `L1..L5` MLPs** with each family's own key-swapped activations (denominator = intact `m_clean - m_ks`, no clean batch means):
  - **Primary `rs6` (`n = 256`):** **`0.727` (`95% CI [0.704, 0.750]`)** (`>= 0.70` even at lower 95% bound), `SEVER_PREV = 0.028`, `SEVER_IND = 0.000`, `RAND = -0.005`.
  - **Secondary `nl6` (`n = 256`):** `0.684` (`95% CI [0.668, 0.702]`), `SEVER_PREV = 0.010`, `SEVER_IND = 0.001`, `RAND = -0.003`.

---

## 3. Zero-Retry Sealed Evaluation (`pool[1000:1500]`, `n = 256` per Format)

Evaluated **once** in a zero-retry run on 500 strictly untouched single-token words (`pool[1000:1500]`, `0` overlap with `pool[0:1000]`), independent random bindings per format:

| Format | Donor | STOCK | **REAL (15L)** | RAND₅ Mean | RAND KL-Matched | Edge-Only Cut | Fixed `+2` Copy (Lexical) | Gap Rec vs STOCK / RAND₅ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`rs6` (primary, no `\n`)** | 50.0% (128) | 33.6% (86) | **42.6% (109)** | 29.4% | 21.9% (56) | 34.0% (87) | 57.0% (146) | **+54.8%** / **+64.0%** |
| **`nl6` (secondary)** | 76.6% (196) | 44.5% (114) | **65.6% (168)** | 36.2% | 25.8% (66) | 53.1% (136) | 71.1% (182) | **+65.8%** / **+72.9%** |
| **`rs12` (secondary, no `\n`)** | 46.1% (118) | 29.3% (75) | **36.3% (93)** | 22.7% | 17.6% (45) | 29.3% (75) | 47.3% (121) | **+41.9%** / **+58.2%** |

### Statistical Significance & Causal Specificity (`pool[1000:1500]`)

| Format | Holm-p vs STOCK | Holm-p vs RAND₅ | Holm-p vs RAND-KL | Holm-p vs Edge-Only Cut | Key-Swap Foil-vs-Target (`t`) | Key-Swap Two-Foil (`t`) | Within-Table Value Acc (`ST → REAL`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`rs6`** | `3.82e-04` | `7.48e-08` | `2.07e-13` | `3.39e-04` | `+1.145` | `+0.373` | `85.2% → 88.3%` |
| **`nl6`** | `7.20e-12` | `6.97e-27` | `1.24e-28` | `2.79e-09` | `+1.210` | `+0.490` | `94.5% → 96.9%` |
| **`rs12`** | `3.82e-04` | `8.51e-10` | `7.11e-14` | `3.05e-05` | `+0.900` | `+0.230` | `80.5% → 81.6%` |

- **Absent-Key Negative Control (`rs6_absent`, query key never appears in context):** `REAL` matched `STOCK` nearly identically (`in_vals = 0.8%`, `in_ctx = 2.7%` vs `STOCK 0.4% / 1.6%`), whereas the fixed `+2` lexical copy-bonus jumped to `3.1% / 14.8%` (**5.5×** spurious in-context copying) and failed to improve within-table binding specificity.
- **KL-Matched `RAND` Control:** Scaling a disjoint random graft to match `REAL`'s exact held-out KL (`0.24`) crashed accuracy well below `STOCK` (`21.9% / 25.8% / 17.6%`), confirming the gain is circuit-specific, not perturbation calibration.
- **Held-Out Retention Cost (7,360 doc- & token-ID-disjoint Python stdlib tokens):** `KL(STOCK || REAL)` `mean = 0.2414`, `p50 = 0.1712`, `p95 = 0.7269`, `p99 = 1.2994`, `max = 3.1603`; `PPL = 26.67` vs `STOCK 20.42` (`+30.7%`).
- **Replication note (prior `pool[450:950]` run independently audited by Astra 6.2 Ultra):** `rs6` `96 → 115/256` (`+46.3%` vs `STOCK`, `+58.3%` vs `RAND₅`), `nl6` `100 → 171/256` (`+87.7% / +88.7%`), `rs12` `58 → 82/256` (`+54.5% / +62.7%`), all 12 Holm-p `<= 1.32e-3`, `KL = 0.2261`, `PPL +29.9%`.

---

## 4. Known Limitations & Post-v1 Follow-Up

1. **High Retention Cost (`KL = 0.24`, `PPL +30.7%`):** Follow-up analysis matching v1's exact `Rh (L5->L3)` alignment (`docs/PYTHIA_V2_RETENTION_AND_LIMITS.md`, `experiments/pythia_v1_1_retention_and_limits.py`, discovery corpus baseline `KL = 0.2094`) showed that directly removing `p` at the suffix entrance leaves `85.5%` of generic-code KL (`0.2094 -> 0.1790`), while an exact `GELU(z) - GELU(-z) = z` parallel-MLP subtractor (`1.15×` alone) combined with stronger ridge shrinkage (`lam = 0.10`) cuts generic-code KL `2.70×` (`0.2094 -> 0.0774`, `PPL 1.108×`, trading off ~36% of the `rs6` gain above `STOCK` and still above the aspirational `<= 0.05 / 1.05×` target).
2. **Augmenting vs. Installing from Zero (`RECIPIENT-SELF-GRAFT` Catch):** `step2000` already exhibits high within-table selection (`80–95%`). Our overlapping Iso-KL discovery audit (`docs/PYTHIA_V2_RETENTION_AND_LIMITS.md`, `n = 256`) showed that duplicating 24 distinct native heads (`RECIPIENT-SELF-GRAFT`) beats the cross-seed donor on `rs6` and `nl6` at matched KL (`rs6 = 0.6016, nl6 = 0.7227` at `KL = 0.0753` vs. donor `0.5000 / 0.6289` at `KL = 0.0774`). In discovery scratch diagnostics, cross-width `pythia-410m` induction failed Gate 2 (`0.185`) and locked-gain Gate 3 (`37.3%`), and on low-skill `step3000` IOI only the hooked donor oracle strongly beat `SELF-GRAFT` (`89.1% / 62.5%` vs `33.3% / 0.0%`), while tested `POST-L11` configurations failed an exact Donor-attention Gate-0 sweep (`fc2 = 0.417, ΔSELF = +0.083` at `KL = 0.091`). See [`PYTHIA_V2_RETENTION_AND_LIMITS.md`](PYTHIA_V2_RETENTION_AND_LIMITS.md) for full details.
