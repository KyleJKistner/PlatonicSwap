# PlatonicSwap v5.0 Stage 1 — Boundary-First Causal Organ Gates (`G0 → G1 → G2`)

**Status:** Completed on Colab NVIDIA L4 (`1,116.7 s`, `0.62 CUs`, `Active assignments: 0`)  
**Decision:** **`STOP_BEFORE_G3_NO_PASSING_ORGAN`** (Transfer Failed — Clean Stop before Stage 2 Native Compilation)  
**Execution/Compliance Review:** Pre-GPU `PASS / PASS` · Post-GPU `PASS / PASS` (Astra 6.2 Ultra + Gemini 4 Neon High — protocol compliance sign-off on a negative stop, NOT transfer success)  
**Script:** [`experiments/v5_0_boundary_organ_gates_g0_g2.py`](../experiments/v5_0_boundary_organ_gates_g0_g2.py) (`ed4d32c`)  
**Results:** [`results/results_v5_0_boundary_organ_gates_g0_g2.json`](../results/results_v5_0_boundary_organ_gates_g0_g2.json)  
**Constraints Honored:** Strict **ZERO SGD** (`strict_zero_sgd = true`), **ZERO task-target-fitted writers** (`zero_task_fitted_writers = true`), single-model GPU phases (`Phase 0: 1.5B`, `Phase 1: 7B`, `Phase 2: 1.5B`), post-unload VRAM `0.009 GiB < 1.0 GiB` across all three phases.

---

## 1. Executive Summary

Following the v4.4.1 private-lane falsifier (`KILL`), v5.0 tested the unanimous Astra 6.2 Ultra + Fable 5.2 Max **boundary-first strict 0-SGD organ architecture** on `Qwen/Qwen2.5-7B-Instruct → Qwen/Qwen2.5-1.5B-Instruct`:
- **Whole verbatim donor components first** (no neuron slicing prior to passing Gate G1),
- **Depth-1 direct-to-L27 output boundary** via the exact raw-coordinate shared-vocab Fisher plug $T_{\text{raw}} = (J_B^\top \bar F J_B + \varepsilon I)^{-1} J_B^\top \bar F J_A$ with $J_H = E_H \operatorname{diag}(\gamma_H)$ and median RMS scale $\gamma_{\text{scale}} = \operatorname{median}_t(\operatorname{rms}_B / \operatorname{rms}_A)$,
- **Demand-weighted supply-filtered input boundaries $(E_r, D_r)$** computed from closed-form zero-backprop Jacobians $G = \mathbb{E}[J^\top F J]$ and $55{,}197$ generic `pos > 0` calibration tokens (no noisy $P_\perp$ regression; testing mean-clamped, native minimum-variance near-constant read $a^*$, and homogeneous zero-intercept $P$).

**Bottom line:** Stage 1 cleanly triggered **`STOP_BEFORE_G3_NO_PASSING_ORGAN`**. None of the four candidates passed full `G0 ∧ G1 ∧ G2`. The run exposed three **task-independent structural walls** for strict 0-SGD Depth-1 organ compilation on Qwen2.5-7B → 1.5B, a crisp **oracle-vs-student causal separation** on 2-Hop late attention heads (`n = 400`), and two **important G0 prompt/scorer confounds** that must be recorded transparently alongside the negative verdict.

---

## 2. Telemetry & Hardware Hygiene

| Metric | Phase 0 (1.5B Screen) | Phase 1 (7B G0/G1) | Phase 2 (1.5B Cal/G2) |
|---|---:|---:|---:|
| Peak Allocated VRAM (GiB) | 3.212 | 17.387 | 3.919 |
| Peak Reserved VRAM (GiB) | 4.062 | 21.400 | 7.428 |
| Allocated After Unload (GiB) | 0.009 | 0.009 | 0.009 |

- **Total wall-clock:** `1,116.7 s` (`host_peak_rss = 9.301 GiB`, `V_shared = 151,665`).
- **Generic calibration corpus:** `55,197` fit / `4,299` tune / `4,376` report `pos > 0` tokens (`gamma_scale_fit = 0.8206`, `gamma_scale_rep = 0.8100`). Note: the `4,376` report positions are drawn from 40 templated held-out passages (`val_report`).

---

## 3. Task-Independent Structural Findings

### 3.1 Large-Scale Shared-Vocab Bridge Verification (`4,376` Held-Out `pos > 0` Tokens)

| Bridge (Donor 7B Final State → Recipient 1.5B Logits) | Top-1 Agreement w/ 7B | Mean KL (nat) | Median KL (nat) | G0 Bar (`>= 80%`) |
|---|---:|---:|---:|---|
| Stock 1.5B (no bridge) | 59.00% | — | — | Baseline |
| Post-norm `M_U` (`eps_rel = 1e-3`) | **64.83%** (2,837/4,376) | **0.8539** | 0.3484 | FAIL |
| Raw pre-norm `T_raw` (`eps_rel = 1e-3`) | **64.51%** | **1.9017** | 1.2921 | **FAIL** |

- **Key revision:** `v4.3.2`'s `92.5%` (`KL = 0.2547`) was measured on only 40 passage-final tokens. Across `4,376` held-out generic `pos > 0` positions, both `M_U` and `T_raw` improve top-1 agreement over Stock 1.5B by only `+5.5` to `+5.8 pp` and fail the preregistered `80%` G0 bridge threshold.

### 3.2 Demand-Weighted Linear Supply Wall (`1.5B → 7B`, Held-Out `val_report`)

| Ingress Site | All-Mode $\kappa_{\text{rep}}$ | Rank-256 $\kappa_{\text{rep}}$ | Median Mode $\rho_i^2$ | Worst Mode $\rho_i^2$ | $a^*$ Held-Out CV | Passes G2 Supply (`κ≥0.85, ρ²≥0.80`) |
|---|---:|---:|---:|---:|---:|---|
| `L27_MLP` (post-attn LN) | 0.76683 | 0.76544 | 0.2532 | −0.07303 | 2.0002% (miss by 0.0002 pp) | **FAIL** |
| `L26_MLP` (post-attn LN) | 0.64635 | 0.64067 | 0.3166 | −0.21072 | 2.7745% | **FAIL** |
| `L25_Attn` (input LN, 2-Hop head $G$) | 0.40940 | 0.40940 | 0.1218 | −0.16989 | n/a (exact Q/K/V bias) | **FAIL** |
| `L26_Attn` (input LN, 2-Hop head $G$) | 0.47051 | 0.47047 | 0.1259 | −0.14596 | n/a (exact Q/K/V bias) | **FAIL** |

- Every ingress site missed both the `0.85` aggregate and `0.80` minimum-mode criteria, with at least one retained mode predicting held-out donor activity worse than the fit mean ($\rho_i^2 < 0$).
- Astra's zero-intercept plug $P$ similarly failed on task states (`pooled R² = −2.72` to `−5.50`, `differences pooled R² = −0.21` to `−0.55`).

### 3.3 Native SwiGLU Superposition / Cancellation (`χ²`, 512 Generic `pos > 0` Tokens)

| Donor MLP (`18,944` neurons) | Native $\chi_{\text{nl}}$ | Native $\chi_{\text{nl}}^2$ | Clamped $\chi_{\text{nl}}^2$ | Importance-Sampling Error Bound at $k=256$ |
|---|---:|---:|---:|---:|
| `L27_MLP` (Airports code) | 59.60 | **3,552.55** | 1,526.79 | 13.87× (native) / 5.96× (clamped) |
| `L27_MLP` (2-Hop code) | 59.93 | **3,591.58** | 1,402.39 | 14.03× (native) / 5.47× (clamped) |
| `L26_MLP` (Airports code) | 89.51 | **8,012.03** | 4,005.11 | 31.29× (native) / 15.64× (clamped) |

- This quantifies the random-component-sampling upper error bound `(χ² - 1) / k` under naive neuron sampling (it is an upper bound on random sampling, not a universal impossibility proof against deterministic, cancellation-aware, or low-rank compression). The width-compression ladder was skipped because whole-MLP G1 did not pass.

---

## 4. Task Gates (`G0 → G1 → G2`) & Honest G0 Confound Audit

### 4.1 In-Prompt 2-Hop (`LateHeads_5h` in `L25–L26`, `n_eval = 400` from 200 tables)

- **G0 Headroom Confound (FAIL):** With raw non-chat 0-shot prompts, exact first-word accuracy was `7.00%` (7B) vs `12.75%` (1.5B) on 4-row and **`6.25%` (7B) vs `9.75%` (1.5B)** (`−3.50 pp`) on pilot-selected 6-row. Neither model mastered bare-word 0-shot formatting, so this run does **not** test a donor-mastered behavioral skill.
- **Controlled Causal Separation on the Reproducible Margin Signal:** Despite low bare-word EM, Donor 7B exhibited a clear `+1.2067`-nat mean answer-vs-foil margin, with **5 late heads** (`[26,17], [26,3], [25,9], [26,5], [26,2]`) carrying **`52.64%`** (`+0.6352` nat; `+0.7648` nat on the 361 items 1.5B missed):
  - **G1 Donor Fidelity (`r = 256`, `pos = 0` preserved): PASS** — $Q_{\text{causal}} = 0.9065$, $R^2_c = 0.8644$, 20-draw `SHUF-π` mean $R^2 = -0.3276$ / $Q = -0.0021$. (Full G1 failed on behavioral generation recovery: `25 → 11 → 20` hits = `64.29%` recovery and `3.50 pp < 5 pp` ablation drop.)
  - **G2 Donor Replay on Recipient-Suppliable $D_{256}$: PASS** — $Q_{\text{causal}} = 0.8736$, $R^2_c = 0.8238$.
  - **Test 2A (True-Donor Outlet Oracle via $T_{\text{raw}}$ at Live 1.5B L27):** Recipient first-token accuracy rose **`9.25% → 20.50%` (`+11.25 pp`)**, `WIR = +42.76%`, $Q_{\text{outlet}} = 0.6587$, $R^2_c = 0.5842$.
  - **Test 2B (Student-Supplied 1.5B Codes into the Exact Same 5 Heads):** Collapsed completely to **`WIR = −0.33%`**, $Q = 0.0075$, $R^2_c = -0.1574$, first-token lift `−1.25 pp`, live generation EM `9.75% → 8.25%` (`−1.50 pp`, vs `SHUF-π −2.00 pp`), while `pos = 0` sink attention mass more than doubled (`0.3034 → 0.6730`). **Verdict: `KILL`.**
  - (The 2-Hop `L27_MLP` candidate returned `INCONCLUSIVE` because `mean_c_true_wrong = −0.0417 < 0.25`, validating the signed denominator guard.)

### 4.2 OurAirports (`L27_MLP` & `L26_MLP`)

- **G0 Headroom Confound (FAIL):** On raw non-chat `p0`, strict first-field scoring (`check_iata_strict_field`) accepted only **`18/1,000`** donor completions (`n_disc = 6`, `n_eval = 12`, **`n_eval_p0_blindspot = 10 << 96`**). On the 994 non-discovery airports on `p1`, 7B vs 1.5B scored `20.42% vs 5.63%` (`+14.79 pp`, just under the `15 pp` bar). A 1-hit batch-composition discrepancy also appeared on the 12 `p1` eval items (`9/12` in G0 vs `8/12` in G1).
- **Underpowered Behavioral Signal (Consistent Direction):** Because `n_eval = 12 < 20`, causal $Q$ was properly guarded as `null`. On those 12 items, `L27_MLP` showed `-58.33 pp` ablation drop but failed donor G1 restoration (`rec_gen = 14.29%`, `R2_c = −0.589`), and G2 reproduced the same oracle-vs-student cliff (Test 2A Oracle `WIR = +76.13%`, `+16.67 pp` tok-0 vs Student `WIR = −2.60%` clamped / `−1.56%` `a*`, `−10.0 pp` live blindspot EM; **`KILL`**). Twelve items cannot support a general behavioral claim, and are reported strictly as underpowered secondary telemetry.

---

## 5. Conclusion & Next-Step Guidance

1. **Why Stage 2 (Gate G3) was NOT run:** The `STOP_BEFORE_G3_NO_PASSING_ORGAN` stop was signed off for execution compliance (`PASS / PASS`). Under the tested Depth-1 linear boundary family on `Qwen2.5-7B → Qwen2.5-1.5B`, **both** the generic L27 output bridge (`T_raw = 64.51% < 80%`) **and** every ingress site's held-out demand-weighted linear supply (`κ_rep = 0.41–0.77 < 0.85`, worst mode `ρ² <= −0.07 < 0.80`) failed our project-declared G0/G2 thresholds (which justify stopping this tested implementation, without constituting a universal impossibility theorem). Note: `WIR` throughout is a ratio of average answer-minus-foil margin changes on originally wrong items, not the fraction of mistakes fixed.
2. **What would be required for any future behavioral follow-up:** A prompt-format study using identical 1-shot/2-shot or chat-formatted prompts (pilot-selected, saving per-item completions) would be needed to certify G0 behavioral mastery on both tasks — and even then, overcoming the `64.5%` output bridge and `0.41–0.77` linear supply walls points beyond strict 0-SGD linear boundary plugs (e.g., toward tiny-budget native adaptation / Tier 2 polish or deeper architectural changes).
