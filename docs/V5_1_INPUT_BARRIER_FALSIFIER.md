# PlatonicSwap v5.1 — 100% Zero-SGD Input-Barrier Diagnostic & Falsifier

**Date:** 2026-11-27  
**Script:** `experiments/v5_1_input_barrier_falsifier.py` (commit `41f8f0a`)  
**Canonical Results JSON:** `results/results_v5_1_input_barrier_falsifier.json` (`219,365` bytes; `0` NaN/Inf)  
**Hardware / Runtime:** Colab `NVIDIA A100-SXM4-40GB` | Wall time: `3248.9s` (`54.1 min`) | Host peak RSS: `36.668 GiB`  
**GPU VRAM Hygiene (Single-Model Phases):** Phase 0 peak `3.346 GiB` (post-unload `0.0079 GiB`), Phase 1 peak `16.138 GiB` (post-unload `0.0079 GiB`), Phase 2 peak `13.779 GiB` (post-unload `0.0079 GiB`)  
**Optimization / Fit Scope:** `zero_sgd_strict = true`, `zero_task_fitted_writers = true` *(Note, Audit F06: zero SGD and zero full-model backpropagation; attention calibration uses deterministic `L-BFGS-B` with `jac=True` on generic data, folded into Q/K matrices.)*  
**Execution/Compliance Review:** Pre-GPU `PASS / PASS`; Post-GPU `PASS / PASS` (Gemini 4 Neon High + Astra 6.2 Ultra — protocol compliance on a negative run, NOT transfer success)  
**Final Decision:** **`STOP_BEFORE_G3`** (Autonomous student-fed transfer **FAILED**; do **not** launch Stage-2 native G3 compilation under the late-layer student readout recipe.)

---

## 1. Executive Summary & Regime Verdicts

`v5.1` tested whether the `v5.0` input barrier (`Q_in ~ 0.01`, `R2_c < 0`) between Donor **`Qwen/Qwen2.5-7B-Instruct`** and Recipient **`Qwen/Qwen2.5-1.5B-Instruct`** is caused by:
- **H1 (BOS / RoPE Calibration Failure):** uncalibrated position-0 sink mass or RoPE shrinkage in `Q/K`,
- **H2 (Late-Layer Representation Deficit):** missing donor `Q/K/V` features in the recipient's late residual stream, or
- **H3 (Upstream Nonlinear Computation Deficit):** intermediate donor circuits (`L20_MLP`, `L22–L23` heads) whose outputs are never computed in `1.5B`.

| Regime / Gate | Outcome | Headline Metric |
|---|---|---|
| **G0-Clean (1-Hop, `chat_2shot`, r8, `n=400`)** | **PASS (`15 pp` & `20 pp`)** | Donor `97.75%` (`391/400`, whole `97.75%`) vs Recipient `66.25%` (`265/400`, whole `66.25%`) — gap **`+31.50 pp`** |
| **G0-Clean (2-Hop, `2shot_fixed`, r4, `n=400`)** | **PASS (`15 pp` & `20 pp`)** | Donor `76.00%` (`304/400`, whole `0.00%`*) vs Recipient `11.75%` (`47/400`, whole `0.00%`*) — gap **`+64.25 pp`** |
| **1-Hop Policy Gate (Best Student: `Arm_2a_multitap_L0`)** | **FAIL** | Live EM lift `-0.75 pp` (`262/400` vs `265` none, `267` worst `SHUF-π`), `Q_outlet = 0.0206`, `R2_c = -0.1638`, `WIR = -0.0119` |
| **2-Hop Policy Gate (Best Student: `Arm_2a_multitap_L0`)** | **FAIL** | Live EM lift `+1.00 pp` (`51/400` vs `47` none, tied `51` worst `SHUF-π`), `Q_outlet = 0.0204`, `R2_c = -0.0101`, `WIR = 0.0129` |
| **2-Hop Binding & Null Gate (Best Student: `Arm_2a_multitap_L0`, `n=200`)** | **FAIL** | Live EM lift `+1.00 pp` (`21/200` vs `19` none, `p = 0.131`), `Q_outlet = 0.0089`, `R2_c = -0.3525`, `WIR = -0.0027`, null leakage `0.2789` |
| **Arm 3c: Live Native `L0` Tied-Embedding Ingress (`K=64`, `alpha=2^-6`)** | **PASS** | `kappa_L0_post_ln0 = 0.999988`, `cosine = 0.999997`, `generic_kl = 0.00117`, `tied_preserved = true`, `final_norm_suffix_zero = true`, **all 3 bit-exact rollback checks `true`** |

*\*Both first-word EM and strict whole-completion EM are reported. On `2hop` (`2shot_fixed`), the raw few-shot prompt has no chat EOS delimiter and continues `\n\nRule:` after the target answer within `max_new_tokens=4` (confirmed in pilot samples such as `" pure\n\nRule:"`), producing `0.00%` whole-completion EM despite `76.00%` first-word EM. On `1hop` (`chat_2shot`), donor whole-completion EM is `97.75%` (and `67.19%` on 2-Hop `chat_2shot` pilot).*

### Bottom-Line Diagnosis

1. **Autonomous student-fed late-layer readout FAILED (`STOP_BEFORE_G3`).** Across 1-Hop, 2-Hop Policy, and 2-Hop Binding, every tested late-layer student arm (`Arm 0`, `Arm 1a`, `Arm 1b`, `Arm 2a`, `Arm 2b`, `Arm 2c`) failed the behavioral gates (`Q_outlet <= 0.034`, `WIR <= 0.022`; selected `Arm_2a_multitap_L0` scored `51/400 = 12.75%` on 2-Hop Policy, tying the strongest shuffled control `51/400`, and dropped 1-Hop from `265` to `262/400`).
2. **Within the tested late-layer adapter family, failure occurs in BOTH content routing (`Q, K`) and value supply (`V`) — NOT BOS calibration alone.** Restoring exact donor BOS sink mass alone (`S1_donor_sink_only`) yields virtually zero recovery (`WIR = -0.0132 / 0.0199 / 0.0012`). Even supplying exact donor attention weights (`S3_donor_attn_student_V = QA_KA_VB`) recovers only `~25%` of oracle `WIR` (`0.1296` vs `0.5222` on 1-Hop; `0.0880` vs `0.3344` on 2-Hop Policy; `0.0705` vs `0.2793` on 2-Hop Binding). *(Metric note, Audit F22: `c_true` is a direct-write final-residual subtraction surrogate, not full downstream ablation; `WIR` is a ratio of average wrong-item margin changes, not the fraction of wrong items flipped — e.g., two-hop oracle `WIR = 0.3344` corresponds to `25.21%` formerly-wrong first-token accuracy.)*
3. **Three bounded diagnostic findings (NOT autonomous organ transfer):**
   - **Donor-assisted outlet oracle (`outlet_oracle_2A`, donor computation required):** Injecting cached outputs from actual Donor 7B forward passes on the eval prompts for 5 heads (`L26H17, L26H26, L27H24, L26H3, L25H9`) via `T_raw` lifts `1.5B` live greedy EM by **`+20.00 pp`** (`11.75% -> 31.75%`, `127/400` vs `47/400`, `Q_outlet = 0.5056`, `R2_c = 0.3978`, `WIR = 0.3344`), and by **`+14.00 pp`** on 2-Hop Binding (`9.50% -> 23.50%`). This validates that the recipient can exploit donor-computed outlet signals; it is **not** an installed autonomous reasoning organ.
   - **Arm 3c — Copied-coordinate `L0` tied-embedding storage/readback check:** Installing projected donor embedding coordinates into `K = 64` added student embedding columns at `alpha = 2^-6` and reading them back immediately after `LN_0` passes at `kappa = 0.999988`, `generic_kl = 0.00117`, with bit-exact rollback. No downstream attention/MLP organ reads these channels yet.
   - **Arm 3a + Arm 3b — Last-token donor mediation inside intact 7B (untested for autonomous extraction/transfer):** Replacing ONLY the last token's activation vector in `L20_MLP` with generic mean activations drops 7B's 2-Hop policy margin (`4.3557 -> 1.5342`, while ablated 7B still answers `57.00%` correctly). Retaining `k = 16` (`0.08%`) or `k = 64` (`0.34%`) `disc`-ranked neurons at that last position inside the intact full donor recovers `67.55%` / `83.93%` of that margin drop (`75.75%` / `78.75%` `tok0_acc`), with negative binding recovery at `k=16` (`-13.82%`), `23.53%` at `k=64`, and no matched random-neuron control.

---

## 2. Generic Calibration Corpus & Vocab-Anchored Fisher Bridge (`T_raw`)

- **Corpus Independence & Diversity:** Built prior to any task dataset from strictly disjoint splits:
  - `fit`: **`60,198`** `pos > 0` tokens (`30,099` WikiText prose, `18,034` Python stdlib code, `12,065` structured schemas); `6,758` unique token IDs; `9.3052` bits entropy; `has_real_wikitext = true`.
  - `tune` & `report`: **`4,318`** `pos > 0` tokens each (`2,159` prose, `1,270` code, `889` structured); disjoint document pools (`wiki [170,18,15]`, `code [180,30,30]`, `structured [61,15,16]`).
- **Frozen Fisher Bridge (`T_{raw, eta}`):** Selected strictly on generic `val_tune` (`frozen_eta = 0.0001`, `gamma_scale = 0.82802`):
  - Held-out `val_report`: stock 1.5B `top1 = 71.97%` (`KL = 0.5786`); `M_U` `top1 = 66.50%` (`KL = 0.9491`); `T_raw` (donor RMS) `KL = 0.9407`; `T_raw` (renorm) `top1 = 66.94%` (`KL = 1.9457`).
- **28-Layer Supply Profile `kappa(l)` (`1.5B -> 7B` on generic `report`):**
  - Peaks early at **`L1 = 0.6795`** (`L0 = 0.5365`, `L2 = 0.5973`, `L3 = 0.5879`), drops sharply through mid-layers (**`L9 = 0.3562`**, **`L13 = 0.3565`**, **`L18 = 0.3569`**), and recovers only to **`0.42–0.45`** at late readout layers (`L25 = 0.4388`, `L26 = 0.4219`, `L27 = 0.4361`).
- **Incremental Multi-Tap Ladder (`L26H17`, held-out `report`):**
  - `L_org_only` (`1536-D`): `q_r2 = 0.5469`, `v_r2 = 0.3566`.
  - `+L0_exact_donor_emb` (`1600-D`): `q_r2 = 0.5485` (`+0.0016`), **`v_r2 = 0.4155` (`+0.0589`)** — the single `64-D` `L0` donor embedding lift adds more `v_r2` than all four mid-layer taps (`L23, L19, L14, L8`) combined (`+0.0290`).
  - Full 5-tap (`+L0+L23+L19+L14+L8`, `2112-D` using 128-D PCA taps): `q_r2 = 0.5683` (`+0.0214` total), `v_r2 = 0.4445` (`+0.0879` total). Within this tested linear multi-tap 128-D PCA estimator, over `43%` of `Q` variance and `55%` of `V` variance remain unpredicted (this diagnoses a linear-probe gap, not an information-theoretic proof of absence under arbitrary nonlinear readouts).

---

## 3. Main Arm Comparison & Controls (1-Hop, 2-Hop Policy, 2-Hop Binding)

### 3.1 Task 1 — 1-Hop Positive Control (`chat_2shot`, r8, `n = 400`, 200 tables)

Primary heads (`disc`): `L26H17, L26H26, L27H24, L26H1, L25H9`. Donor `97.75%` (`391/400`), clean Recipient `66.25%` (`265/400`), `n_wrong = 135`. Primary qualifying arm selected by unrounded generic `val_tune_attn_kl`: **`Arm_2a_multitap_L0`** (`0.35769`).

| Arm / Control | Live EM (hits/400) | Live Lift vs None | `Q_outlet` | `R2_c` | `WIR` | `pos0` Sink (Donor `0.000`) |
|---|---:|---:|---:|---:|---:|---:|
| `outlet_oracle_2A` (True Donor `Q,K,V`) | `71.50%` (`286`) | `+5.25 pp` | `0.6802` | `0.4785` | `0.5222` | `0.0000` |
| `Arm_0_v50_baseline_diverse` | `66.25%` (`265`) | `+0.00 pp` | `0.0053` | `-0.1636` | `-0.0110` | `0.6735` |
| `Arm_1a_convex_m63` | `66.25%` (`265`) | `+0.00 pp` | `0.0107` | `-0.1682` | `-0.0133` | `0.5871` |
| `Arm_1b_rope_beta0` | `66.25%` (`265`) | `+0.00 pp` | `0.0337` | `-0.1367` | `-0.0306` | `0.5784` |
| **`Arm_2a_multitap_L0` (Selected)** | **`65.50%` (`262`)** | **`-0.75 pp`** | **`0.0206`** | **`-0.1638`** | **`-0.0119`** | **`0.4903`** |
| `Arm_2b_multitap_quad` (`0/15` quad active) | `65.50%` (`262`) | `-0.75 pp` | `0.0206` | `-0.1638` | `-0.0119` | `0.4903` |
| `Arm_2c_S_real_diagnostic` (diff. heads) | `65.50%` (`262`) | `-0.75 pp` | `null` | `null` | `null` | `0.4456` |
| `SHUF_pi_20draw` (top-3 worst live gen) | `66.75%` (`267`) | `+0.50 pp` | — | — | `max 0.0162` | — |
| `SHUF_layer_matched` (`L24–L26`) | `66.25%` (`265`) | `+0.00 pp` | `0.0000` | `-0.1784` | `-0.0013` | `0.7543` |

Bootstrap 95% CIs (`200` table clusters): vs None `[-1.75 pp, 0.00 pp]` (`p = 1.0`), vs `SHUF-π` `[-2.25 pp, -0.25 pp]`, vs `SHUF-layer` `[-1.75 pp, 0.00 pp]`.

### 3.2 Task 2A — 2-Hop Policy (`2shot_fixed`, r4, `n = 400`, 200 tables)

Primary heads (`disc`): `L26H17, L26H26, L27H24, L26H3, L25H9`. Donor first-word EM `76.00%` (`304/400`, whole `0.00%`), clean Recipient `11.75%` (`47/400`, whole `0.00%`), `n_wrong = 353`. Selected qualifying arm: **`Arm_2a_multitap_L0`** (`val_tune_attn_kl = 0.33923`).

| Arm / Control | Live EM (hits/400) | Live Lift vs None | `Q_outlet` | `R2_c` | `WIR` | `pos0` Sink (Donor `0.1429`) |
|---|---:|---:|---:|---:|---:|---:|
| **`outlet_oracle_2A` (True Donor `Q,K,V`)** | **`31.75%` (`127`)** | **`+20.00 pp`** | **`0.5056`** | **`0.3978`** | **`0.3344`** | **`0.1429`** |
| `Arm_0_v50_baseline_diverse` | `12.00%` (`48`) | `+0.25 pp` | `0.0011` | `-0.0132` | `-0.0009` | `0.6078` |
| `Arm_1a_convex_m63` | `12.00%` (`48`) | `+0.25 pp` | `0.0037` | `-0.0153` | `0.0012` | `0.4481` |
| `Arm_1b_rope_beta0` | `14.25%` (`57`) | `+2.50 pp` | `0.0287` | `0.0226` | `0.0218` | `0.3455` |
| **`Arm_2a_multitap_L0` (Selected)** | **`12.75%` (`51`)** | **`+1.00 pp`** | **`0.0204`** | **`-0.0101`** | **`0.0129`** | **`0.3753`** |
| `Arm_2b_multitap_quad` (`0/15` quad active) | `12.75%` (`51`) | `+1.00 pp` | `0.0204` | `-0.0101` | `0.0129` | `0.3753` |
| `Arm_2c_S_real_diagnostic` (diff. heads) | `12.00%` (`48`) | `+0.25 pp` | `null` | `null` | `null` | `0.3218` |
| `SHUF_pi_20draw` (top-3 worst live gen) | `12.75%` (`51`) | `+1.00 pp` | — | — | `max 0.0032` | — |
| `SHUF_layer_matched` (`L24–L26`) | `12.00%` (`48`) | `+0.25 pp` | `-0.0004` | `-0.0243` | `-0.0023` | `0.6927` |

Bootstrap 95% CIs (`200` clusters): vs None `[+0.25 pp, +2.00 pp]` (`p = 0.018`), vs `SHUF-π` `[-1.00 pp, +1.00 pp]` (`p = 0.6005`), vs `SHUF-layer` `[0.00 pp, +1.75 pp]` (`p = 0.0545`).

### 3.3 Task 2B — 2-Hop Binding & Null-Swap (`n = 200` tables)

Donor Binding first-word EM `74.00%` (`148/200`), Donor Null `77.50%`, Donor own null/binding swap ratio `0.0915` (`0.6934 / 7.5788` nats). Clean Recipient Binding `9.50%` (`19/200`, `n_wrong = 181`).

| Arm / Control | Live EM (hits/200) | Live Lift | `binding_Q` | `binding_R2_c` | `WIR` | Null/Bind Leakage |
|---|---:|---:|---:|---:|---:|---:|
| **`outlet_oracle_2A` (Policy Heads)** | **`23.50%` (`47`)** | **`+14.00 pp`** | **`0.4934`** | **`0.1769`** | **`0.2793`** | **`0.0974`** |
| **`Arm_2a_multitap_L0` (Selected)** | **`10.50%` (`21`)** | **`+1.00 pp`** | **`0.0089`** | **`-0.3525`** | **`-0.0027`** | **`0.2789`** |
| `SHUF_pi_20draw_binding` (worst) | `9.50%` (`19`) | `+0.00 pp` | — | — | `max 0.0061` | — |
| `SHUF_layer` | `9.50%` (`19`) | `+0.00 pp` | `-0.0002` | `-0.3686` | `0.0012` | `0.7038` |

All three binding bootstrap 95% CIs have lower bound `0.00 pp` (`[0.00 pp, +2.50 pp]`, `p = 0.131`).

---

## 4. H1 vs H2 Decomposition: 8-Cell Factorial & 4-Way Sink/Content Ladder

On generic `val_tune`, BOS calibration worked cleanly (`0.5714` student vs `0.5712` donor on 1-Hop heads; `0.5705` vs `0.5704` on 2-Hop heads; `pos_gt0_kkt_logit_rmse <= 0.018`). On structured task prompts (`pos0 = <|im_start|>` / `Rule`), donor `pos0` sink dropped to `0.0000` / `0.1429` while student remained `0.4903` / `0.3753`. Crucially, the offline 4-way sink/content ladder and 8-cell factorial **refute BOS-only explanations** and demonstrate a multiplicative double-wall in both `Q/K` routing and `V` values:

| Intervention (`Arm_2a_multitap_L0`) | 1-Hop `WIR` / `tok0` (`hits`) | 2-Hop Policy `WIR` / `tok0` (`hits`) | 2-Hop Binding `WIR` / `tok0` (`hits`) |
|---|---:|---:|---:|
| `S0_pure_student` (`QB_KB_VB`) | `-0.0119` / `65.50%` (`262`) | `0.0129` / `12.75%` (`51`) | `-0.0027` / `10.00%` (`20`) |
| `S1_donor_sink_only` (Exact donor `alpha_0`) | `-0.0132` / `66.25%` (`265`) | `0.0199` / `13.75%` (`55`) | `0.0012` / `10.50%` (`21`) |
| `S2_donor_content_pi_only` (Donor `pi_{>0}`) | `0.0662` / `67.25%` (`269`) | `0.0561` / `17.00%` (`68`) | `0.0368` / `13.00%` (`26`) |
| `S3_donor_attn_student_V` (`QA_KA_VB`) | `0.1296` / `66.00%` (`264`) | `0.0880` / `19.00%` (`76`) | `0.0705` / `15.50%` (`31`) |
| `QB_KB_VA` (Student attn + Donor `V`) | `0.0977` / `67.00%` (`268`) | `0.0713` / `14.00%` (`56`) | `0.0231` / `11.50%` (`23`) |
| `QB_KA_VA` (Student `Q` only) | `0.1687` / `67.50%` (`270`) | `0.1475` / `19.00%` (`76`) | `0.0606` / `13.50%` (`27`) |
| `QA_KB_VA` (Student `K` only) | `0.1730` / `67.75%` (`271`) | `0.1326` / `16.50%` (`66`) | `0.0533` / `12.50%` (`25`) |
| **`S4_full_donor_oracle` (`QA_KA_VA`)** | **`0.5222` / `71.25%` (`285`)** | **`0.3344` / `31.75%` (`127`)** | **`0.2793` / `24.00%` (`48`)** |

---

## 5. Arm 3 Breakthroughs: Upstream Circuit, Sparse `L20_MLP` Community, & Native `L0` Ingress

### 5.1 Arm 3a — Donor Backward ISA (`disc`) & Live Solo Ablations
- **Bottleneck MLP:** `L20_MLP` ranks `#1` by policy ISA (`1.8531`) AND `#1` by solo ablation on `disc`, where mean-ablating `L20_MLP` alone drops donor policy margin by **`2.4016` nats (`54.91%` of total `4.3738` margin)** despite negligible direct logit attribution (`mean_dla = 0.0129`), confirming it is a deep intermediate computation feeding `L22–L27`. On binding, `L22_MLP` leads ISA (`3.0647`), ahead of `L20_MLP` (`1.8186`) and `L24_MLP` (`1.7863`).
- **Upstream Heads:** Among ISA-ranked heads, live solo ablations on `disc` show positive causal drops for **`L22H1`** (`+0.6694` policy, `+0.7001` binding), **`L23H11`** (`+0.3236` policy, `+0.4431` binding), and **`L22H24`** (`+0.6359` binding), whereas `#1` ISA head `L22H25` had `-0.0003` policy and `-1.4471` binding drops (highlighting that ISA is a candidate ranking requiring independent mediation verification).

### 5.2 Arm 3b — `L20_MLP` Linearization vs Goodfire/Anthropic Sparse `J`-Lens Community (`M = 18,944`)
- **Generic Cancellation Constant:** `chi_nl = 93.917`, `chi_nl^2 = 8820.41`, `nl_energy_frac = 0.5534` (`slicing_rel_err_bound_k256 = 34.45` — an upper bound for naive random component sampling, not a universal lower bound against deterministic subsets, cancellation-aware methods, or SVD).
- **Disjoint `eval` Evaluation (zero eval leakage; neuron ranking computed strictly on `disc` contrasts, common generic baseline `mean_silu_u_gen`):**

| Condition on `L20_MLP` (Donor 7B) | Policy Margin (`tok0_acc`) | Policy Recovery | Binding Margin (`tok0_acc`) | Binding Recovery |
|---|---:|---:|---:|---:|
| `intact` (`k = 18,944`) | `4.3557` (`75.25%`) | `100.00%` | `10.9990` (`74.00%`) | `100.00%` |
| `mean_ablate` (`mean_silu_u_gen`) | `1.5342` (`57.00%`) | `0.00%` | `9.3812` (`75.50%`) | `0.00%` |
| `linear_jbar_E_mlp` (generic Jacobian) | `1.2915` (`55.50%`) | `-8.60%` | `10.4454` (`69.00%`) | `+65.78%` |
| `linear_jbar_mlp_E_x` | `1.2157` (`54.50%`) | `-11.29%` | `9.8375` (`69.00%`) | `+28.21%` |
| Sparse `J`-lens `k = 0` | `1.5346` (`57.50%`) | `0.01%` | `9.3858` (`76.00%`) | `0.28%` |
| **Sparse `J`-lens `k = 16` (`0.08%`)** | **`3.4403` (`75.75%`)** | **`67.55%`** | `9.1577` (`70.00%`) | `-13.82%` |
| **Sparse `J`-lens `k = 64` (`0.34%`)** | **`3.9023` (`78.75%`)** | **`83.93%`** | `9.7619` (`74.50%`) | `+23.53%` |
| Sparse `J`-lens `k = 128` (`0.68%`) | `3.5948` (`73.50%`) | `73.03%` | `9.3926` (`76.50%`) | `+0.70%` |
| Sparse `J`-lens `k = 256` (`1.35%`) | `3.8476` (`75.75%`) | `81.99%` | `9.7290` (`75.00%`) | `+21.50%` |
| Sparse `J`-lens `k = 1024` (`5.41%`) | `3.9398` (`74.00%`) | `85.26%` | `10.1864` (`74.50%`) | `+49.77%` |

### 5.3 Arm 3c — Live Native `L0` Tied-Embedding Ingress on `Qwen2.5-1.5B-Instruct`
- `K = 64`, `alpha = 0.015625` (`2^-6`), `tied_preserved = true`, `final_norm_suffix_zero = true`.
- Post-`LN_0` recovery on 16 held-out generic `report` sequences: **`kappa_L0_post_ln0 = 0.999988`**, **`cosine_L0_post_ln0 = 0.999997`**, **`generic_kl = 0.00117` nats**.
- Rollback verification: **`bit_exact_all_params_rollback = true`**, **`bit_exact_norm_eps_rollback = true`**, **`bit_exact_logits_rollback = true`** (`passes_all = true`).

---

## 6. Post-GPU Grading Council Qualifications & Caveats (Mandatory Disclosures)

Both Post-GPU Council seats (**Gemini 4 Neon High** and **Astra 6.2 Ultra**) issued **`VERDICT: PASS`** (`0 P0` defects) with the following explicit qualifications:

1. **Oracle Ceiling Limits (Do Not Over-Attribute 100% of Failure to Input Alone):** While the 5-head donor outlet oracle delivers `+20.00 pp` on 2-Hop Policy (`Q = 0.5056`, `R2_c = 0.3978`, `WIR = 0.3344`), its 1-Hop live EM lift (`+5.25 pp`) is below the `+10 pp` EM gate, and its 2-Hop Binding `WIR` (`0.2793`) and `R2_c` (`0.1769`) are below `0.30` (note that Binding evaluated the policy-selected 5 heads rather than binding-specific top-5 heads `L26H17, L26H3, L27H24, L26H5, L26H26`). `v5.2` should verify binding-specific head sets alongside policy heads.
2. **Arm 3b Policy vs Binding Asymmetry & Selection Scope:** `L20_MLP`'s ultra-sparse recovery (`67.55%` at `k=16`, `83.93%` at `k=64`) holds for **2-Hop Policy**, NOT Binding (where `L20_MLP` ablation drops margin only `14.7%` and raises `tok0_acc` `74.0% -> 75.5%`, `linear_jbar_E_mlp` recovers `65.78%`, and `L22_MLP` leads ISA). Moreover, neuron ranking used **0-SGD task-`disc` contrasts** (strictly out-of-sample on disjoint `eval`, zero eval leakage, but not unlabeled generic text), measured last-position answer-foil margin, and lacked a matched random-subset control; freezing the neuron IDs and `k` on a fresh split with matched random controls is required before claiming generalizable sparse organ extraction.
3. **Arm 3a ISA Qualification:** Ranking and solo ablations shared the `disc` split; treat ISA scores as candidate rankings pending independent mediation tests.
4. **Arm 3c Scope:** Verified on 16 generic report sequences with directly encoded donor embedding projections; end-to-end downstream task transport over the native `L0` lane remains to be tested in `v5.2`.
5. **Metric Definitions:** Both first-word EM and whole-completion EM are reported; `2hop` `whole_em = 0.0` is an expected artifact of the `2shot_fixed` prompt continuing `\n\nRule:` within `max_new_tokens=4`.

---

## 7. Next-Step Implication (`v5.2` Pivot)

Keep **G3 closed** for the tested late-layer-only linear readout recipe. Within the tested late-layer linear multi-tap adapter family (`L25–L27`), student-fed readout failed on `1.5B` (this motivates pivoting upstream rather than proving autonomous transfer impossible). *(Reproducibility note, Audit F25: the original `v5.1` run cached intermediate `.npz` tensors in `/tmp/pswap_v51_eval_cache_*.npz` outside git; the script now defaults to `experiments/cache/` via `PLATONICSWAP_CACHE_DIR`.)* The next-step hypothesis for `v5.2` is:
1. Verify frozen sparse `L20_MLP` (`k=16/64`) + `L22_MLP` / `L22H1, L22H24, L23H11` multi-stage donor circuits with matched random-subset controls and binding-specific readout heads, and
2. Test whether carrying the **upstream nonlinear + attention sub-circuit** through the now-verified **native `L0` tied-embedding private lane** (`kappa = 0.999988`) breaks the multiplicative `Q/K/V` input wall without SGD.
