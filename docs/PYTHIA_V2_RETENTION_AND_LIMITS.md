# Pythia v1.1 Retention Repair & v2 Falsification / Boundary Report

> **Executive Summary & Provenance Boundary:** Following the verified, zero-retry sealed Pythia-160m v1 standalone transplant ([PYTHIA_V1_STANDALONE_TRANSPLANT.md](PYTHIA_V1_STANDALONE_TRANSPLANT.md)), we investigated its three primary limits—(1) generic-code retention cost, (2) cross-width compilation (`pythia-410m -> pythia-160m-seed2`), and (3) true low-skill recipients where native task ability is absent—adding an adversarial **Recipient-Self-Graft** control (duplicating the recipient's own best heads under identical discovery, head, and retention budgets).
> - **JSON-backed reproducible benchmark (§1.1–§1.2):** [`experiments/pythia_v1_1_retention_and_limits.py`](../experiments/pythia_v1_1_retention_and_limits.py) → [`results/results_pythia_v1_1_retention_and_limits.json`](../results/results_pythia_v1_1_retention_and_limits.json) (single-run discovery estimates on `pool[150:300]`, `n = 256`, binomial SE ≈ 3 pp, no hypothesis tests; reserved induction prompts `pool[1500:]` and IOI prompts `names[64:117]` were never evaluated, though the stdlib code token filter for this discovery run excluded `pool[:600]` rather than `pool[1500:]`).
> - **Discovery scratch diagnostics (§§1.3, 2, 3):** Recorded from local discovery-only runs (not packaged as standalone repo JSONs) to transparently document which v2 paths failed entry/kill criteria specified in advance and why we halted without unsealing `pool[1500:]` or `names[64:117]`.
>
> **Summary of Verified Findings:**
> 1. **Exact `GELU(z) - GELU(-z) = z` Parallel-MLP Subtractor & Direct-`p` Counterfactual (§1.1–§1.2, JSON-backed):** Matching published v1's exact `Rh` alignment (`recipient L5 -> donor L3`), folding a closed-form linear subtractor into the unused parallel MLP of `Block_Ind_Late` (`2*d = 1,536/3,072` units, paired input biases `[-V^T x_bar, +V^T x_bar]`, unit-test `max_abs_err = 1.91e-6` at `d = 64` and `9.30e-6` at `d = 768`, zero hooks, `148/148` frozen recipient tensors byte-identical, `0.0` rollback) reduces generic-code KL `1.15×` on its own (`0.2094 -> 0.1824`, recovering `88.8%` of the `0.0305` drop observed when directly subtracting `p` at the suffix entrance, which leaves `85.5%` of KL). Combining `L8 MLP(-p)` with stronger ridge shrinkage (`lam = 0.01 -> 0.10`) cuts generic-code KL **`2.70×`** on this corpus (`0.2094 -> 0.0774`, `PPL 1.296× -> 1.108×`, `p99 KL 1.11 -> 0.45`, still above the aspirational `KL <= 0.05 / PPL <= 1.05×` target); most of this `2.70×` reduction comes from ridge shrinkage (holding the MLP enabled, ridge shrinkage alone reduces `RAND24` KL `2.55×` from `0.0743 -> 0.0292`, and trades off ~36% of the `rs6` gain over `STOCK`, from `+15.2 pp` to `+9.8 pp`).
> 2. **Mandatory Control Catch — `RECIPIENT-SELF-GRAFT` Beats Cross-Seed Donor on Induction at Overlapping KL (§1.2, JSON-backed):** Because `seed2@step2000` already has strong native induction (`allVal = 91.4%`), duplicating 24 distinct native heads (`8 + 7 + 9` + `L8 MLP(-p)`) beats the cross-seed donor on both `rs6` and `nl6` at directly overlapping KL: at `KL = 0.0753`, `SELF-GRAFT` reaches **`rs6 = 0.6016, nl6 = 0.7227`** (`allVal = 0.9336`) vs. Donor `v1.1` (`lam = 0.10, KL = 0.0774`: **`rs6 = 0.5000, nl6 = 0.6289`**, `allVal = 0.9336`); at `KL = 0.0940`, `SELF-GRAFT` reaches **`rs6 = 0.6328, nl6 = 0.7617`**, exceeding even the unedited donor (`0.6172 / 0.7422`). We therefore **did not seal v1.1 for cross-seed donor superiority** and treat `RECIPIENT-SELF-GRAFT` as a mandatory baseline.
> 3. **Cross-Width `pythia-410m -> seed2@step2000` Induction Stopped at Kill Gates Specified in Advance (§2, scratch diagnostic):** Cross-width `1024 <-> 768` linear folding verified to `3.46e-6` parity and cleared Gate 1 (`+29.3 pp`), but **failed Gate 2** (`8 + 7 + 9` key-swap isolation = `0.185`; even all 384 heads with `L1..7` MLPs corrupted = `0.683 < 0.70`) and **failed locked-gain Gate 3** (`37.3% < 50%`; exploratory `1.5/1.5` gains reached `61.3%`). Stopped cleanly.
> 4. **True Low-Skill IOI (`pythia-160m -> seed2@step3000`, §3, scratch diagnostic):** `seed2@step3000` genuinely lacks IOI (`STOCK fc2 = 20.1%, top1 = 2.1%, m = -1.47`; `SELF-GRAFT fc2 = 33.3%, top1 = 0.0%, m = -0.67`). Donor IOI cleared Gate 1, Gate 2 (`Top-20` heads: `111.3%` attn / `79.7%` all-MLP isolation; 19 heads compiled), and hooked donor-oracle Gate 3 (`89.1% / 62.5% / +2.48` via zero-data `W_U` Procrustes `R_U in O(768)`). Only the hooked donor oracle strongly beat `SELF-GRAFT`; no standalone compiled model did. Tested standalone designs hit a cross-depth tradeoff (`L7` injection collapsed `top1` to `0.016`), and an exact Donor-attention Gate-0 sweep on the tested `POST-L11` (`idx = 12`) configurations (with `MLP_14` refit per gain under donor attention) found zero operating points meeting `fc2 >= 0.75, fc2 - SELF >= 0.40, KL <= 0.10, minCell > 0, top1 >= 0.30` (at `KL = 0.0906`: `fc2 = 0.417, ΔSELF = +0.083, top1 = 0.083, minCell = -2.03`; at `a = 1.50, KL = 0.3651`: `fc2 = 0.708, ΔSELF = +0.375, top1 = 0.243, minCell = -0.08`). We halted this `POST-L11` design and kept `names[64:117]` unevaluated.

---

## 1. v1.1 Retention Repair & Overlapping Iso-KL `SELF-GRAFT` Benchmark (§1.1–§1.2 JSON-Backed; §1.3 Scratch)

**Reproducible script & artifact (§1.1–§1.2):** [`experiments/pythia_v1_1_retention_and_limits.py`](../experiments/pythia_v1_1_retention_and_limits.py) (`--unit-test` PASS: `d=64 err=1.91e-6`, `d=768 err=9.30e-6`, `RAND = 24` distinct disjoint heads) · [`results/results_pythia_v1_1_retention_and_limits.json`](../results/results_pythia_v1_1_retention_and_limits.json).

In `GPTNeoXLayer` (`use_parallel_residual = True`), the third inserted block (`Block_Ind_Late`, Layer 8) has an unused `4*d = 3,072`-unit GELU MLP reading `x = LN(h + p + e)` right before frozen `L6..L11`. Using $\mathrm{GELU}(z) - \mathrm{GELU}(-z) = z$, any closed-form linear ridge predictor $\hat{t}(x) = \bar{t} + A(x - \bar{x})$ ($A = U\Sigma V^\top$) compiles into `2*d = 1,536` native MLP units with input rows `[V^T; -V^T]`, paired input biases `[-V^T x_bar; +V^T x_bar]`, output columns `[-U\Sigma, +U\Sigma]`, and output bias `-\bar{t}` (zero hooks, zero SGD; verified `148/148` frozen recipient tensors byte-identical and bit-exact `0.0` rollback).

### 1.1 Direct-`p` Removal Counterfactual (`exact v1 Rh = h3`, 40 disjoint stdlib `.py` files vs. 65 fit files)
- **Full v1 (`h + p + e + l`, `lam = 0.01`, no MLP):** `KL(STOCK || REAL) = 0.2094`, `PPL = 1.296×` (`STOCK PPL = 29.98`; corpus differs from sealed v1's `0.2414`).
- **Counterfactual Direct-`p` Removal (`p` subtracted from Layer-8 output before `L6..L11`, leaving `e` and `l` intact):** `KL = 0.1790` (**`85.5%`** remaining; because `e` and `l` already read `p` and KL is non-additive, this measures remaining KL after direct `p` removal rather than an additive causal share or syntax mechanism).
- **Deployed `L8 MLP(-p)` (`lam = 0.01`, `fit R² = 0.86`):** `KL = 0.1824` (`1.15×` reduction alone), recovering **`88.8%`** of the `0.0305` direct-`p` KL drop while nudging `rs6` from `0.5547` to `0.5664`.

### 1.2 Donor v1.1 Retention Sweep vs. Overlapping Iso-KL `RECIPIENT-SELF-GRAFT` (`pool[150:300]`, `n = 256`)

All models share the exact same `15L` vanilla `GPTNeoXForCausalLM` budget (`8 + 7 + 9 = 24` distinct heads + `L8` exact-GELU `MLP(-p)`), with `RAND24` sampled **without replacement** (24 distinct disjoint heads matching exact per-layer counts) and clean `EdgeCut` zeroing both `L6` Prev attention and `L8 MLP(-p)`. Single-run `disc_val` estimates (`n = 256`, binomial SE ≈ 3 pp, no significance tests; at `lam = 0.10`, `rs6 = 0.5000` vs `EdgeCut = 0.4922` differs by only 2 families).

| System (`seed2@step2000`, `disc_val n=256`) | Code KL (mean / p99) | Code PPL Ratio | `rs6` Top-1 | `rs6` `allVal` | `nl6` Top-1 | `EdgeCut rs6` | `RAND24 rs6` (KL) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DONOR` (`pythia-160m` final) | — | — | 0.6172 | 0.9062 | 0.7422 | — | — |
| `STOCK` (`seed2@step2000`) | 0.0000 / 0.00 | 1.000× | 0.4023 | 0.9141 | 0.4258 | — | — |
| **Donor v1** (`lam=0.01`, no MLP) | 0.2094 / 1.11 | 1.296× | 0.5547 | 0.9453 | 0.6953 | 0.5195 | 0.3203 (0.0788) |
| **Donor v1.1** (`lam=0.01`, `MLP(-p)`) | 0.1824 / 0.95 | 1.255× | 0.5664 | 0.9492 | 0.6914 | 0.5195 | 0.3047 (0.0743) |
| **Donor v1.1** (`lam=0.05`, `MLP(-p)`) | 0.1082 / 0.62 | 1.151× | 0.5273 | 0.9336 | 0.6523 | 0.5078 | 0.3242 (0.0411) |
| **Donor v1.1** (`lam=0.10`, `MLP(-p)`) | **0.0774** / 0.45 | **1.108×** | 0.5000 | 0.9336 | 0.6289 | 0.4922 | 0.3359 (0.0292) |
| `SELF-GRAFT` (`scale=0.25`, `MLP(-p)`) | 0.0095 / 0.04 | 1.019× | 0.4375 | 0.9062 | 0.5117 | — | — |
| `SELF-GRAFT` (`scale=0.40`, `MLP(-p)`) | 0.0148 / 0.07 | 1.029× | 0.4688 | 0.9102 | 0.5391 | — | — |
| `SELF-GRAFT` (`scale=0.55`, `MLP(-p)`) | 0.0224 / 0.11 | 1.042× | 0.5039 | 0.9102 | 0.5820 | — | — |
| `SELF-GRAFT` (`scale=0.70`, `MLP(-p)`) | 0.0323 / 0.17 | 1.057× | 0.5234 | 0.9141 | 0.6328 | — | — |
| `SELF-GRAFT` (`scale=0.85`, `MLP(-p)`) | 0.0445 / 0.23 | 1.076× | 0.5586 | 0.9219 | 0.6602 | — | — |
| `SELF-GRAFT` (`scale=1.00`, `MLP(-p)`) | 0.0588 / 0.31 | 1.096× | 0.5820 | 0.9258 | 0.6992 | — | — |
| **`SELF-GRAFT`** (`scale=1.15`, `MLP(-p)`) | **0.0753** / 0.41 | **1.120×** | **0.6016** | 0.9336 | **0.7227** | — | — |
| **`SELF-GRAFT`** (`scale=1.30`, `MLP(-p)`) | **0.0940** / 0.51 | **1.147×** | **0.6328** | 0.9336 | **0.7617** | — | — |

**Takeaway:** `v1.1` reduces generic-code KL **`2.70×`** (`0.2094 -> 0.0774`, driven primarily by ridge shrinkage `lam = 0.01 -> 0.10`, which trades off ~36% of the `rs6` gain above `STOCK` and still misses `KL <= 0.05 / PPL <= 1.05×`). At directly overlapping KL (`0.0753` vs `0.0774`, and `0.0940` vs `0.1082`), **`RECIPIENT-SELF-GRAFT` beats the cross-seed donor by `+10.2 to +10.5 pp` on `rs6` and `+9.4 to +10.9 pp` on `nl6`** at identical `allVal = 0.9336`—because `step2000` already possesses strong native induction (`allVal = 91.4%`).

### 1.3 Lower Induction Checkpoints (Discovery Scratch Diagnostic, Not JSON-Backed, `disc_val n = 128`)
- **`seed2@step1000` (`8b6c38e87a25`, `PPL = 43.79`):** `STOCK rs6 = 0.039 (allVal = 0.617)`. Compiled donor `(1.0, 1.2, 1.2)` reached `rs6 = 0.070 (allVal = 0.688, KL = 0.3261)`—still beaten by `RECIPIENT-SELF-GRAFT` (`rs6 = 0.148, allVal = 0.680, KL = 0.1381`).
- **`seed2@step512` (`a91d795fcc1b`, `PPL = 157.45`):** `STOCK rs6 = 0.000 (allVal = 0.109)`. Both compiled donor (`0.000`) and `SELF-GRAFT` (`0.000`) failed.

---

## 2. Cross-Width Qualification (`pythia-410m -> seed2@step2000`, Discovery Scratch Diagnostic)

We evaluated cross-width compilation from `EleutherAI/pythia-410m` (`9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`, `24L, d=1024, 16H`, `hd=64, rot_dim=16`) into `pythia-160m-seed2@step2000` (`12L, d=768, 12H`):
- **Cross-Width Fold Parity:** Exact `1024 <-> 768` linear folding verified to `max_abs_err = 3.46e-6`.
- **Gate 1 (Donor Lead):** PASS (`+29.3 pp` on `disc_val rs6`).
- **Gate 2 (Strict Family-Matched Key-Swap Isolation, threshold `>= 0.70`):** **FAIL (KILL)** — the nominated `8 + 7 + 9` circuit retained only **`0.185`** of donor clean-vs-swap margin; even keeping all 384 heads while corrupting `L1..L7` MLPs reached only **`0.683 < 0.70`**.
- **Gate 3 (Frozen Suffix Oracle, threshold `>= 50%` at locked gains):** **FAIL** at locked gains (**`37.3%`**); exploratory `1.5/1.5` gains reached `61.3%`. Direct compiled `410m -> 160m` checks showed `rs6` attention collapse (`0.08 / 0.06 / 0.05` at `idx = 6`). Stopped per kill rules specified in advance.

---

## 3. Low-Skill IOI (`pythia-160m -> seed2@step3000`, Discovery Scratch Diagnostic)

Screening four circuits (`IOI`, `Greater-Than`, `Binding-Update`, `SVA`) on discovery splits eliminated Greater-Than (`step1000 STOCK 0.000/-0.162 -> oracle 0.336/+0.065`; `step2000 STOCK 0.625` vs oracle `0.633`) and identified **IOI on `seed2@step3000`** (`a6124d29ca47`, counterbalanced 50/50 `ABBA`+`BABA`, 3 templates `T1–T3`, `disc_rank = names[0:32]`, `disc_val = names[32:64]`, reserved `names[64:117]` never evaluated):
- **Gate 1 PASS (6-cell balanced `n = 144`):** `DONOR fc2 = 1.000, top1 = 0.799, m = +4.43 (minCell = +3.01)` vs. `STOCK fc2 = 0.201, top1 = 0.021, m = -1.47 (minCell = -3.15, Code PPL = 21.80, SVA = 100.0%)`.
- **`RECIPIENT-SELF-GRAFT` Fails on `step3000`:** Across `a in {0.5, 1.0, 1.5, 2.0}` (`KL = 0.0060–0.0795`), duplicating `step3000`'s own best 4 DLA heads reached at most **`fc2 = 0.333, top1 = 0.000, m = -0.67 (minCell = -2.15)`**. (On `step5000`, `SELF-GRAFT` reached `fc2 = 0.781, m = +1.57` at `KL = 0.0193`, disqualifying `step5000`.)
- **Gate 2 PASS (`disc_val n = 128`, `Top-20` LOO heads):** `111.3%` attention-only isolation and `79.7%` isolation even when corrupting all 11 MLPs (`L1..L11`). *(The 3-stage compiler used 19 heads: 7 Dup + 4 S-Inh + 8 NM, or smaller NM2/NM4 subsets.)*
- **Hooked Donor-Oracle Gate 3 (`L11 + 1.5 * M_RU @ y_don`, zero-data `W_U` Procrustes `R_U in O(768)` on 50,187 non-name tokens, `R² = 0.799`):** `NM8` reached `fc2 = 0.891, top1 = 0.625, m = +2.48` (`NM4 = [(8,9),(8,10),(8,2),(9,8)]`: `fc2 = 0.859, top1 = 0.531, m = +2.18`).
- **2×2 Q/K-vs-V Autopsy (`NM4`, `dv_fixed n = 64`):** `(Donor Attn, Donor V)` = `0.859 / 0.531 / +2.18`; `(Donor Attn, Compiled h_7 V)` = `0.938 / 0.578 / +3.07` (exceeds Donor V); `(Donor Attn, Compiled h_11 V)` = `0.781 / 0.469 / +1.75` (below Donor V); `(Compiled Attn @ idx=6, Compiled h_5 V -> direct L11)` = `0.703 / 0.500 / +1.41`.

### 3.1 Tested Cross-Depth Failure Modes & `POST-L11` Gate-0 Kill
Only the hooked donor oracle strongly beat `SELF-GRAFT`; no standalone compiled model cleared the KL-budgeted target:
1. **Mid-layer (`L7`, `idx = 8`) injection:** Adding `1.5 * M_RU @ y_don` at `L7` preserved `fc2 = 0.828, m = +2.29`, but collapsed `top1` to `0.016` (and `0.000` even after projecting `M_RU` orthogonal to the top `384` input-sensitive directions of `L8..L11`).
2. **Residual Nullspace Bus (`L10 -> L15`):** A `||c|| = 5–6` signal in the bottom `32–128` singular vectors of `[h_7..h_11; W_in(L8..11)]` traversed `L8..L11` with `KL = 0.011–0.016` and `0.83–0.85` gain, but `h_11_B` retained `e_ratio = 0.21` (`bg_rms = 1.52–4.27`), causing `1/gamma` decoding to inflate `CodeKL = 0.61–1.65`.
3. **Exact Donor-Attention Gate-0 Sweep on Tested `POST-L11` (`idx = 12`) Configurations:** Replacing Layer-14 `NM4` attention probabilities with exact Donor causal attention matrices at every position on both 6-cell IOI (`n = 144`) and held-out code (`80` chunks), with `MLP_14(-y13, -y14)` refit per gain under donor attention (which lowered `a = 1.50` KL from `0.5396` to `0.3651` while lifting `fc2` from `0.653` to `0.708`), yielded **zero operating points** passing the criteria specified in advance (`fc2 >= 0.75, fc2 - SELF >= 0.40, KL <= 0.10, minCell > 0, top1 >= 0.30`):

| Tested `POST-L11` Graft + Exact Donor `NM4` Attn (`MLP14` refit per gain) | Code KL (p99) | Code PPL Ratio | `fc2` (`ΔSELF`) | Full-Vocab `top1` | Mean Margin | Worst Cell (`T3_BA`) | Gate-0 |
|---|---:|---:|---:|---:|---:|---:|---|
| `a = 0.25` | 0.0665 (0.38) | 1.042× | 0.306 (-0.028) | 0.056 | -0.87 | -2.63 | FAIL |
| `a = 0.40` | 0.0790 (0.43) | 1.051× | 0.396 (+0.062) | 0.069 | -0.54 | -2.27 | FAIL |
| `a = 0.50` *(highest gain with `KL <= 0.10`)* | **0.0906** (0.50) | 1.062× | **0.417** (+0.083) | **0.083** | -0.34 | -2.03 | **FAIL** |
| `a = 0.80` | 0.1423 (0.77) | 1.114× | 0.549 (+0.215) | 0.132 | +0.22 | -1.33 | FAIL |
| `a = 1.00` | 0.1912 (1.03) | 1.169× | 0.625 (+0.292) | 0.160 | +0.54 | -0.92 | FAIL |
| `a = 1.50` *(3.65× over KL budget)* | 0.3651 (1.91) | 1.392× | 0.708 (+0.375) | 0.243 | +1.15 | -0.08 | **FAIL (all 5)** |

These measurements refute the tested linear `POST-L11` operating points on `seed2@step3000` (they do not prove a universal impossibility across nonlinearly gated or alternative cross-depth architectures). We halted `POST-L11` IOI compilation cleanly and kept `names[64:117]` unevaluated.
