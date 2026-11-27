# PlatonicSwap Stage 1 — Pure Native In-Weights Width Compression (`Qwen2.5-7B-Instruct → Qwen2.5-1.5B-Instruct`)

**Status:** Audited on Colab NVIDIA L4 across three sequential runs (`v4.2`, `v4.3.2`, `v4.4.1`) with mandatory 2-model Grading Council (**Astra 6.2 Ultra** + **Gemini 4 Neon High**) sign-off before and after every GPU run.
**Architecture Contract:** 100% stock HuggingFace `Qwen2ForCausalLM` matrices — **zero external bolted-on PyTorch sidecars**, **zero runtime hooks at inference**, **Donor 7B 100% unloaded before compilation/training/evaluation**, and **100% bit-exact `torch.equal` rollback** verified on every evaluated pack (`54/54` rollback probes across `v4.2`, `v4.3.2`, and `v4.4.1`).

---

## 1. Executive Summary (Plain Language)

We isolated pure width compression (`D = 3584 → 1536`, identical 28-layer depth, `head_dim = 128`, `rope_theta = 1e6`, and shared tokenizer `V_shared = 151,665`) to test whether organs from a 7B model can be transplanted natively into a 1.5B model **without backpropagation (0-SGD)** or with **light native polishing (Tier 2)**:

1. **Pure native widened organs (`+1 KV / +6 Q` GQA heads + `+256` SwiGLU neurons inside stock `Qwen2ForCausalLM`) work and beat Merged Teacher-LoRA:**
   - **Text-to-SQLite (`40` unseen schemas, `Base 1.5B = 52.5%` / `Donor 7B = 90.0%`):** In `v4.3.2`, **0-SGD `Tier 1B Observable Widened`** jumped to **`60.0%` (`+7.5 pp` over Base 1.5B, `KL = 0.0102`, `13/313` locality broken)** — beating 40-step `Tier 3 Merged Teacher-LoRA` (`50.0%`) by **`+10.0 pp`** with **zero backpropagation**. (Caveat: `n = 40` has ±7.8 pp binomial SE, and after 40 steps of SFT polish, `Tier 2 Init` and random-init `Tier 2 Ctrl` both sat at `60.0%` with `6` vs `2` broken locality facts.)
   - **OurAirports Implantation (`143` blindspots):** Widened native capacity (`Tier 2 Ctrl`) hit **`75.5%` implanted `p0` / `65.7%` unseen `p1/p2` / `31.3%` ordered routes** (`3/313` broken), crushing equal-step `Tier 3 Merged Teacher-LoRA` (`46.9% / 33.9% / 4.2%`) by **`+31.8 pp`** on unseen paraphrases. However, donor initialization underperformed random-zero initialization (`68.5% / 58.4%`, `8` broken).
   - **2-Row 2-Hop Policy Swap (`256` sealed test, 100% zero-label dev selection):** Global-stream 0-SGD failed (`0.0%` joint EM). With 80 steps of Teacher-KL polish, `Tier 2 Init` hit **`28.9%` EM / `27.0%` FC4** (`v4.2`: `41.0% / 40.6%`) vs `Tier 2 Ctrl` `26.6% / 26.2%` (`v4.2`: `34.4% / 34.0%`) and `Tier 3 LoRA` `26.6% / 25.8%` (`v4.2`: `27.0% / 25.4%`); `Shuffled Supervision = 0.0%`.
2. **The 5-Minute Private-Lane Falsifier (`v4.4.1`, `L18 → L20`) cleanly KILLED 20-layer 0-SGD compilation — while validating a near-lossless native `+128` private-lane substrate:**
   - **Substrate Validated:** Adding `K = 128` native private residual lanes (`D = 1536 → 1664`) with the **Free RMSNorm Gauge Trick (`α = 0.1`)** and **tied embeddings (`embed_tokens` / `lm_head` zero-padded, same `data_ptr`)** achieved **`KL = 0.00171` nat/tok**, **`99.4%–99.7%` 380-fact locality retention (`311–312/380`, only 1–2 broken)**, and `torch.equal == True` rollback. Excluding position 0 cured the `erank = 1.03` rank-1 spike (`erank_pos_gt0 = 44–72`, `pos0_sink_rel_err = 0.0`) and lifted L20 gate/up direction cosine from `0.0095` to `0.9794`.
   - **0-SGD Multi-Layer Relay Decisively KILLED (`VERDICT = KILL`):** Although Metric 1 (`c = 0.9563 ≥ 0.80`) and Metric 2 (`β_refit = 0.9404 ≥ 0.70`, `donor_copied = -2.6543`) hit **GO**, strict copied writes failed (`REAL_COPIED η_direct = -7.0327`), and refitted writes (`REAL_REFIT η_direct = 0.6890`) tied the **wrong-layer `L6/L7` `SHUF` control (`η_direct = 0.6784`, `REAL - SHUF = +0.0106 ≪ 0.15`)** — proving a reservoir/random-feature regression effect rather than layer-specific circuit transfer. Native BF16 SwiGLU product `R²_Δ` also failed (`-46.41` vs `SHUF -44.37` and `NONE` local reader `+0.1744`), and RoPE `attn_logit_rmse = 2.506 nat > 1.0 nat`.

---

## 2. Exact Math Diagnostics (`v4.3.2` — `results_v4_3_stage1_exact_math_native.json`)

Streamed FP64 second moments over **`N_fit = 71,055` tokens** (`N_val_tune = 3,200`, `N_val_report = 3,200`, **0 shared prefixes** across both `fit vs val` and `val_tune vs val_report`), zero-copy BF16 chunked shared-vocab Fisher (`17.789 GiB` Phase 1 peak alloc, `2.884 GiB` post-unload, `6.732 GiB` Phase 2 peak alloc):

| Diagnostic | Result | Interpretation & Caveat |
|---|---|---|
| **Gate 0 (`L27`): `M_U` vs `M_out` vs Base 1.5B** | `M_U`: **`0.2547` KL / `92.5%` top-1**<br>`M_out`: `0.6263` KL / `42.5%` top-1<br>`Base 1.5B`: `0.4341` KL / `45.0%` top-1 | Exact zero-data shared-vocab Fisher bridge `M_U = (G_B + εI)^{-1} G_BA` beats Base 1.5B and more than doubles `M_out` top-1 agreement when supplied Donor 7B's final state. |
| **Floor #1 (`L27` Fisher-RRR at `k=1536`)** | **`99.8305%`** retained (`k=256: 96.74%`, `k=512: 98.62%`, `k=1024: 99.58%`) | Width (`3584 → 1536`) is **not** an information bottleneck for next-token Fisher information. |
| **Floor #2 (Observability `o_j` & Patched KL `R`)** | `L20`: `R_push = 97.68%`, `R_pull = 98.12%`, `R_rt = 98.47%`<br>`L27`: `R_push = 91.45%`, `R_pull = 88.92%`, `R_rt = 89.12%` | **Caveat:** `v4.3.2` included position 0 (BOS/sink outlier), inflating pooled `L14–L24` `R² ≈ 0.998` and putting `18,944/18,944` MLP units above `o_j ≥ 0.25` (whereas `L26–L27` `R² = 0.40–0.42`). Fixed by pos-0 exclusion in `v4.4.1` (`val_r2_pos_gt0 = 0.27–0.44`). |
| **Floor #3 (`Delta_NL` Nyström Kernel)** | `L20`: **`+0.0000 R²`** / `L27`: **`+0.0015 R²`** | Linear maps capture essentially all predictable cross-model state geometry; nonlinear bridges add ~0. |

---

## 3. Head-to-Head Task Comparison (`v4.3.2` vs `v4.2`)

All `v4.3.2` Tier 1 packs satisfied the fail-closed `KL_drift ≤ 0.05` OOD guard (`final_ood_kl ∈ [0.0000, 0.0350]`) and every pack passed `logits_bit_equal = true, max_abs_diff = 0.0, zero_runtime_hooks = true`.

### Task A — Text-to-SQLite (`40` Unseen Schemas; Donor 7B = `90.0%`, Base 1.5B = `52.5%`)

| Tier (`v4.3.2`) | Unseen Exec | Unseen Syntax | OOD KL (`α`) | Locality Broken (`/313`) |
|---|---:|---:|---:|---:|
| `Tier 1A` OOD-Guarded Seq `ΔW` (**0 SGD**) | `37.5%` | `50.0%` | `0.0350` (`0.05`) | `29` |
| `Tier 1B` Observable Widened (**0 SGD**) | **`60.0%`** | `70.0%` | `0.0102` (`0.25`) | `13` |
| `Tier 2` Observable Native Init + 40-step SFT | **`60.0%`** | `75.0%` | — | `6` |
| `Tier 2 Ctrl` Random/Zero Native + 40-step SFT | **`60.0%`** | **`82.5%`** | — | **`2`** |
| `Tier 3` Merged Teacher-LoRA (Rank 16, 40 steps) | `50.0%` | `72.5%` | — | `6` |

### Task B — OurAirports Implantation (`143` Blindspots, `286` Unseen `p1/p2`, `48` Unseen Routes)

| Tier (`v4.3.2`) | `implanted_p0` | Unseen `p1/p2` | Unseen Routes (Ord / Either) | Locality Broken (`/313`) |
|---|---:|---:|---:|---:|
| `BASE 1.5B` | `0.0%` | `9.8%` | `2.1% / 18.8%` | `0` |
| `Tier 1B` Observable Widened (**0 SGD**, `α=0.25`) | `3.5%` | `11.2%` | `0.0% / 18.8%` | `8` |
| `Tier 1-GN` Alignment-Free L27 (**0 SGD**, `α=0.0`) | `0.0%` | `9.8%` | `2.1% / 18.8%` | `0` |
| `Tier 2` Observable Native Init + 40-step SFT | `68.5%` | `58.4%` | `12.5% / 77.1%` | `8` |
| `Tier 2 Ctrl` Random/Zero Native + 40-step SFT | **`75.5%`** | **`65.7%`** | **`31.3% / 79.2%`** | `3` |
| `Tier 3` Merged Teacher-LoRA (Rank 16, 40 steps) | `46.9%` | `33.9%` | `4.2% / 56.3%` | **`2`** |

### Task C — 2-Row 2-Hop Policy Swap (`256` Sealed Test; 100% Zero-Label Dev Selection; Donor 7B = `37.5%` EM/FC4)

| Tier (`v4.3.2`) | Joint EM | Joint FC4 | Joint 2-Way FC | Locality Broken `(L, R)` |
|---|---:|---:|---:|---:|
| `Tier 1B` Last-Tok / Shared-Suffix / `Tier 1-GN` (**0 SGD**) | `0.0%` | `0.0–1.2%` | `0.0–2.0%` | `(0–1, 0–4)` |
| `Tier 2` Observable Native Init + 80-step KL (`α_init=0`) | **`28.9%`** | **`27.0%`** | **`100.0%`** | `(22, 7)` |
| `Tier 2 Ctrl` Random/Zero Native + 80-step KL | `26.6%` | `26.2%` | `100.0%` | `(10, 7)` |
| `Tier 2 Ctrl` Shuffled Supervision (Sanity Control) | `0.0%` | `2.0%` | `16.4%` | `(19, 17)` |
| `Tier 3` Merged Teacher-LoRA (Rank 16, 80-step KL) | `26.6%` | `25.8%` | `99.2%` | **`(2, 8)`** |

*(In `v4.2` with a different dataset seed where Donor 7B ceiling was `43.8%`: `Tier 2 Init` hit `41.0% / 40.6%` vs `Tier 2 Ctrl` `34.4% / 34.0%` vs `Tier 3 LoRA` `27.0% / 25.4%`; `Shuffled = 0.0%`. In `v4.3.2`, because `t1b_best_packs` entered polish with `α = 0.0` (zero output weights), the `+2.35 pp` directional edge (`+6/256`, within ~1 SE) came strictly from pulled-back `Q/K/V/gate/up` input features, at the cost of +12 left-policy locality breaks.)*

---

## 4. 5-Minute Go/No-Go Private-Lane Falsifier (`v4.4.1` — `results_v4_4_private_lane_falsifier.json`)

Run on Colab NVIDIA L4 in **`480.6s`** (`14.558 GiB` Donor peak → `0.009 GiB` post-unload → `6.412 GiB` Recipient peak; `400` fit / `400` unseen-key test 2-Hop pairs; `20,000` `pos > 0` generic tokens; `K = 128` private lanes; `α = 0.1`; leak-free two-pass inner-320 tuning → final-400 refit).

- **Metric 1 (Interface Capture `c` on unseen `δΔ = δ(L20_in − L18_in)`):** **`0.9563`** (**GO ≥ 0.80**; endpoint `L20 = 0.7381`).
- **Metric 2 (Organ Compute Budget `β` on unseen `δΔ`):** **`donor_refit R² = 0.9404`** (**GO ≥ 0.70**; `donor_copied = -2.6543`).

| Arm (`v4.4.1`) | `η_direct` (`δΔ`) | `η_readout` | `L20` Med Cos (`gate/up`) | Native BF16 SwiGLU `R²_Δ` | RoPE RMSE (`pos>0`) | Generic KL | `ρ` Shift | Locality (`/380`, Ret.) | Rollback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `NONE` (0 writes) | `-7.7981` | `-7.7981` | `0.8472` | `-50.43` (local reader **`+0.1744`**) | `2.455 nat` | `0.00170` | `1.51e-3` | `311` (`99.4%`) | `True` |
| `REAL_COPIED` | `-7.0327` | `0.6702` | `0.8649` | `-50.30` | `2.350 nat` | `0.00171` | `1.52e-3` | `311` (`99.4%`) | `True` |
| **`REAL_REFIT` (Primary)** | **`0.6890`** | `0.6928` | `0.9794` | `-46.41` | `2.506 nat` | `0.00171` | `1.52e-3` | `311` (`99.4%`) | `True` |
| `SHUF` (`L6/L7` control) | **`0.6784`** | `0.6752` | **`0.9806`** | **`-44.37`** | `2.587 nat` | **`0.00170`** | `1.53e-3` | **`312` (`99.7%`)** | `True` |

- **Paired Key-Group Cluster Bootstrap (`56` unseen key groups, 95% CI on `REAL_REFIT` SwiGLU `R²_Δ`):**
  - vs `NONE_pulled`: **`+4.02` `[+3.76, +4.33]`**
  - vs `NONE_local_reader`: **`-46.59` `[-50.24, -43.55]`** (FAIL)
  - vs `SHUF`: **`-2.04` `[-2.34, -1.79]`** (FAIL)
- **Verdict:** **`KILL`** on both `REAL_REFIT` (`REAL - SHUF = +0.0106 ≪ 0.15`) and `REAL_COPIED` (`-7.0327 < 0.25`).

---

## 5. Grading Council Audit Trail & Predeclared Next Pivot

- **Post-GPU Audit (`v4.2` + `v4.3.2`):** **Gemini 4 Neon High** `PASS`; **Astra 6.2 Ultra** confirmed `v4.3.2` passed all execution/rollback/OOD-KL/zero-label checks (`v4.2` pre-dated OOD-KL & zero-label selection) and endorsed proceeding to `v4.4`.
- **Post-GPU Audit (`v4.4.1`):** **Astra 6.2 Ultra** `POST-GPU AUDIT VERDICT: PASS` & **Gemini 4 Neon High** `POST-GPU AUDIT VERDICT: PASS` — unanimous agreement that `v4.4.1` cleanly **kills 20-layer 0-SGD compilation** and validates the **predeclared pivot**:
  1. **Pivot Target (`v4.5`):** **Private-lane lightly polished native organs (`Tier 2` with `+128` private lanes + Free RMSNorm Gauge Trick `α = 0.1` + tied embeddings)** to capture the near-zero locality loss (`1–2` broken vs `22` broken in shared-stream `v4.3.2`) and strong task gains (`+31.8 pp` over LoRA on Airports; `+7.5–10 pp` on SQL).
  2. **Mandatory Rigor Controls for `v4.5`:**
     - **Layer-shuffled donor control (`L6–L13` / wrong-layer init)** alongside **random-init** and **Rank-16 Merged Teacher-LoRA** (directly defeating the reservoir effect!).
     - **Matched `0 / 10 / 40 / 80`-step training curves** from a **zero-write function-preserving start** (verifying whether 7B input init genuinely accelerates convergence or raises the asymptote over shuffled-donor & random-init!).
     - Report both absolute `ρ` and incremental `Δρ = ρ(arm) − ρ(NONE)` alongside `KL`, 380-fact retention, and bit-exact `torch.equal` rollback.

---

## 6. Reproducibility & Artifacts

- `v4.2` code & results: `experiments/v4_stage1_qwen7b_to_1p5b_native.py`, `results/results_v4_stage1_qwen7b_to_1p5b_native.json`
- `v4.3.2` code & results: `platonicswap/exact_math_v43.py`, `platonicswap/native_weight_translation.py`, `experiments/v4_3_stage1_exact_math_native.py`, `results/results_v4_3_stage1_exact_math_native.json`
- `v4.4.1` code & results: `experiments/v4_4_private_lane_falsifier.py`, `results/results_v4_4_private_lane_falsifier.json`
