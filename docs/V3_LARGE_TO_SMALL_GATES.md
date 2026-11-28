# PlatonicSwap v3: Real Large → Small 48-Hour Gate Results

**Transfer Pair:** `Qwen/Qwen2.5-7B-Instruct` (`3584×28`, 14.2 GiB BF16, unloaded 100% at inference) → `HuggingFaceTB/SmolLM2-1.7B-Instruct` (`2048×24`, BF16)  
**Hardware & Compute Hygiene:** Colab NVIDIA L4 (24 GB VRAM, BF16). Total spend across both gates: `0.86` Compute Units (`Active assignments: 0` verified post-run).  
**Contract Compliance:** All 5 locked `v1.0` SHA-256 artifacts verified byte-identical. Every tier across both gates verified exact bit-identical rollback (`torch.equal == True`, `max_abs_diff = 0.0`) on matched-batch fresh-cache logits (`use_cache=False`).

---

## Executive Summary

1. **Pure Zero-Backprop Linear Cross-Family Mapping Is Cleanly Falsified on 7B → 1.7B:**
   - **Gate 1a (Knowledge Enrichment — OurAirports):** Even with full-rank bridge conditioning (`N = 3,945 > D_A = 3,584`, `cos = 0.976`, `R² = 0.957`, strictly zero blindspot leakage), zero-shot linear state pushforward recovers **`1.6%` (`1/61`)** vs `1.6%` `SELF` and `3.3%` `PERMUTED` (**FAIL**, `-1.6 pp` vs `+20 pp` bar, McNemar `p = 1.0`). Both zero-backprop Tier 1 methods (`CLOSED_FORM_ΔW`: `3.3%`, `ROUTED_FACT_BANK`: `1.6%`) collapse at noise floor.
   - **Gate 2 Part A (2-Hop Policy Swap):** Pure zero-backprop sequential closed-form `ΔW` (`Tier 1`) scores **`0.0%` joint swap EM** and **`8.6%` joint swap FC** (**FAIL**).
2. **Gate 2 Part A (2-Hop Policy Swap — Teacher-Distilled Sidecar, FAILED Pre-Registered EM Gate, 2-Way FC Metric Defect):**
   - **Metric validity warning (Audit F05):** The originally reported **`93.0%` joint swap FC** used a 2-way same-row metric (`val_L` vs `val_R` only) that **can be aced (`100%`) by a query-independent column preference without solving either hop**. That metric has been patched in `experiments/v3_gate2_skill_and_sql.py` to score against all 8 candidate leaves.
   - **Actual behavioral result:** Trained via **AdamW teacher distillation (`14` epochs, `336` gradient steps per policy; zero human ground-truth labels, NOT zero training/SGD)**, the `3.68 MB` BF16 `Removable2BlockSidecarOrgan` with mid-layer hint (`Tier 2A`) reached **`11.3%` (`29/256`) joint swap EM** — **failing** the pre-registered `>= 70%` EM / `>= 30 pp` shuffle-margin gate (`gate2a_pass = false`). Its `+3`-hit difference vs. Rank-16 Teacher-LoRA (`26/256 = 10.2%` at `6.49 MB`, trained `10` epochs at `lr = 1e-3` vs sidecar `14` epochs at `lr = 3e-3`) is **not statistically significant** (best possible exact McNemar `p >= 0.25`); the confirmed advantage is smaller byte footprint (`57%` of LoRA), not an established quality win.
   - **Hint ablation & control limit (Audit F13):** Removing the mid-layer hint (`Tier 2B` Output-Only Control) dropped joint swap EM from `11.3%` (`29/256`) to `0.4%` (`1/256`), showing the auxiliary mid-layer loss materially affects training. However, `Tier 2C` shuffled both hints AND teacher targets simultaneously (no fixed-output shuffled-hint or policy-mean control), so item-specific cross-model computation transfer is not isolated from policy steering or regularization.
3. **Where Reversible Teacher-Distilled LoRA Wins (Knowledge Enrichment):**
   - On **Gate 1 (OurAirports)**, **Tier 3 (`TEACHER_LORA_R16`, `4.13 MB`, `3.5s` on L4)** hits **`91.8%` seen `p0`**, **`63.9%` unseen phrasings (`p1/p2`)** (McNemar vs Tier 2 `p = 6.6e-24`), **`22.9%` both / `81.3%` either** on unseen 2-endpoint routes, only **`2/346` locality facts broken (`99.4%` retention)**, and exact `torch.equal` rollback.
4. **Verified Real-World Skill Headroom (Text-to-SQLite):**
   - Across **80 live `sqlite3` queries × 8 unfamiliar multi-table schemas**, `Qwen2.5-7B-Instruct` hits **`88.8%` (`71/80`)** execution match (`97.5%` syntax valid) vs `SmolLM2-1.7B-Instruct` at **`41.3%` (`33/80`)** (`61.3%` syntax valid) — a **`+47.5 pp` real-world skill headroom gap**.

---

## Gate 1: Real Large → Small Knowledge Enrichment (OurAirports)

Source artifact: [`results/results_v3_gate1_airports.json`](../results/results_v3_gate1_airports.json)  
Screening: `1,000` live OurAirports → `196` Donor-7B `3/3` exact IATA → `55` shared-known + `85` strict blindspots (`24 dev`, `61 eval`) + `48` unseen 2-endpoint routes + `380` certified locality facts (`346/380` base-correct).

| Arm | Seen `p0` (`n=61`) | Unseen `p1/p2` (`n=122`) | Unseen Routes Both / Either (`n=48`) | Locality (346 base-OK) | Size (BF16) | Wall Time | `torch.equal` Rollback |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Base `SmolLM2-1.7B-Instruct` | `1.6%` (`1/61`) | `0.0%` (`0/122`) | `2.1%` / `14.6%` | `0/346` (`100.0%`) | — | — | — |
| **Gate 1a** (Linear map `N=3,945`, `cos=0.976`) | `1.6%` (vs `1.6%` self / `3.3%` perm) | — | — | — | — | — | — |
| **Tier 1A** `CLOSED_FORM_DELTA_W` (0-SGD) | `3.3%` (`2/61`) | `2.5%` (`3/122`) | `0.0%` / `0.0%` | `6/346` (`98.3%`) | `33.55 MB` | `11.0s` | `True` (`0.0`) |
| **Tier 1B** `ROUTED_FACT_BANK` (0-SGD, `τ=0.743`) | `1.6%` (`1/61`) | `0.0%` (`0/122`) | `0.0%` / `10.4%` | **`0/346` (`100.0%`)** | `0.72 MB` | `10.7s` | `True` (`0.0`) |
| **Tier 2** `ROUTED_BANK_TINY_POLISH` (25 steps) | `4.9%` (`3/61`) | `0.0%` (`0/122`) | `0.0%` / `12.5%` | **`0/346` (`100.0%`)** | `0.72 MB` | `1.9s` | `True` (`0.0`) |
| **Tier 2 Ctrl** `RECIPIENT_ONLY_CTRL` (25 steps) | `4.9%` (`3/61`) | `0.8%` (`1/122`) | `2.1%` / `14.6%` | **`0/346` (`100.0%`)** | `0.72 MB` | `1.6s` | `True` (`0.0`) |
| **Tier 3** `TEACHER_LORA_R16` (45 steps, L12–18) | **`91.8%` (`56/61`)** | **`63.9%` (`78/122`)** | **`22.9%` / `81.3%`** | `2/346` (`99.4%`) | `4.13 MB` | `3.5s` | `True` (`0.0`) |

---

## Gate 2 Part A: Counterfactual 2-Hop Routing Policy Swap (Zero Ground-Truth Labels)

Source artifact: [`results/results_v3_gate2_skill_and_sql.json`](../results/results_v3_gate2_skill_and_sql.json)  
Protocol: `192` train / `256` sealed test shuffled 4-row 2-hop tables (`53` train / `54` test strictly disjoint key words). Donor 7B receives 2-shot `"FIRST"` (Policy L) vs `"SECOND"` (Policy R) prompts; Recipient `SmolLM2-1.7B` receives the **identical 0-shot instruction-free prompt** for both policies. Contrast bridges (in-sample **training fits** on the 192 training items, NOT held-out preservation): `L14→L12` (`train_mean_cos=0.991, train_r2=0.975`), `L21→L18` (`train_mean_cos=0.990, train_r2=0.967`). Note: the `FC` columns below reflect the legacy 2-way same-row metric (`val_L` vs `val_R`), which does not test hop 1 or query reading (Audit F05).

| Arm | `L_em` | `R_em` | **Joint Swap EM** | `L_fc` | `R_fc` | **Joint Swap FC** | Size (BF16) | Wall Time | Hard Bypass / Rollback |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Donor `Qwen2.5-7B` (2-shot ceiling) | `16.0%` | `34.8%` | `11.7%` | `94.9%` | `95.3%` | `90.2%` | — | — | — |
| Base `SmolLM2-1.7B` (0-shot, no organ) | `0.4%` | `0.0%` | `0.0%` | `78.9%` | `19.9%` | — | — | — | — |
| **Tier 1** `CLOSED_FORM_SEQ_DELTA_W` (0-SGD) | `1.2%` | `0.0%` | `0.0%` | `80.1%` | `23.1%` | `8.6%` | `67.11 MB` | `36.3s` | `—` / `True` |
| **Tier 2A** `SIDECAR_2BLOCK_WITH_HINT` | **`23.1%`** | `26.2%` | **`11.3%`** | `95.3%` | **`97.7%`** | **`93.0%`** | **`3.68 MB`** | `68.7s` | **`True` / `True`** |
| **Tier 2B** `SIDECAR_OUTPUT_ONLY_CTRL` | `22.3%` | `1.2%` | `0.4%` | `97.7%` | `54.7%` | `53.1%` | `3.68 MB` | `67.9s` | `True` / `True` |
| **Tier 2C** `SIDECAR_SHUFFLED_CTRL` | `2.3%` | `2.0%` | `0.0%` | `48.8%` | `50.4%` | `28.5%` | `3.68 MB` | `67.6s` | `True` / `True` |
| **Tier 3** `TEACHER_LORA_R16` | `19.9%` | **`32.0%`** | `10.2%` | **`98.4%`** | `96.5%` | **`94.9%`** | `6.49 MB` | `49.1s` | `—` / `True` |

> **Honest Audit Caveats (Gate 2 Part A):**
> 1. **Donor 7B 1-token non-CoT EM ceiling (`11.7%`):** On 4-row shuffled 2-hop tables without chain-of-thought, Donor 7B strongly separates Policy L vs Policy R (`90.2%` joint FC), but distractor rows in the same column cap raw next-token EM at `11.7%` — making the pre-registered `>=70%` joint EM bar impossible even for the 7B donor. Evaluating on 2-row tables (or CoT/multi-token output) is needed to lift the Donor EM ceiling above `80%`.
> 2. **Part A 100-fact locality harness:** `base_recipient.locality_100` scored `0/100` on raw `dataset.json` `direct_prompt` completions under `SmolLM2-1.7B-Instruct`, so Part A's locality column is uninformative (`0/0`) and must be replaced with Gate 1's certified 380-fact locality suite (`346/380 = 91.1%` base-correct) in the next iteration.

---

## Gate 2 Part B: Real Text-to-SQLite Headroom (80 Live Queries × 8 Unfamiliar Schemas)

| Model | Live `sqlite3` Execution Match | SQL Syntax Valid | Per-Schema Min / Max | Wall Time |
| :--- | :---: | :---: | :---: | :---: |
| **`Qwen/Qwen2.5-7B-Instruct`** | **`71/80` (`88.8%`)** | **`78/80` (`97.5%`)** | `70%` / `100%` (3/8 schemas at `100%`) | `202.1s` |
| **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** | **`33/80` (`41.3%`)** | **`49/80` (`61.3%`)** | `20%` / `70%` | `91.0s` |
| **Headroom Gap (7B − 1.7B)** | **`+47.5 pp`** | **`+36.3 pp`** | — | — |

Per-schema execution match (`7B` vs `1.7B`): `logistics_shipments` `10/10 vs 4/10`, `university_courses` `10/10 vs 2/10`, `flight_manifests` `10/10 vs 3/10`, `library_loans` `9/10 vs 3/10`, `sensor_telemetry` `9/10 vs 6/10`, `hospital_appointments` `8/10 vs 4/10`, `solar_power_grid` `8/10 vs 7/10`, `retail_inventory` `7/10 vs 4/10`.
