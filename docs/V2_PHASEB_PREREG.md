# PlatonicSwap `v2` Phase B — One-Page Pre-Registration

> **Date:** 2026-11-26 | **Frozen `v1.0` Anchor:** tag `v1.0` (`6102bf8`), `platonic_organ_engine.py` (`SHA-256: 22ef57c9…`) untouched.
> **Design Basis:** Step 0 (`results_v2_step0_rw_diagnostic.json`), Falsifier A (`results_v2_falsifierA_rope_torus.json`), Falsifier B (`results_v2_falsifierB_carry_transplant.json`), and Fable 5.2 Max's adversarial pressure test.

---

## 1. Measured Ceilings & Certified Sample Sizes (Locked Before Phase B Code)

| Metric | Measured Ceiling (`gold × NAT_W` from Step 0) | Phase B Dev / Cap-3 Sample Size | Certification Power (One-Sided 95% Clopper-Pearson) |
| --- | --- | ---: | --- |
| **Direct Full-FC** | `95.8%` Smol · `54.2%` Gran · `97.9%` Falc · `100.0%` Q1.5 | **`n ≥ 96`** pooled (`48/rule`, genus-disjoint) | Standard error `≤ 3.5 pp` |
| **Direct EM** | `89.6%` Smol · `35.4%` Gran · `68.8%` Falc · `83.3%` Q1.5 | **`n ≥ 96`** pooled | Gate relative to `gold × NAT_W` (`≥ 0.8 ×` ceiling) |
| **Ripple Full-FC** | `92.7%` Smol · `72.9%` Gran · `83.3%` Falc · `74.0%` Q1.5 | **`n ≥ 192`** (4 attrs × 48) | Reported alongside Direct |
| **Ripple EM** | `46.9%` Smol · `41.1%` Gran · `39.6%` Falc · `29.2%` Q1.5 | **`n ≥ 192`** | **Reported without gating** |
| **HardNeg False Flips (FC)** | `BASE = 0–4%` (`n=24` only bounds `11.7%`!) | **`n ≥ 96`** (`≥ 93` required) | `≤ 1/96` certifies **`≤ 4.9%`** (beats `5%` bar!) |
| **Out-of-Domain Locality** | `BASE = 100%` (`n=196` only bounds `1.5%`!) | **`n ≥ 300`** base-correct facts | `0/300` certifies **`≤ 1.0%`** (meets `≤ 1%` bar!) |

---

## 2. Anti-Leakage & Split Discipline
1. **Cap-2 `BIO-RIPPLE` is burned** (Step 0 exposed per-item telemetry): NEVER used as a tuning gate; run at most once at the end as a regression check.
2. **Fresh genus-disjoint Dev Split:** Donor-side keys/thresholds fit on `≥ 200` genus-disjoint species per class (strictly disjoint from Cap-1, Cap-2, and future Sealed Cap-3; 0 overlap asserted in unit tests).
3. **Recipient Holdout:** Tune Track 1 ONLY on **`smollm2-1.7b`** and **`falcon3-3b`** (`≤ 3` rule-set scalar hyperparameters: donor-matched threshold quantile, capped dose scale `α`, fixed Fisher projector rank `k`). **`granite-3.1-2b`** and **`qwen2.5-1.5b`** are held out and run ONCE with frozen rules.
4. **Sealed Cap-3 Lock:** Not executed until Track 1 Dev Gate passes cleanly. Must include at least one novel class pair not in Cap-2, with benchmark, engine (`platonicswap/platonic_organ_engine_v2.py`), and SHA-256 hash committed and pushed as a git tag *before* first execution.

---

## 3. Mandatory Free Checks on Existing Caches (Executed FIRST, Before New Fits)
1. **Free Check 1 — Split-Half Write Reliability (`r = cos(v_1, v_2)`):** Fit the concept write bridge on two disjoint halves of the cached 1,200 concept pairs (`universal_write_concept_*.pt`) and compute `r = cos(v_1, v_2)` on Cap-2 `v_A`, inferring `N_0/N = (1/r - 1)/2` and asymptotic ceiling `c_∞ = c(N) * sqrt(1 + N_0/N)`.
   - **Pre-registered Branch Rule:** If **`r > 0.20`** (meaning `10×` generic data is asymptotically capped below `~0.47`), **DROP** the `10×` generic corpus arm and switch directly to a **domain-dense contrast write corpus** + **Fisher readout-gradient projector `v' = P_k J v`** (rank `k ∈ {64, 128}` fixed in advance; never output-metric Fisher weighting, which is mathematically inert on unconstrained `J`).
2. **Free Check 2 — Per-Position `φ` Telemetry & Gold Scale Sweep:** Log `φ` split across (i) species tokens, (ii) non-species prompt tokens, and (iii) generated tokens, PLUS sweep `α ∈ {1, 2, 3, 4, 6}` at gold species position with `PUSH_W` to verify the dose-stacking model (`D = Σ φ·ρ`).
3. **Free Check 3 — Top-3 HardNeg Inspection:** Log exact top-3 HardNegs per rule×recipient cell and their rank inside the positive distribution.

---

## 4. Track 1 — 7-Arm Pinned Dev Matrix & Pre-Registered Pass Bars

| Arm | Read | Write | Isolates | Dose-Model Prediction (SmolLM2) |
| ---: | --- | --- | --- | --- |
| **0** | none | none | BASE HardNeg & 300-fact locality floor | Dir `0%`, HN `~0–4%`, Loc `0/300` |
| **1** | `v1.0 PUSH` | `v1.0 PUSH` | `v1.0` baseline on fresh dev split | Dir FC `~40%`, EM `~10%`, HN FC `~10%` |
| **2** | `gold` (species token) | `native` | Single-site ceiling | Dir FC `≥95%`, EM `~85–90%`, Rip FC `~90%` |
| **3** | `v1.0 PUSH` | `native` | Read reference (exposes HardNeg leak) | Dir FC `~98%`, EM `~80%`, HN FC `~55–75%` |
| **4** | **`NEW_GATE`** *(2-layer LDA + steep gate + prompt latch)* | `native` | **Isolates Read Upgrade** | Dir FC within 5 pp of Arm 2; HN `≤1/96`; Loc `0/300` |
| **5** | `gold` | **`NEW_WRITE`** *(domain-dense + `P_k J v`)* | **Isolates Write Upgrade** | Dir FC `≥85%`, `EM/FC ≥ 0.75` |
| **6** | **`NEW_GATE`** | **`NEW_WRITE`** | **Full Zero-SGD Candidate** | Meets both Read & Write bars |
| **6L** | **`NEW_GATE`** | `v2-Lite v*` (`≤25` frozen-B steps) | Upper-bound control; MUST log `cos(v*, v_push)` | Dir FC `≥90%`, EM `≥80%` |

- **Pass Bars:**
  - **Read (Arm 4):** `≤ 1/96` HardNeg FC flips above Arm 0, Direct FC within `5 pp` of Arm 2, Direct EM `≥ 0.9 ×` Arm 2, `0/300` locality breaks.
  - **Write (Arm 5):** Direct FC `≥ 85%` and `EM / FC ≥ 0.75`.
  - **Candidate (Arm 6):** Direct EM `≥ 0.8 ×` Arm 2, Direct FC `≥ 85%`, `≤ 1/96` HardNeg above Arm 0, `0/300` locality breaks.

---

## 5. Track 2 — 15-Minute Pre-Flight Only (Gates K0, K1 & Headroom Census)
Full Falsifier C (Arms A1–A6 + P, 256 seqs/lag, Falcon3 lag×arm interaction `[R(A1)-R(A2)]_{Δ=64} - [R(A1)-R(A2)]_{Δ=4} ≥ 0.20`) is **strictly gated** behind passing ALL THREE pre-flight checks on Colab L4 #2:
1. **Headroom Census:** Mean-ablate recipient native induction heads until repeated-random-token copy top-1 drops to `10–30%`. *(If it won't drop below `50%`, pivot to donor-only 2-head-slot SGD skip-induction `… A B C … A → C`, leaving Recipient B 100% zero-SGD.)*
2. **Gate K0 (Algebra, minutes):** Transplanted head-1→head-2 composition score `≥ 0.5×` donor AND mean-centered active-prompt OV copying rate `≥ 0.5×` donor, enforcing the exact cocycle equality constraint **`J_2 U = J_1 U`** on `U = Im(W_OV)` of head 1 (plus exact `1/√d_h` `√2` compensation for `d_h = 128 → 256`).
3. **Gate K1 (Positive Control):** Re-inserting recipient's own ablated heads through the exact same hooks restores **`≥ 90%`** of the ablation gap, and rotated self-transplant restores **`≥ 80%`** (plus exact batch-matched `unhook` bit-equality check!).
