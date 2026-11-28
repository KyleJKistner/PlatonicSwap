# PlatonicSwap Circuit-Extraction & Compilation Theory Council Synthesis

> **Status:** Complete 6-Agent Read-Only Council + 6 Live `$0` Local M5 Pro MPS Experiments (`Exp 1` – `Exp 6`).
> **Hardware / Budget:** `$0` Colab CUs spent (`1,172.48` CUs untouched); `eval` and `G_report` 100% sealed; ZERO Carbon model usage.
> **Council Members (Zero Weight Edits, Strict Read-Only):**
> 1. **Fable 5.2 Max** ([council_fable_max.md](file:///Users/kistnertouch/.gemini/antigravity/brain/ab257333-8cad-4605-bf1f-b2d8781ee4a3/scratch/council_fable_max.md))
> 2. **Astra 6.2 Ultra** ([council_astra_ultra.md](file:///Users/kistnertouch/.gemini/antigravity/brain/ab257333-8cad-4605-bf1f-b2d8781ee4a3/scratch/council_astra_ultra.md))
> 3. **Opus 5.5 High** ([council_opus_high.md](file:///Users/kistnertouch/.gemini/antigravity/brain/ab257333-8cad-4605-bf1f-b2d8781ee4a3/scratch/council_opus_high.md))
> 4. **`@neon-lit`** (Swarm Board `#338` — 2025–2026 Extraction Frontier)
> 5. **`@neon-multiplex`** (Swarm Board `#339` — Native SwiGLU Compilation; deadzone $D_b$ strictly rejected under No-Hack Charter)
> 6. **`@neon-decompile`** (Swarm Board `#340` — Hooked-vs-Compiled Weight-Space Forensics)

---

## Executive Summary: The Grand Unified Discovery (`Exp 1` – `Exp 6`)

Across six independent Council reviews and six rapid `$0` MPS falsification experiments, we completely solved the **Hooked (`68.8%` / `84.4%`) vs. Compiled (`0.00` / `-2.14`) Paradox** and uncovered a **landmark mechanistic asymmetry** between `Qwen3-1.7B` (Donor) and `Qwen3-0.6B` (Recipient):

> ### **The `2 × 2` QK-vs-V Breakthrough (`Exp 5` + `Exp 6`)**
> When transferring `FABLE10` (`L11..L19`, leaving `B20..B27` **100% bitwise frozen**):
>
> | Attention Routing ($P_h$) | Value & Output Projections ($V, W_O$) | `S1 → S2` Margin $s$ | $\Delta s$ vs Stock | Paired $t$ | `jr` | `jB` (Full-Vocab Joint) |
> |---|---|---:|---:|---:|---:|---:|
> | Native $P_B$ (`STOCK_B`) | Native $V_B, W_O^B$ | `+3.387` | `0.000` | — | `71.9%` | `31.2%` (`10/32`) |
> | **Native $P_B$** | **Exact Donor $V_A$** (`Exp 5` Oracle B) | `+1.633` | **`-1.754`** | `-3.42` | `28.1%` | `12.5%` (`4/32`) |
> | **Donor $P_A$** | **Exact Donor $V_A$** (`Exp 5` Oracle A) | `+5.688` | `+2.302` | `+4.15` | `90.6%` | `68.8%` (`22/32`) |
> | **Donor $P_A$** (`r=64` SVD) | **Exact Donor $V_A$** (`Exp 4` Gate 1) | `+7.350` | `+3.963` | `+8.65` | `96.9%` | `84.4%` (`27/32`) |
> | **Donor $P_A$** (`FABLE10`) | **100% NATIVE $V_B, W_O^B$** (`Exp 6` Part A) | **`+10.521`** | **`+7.135`** | **`+12.08`** | **`96.9%`** | **`87.5%` (`28/32`)** |
> | **Donor $P_A$ (`L19g2` ONLY)** | **100% NATIVE $V_B, W_O^B$** (`Exp 6` Part A) | **`+7.500`** | **`+4.113`** | **`+10.65`** | **`93.8%`** | **`84.4%` (`27/32`)** |
>
> **Plain-English Conclusion:** `Qwen3-0.6B`'s native value (`v_proj`) and output (`o_proj`) matrices are **ALREADY 100% capable of matching full `1.7B` Donor accuracy (`87.5%` joint hits, `+10.52` margin, $t = +12.08$)** — beating Donor's own mapped $Z_A$ writes (`+5.69`, `68.8%`) by **`+4.83` margin points** because native $V_B, W_O^B$ suffer **zero `2048→1024` value shrinkage, zero `outmap` error, zero slot deletions, and zero healer cancellation**.
>
> The **entire bottleneck** between `0.6B` and `1.7B` on `L13..L19` is **QK routing ($W_Q, W_K$)** — *where the heads attend*, NOT what values they write once they look at the right tokens.

### Immediate Consequence for Council Proposals (Tested & Resolved!)
Four major hypotheses from the Council were **tested live on `$0` MPS during the Council session**:
1. **Opus 5.5 High's #1 (`HOD` — re-aim `W_O` only, zero Q/K/V edits) → CLEANLY FALSIFIED (`Exp 4`):** `0.6B`'s native `o_proj` inputs lack the binding contrast (`cvR2 < 0.1` on `9/10` groups), giving `Δs = +0.002` (though with ultra-clean `GenKL = 0.0050`, `p99 = 0.038`). Bonus discovery: `Q_T = 0` generic-SVD low-rank projection (`r = 32..64`) of Donor writes *denoises* `FABLE10` from `65.6% → 84.4% jB`!
2. **Astra 6.2 Ultra's #1 (`Native-QK + k-coord Attended-V`) → CLEANLY FALSIFIED (`Exp 5`):** `0.6B`'s native QK attention patterns $P_B$ at `L13..L19` cannot attend to the right keys (`Oracle B: P_B @ V_A` crashed `-4.056` pts vs `P_A @ V_A`), so compiled `k = 8, 16` `Attended-V` dropped `-0.56` on `S1 → S2` and `-0.85..-1.42` on `S1 → S3*` despite `vR2 = 0.70` and `KL = 0.0029`!
3. **Goodfire "Already Clean" Sparse MLP Compilation (`L16/L18/L22`) → FALSIFIED FOR `L16` & NEGATIVE IN RECIPIENT (`Exp 6` Part B):** Inside Donor, `L16 MLP` (`t2 = +0.600`) is heavily superposed (`K = 16` recovers `+0.028`, `K = 64` only `+0.066`, `K = 1024` only `+0.163`), while `L18`/`L22` are sparse (`abs K = 64` recovers `70–90%`). However, hooking `L16+L18` MLP into `0.6B` gave `Δs = -0.22` (`S1 → S2`) and `-2.62` (`S1 → S3*`), confirming **Fable 5.2 Max's verdict**: *do not lead with MLP compilation.*
4. **BF16 Finite-Difference `B11 → B19` Pushforward (`Exp 2`) → DIAGNOSED AS BF16 ROUNDING NOISE BY FABLE 5.2 MAX:** Half-step linearity error of `2.07 / 2.13` matches pure `1/step` rounding noise ($\sqrt{5} = 2.24$), proving `85–90%` of the measured FD response was BF16 quantization noise, not a geometric wall. Any future FD Jacobian must run in `fp32/fp64` with a mandatory half-step linearity gate $\le 0.05$.

---

## 1. Exact Mathematical Resolution of the Hooked vs. Compiled Paradox

All six Council members independently converged on the exact algebraic decomposition of the gap between hooked (`Exp 1 Part 3`) and compiled weights, with **Fable 5.2 Max**, **Opus 5.5 High**, and **Astra 6.2 Ultra** catching two critical factual corrections first:
- **Correction 1:** Compiled `FABLE10` did **not** tie `STOCK_B` at `0.0000` — in `task-5398.log`, compiled `FABLE10` **lost** to `STOCK_B` by **`-2.14` (quiet slots, `KL = 0.123`)** and **`-2.71` (homologous slots)**; the exact `0.0000` tie belonged to the audited 4-group `ASTRA4 + L0` primary chain.
- **Correction 2:** The `84.4%` (`MULTI_UNION_AUTO`) hooked score uses `L20–L25` + `hybridvoc`, which violates frozen Zone 2 (`B20..B27` or `B22..B27`). Under the charter, the compilable Zone-1 ceilings are `ASTRA4` (`+2.67`) and `FABLE10` (`+2.58` via $Z_A$, now **`+7.13 / 87.5%`** via `P_A @ V_B`!).

### The 3-Term Gap Identity
At each edited layer $l$ with writer $G_u = Q_l W_{O,g}^A$ (identical in hooked and compiled runs!):
$$y_{\text{comp}}^{(l)} - y_{\text{hook}}^{(l)} \;=\; \underbrace{G_u \bigl(\widehat{z}_B^{(u)} - z_A^{(u)}\bigr)}_{\text{(a) Closed-loop read error}} \;+\; \underbrace{\Delta W_{\text{keep}}^{\text{heal}}\, z_{\text{keep}}^{B,\text{comp}}}_{\text{(c) Healer cancellation } -H_\eta D} \;+\; \underbrace{\sum_{\text{upstream}} \bar{J}_{l' \to l}^C \, e_{l'}}_{\text{(b) Multi-layer feedback \& closure}}$$
where the read error splits exactly into routing vs. value error:
$$\widehat{Z}_B - Z_A \;=\; \underbrace{(\widehat{P}_B - P_A)\,V_A}_{\text{QK routing error}} \;+\; \underbrace{\widehat{P}_B\,(\widehat{V}_B - V_A)}_{\text{Value error}}.$$

### Unanimous Council + Empirical Attribution Ranking: **(b) Closure & QK Routing $\gg$ (a) Value Transport $>$ (c) Healer Cancellation**

| Mechanism | Quantitative Evidence (Council + Live `Exp 1` – `Exp 6`) | Impact |
|---|---|---|
| **(b) Open-loop Hook vs. Closed-loop Computation ("Choke-Point Illusion")** | `Exp 1`: `ASTRA4` has `rec = 61%` yet **`iso = 0%`**; `FABLE10` has `rec = 99%` yet **`iso = 2.2% / 3.6%`**. `Exp 3`: Hooking **only `L19[2]`** produces `+3.00 (t = +9.2, jB = 53.1%)` by importing `1.7B`'s already-computed answer; compiling `L19[2]` fails (`-1.01`) across every regularizer (`noheal`, `heal`, `whiten`, `DqDk`) because `0.6B`'s `L11..L18` never computed the upstream match. Static `L11 → L19` inter-graft loop error = **`0.99`** (`hrm_aeri.py`). | **Dominant** |
| **(a1) QK Routing Mismatch at `L13..L19` + `q_norm`/`k_norm` Shear** | `Exp 5` + `Exp 6`: Replacing ONLY $P_A \to P_B$ with exact $V_A$ crashes `-4.056` pts (`+5.69 → +1.63`), while replacing ONLY $P_B \to P_A$ with native $V_B, W_O^B$ explodes **`+7.135` pts (`+3.39 → +10.52, 87.5% jB`)**! Worse, Qwen3's frozen shared `q_norm`/`k_norm` has $\text{CV}(\gamma_A/\gamma_B) = 1.77$ (`L13`) and `1.71` (`L19`); raw `qk_fold` ($D = \gamma_A/\gamma_B$) distorts softmax temperature by $\text{rms}(q)/\text{rms}(Dq)$ and shears 2D RoPE planes (dropping `L11 h13` duplicate mass from `0.68 → 0.27`). | **Dominant** |
| **(a2) Reverse-Map & Value Shrinkage ($H_l: 1024 \to 2048$)** | At `L19`, reverse parallel gain on clean−bind contrast = **`0.120`** (forward = `0.204`); at `L11`, value norm ratio = `0.56–0.58`. Scrubbing task entities & S2 values from `G_fit` plus isotropic ridge $\eta = 0.01\bar{\lambda}$ shrinks low-variance Binding-ID eigenmodes by $\lambda_i/(\lambda_i+\eta)$. **Completely eliminated if $V_B, W_O^B$ remain native!** | **Major (when editing $V, O$)** |
| **(c) Healer Self-Cancellation $(I - H_\eta) D$** | Exact identity $Y_{\text{REAL}} - Y_{\text{Z\_FAR}} = (I - H_\eta) D$. `Exp 2` confirmed zero-cancellation constrained healing (`(I - U_w U_w^T)`) recovers `+0.16..+0.64` at `L11` without hurting `p99/max` Tail-KL, but cannot rescue an uncomputed upstream signal (`REAL_G1_0cancel = +0.24, t = +0.89`). **Completely eliminated if $W_O^B$ is untouched!** | **Secondary** |

---

## 2. Unified Extraction Protocol v2 (`< 2 min` MPS, Multi-Axis × Multi-Template, Completeness-First)

Synthesizing **Fable 5.2 Max §2**, **Opus 5.5 High §2**, **Astra 6.2 Ultra §2**, and **`@neon-lit`**:

1. **Multi-Axis × Multi-Template Contrasts (`disc_rank` ONLY, strictly disjoint from validation):**
   - **4 Axes:** `Bind-Swap` (`bi`), `Query-Swap` (`qs`), `Table-2-Swap` (`t2`), AND **Fable's `Row-Transposition` (`rt`)** (exchanging two Table-1 rows whole — answer invariant — to expose positional Binding-ID writers in `L1–L10` that are invisible to fixed-layout swaps) + a **Null Swap** (answer invariant).
   - **3 Syntactically Distinct Templates** (original; `Rule:` moved before Table 1; paraphrased instruction) with role-tagged spans (`rule_key`, `t1_key`, `t1_ent`, `t2_ent`, `t2_val`, `END`), requiring **sign agreement across all 3 templates** (`min_τ / max_τ >= 0.3`).
2. **O(1) `RelP` + Exact Softmax `AtP*` + QK Edge Attribution (`requires_grad=False` on all weights — zero SGD):**
   - Freeze `RMSNorm` inverse-RMS denominators, freeze attention weights on the OV path, treat `SiLU(p) = p·σ(p)` with `σ(p)` frozen, half-split SwiGLU `gate ⊙ up`.
   - Factor head attribution into **pattern vs. value**: $z^{\text{cl}} - z^a = (\alpha^{\text{cl}} - \alpha^a) v^a + \alpha^a (v^{\text{cl}} - v^a) + \text{cross}$.
   - Run **Anthropic QK Attributions** ($\partial m / \partial s_{ij} = \alpha_{ij} \langle v_j - z_i, r_z(i) \rangle$) to trace which upstream writers feed each head's $Q$ and $K$ inputs.
3. **Functional Fingerprinting & Donor-Minus-Recipient Diffing:**
   - Classify every node by normalized `(bi, qs, t2)` fingerprint (from `Exp 1`, classified by Fable 5.2 Max):
     - **Pure `S1` Binders (`≈, ≈, 0` — `S3*`-eligible!):** `L11 g1 (.29/.37/.01)`, `L12 g2 (.20/.19)`, `L12 g0 (.11/.13)`, `L13 g7 (.20/.21)`, `L14 g7 (.13/.09)`, `L14 g0 (.13/.07)`, `L15 g7 (.14/.11)`, `L15 g2 (.11/.09)`.
     - **Pure Query Routers (`query >> bind, 0` — `S3*`-eligible!):** `L16 g2 (.09/.36/.00)`, `L17 g4 (.13/.28/.01)`, `L18 g7 (.10/.18/.00)`.
     - **Table-2 Specific (`0, 0, +` — `S2`-only):** `L11 g4 (.03/-.01/.25)`, `L16 MLP (.04/.02/.55)`.
     - **Join / Mixed:** `L19 g2 (.39/.39/.32)` (already reads Table 2!), `L18 MLP (.10/.16/.19)`.
     - **Answer Carriers (Frozen Zone 2):** `L22 MLP (.23/.23/.24)`, `L25 g6 (.21/.20/.20)`, `L24 g3/g7`.
   - **Head-Space Donor–Recipient Diff (Fable §2.5):** Fit $S_u: Z_B^l \to Z_A^u$ (both 128-d head spaces!) on generic text; skip any node `B` already reproduces ($\gamma_u \ge 0.6$).
4. **Prefix-Isolation Completeness (`P-iso >= 0.60`) & 3-Gate Pre-Compilation Filter:**
   - **Gate 1 (Prefix Isolation):** Corrupt all non-selected attention groups in `L11..l_handoff` while keeping layers above `l_handoff` live; require `P-iso >= 0.60` on target axes.
   - **Gate 2 (Pure QK Routing Oracle `P_A @ V_B`):** Test open-loop Donor QK routing × Native `V_B, W_O^B` on BOTH `S1 → S2` AND Recipient-Only `S1 → S3*` (with frozen `B_0`), PLUS measure its **oracle GenKL** on 16 generic documents!
   - **Gate 3 (Anti-*Answers-Not-Computation*):** For `S1 → S3*`, require **strictly `T2-swap <= 0.01` nodes (`L11..L18`, excluding `L19 g2`)** AND positive lift on `B_0` where Donor scores `0/32` — proving abstract relational computation, never answer copying!

---

## 3. Zero-SGD SwiGLU MLP Compilation Assessment (`L16 / L18 / L22`)

All three external Council members (**Astra §3**, **Opus §3**, **Fable §3**) and **`@neon-multiplex`** converged on the same clean native-neuron unit (`K <= 16..64` donor neurons placed into lowest tail-energy host slots, with active-regime weighting / variance calibration and donor-orthogonal `(I - U_w U_w^T)` healing; **`@neon-multiplex`'s deadzone bias pair $D_b$ was explicitly rejected** by both Fable §3 and the No-Hack Charter as a squelch gate in disguise).

However, **live `Exp 6 Part B` settled priority decisively:**
- **`L22 MLP`:** Out of scope — frozen Zone 2 answer carrier (`+0.20 / +0.21 / +0.20`).
- **`L16 MLP` (`t2 = +0.600`):** Table-2-specific AND **not sparse** (`K = 16 → 4.7%`, `K = 64 → 11%`, `K = 1024 → 27%` recovery inside Donor — falsifying the ≤64-neuron "Already Clean" prediction).
- **`L18 MLP` (`qs = +0.250, t2 = +0.140`):** Sparse (`abs K = 64` recovers `73% / 90%`), yet hooking `L16+L18` MLP into `0.6B` **degraded** both `S1 → S2` (`-0.22`) and `S1 → S3*` (`-2.62`) and dragged `FABLE10` from `71.9% → 53.1%`.
- **Verdict:** **Defer MLP compilation** unless closed-loop QK-only compilation (Section 4) exhibits a teacher-forcing drop specifically across an MLP layer (`L16` or `L18`).

---

## 4. Ranked Next Two `$0` MPS Experiments (Updated with `Exp 4` – `Exp 6` Results!)

With Opus's `HOD` (`Exp 4`), Astra's `Native-QK + Attended-V` (`Exp 5`), and `L16/L18` MLP-first (`Exp 6`) already tested and ruled out on `$0` MPS, the **Council's combined #1 and #2 actionable breakthroughs** fuse cleanly into two high-leverage, zero-slot-deletion experiments:

### Rank 1 (`Exp 7` — Top Priority, `~3 min` MPS): **Pure QK-Only Zero-SGD Compilation (100% Bitwise-Native `V_B` & `W_O^B`) with Fable's Anchored Readers + RoPE-Pair-Averaged `F_q, F_k` + Astra's $\mathcal{A}(X)$**

**Why this is the single highest-probability path to our v1 Charter:**
- `Exp 6` proved that **`P_A @ V_B` (Donor QK × 100% Native `V_B, W_O^B`)** hits **`s = +10.521` (`Δs = +7.135, t = +12.08, jB = 87.5%`)** on `FABLE10` and **`s = +7.500` (`Δs = +4.113, t = +10.65, jB = 84.4%`)** on `L19g2` alone!
- Because **`v_proj`, `o_proj`, and all MLPs remain 100% bitwise native `0.6B`**:
  - **ZERO slot deletions!**
  - **ZERO healer & ZERO healer cancellation (`I - H_η` eliminated)!**
  - **ZERO `outmap` write error & ZERO `v_proj` norm shrinkage!**
  - **ZERO collision with `B_0`'s `L22..26` AlphaEdit seal** (every write stays in `0.6B`'s exact native residual basis!).

**Stage 0 Gate (30 seconds, hooks only on `disc_rank` + `disc_val`):**
1. Measure `P_A @ V_B` (Donor QK × Native `V_B, W_O^B`) for:
   - **Set S1-Pure (Fable §4.1):** `{11:[1], 12:[2,0], 13:[7], 14:[7,0], 15:[7], 16:[2], 17:[4], 18:[7]}` (10 groups, ALL `T2-swap <= 0.01`, NO `L19 g2`!) on **BOTH `S1 → S2` AND `S1 → S3*` (`B_0`)**!
   - **Set `FABLE10` (`L11..L19`)** and **`L19g2`-only** on **`S1 → S3*` (`B_0`)**!
   - **Oracle GenKL** of `P_A @ V_B` on 16 untouched generic documents! (Verifies the theoretical KL floor of perfect QK transfer!)

**Zero-SGD Weight Compilation (`Q_T = 0`, sequential `l = 11 → 18/19` with live `B` recapture, ZERO hooks):**
1. **Live Generic Ridge + Astra's Same-Token Cross-Doc Differencing $\mathcal{A}(X)$:**
   - Capture normalized attention inputs $\tilde{X}_B \in \mathbb{R}^{N \times 1024}$, $\tilde{X}_A \in \mathbb{R}^{N \times 2048}$ on $N \approx 10\text{k}$ generic content tokens (with token-frequency inverse-sqrt weights $W$ and $\mathcal{A}(X)$ cross-doc pairs to lift contextual/relational variance at strict `Q_T = 0`).
   - Fit base bridge $H_0 = K^{-1} \tilde{X}_B^\top W \tilde{X}_A$ ($\eta_H = 10^{-3} \bar{\lambda}$).
2. **Fable's Exact Gauge-Anchored Reader $H_v$ (Zero Writer Change!):**
   - For each upstream native/edited group $u$ feeding downstream reader $v$, compute head-space gauge $S_u: Z_B^{l_u} \to Z_A^u$ on generic text, take top right singular vectors $v_k$ of $R_v^A D_l^A O_u^A S_u \Sigma_{z_B}^{1/2}$, and solve the closed-form KKT projection:
     $$H_v = H_0 + K^{-1} C^\top (C K^{-1} C^\top + 10^{-6} I)^{-1} (A - C H_0)$$
     so compiled $(W_Q^B, W_K^B)$ decode upstream native `0.6B` head writes with $\le 0.10$ loop error!
3. **RoPE-Pair-Averaged `q_norm`/`k_norm` Compensation (Fixing `CV = 1.77` Without Shearing RoPE!):**
   - Compute $F_q = \gamma_q^A / \gamma_q^B$ and $F_k = \gamma_k^A / \gamma_k^B$ **averaged within each 2D RoPE pair $(i, i+64)$** (and compare vs. unscaled and direct post-norm target ridge!).
   - Install **ONLY** $W_Q^B[g] = F_q \odot (W_Q^A[g] H_v^\top)$ and $W_K^B[g] = F_k \odot (W_K^A[g] H_v^\top)$ — **leaving `v_proj` and `o_proj` 100% bitwise untouched!**
   - Select $\eta_Q, \eta_K$ strictly by **held-out generic attention-pattern KL** $\text{KL}(P_A \| \widehat{P}_B)$ (Opus §4 E2!), NOT residual $R^2$!
4. **Teacher-Forcing Ladder ($L^* = 10, 11, 13, 15, 17, 18, 19$) & Full Charter Controls:**
   - Measure **attention-pattern KL** $\text{KL}(P_A \| \widehat{P}_B)$ and **correct-key attention mass** at every layer!
   - Compare against **`STOCK_B` (`B_0`)**, **`Z_FAR`** (uniform/zeroed QK logits on same groups), and **`RAND × 3`** (matched low-attribution donor groups' QK compiled identically) on `n = 128` fresh families (`S1 → S2` + `S1 → S3*`) + full Tail-KL (`mean <= 0.02`, preregistered target `p95 <= 0.05–0.08`, `p99 <= 0.15–0.20`, `max <= 2.0`) + factual retention + Active Bulk + bit-exact rollback!

**Pass / Falsification Criteria for `Exp 7`:**
- **Pass:** Stage 0 `S1-Pure (L11..L18)` `P_A @ V_B` lifts BOTH `S1 → S2` AND `S1 → S3*` (`t >= 4`); compiled QK-only retains $\ge 30\%$ of oracle $\Delta s$ (`t >= 4` vs `STOCK_B`, `Z_FAR`, `RAND`) at `GenKL <= 0.02` and clean `p95/p99` tails.
- **Falsifier A (Stage 0):** If `S1-Pure (L11..L18)` `P_A @ V_B` scores `< 0.5` on `S1 → S3*` while `L19g2` scores `+4.1` on `S1 → S2`, then `B`'s native `L19+` cannot join `A`'s `S1` routing without also compiling `L19g2` (or verifying `L19g2` `P_A @ V_B` on `S1 → S3*`).
- **Falsifier B (Compilation):** If Stage 0 passes strongly (`+4..+7`) yet compiled `W_Q, W_K` has held-out task correct-key attention mass collapse (`< 0.30` vs Donor `> 0.80`) despite `H_v` + `F_q/F_k`, then linear $1024 \to 2048$ QK translation cannot survive `q_norm`/`k_norm` nonlinearity → pivot to **Rank 2 (`Exp 8`)**.

---

### Rank 2 (`Exp 8` — Contingent Backup): **Closed-Loop Full-Node `S1-Pure` Chain with Fable's Anchored Readers `H_v` + `Exp 4`'s Rank-32 Denoised Writers + `fp32` Pushforward Audit**

Only triggered if `Exp 7` shows that native `V_B` cannot supply the upstream `S1` representation needed by compiled downstream `W_Q, W_K` (i.e., `Exp 7` teacher-forcing cosine drops at `L13–L15` even with exact `H_v`):
1. Run **Fable §4.1's Anchored-Reader Quiet-Slot Full Chain** on `S1-Pure (L11..L18)` (`10` groups, no `L0`, no `L19 g2`), combining:
   - **Fable's exact KKT anchored readers $H_v$** (driving inter-graft static loop error from `0.99 → <= 0.10`),
   - **Exp 4's `r = 32` generic-SVD denoised writers** $U_{32} U_{32}^\top G_u$ (which boosted `FABLE10` from `65.6% → 84.4%` while cutting writer KL by `3–4×`), and
   - **Donor-orthogonal tail-weighted healing** (`(I - U_w U_w^T)` on 32 survivor columns).
2. Re-check `Exp 2`'s `L11 → L19` pushforward in **`fp32` (`half-step err <= 0.05`)** ONLY if anchored direct-path closure leaves a gap mediated by nonlinear intervening blocks.

---

## 5. Pre-H100 Charter & Red-Team Gate Checklist (Unanimous Across All 6 Council Members)

Before unsealing `eval` / `G_report` or spending **1 single Colab H100 CU**:
1. **Strict `Q_T = 0` Audit:** Zero task prompts, zero task-derived singular vectors, and zero registry keys enter any bridge, anchor, or healer matrix (discovery supplies discrete component IDs only).
2. **Dual-Endpoint Win (`n = 128` fresh families, held-out wording, unique Table-2 seed + second independent `S3*` registry):** `REAL` beats `STOCK_B` (`B_0`), fair independently-healed `Z_FAR`, AND `RAND × 3` with paired `t >= 4` and positive joint-hit (`jB`) lift on **BOTH `S1 → S2` AND Recipient-Only `S1 → S3*`** (where Donor = `0/32`).
3. **Full Preservation & Tail-KL Audit (untrimmed generic corpus, ≥ 5k tokens):**
   - `GenKL <= 0.02` (preregistered clean tails: `p95 <= 0.08`, `p99 <= 0.15–0.20`, `max <= 2.0`), worst-1% share, baseline-surprisal stratification.
   - `100%` baseline factual retention + direct `S3*` registry retention.
   - Active Bulk (`X7 >= 5%`, $\bar{\Lambda} \ge 0.25$, excluding `pos == 0`) on every edited block.
   - Vanilla `Qwen3ForCausalLM` reload with `sa.assert_zero_hooks`, frozen-Zone-2 SHA-256 verification, tied embeddings, and **bit-exact rollback (`0.0` diff on prefill + 3-step KV decode)**.
4. **Mandatory Independent Dual Red-Team Sign-Off:** Fresh `PASS / PASS` from **Opus 5.5** AND **`-s read-only` Astra 6.2 Ultra**.

---

## 6. Post-Council Resolution: `Exp 7b` Hard Stop & Pivot to Matched-Width Pythia-160m v1

`Exp 7` (Stage 0 + `Exp 7b` QK-Only Compile) was executed locally on `$0` MPS immediately after the Council synthesis:
- **Stage 0 (`P_A @ V_B` Hooked Oracle) PASSED:** `S1-Pure (L11..L18, no L19g2)` hit `S1 → S2 +4.955 (t = +10.77, jB = 87.5%)` and `S1 → S3* +2.365 (t = +4.28)` at `OracleKL = 0.0193`.
- **`Exp 7b` (Compiled QK-Only, both Zero-SGD and Generic-Text Adam on Foldable `H_l`) FAILED:** Even with `GenKL` reduced 12× to `0.0014` (`L19g2_AdamH`), held-out task attention KL blew up `1.22 → 4.23` (`S1 → S2 Δs = +0.05`, `S1 → S3* -1.55`; `S1Pure_AdamH -3.14 / -5.14`), triggering the pre-registered Hard Stop on `Qwen3-1.7B → 0.6B` in-place cross-width compilation (demoted to v2 stress test).
- **Clean Matched-Width v1 Pivot (`EleutherAI/pythia-160m → pythia-160m-seed2@step2000`):** Following Astra 6.2 Ultra's restrictions review (matched width `d = 768`, additive standard `GPTNeoXLayer` blocks upstream of bitwise-frozen `L6..11`), a **100% closed-form Zero-SGD, Zero-Hook** standalone 15-layer transplant succeeded on sealed untouched `pool[1000:1500]` (`n = 256` per format, all 12 Holm-p `<= 3.82e-4`, primary `rs6 +54.8%` vs `STOCK` / `+64.0%` vs `RAND₅`, bit-exact rollback `0.0`). See [PYTHIA_V1_STANDALONE_TRANSPLANT.md](PYTHIA_V1_STANDALONE_TRANSPLANT.md).
