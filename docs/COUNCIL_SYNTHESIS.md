# PlatonicSwap — 6-Seat LLM Council Literature Synthesis (v1 Hardening & v2 Hidden Rule)

> **Date:** 2026-11-26 | **Corpus Audited:** All 24 full-text works in [`docs/papers/`](papers/) + `/tmp/PlatonicSwap/` code + `/tmp/platos_key_lab/master_audited_results.md` + local HF model configs.
> **Council Roster (100% read-only audit; zero files modified during review):**
> 1. **Opus 5.5 Max #1** — v1 Reviewer #2 Red-Team (Weight/Adapter/Merging & Editing Baselines)
> 2. **Opus 5.5 Max #2** — v1 Reviewer #2 Red-Team (Activation Transfer, Probing & Foundations)
> 3. **Opus 5.5 Max #3** — v2 Multi-Layer Mechanism Architect (Attention, KV, OT, Subspace, Geometric Flow)
> 4. **Opus 5.5 Max #4** — v2 Triggers, Low-FPR Calibration, Quant/LoRA Staleness & Donor Harvesting
> 5. **Fable 5.2 Max** — Mathematical Unifier & Impossibility Critic
> 6. **Astra 6.2 Ultra** (with parallel read-only subagents) — Full 24-Paper Lead Architect Cross-Audit

> **Round 2 Addendum (2026-11-26):** Round 1 ran prior to completion of Sealed Cap-2 `BIO-RIPPLE` (`results_bio_ripple_sealed.json`) and repo commit `7745b81`. For the updated 5-seat Round 2 audit — including full Cap-2 results (`0/722` far-locality breaks, bit-exact modular removal of two working co-implanted capabilities), quantitative 85–95% ceiling math, and the definitive zero-GPU v1 closeout checklist — see [`V1_CLOSEOUT_COUNCIL_R2.md`](V1_CLOSEOUT_COUNCIL_R2.md).

---

## 1. Executive Consensus: Immediate Zero-Compute v1 Upgrades (Before Pushing)

All v1 red-team seats (**Astra 6.2 Ultra**, **Opus #1**, **Opus #2**) independently converged on **7 release-critical fixes** that require **zero extra MPS runs** and eliminate the sharpest Reviewer #2 objections:

1. **Retire the "+24.49 pp vs MEMIT" strawman framing** *(Astra + Opus #1)*:
   - Canonical MEMIT (`meng_2023_memit.md`, §4.3 / Eq. 14) edits mid-layers `R={3..8}` with $\lambda=15{,}000$ over 100k keys (moving CounterFact NS only `83.5 -> 73.7` at 10,000 edits). Our L23 foil used single-layer $\lambda \in \{10, 50, 200\}$ and $\alpha=1.5$, giving `24.49 pp` at *every* $\lambda$ (random map: `32.14 pp`).
   - Moreover, `+0.00 pp` belongs to the **L23 hooked gated sidecar** (0% ripple), whereas Cap-1 mid-layer native organs break **15/175 to 34/155 (`8.6%–21.9%`)** of base-correct unrelated facts.
   - **Fix:** Label the foil explicitly as *"non-canonical single-layer (L23) closed-form `W_down` overwrite"*, report unrelated-fact retention (not CounterFact neighborhood specificity), and keep sidecar vs. native-organ numbers strictly separated.
2. **Retire "41% → 83–92% over Plato's Key"** *(Astra + Opus #2)*:
   - Our own 20-pair PK-protocol replication (`results_baseline.json`) matches Plato's Key retention almost identically (**70.9%** vs. PK **73%**). Zero-QA `N=10k` maps add `+0.6` to `+5.8 pp` over QA-fit Procrustes (overlapping 95% CIs).
   - Accuracies `>=85%` require a **second, recipient-side intervention** (`full_knockout` attention masking of the original subject span), diagnosing the residual-subject route PK identified rather than beating PK's 41% like-for-like. Also note that swap layers were chosen via labeled validation transfer ("zero-QA" refers to map fitting only).
3. **Reframe T3 Honestly: Latent-Knowledge Unlocking with Donor-Attributable Content** *(Opus #2 + Astra)*:
   - On the 21 SmolLM2 T3 "gaps", **10/21 were region-first answers already naming the correct country** (e.g., `' Tuscany, Italy.'`, `' Western Australia...'`) flagged as misses by `startswith`, and unpatched SmolLM2 already scored `11/21` capital, `16/21` currency, and `15/21` language.
   - On the strict blind-spot cohort (`n=8`), country recovery is `5/8` vs. `0/8` (exact McNemar `p=0.0625`, n.s.), while pooled strict 3-attr ripple is `8:0` (`p=0.0078`) and wrong-entity control `e'` tracks `e'` in **81.0%** of cases.
   - **Fix:** Claim **latent-knowledge unlocking with donor-attributable attribute content** (not absent-knowledge transplant).
4. **Fix the `HardNeg-FF` 12.5% Floor Bug & Per-Attribute Ripple Scorecard** *(Astra + Opus #1)*:
   - `Kyoto` and `Nagoya` appear in the 16 hard negatives while answering `Japan`, giving every untouched baseline `12.5%` on `HardNeg-FF`. Report net of the 2/16 Japanese-entity floor (or exclude them).
   - Because the geographic write explicitly contains both `country_dst - country_src` AND `capital_dst - capital_src`, publish per-attribute columns (*direct country*, *capital*, *currency*, *language*, *joint 4/4*) — **Currency and Language** are the cleanest tests of propagation beyond authored vector terms. Also note Smol's `p=0.0039` pools 3 attribute prompts per entity (`<=5` independent entities -> entity-level `p >= 0.0625`).
5. **Fix Scope of "Implant-Anywhere", "Multi-Donor Composition", & "Exact Unlearning"** *(Astra + Opus #1)*:
   - Cap-1 unseen ripple beats shuffled controls in **3/4 recipients** (`SmolLM2`, `Granite`, `Qwen-1.5B`; Falcon3 `p=0.25`); direct EM is significant only intra-family (`Qwen-1.5B`, `p=0.039`); Stage A transferred only to SmolLM2.
   - In native two-donor composition, the Falcon rule scored `0%/0%/0%`; bit-exact `max |Delta logit| == 0.0` is verified for `UNLEARN_ALL`. Claim **structural reversibility** and **reversible co-installation**; scope functional 2-step multi-donor composition to the L23 output sidecar (Table 3B).
   - Call the abstention predicate *"frozen design thresholds"* (`R^2 <= 0`, `r <= 0.25`, `a <= 0.15`), noting it caught **4/4 shuffled-read controls** while **4/4 shuffled-write controls compiled** (so write quality needs a separate check).
6. **Fix Prior-Art Descriptions & Code/Builder Bugs Before Release** *(Astra + Opus #1 + Opus #2 + Opus #4)*:
   - **Latent code bug (`pilot_organs_p0.py:907-955`):** `apply_exact_bos_null_affine` declares `target_bos_gate=-20` but never applies it (only edits `W_u/b_u`), leaving BOS silence dependent on a brittle exact linear null that fails under 4-bit quantization.
   - **Table builder bug (`build_repo_tables.py:370, 243, 698`):** Looks for wrong Stage A/B keys (`stage_A_multifamily` and `stage_B_multidonor_composition` exist in JSON!), mixes Granite native ceilings (`91.5%` displayed while efficiency uses per-config ceilings -> `107%`), and resolves JSONs relative to `experiments/` instead of `results/`.
   - **Prior-art accuracy:** Corrected in [`docs/README.md`](README.md) — `Crosscoders` *does* perform causal Llama→Qwen steering (§3.1); `Transport & Merge` is masked top-128 OT interpolation (not destructive global merge); `LoRA-X` is same-family only; `Oozeer` already built an in-weights autoencoder switch (`0.32%` params); drop the `Agarwal >=1.7B` claim (our own `BASE` is non-monotone in size: SmolLM2 1.61B `0.831` > Qwen-3B 2.77B `0.643`), and cite `Piepereit` strictly for *"alignment ≠ causal transfer"* (no Granite, no abstention).
7. **High-Value Related-Work Ammunition (Unanimously Proved by Opus #1, Opus #3, and Fable 5.2 Max):**
   - **Cross-LoRA Collapse Lemma:** In `xia_2025_cross_lora_heterogeneous_transfer.md` Eqs. 3–5, because $U_s$ has orthonormal columns, $P = U_t U_s^\top$ achieves zero residual, forcing $\tilde U_s \equiv U_t$ and $\tilde V_s \equiv V_t$. Eq. 7 collapses to $\Delta W_t = U_t U_t^\top \Delta W_s V_t V_t^\top$ (undefined when $m \neq m'$, carrying **zero** cross-model correspondence) — explaining why 4/16 Cross-LoRA cells are `+0.00%` and proving mathematically why **paired activations are necessary**.
   - **Independent Qwen2.5-3B Asymmetry Corroboration:** Both our `BASE` (Qwen2.5-3B = best donor `0.779`, worst recipient `0.643`) and `bakker_2026` C.14 (Qwen2.5-3B = strong source `84.5`, weakest target `71.3`) find the exact same directional asymmetry on independent tasks.

---

## 2. The Mathematical "Hidden Rule" for v2 (Higher-Level Skill Translation)

Synthesizing **Fable 5.2 Max**, **Opus #3**, **Opus #4**, and **Astra 6.2 Ultra**:

### 2.1 Unified Invariant & Gauge Hierarchy
All five literature transfer families (`Plato's Key`, `Relative/Indra/Bakker`, `Heo KV`, `Cross-LoRA`, `Transport & Merge`) are **degree-0 projections of one cross-covariance statistic $C_{BA} = \mathbb{E}[h_B h_A^\top]$ onto the gauge group of the port they touch** ($CO(d)$ for pre-RMSNorm residuals, $GL(d_h)$ for RoPE-stripped heads, $S_n$ for neuron bases, or the quotient by $CO(d)$).

Higher-level multi-layer skills live in **degree $\ge 1$ gauge-invariant circuit amplitudes**:
$$\mathcal{A}_w(x, x') = h(x)^\top w(T, T^\top) h(x'), \qquad T \in \{W_O W_V,\; W_Q^\top \rho(\Delta) W_K\}.$$

**Core Selection Principle (confirmed across `Heo §4.5`, `Cheng §1`, `Bakker C.10`, `Piepereit Table 3`, and our Table 1B):** Judge every interface by the *downstream operation that consumes it* — calibration $R^2$ correlates negatively with KV retention (`r = -0.20`), while recipient-query attention-output cosine predicts it (`r = +0.57`).

### 2.2 Three Mathematical Obstructions Blocking Single-Layer v1 on Higher-Level Skills
1. **O1 — RoPE Torus-Commutant Rigidity (Q–K Phase Mismatch):**
   - A position-independent map on rotated keys must commute with RoPE: $S \rho_A(t) = \rho_B(t) S$, restricting $S$ to shared frequencies only.
   - **Verified from our cached HF configs (`hf_cache/*/config.json`):**
     - `qwen2.5-3b` ($d_h=128, b=10^6, s=0.2159$) → **`falcon3-3b`** ($d_h=256, b=1000042, s=0.1079$) is **nearly RoPE-nested** ($i \mapsto 2i$, relative frequency error $\le 4.1 \times 10^{-5}$)!
     - `qwen2.5-3b` → **`smollm2-1.7b`** ($d_h=64, b=130000, s=0.3680$) and **`granite-3.1-2b`** ($d_h=64, b=5\times 10^6, s=0.4820$) are **RoPE-incommensurate** (step ratios `1.705` and `2.233`), predicting exact-offset decay beyond $\Delta_{\text{coh}} \approx 4\text{--}10$ tokens — the **exact reverse** of v1's residual-socket recipient ranking!
2. **O2 — Fibre-Locality & Composition Kernel (Depth-Distributed Circuits):**
   - An MLP organ acts fibrewise at a single token and cannot add a Q–K attention form.
   - Worse, for a 2-site circuit ($h + \mathcal F_1 + \mathcal F_2 + D\mathcal F_2[\mathcal F_1]$), any statically fitted layer-2 read socket has **zero gain** on organ 1's novel write because untouched $B$ never produced it.
3. **O3 — Mehler Attenuation & Discrete Activation Commutant:**
   - Under Gaussian channel noise with correlation $\rho$, Hermite-degree-$p$ features decay as $\rho^p$ and $p$-way conjunctions decay as $\prod_i \rho_i$. **Check against our own Table 2A medians:** $r(a) \approx r(g)^2$ tracks all three drift-corrected arms closely (`0.6889 vs 0.671`, `0.6116 vs 0.600`, `0.8484 vs 0.835`)!
   - Elementwise SiLU/GELU only commutes with permutations, so soft OT plans cannot preserve nonlinearities without **regenerative saturation**.

### 2.3 The Minimum Sufficient v2 Operator ($\mathfrak{S}_{v2}$) — Still Zero Recipient SGD
To overcome O1–O3 without training Recipient $B$:
1. **Monotone Depth Warp ($\varphi^\star$):** DTW over `1 - CKA_lin` across layers (`maps.py` + `Transport & Merge` Eq. 10–13, monotone to respect `Liang` Eq. 59 torsion).
2. **Congruent Attention-Head Organ + Forced RoPE Intertwiner ($\mathcal J$):**
   $$W_Q^B = \tau\,\mathcal J\,W_Q^A M^{(j)}_{in},\quad W_K^B = \mathcal J\,W_K^A M^{(j)}_{in},\quad W_V^B = \mathcal J_V W_V^A M^{(j)}_{in},\quad W_O^B = \rho_B^{-1} M^{(j)}_{out} W_O^A \mathcal J_V^\top$$
   where $\mathcal J$ is the zero-parameter 1-D optimal frequency assignment minimizing $\sum_i \omega_i |\ln \theta^A_i - \ln \theta^B_{j(i)}|$ (or organs born in recipient RoPE for sidecar/MHA widening), $\tau$ matches donor attention entropy via 1-D bisection, and $(W_O, W_V)$ gets an optional closed-form reduced-rank ridge refit matching post-`o_proj` outputs (`Attention Editing` Stage I without SGD).
3. **Sequential Depth-Cocycle Refit:** Implant organs in depth order $\ell_1 < \dots < \ell_K$, refitting each $M^{(j)}_{in}$ in closed form on the **organ-augmented recipient $\tilde B$** (gives nonzero gain on upstream private channels!).
4. **Regenerative Triggers (`V5-RAX`) + Update Resilience (`ROU`):**
   - **Local Cosine Anchor Gate:** Compiles directly into bias-free SwiGLU rows (`w_B = diag(sqrt(d)*gamma)^{-1} A_B^T D^{-1} w_rel + b' c_B`), mathematically equal to **Procrustes with singular values $\Sigma$ kept** (~300 anchors vs. 10k tokens). Reserve Bakker's global all-layer `arccos` probe as an offline diagnostic / causal-prefix sidecar.
   - **Sequence-Max Calibration + AlphaEvolve Regularizers (`Kramár 2026`):** Calibrate thresholds on **sequence maxima** (`T=8k` at `1%` seq-FPR requires `p_tok ~ 1.2e-6`), add L1 + orthogonality regularizers from Gemini's AlphaEvolve GLU probe, and enforce **saturating** `target_bos_gate = -20` (`SiLU(-20) ≈ 0, SiLU' ≈ 0`) so BOS silence survives 4-bit quantization.
   - **Organ Manifest + Re-Conjugate-on-Update (`ROU`):** Ship organs unmerged with a ~8 MB reference activation bank + **frozen original anchors** (`Bakker` Table 65 shows re-encoding anchors on LoRA'd models *degrades* transfer, whereas `Duan 2026` label-free activation realignment fixed `111/111` cells). Pad `d_ff + k` to multiples of 256 with exact-zero rows for `ggml` K-quant compatibility.
   - **Organic Donor Harvesting (`Chantless` + `Crosscoders` -> `PlatonicSwap`):** Use Chantless necessity masks or Crosscoder exclusive features to *locate* organic donor neurons, express payloads in the *shared* decoder basis (never linearly transport exclusivity-5 A-only directions), and retrofit silence via closed-form quantile bias shifts.
   - **Alternative State-Transplant Path (Architecture B):** Gauge-fixed KV "skill cartridge" (span-overlap Sinkhorn tokenizer alignment $\bar\Pi$ + RoPE strip/re-encode + query-metric $\Gamma=\Sigma_Q$ reduced-rank ridge keys).

---

## 3. Ranked 1-Hour Apple Silicon M5 Pro v2 Falsification Experiments

| Rank | Experiment & Lead Seat | Models & Skill | What It Tests / Kills | Estimated MPS Time |
|---:|---|---|---|---|
| **1** | **Organic Carry-Circuit Neuron Transplant** *(Astra 6.2 Ultra)* | `Qwen2.5-3B L17` → `SmolLM2-1.7B L9` (cached v1 sockets, zero training). Top-32 causal units-to-tens **arithmetic carry** rows harvested via paired lesion (`47+35` vs `41+32`). | Tests whether *organic pretrained* procedural MLP rows transfer zero-SGD. Built-in 25-min donor-lesion kill gate + B-native amplification & constant-mean controls. | **~60 min** |
| **2** | **RoPE Torus-Dissociation Test (Prediction P1)** *(Fable 5.2 Max)* | `Qwen2.5-3B` → RoPE-nested **`Falcon3-3B`** vs. RoPE-incommensurate **`SmolLM2-1.7B`**. Transplant 1 positional previous-token head via `QKOV` + $\mathcal J$ vs. Haar-random $\mathcal J$. | Cleanest direct falsification of Obstruction O1 & the RoPE intertwiner law (predicts exact reversal of v1's recipient ranking!). | **~30–45 min** |
| **3** | **Full Multi-Layer Procedural Skill (Arch A + B)** *(Opus 5.5 Max #3)* | `Qwen2.5-3B` → `SmolLM2-1.7B`. **Dual-trigger cyclic digit rotation** (`->` ROTL / `<-` ROTR), 6-layer heads + SwiGLU organs + sequential depth-cocycle refit, scored on `START` (routing) / `MID` (control) / `WRAP` (state). | End-to-end test of multi-layer attention+MLP operator splicing vs. Cross-LoRA control, broken Q-K control, single-layer v1 control, and KV cartridge. | **~49 min** |

---

## 4. Literature Blind Spots to Fetch Next

From [`WANT_TO_READ.md`](WANT_TO_READ.md), the council identified **3 priority manual fetches** (everything else can wait):
1. **Zimmermann & Boussard (AAAI 2025 CoLoRAI, OpenReview)** — *Transplanting Knowledge: Layer-Specific Grafting* (check recipient training & translator controls before v2 multi-layer grafting).
2. **Bansal, Nakkiran & Barak (NeurIPS 2021)** — *Model Stitching* (methodology for verifying the connector isn't carrying the task itself).
3. **Neuron-addition / side-memory model editors (`SERAC`, `WISE`, `T-Patcher`)** — check before claiming "append-only neuron addition" novelty in formal paper drafts.
