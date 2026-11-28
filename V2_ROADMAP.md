# PlatonicSwap `v2` Roadmap — Beyond Single-Layer Concept Swaps

> **Baseline Anchor:** `v1.0` (tag `v1.0`, frozen engine `SHA-256: 22ef57c9d196e12f…`) proved that closed-form cross-family rule compilation into 4 appended bias-free SwiGLU rows works zero-SGD across 4 pretraining families on two disjoint capabilities (`Cap-1` Geography and single-execution `Cap-2 BIO-RIPPLE`), with `0/722` out-of-domain locality breaks, `8/8` automatic shuffled-channel abstention, and bit-exact modular removal (`max |Δlogit| = 0.0`).
>
> **Round 2 Council Verdict (`docs/V1_CLOSEOUT_COUNCIL_R2.md`):** Single-layer zero-SGD `v1.0` cannot cross `85–95%` at `<1%` locality & near-zero hard-negative flips because of three structural bottlenecks:
> 1. **Write-bridge starvation:** `N = 1,200` concept sentences in `D = 2,048` yields held-out write cosine `0.38–0.50`, explaining `η = 0.34–0.53` of the Native-B oracle log-odds shift with zero free parameters.
> 2. **One-sided general-text gating:** `t1` is set from general text only (`alpha_neg` never enters `t1`), putting other animals at `5.1–6.7σ` on the oracle and forcing an efficacy–selectivity frontier (`0/722` broken vs. `157/722` broken on Oracle).
> 3. **Single-site & MLP fibre-locality obstructions (`O1/O2/O3` in `docs/COUNCIL_SYNTHESIS.md`):** Unmasked single-layer edits leave the original subject readable in other tokens/layers, and a single-layer MLP organ cannot install cross-token attention routing or multi-layer procedural state transitions.

---

## Completed `v2` Phase-A Falsification Results (Audited JSONs in `results/`)

| Experiment | Compute & Artifact | Outcome | Headline Audited Finding |
| :--- | :--- | :---: | :--- |
| **Step 0 — Read×Write Decomposition & Per-Item Gate Telemetry** | Local M5 Pro MPS (`1,615 s`) · [`results_v2_step0_rw_diagnostic.json`](results/results_v2_step0_rw_diagnostic.json) | **DIAGNOSTIC (Write-Limited, High HardNeg Bleed)** | **`0 / 192` gate misses** across all 4 recipients (`pos φ ≥ 0.90` on `92–100%`). Keeping the untouched frozen `v1.0` Pushforward Reader and substituting the **recipient's own native write vector** (`PUSH_READ × NATIVE_WRITE` — diagnostic isolation, not donor-knowledge write construction) raises Direct Full-FC to **`100.0%` on 4/4 recipients** (`SmolLM2`: `100.0% FC / 83.3% EM`, Ripple `97.4% FC / 51.0% EM`; `Granite`: `100.0% Full-FC / 47.9% EM`; `Falcon3`: `100.0% FC / 39.6% EM`, Ripple `87.5%`; `Qwen1.5B`: `100.0% FC / 54.2% EM`), **at the cost of severe hard-negative selectivity loss: hard-negative false-flip Full-FC jumps to `58.3% / 58.3% / 66.7% / 75.0%` (EM `25.0% / 16.7% / 16.7% / 29.2%`)** vs. original `PUSH_READ × PUSH_WRITE` (`8.3% / 12.5% / 25.0% / 33.3%` FC). Actual write alignment `cos(v_push, v_nat) = 0.24–0.40`. Empirical Pushforward gate `AUROC` (species vs. HardNegs) = `0.80–0.96`. |
| **Falsifier A — RoPE Torus-Dissociation Test (Prediction P1)** | Colab NVIDIA L4 (`277 s`) · [`results_v2_falsifierA_rope_torus.json`](results/results_v2_falsifierA_rope_torus.json) | **CONFIRMED (Attention Alignment Only; Copying Drops)** | **Positional attention-mass dissociation on long-range duplicate-token attention (`L05.H09`):** nearly RoPE-nested **`Falcon3-3B`** (`OPTIMAL_ROPE_J`, `i → 2i`) holds **`0.88 / 0.86 / 0.85 / 0.81`** target-lag attention mass at `Δ = 8 / 16 / 32 / 64` (`r = 0.62` at `Δ=64`) vs. `NAIVE_INDEX_J` (`i → i`) collapsing to `0.18` (`Δ=32`) → `0.0083` (`Δ=64`) (**`~81–97×` attention-mass ratio**). **Task-endpoint caveat:** on the same featured `Falcon3 L05.H09` lag-64 benchmark, downstream **copying accuracy drops below base** (`base 85.12%`, `optimal 83.53%` [`-1.59 pp`], `naive 84.33%`, `Haar random 83.53%`), confirming that attention-pattern alignment alone does not translate into copying gains. RoPE-incommensurate **`SmolLM2-1.7B`** attention mass decays past `Δ ≈ 8` (`0.61 → 0.42 → 0.28 → 0.21`), while `ROPE_STRIPPED_REF` recovers `0.92 / 0.92 / 0.85`. Short-lag (`Δ=1`) previous-token heads (`L01.H01`, `L19.H11`) hit `r = 0.98–0.99`, `top1 = 100%`, `KL = 0.03–0.16` on both recipients. 2-hop induction heads (`L20.H01/H03`) fail even under `ROPE_STRIPPED_REF` (`r = 0.09–0.35`), confirming **O2**. |
| **Falsifier B — Organic Arithmetic-Carry Neuron Transplant** | Colab NVIDIA L4 (`746 s`) · [`results_v2_falsifierB_carry_transplant.json`](results/results_v2_falsifierB_carry_transplant.json) | **FALSIFIED (Null vs. Controls)** | Donor `qwen2.5-3b` carry is multi-layer distributed (`L24 k=64` lesion: `Δcarry_LO = -0.804` vs. `RAND-64 +0.002`, but only `-1.7 pp` EM and `Δnocarry_LO = -0.498`). Conjugating top-32/64 raw pretrained carry rows into `smollm2-1.7b` (`L17→L9` & `L24→L16`, `0/181` locality broken, `torch.equal = True`) fails to beat **`RANDOM_DONOR_NEURONS`** (`70.8%` vs. `71.7%` EM on `L17→L9`, `p=1.0`; `67.5%` vs. `65.8%` on `L24→L16`, `p=0.625`) or **`CONSTANT_MEAN_WRITE`** (`66.7%` / `69.2%`, `p ≥ 0.18`). Even `NATIVE_B_CARRY_AMPLIFY_k32` gains carry FC (`90.8% → 97.5%`) at the expense of No-Carry accuracy (`56.7% → 50.0%`; `43.3%` at `k=64`, `p=0.005`). |

**Strategic Pivot Informed by Phase A:**
1. **Drop** naive single-layer raw pretrained procedural neuron harvesting (Falsifier B falsified).
2. **Double down on the two empirically validated `v2` winners:**
   - **Write-Bridge + Two-Sided Gate `v2` Compiler** (Step 0 isolated a weak write direction with recipient-native writes hitting `100.0%` Direct Full-FC at `58–75%` HardNeg false-flip FC). Next: scale the unlabeled write corpus `10×` + readout-weighted write bridge AND test `<= 25`-step gradient optimization of appended write vectors (`v2-Lite`) + two-sided `t1/t2` on a fresh externally pre-registered **Sealed Cap-3**.
   - **Congruent Attention-Head Organs with Optimal RoPE Intertwiner $\mathcal{J}$ + Depth-Cocycle Refit** (Falsifier A validated attention-pattern alignment on RoPE-nested `Qwen → Falcon3`, while copying accuracy fell `-1.59 pp`, showing OV/value transfer remains unresolved). Next: **Falsifier C** (2-site `prev_token → induction` / cyclic digit rotation with sequential depth-cocycle refit to overcome O2).

---

## Executed & Planned `v2` Falsification Ladder (Apple Silicon M5 Pro + Colab NVIDIA L4)

### Step 0 — Per-Item Gate Telemetry & `2 × 2` Native/Transferred Read×Write Decomposition (`~15 min MPS`)
- **Script:** `experiments/v2_step0_rw_diagnostic.py` → `results/results_v2_step0_rw_diagnostic.json`
- **Strict Read-Only Contract:** Imports frozen `v1.0` `platonic_organ_engine.py` (`22ef57c9…`) and `bio_ripple_sealed.py` untouched; reuses frozen `v1.0` cached sockets.
- **What it resolves:**
  1. Logs real per-item gate pre-activation `z = w_c · x_B` and gate amplitude `φ ∈ [0, 1]` at the subject token on all **48 Cap-2 species** and **24 HardNegs** across all 4 recipients (`SmolLM2-1.7B`, `Granite-3.1-2B`, `Falcon3-3B`, `Qwen2.5-1.5B`).
  2. Crosses `{Pushforward Read, Native-B Read} × {Pushforward Write, Native-B Write}` + forced-fire gold-token write controls (`φ = 1.0`), cleanly separating **Gate Miss (`φ < 0.05`)**, **Ramp Under-Drive (`0.05 ≤ φ < 0.90`)**, **Write-Limited (`φ ≥ 0.90`, FC fails)**, and **Verbalization Gap (`φ ≥ 0.90`, FC passes, EM fails)** with full per-attribute and generated-string ledgers.
  3. Computes the exact empirical single-direction ROC (TPR on test species vs. HardNeg FPR) for both Pushforward and Native-B Oracle.

### Falsifier A — RoPE Torus-Dissociation Test (`Prediction P1`, `~30–45 min MPS`)
- **Script:** `experiments/v2_falsifierA_rope_torus.py` → `results/results_v2_falsifierA_rope_torus.json`
- **Hypothesis (`docs/COUNCIL_SYNTHESIS.md` Obstruction O1):** A position-independent linear map on RoPE-rotated Q/K heads must commute with the RoPE torus ($S \rho_A(t) = \rho_B(t) S$). From our cached HF configs:
  - `Qwen2.5-3B` ($d_h = 128, b = 10^6, s = 0.2159$) → **`Falcon3-3B`** ($d_h = 256, b = 1{,}000{,}042, s = 0.1079$) is **nearly RoPE-nested** ($i \mapsto 2i$, relative frequency error $\le 4.1 \times 10^{-5}$).
  - `Qwen2.5-3B` → **`SmolLM2-1.7B`** ($d_h = 64, b = 130{,}000, s = 0.3680$) is **RoPE-incommensurate** (step ratio `1.705`), predicting phase-coherence decay beyond $\Delta_{\text{coh}} \approx 4\text{--}10$ tokens — the **exact reversal** of `v1.0`'s residual-socket recipient ranking!
- **Experimental Design:**
  1. Locate top sharp positional-offset / induction / previous-token attention heads in `Qwen2.5-3B` across distance lags $\Delta \in \{1, 2, 4, 8, 16, 32, 64\}$.
  2. Transplant the donor head's `(W_Q, W_K, W_V, W_O)` into `Falcon3-3B` and `SmolLM2-1.7B` via the closed-form congruent attention-head operator with optimal 1-D frequency intertwiner $\mathcal{J}$ vs. (i) **Haar-random $\mathcal{J}$**, (ii) **unpermuted naive truncation/padding**, and (iii) **unrotated (RoPE-stripped) reference**.
  3. Measure both attention-pattern KL / top-1 lag-accuracy curve vs. lag $\Delta$ and downstream causal copy/previous-token logit shift as a function of $\Delta$.

### Falsifier B — Organic Procedural Circuit Transplant (`~60 min MPS`)
- **Script:** `experiments/v2_falsifierB_carry_transplant.py` → `results/results_v2_falsifierB_carry_transplant.json`
- **Hypothesis (Astra 6.2 Ultra Rank-1 Falsifier):** Can *organic pretrained* procedural MLP neurons (not synthetic constant-write concept rules) be harvested from a donor model via causal lesioning and transplanted zero-SGD into a recipient from another family?
- **Experimental Design:**
  1. **25-min Donor-Lesion Kill Gate:** On `Qwen2.5-3B`, evaluate paired 2-digit addition (`a + b` with units carry vs. matched no-carry controls, e.g., `47+35` vs. `41+32`). Identify the causally critical layer & top-`k` (`k ∈ {16, 32, 64}`) carry-selective SwiGLU neurons via mean/zero ablation (`Δ carry acc` vs. `Δ no-carry acc`).
  2. **Zero-SGD Pretrained Neuron Conjugation (`Qwen2.5-3B` → `SmolLM2-1.7B`):** Conjugate the harvested donor MLP rows directly into `SmolLM2-1.7B` (bias-free SwiGLU widening, Donor A unloaded) using our unlabeled read/write sockets, compared against:
     - **Shuffled-socket control**,
     - **Random-donor-neuron control** (matched norm & count),
     - **Constant-mean write control** (isolating input-dependent computation from constant bias shift),
     - **Recipient-native carry amplification** (upper reference).

### Falsifier C — Multi-Layer Procedural Skill & Depth-Cocycle Refit (Follow-Up)
- Dual-trigger cyclic digit rotation (`->` ROTL / `<-` ROTR) across 2–6 layers combining congruent attention-head organs + SwiGLU organs + sequential depth-cocycle refit (refitting downstream sockets on the organ-augmented recipient $\tilde B$), tested against Cross-LoRA, broken Q–K, and single-layer `v1` controls.

---

## Path to `85–95%` on a Fresh Externally Pre-Registered Sealed Cap-3
Once Step 0 and Falsifiers A–B establish the exact loss terms and attention-organ feasibility, build `v2` engine (new SHA-256) and pre-register a fresh **Sealed Cap-3** (pushed signed git tag *before* first execution) fixing every `v1.0` benchmark defect:
- **Benchmark fixes:** Full-continuation log-probability FC (no first-sub-token BPE artifacts like Granite `" re"` vs. `" m"`), base-normalized / flexible EM parser, premise-prefixed hard negatives, zero prompt-entailed floors (replace `hop1_heat_lamp`), `n >= 300` locality facts (certifying `<= 1%`), `>= 48` species/entities per rule, repeated shuffled seeds, and saved per-item outcome vectors for exact paired species-level tests.
- **Engine `v2` upgrades:**
  - **v1.1 / Pure Closed-Form Track:** Bilinear twin-AND (`--ablate-domain-up`) + implemented two-sided `t1/t2` (general + in-domain negative conformal thresholds + sequence-max calibration) + `10×` domain-covering write corpus (`N >= 10,000` concept tokens, resolving Falcon `D=3072` underdetermination) + saturating BOS gate (`-20`) + `ggml` 256-multiple `d_ff` padding + Organ Manifest / Re-Conjugate-on-Update (`ROU`) + compiled subject-route attention-head organs (if Falsifier A OV transfer passes).
  - **v2-Lite Track (Frozen-Base Appended-Weight Optimization — NOT Zero-Training):** `<= 25`-step gradient optimization of only the appended write vector `v*` on the frozen recipient (`ROME`/`MEMIT` potency while retaining append-only modularity and bit-exact removal) to cross `85–95%` on both FC and EM.
