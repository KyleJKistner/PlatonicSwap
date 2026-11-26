# PlatonicSwap

**Architecture-Agnostic Cross-Family Activation Transfer & Native Mid-Layer "Platonic Organs"**

PlatonicSwap investigates and extends Stanford Hazy Research's *Plato's Key* across **5 ungated base LLMs spanning 4 distinct architecture families and 3 hidden widths** (`Qwen2.5-3B` D=2048, `Qwen2.5-1.5B` D=1536, `SmolLM2-1.7B` D=2048, `Granite-3.1-2B` L=40 D=2048 `rm=0.22`, and `Falcon3-3B` D=3072), all reproducible on a single local 48 GB Apple Silicon workstation.

> **Core Verified Claim (Scoped Strictly to Our Audited Evidence):**
> A single frozen, zero-magic-constant **Channel-Pushforward Twin-SwiGLU** construction compiles donor-derived mid-layer entity/concept rules into **4 appended native `mlp_bias=False` SwiGLU rows** (`d_ff -> d_ff + 4`) inside standalone stock recipient checkpoints—with the donor completely unloaded, zero forward hooks, zero attention masks, zero ZCA whiteners, zero lookup tables, zero base-weight overwrites, and zero SGD on the recipient. Using the **exact same frozen engine (`SHA-256: 22ef57c9d196e12f`)** and **exact same task-blind universal sockets**, the method transferred **two completely independent capabilities**—Geography (Cap-1) and a **strictly one-shot sealed Biological Taxonomy benchmark (`BIO-RIPPLE`, `SHA-256: d321fd0255c47cd3`)**—across all **4 distinct recipient architectures** (including rectangular `2048 -> 3072` up-projection and `2048 -> 1536` down-projection, and 40-layer `rm=0.22` Granite), achieving **`0 / 722` broken unrelated facts** (`100.0%` retention) on sealed `BIO-RIPPLE`, **`4/4` automatic unlabeled abstention** on shuffled channels, cross-capability co-implantation, and bit-verified exact unlearning (`torch.equal` on both `state_dict` and deterministic logits).

All numbers below are programmatically audited from the raw JSON artifacts in [`results/`](results/) via [`experiments/build_repo_tables.py`](experiments/build_repo_tables.py) — see [`results/master_audited_results.md`](results/master_audited_results.md) for the complete 21-artifact SHA-256 manifest, confidence intervals, and exact McNemar statistics.

---

## Pillar 1 — Zero-QA Geometry & Breaking 85–92% Cross-Family Activation Transfer

1. **The 84–95% Causal Fisher Tail (`results_pilot0.json`, `results_pilot1*.json`):**
   - `84.1%` (SmolLM2-1.7B) to `95.0%` (Granite-3.1-2B) of causal factual Fisher sensitivity lies **outside** the top-64 prose PCA subspace (`AUC(|g·δ|) = 0.820` vs Euclidean `L2 = 0.631`).
   - Consequently, fitting rigid orthogonal $O(d)$ bridges on `N=10,000` **unlabeled (0-QA)** states beats QA-Procrustes by **`+25 to +42 pp`** on Leave-One-Relation-Out (LORO) generalization, and rigid $O(d)$ outperforms unconstrained Ridge $GL(d)$ by **`+8.7 to +10.4 pp`** despite 3–6× worse Euclidean $R^2$ (because Ridge shrinks the 84–95% causal tail toward $\mu_B$).
   - Low-rank `k=64` Fisher-PLS beats PCA by `2.0–2.3×` across all 3 pairs.

2. **Cross-Family Activation Transfer (`results_pilot1b_*.json`, Zero QA Labels — Natural vs. Knockout Strictly Separated):**

| Donor → Recipient | Natural Single-Layer CF (0 KO) | Matched Natural Ceiling | Natural Eff. | Earlier-Subject Attention KO CF | Powered 5-Donor KO (`n≈750`) | Shuffled-R Ctrl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-3B` → `SmolLM2-1.7B` | **83.2%** (`AD=80.3%`, multi-tok `78.5%`) | 87.9% | **94.7%** | **89.3%** (`LORO=86.5%`) | **88.8%** (`AD=83.8%`) | 0.0% |
| `Qwen2.5-3B` → `Granite-3.1-2B` | **52.9%** (`clamped_1.5`, `LORO=57.1%`) | 53.6% | **98.8%** | **91.5%** (`LORO=84.6%`) | **92.0%** (`AD=88.5%`) | 0.0% |
| `Falcon3-3B` (`3072`) → `SmolLM2-1.7B` (`2048`) | **74.6%** | 83.1% | **89.8%** | **87.6%** (`LORO=90.6%`) | **87.2%** | 0.6% |

   - **Two-Donor Convergence:** Residual errors of `Qwen → SmolLM2` and `Falcon → SmolLM2` have correlation `0.471–0.480`; averaging both donors lifts natural single-layer transfer to **`79.9% CF / 80.3% AD / 83.2% LORO`** (`90.8%` of ceiling).
   - **Scaled Long-Tail Knowledge-Gap Transplant (`results_graft_t3_gap.json` + `results_graft_t3_gap_scaled*.json`):** Across 38 total genuine `A-right, B-wrong` obscure entities on SmolLM2 (`n=17` initial + `n=21` newly curated disjoint cohort), zero-QA `O(d)` transport recovers **`76.2–82.4%`** country (`16:0, p=3.05e-5` vs `0.0%` B-self-roundtrip; `12:0, p=4.88e-4` vs `shuffled_R`; `15:0, p=6.10e-5` vs `wrong_entity e'`) and **`70.8–75.0%`** unseen 3-attribute ripple on strict zero-hint blindspots (vs `0.0%` roundtrip).

---

## Pillar 2 — Native Mid-Layer "Platonic Organs" (Zero-Magic-Constant Channel-Pushforward Twin-SwiGLU)

### Why Naive SGD Ramp SwiGLU Fails Cross-Family & How Channel-Pushforward Twin-SwiGLU Solves It
- **Regime 1 (`results_organs_p0.json`):** Conjugating 2,000 native pretrained `Qwen2.5-3B` neurons into `SmolLM2-1.7B` (`mlp_bias=False`) via pre-$\gamma$ deconvolution + constant pin + exact BOS up-null achieves **`median r(g) = 0.774`**, `|Δ|/σ = 0.0425` (`20×` less zero-point drift than naive `0.831`), and exact `0.0` BOS leakage. Gate G0 confirms `95.8%` top-1 read retrieval on known landmarks and **`88.2%` top-1 (`100%` top-5)** on B-unknown long-tail entities.
- **Why Standard 32-Neuron SGD Ramp SwiGLU Collapsed (`0–6%`):** Unconstrained ramp neurons suffer (i) degree-2 bilinear attenuation + correlated read-noise DC drift, (ii) co-adapted write cancellation across 32 rows, (iii) SiLU negative-lobe exploitation, and (iv) donor-optimal discriminants anti-selecting for cross-family transportability.
- **The Frozen First-Principles Law (`platonicswap/platonic_organ_engine.py`, `ENGINE SHA-256 = 22ef57c9d196e12f`):**
  1. **Universal Task-Blind Channels (reused unchanged across all capabilities):**
     - *Read Channel ($A \to B$):* Ledoit-Wolf generative map $G = \Sigma_{BA}\Sigma_{AA,\text{LW}}^{-1} \in \mathbb{R}^{D_B \times D_A}$ (naturally rectangular across `2048 <-> 3072 <-> 1536`) fitted on a 300-passage multi-register unlabeled corpus (prose + open-domain encyclopedic + code/logs/QA, strictly blacklisted against all Cap-1 and Cap-2 terms), with a strict 50/50 split of the held-out residual bank (`[:mid]` for $N_{\text{LW}}$, `[mid:]` strictly untouched for conformal thresholds & abstention).
     - *Write Bridge ($A \to B$):* Smooth parameter-free Cauchy-weighted Stiefel SVD (`w_i = 1 / (1 + (||h_A||/med_A)^4 + (||h_B||/med_B)^4)`) on 1,200 task-blind short-context declarative concept tokens, with exact projected-variance scale (`gain == 1.000`) and exact architectural residual-multiplier inversion $W_{d,B} = \frac{1}{\mathrm{rm}_B} M_{\text{out}} W_{d,A}$.
  2. **Exact 2-Constraint KKT Channel-Optimal Direction (0 Magic Constants):**
     - Enforces a mandatory **Frame-Counterbalance Certificate** (`frame_only_auroc == 0.5000`).
     - Solves the exact closed-form 2-constraint KKT QP $\min w^\top S_{\text{bg}} w$ subject to $w^\top(\mu_{\text{pos,sim}} - \mu_{\text{neg,sim}}) \ge \Phi^{-1}(1-\alpha_{\text{neg}})$ and $w^\top(\mu_{\text{pos,sim}} - \mu_B) \ge \Phi^{-1}(1-\alpha)$, parameterized **solely** by 3 declared operational tolerances `FROZEN_TOLERANCES = {alpha: 0.002, alpha_neg: 0.03, delta: 0.02}`.
  3. **Real-B Conformal Thresholds, Analytic Sharpness & Unlabeled Abstention:**
     - $t_1 = Q_{1-\alpha}(X_{B,\text{ev}} w_c)$ (exact exchangeability false-fire guarantee $\le \alpha$ on fresh text), $\Gamma = \operatorname{Std}(X_{B,\text{ev}} w_c)$, $t_2 = t_1 + \Gamma$, and exact analytic sharpness $\beta = 0.27846454 / (\delta \cdot \Gamma)$ (bounding off-trough and on-plateau SiLU ripple by $\delta = 0.02$).
     - **Pure Digital `mlp_bias=False` Compilation (`2` rows per rule = `4` rows total):** $U = c_u$ (exact BOS-nulled constant pin). Automatically emits `ABSTAIN` if $R^2_{\text{chan,ev}} \le 0$ or $r_{\text{dir}} \le 0.25$ or $a_{\text{emp}} \le 0.15$ — catching `SHUFFLED_READ` **4/4** times automatically.

### Headline Result A — Strictly One-Shot Sealed Capability 2: `BIO-RIPPLE` (`Bird -> Fish`, `Mammal -> Reptile`)
*Executed ONCE with frozen `ENGINE SHA-256 = 22ef57c9d196e12f` and untouched `SEALED CAP-2 SHA-256 = d321fd0255c47cd3` (48 strictly unseen test species, 408 eval trials per model, 0.0% geography overlap, zero hyperparameter edits, reusing the exact frozen Cap-1 universal sockets).*

| Solo Recipient (`mlp_bias=False`, Donor A Unloaded) | Dim Transport | Direct FC (48 unseen) | **4-Attr Unseen Ripple FC (192 trials)** | **2-Sent Compositional Hops FC / EM (96 trials)** | McNemar Ripple vs Shuffled-R Forced | Cross-Organ Cap-1 Swap Rip | **196-Fact Unrelated Broken** | **General-Text KL** | `SHUFFLED_R` Auto / Bit-UL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **`smollm2-1.7b` L9** (`d_ff 8192→8196`) | `2048 → 2048` | **41.7%** | **57.8%** *(> In-Ctx 55.7%)* | **51.0% / 24.0%** *(> In-Ctx 32.3%)* | **81:0 (`p=8.27e-25`)** | 12.0% | **0 / 181 (100.0%)** | **0.00107** | `ABSTAIN` / `True` |
| **`granite-3.1-2b` L12** (`rm=0.22`, fp32) | `2048 → 2048` | **6.2%** | **34.9%** | **43.8% / 18.8%** *(> In-Ctx 28.1%)* | **40:0 (`p=1.82e-12`)** | 15.6% | **0 / 192 (100.0%)** | **0.00076** | `ABSTAIN` / `True` |
| **`falcon3-3b` L9** (`d_ff 9216→9220`) | `2048 → 3072` | **12.5%** | **51.6%** | **42.7% / 30.2%** | **68:0 (`p=6.78e-21`)** | 12.0% | **0 / 186 (100.0%)** | **0.00071** | `ABSTAIN` / `True` |
| **`qwen2.5-1.5b` L13** (`d_ff 8960→8964`) | `2048 → 1536` | **54.2%** (`33.3%` EM) | **52.1%** | **45.8% / 18.8%** *(= In-Ctx 45.8%)* | **66:1 (`p=9.22e-19`)** | 15.1% | **0 / 163 (100.0%)** | **0.00119** | `ABSTAIN` / `True` |

- **Zero observed forgetting across all 4 families on Cap-2 (`0 / 722` broken unrelated facts)**; invariant retention `85.4–89.6%`; hard-negative (bat/whale/dolphin/moth) false-flip rate `0.0–4.2%`.

### Headline Result B — Capability 1 (Frame-Counterbalanced Geography, Same Frozen Engine `22ef57c9d196e12f`)
- **All 4 architectures fire above `SHUFFLED_READ/WRITE` (`0.0%`) with verified conformal false-fire rates `ff <= 0.00136 <= 0.002` and `4/4` `SHUFFLED_READ_AUTO == ABSTAIN`:**
  - `smollm2-1.7b L9`: `18.8%` dir / `18.8%` rip3 (`p=0.0039`) / `25.0%` DS-8 (`Prose Read × Concept Write` `2×2` ablation: **`56.2%` dir / `50.0%` rip3** vs `0.0% / 8.3%` long-paragraph write, isolating why short-context concept write bridges are essential!).
  - `granite-3.1-2b L12` (`rm=0.22`): **`25.0%` dir / `22.9%` rip3 (`p=0.0010`) / `25.0%` DS-8**, `KL=0.00159`, `ff=0.00000`.
  - `falcon3-3b L9` (`2048 → 3072`): **`37.5%` dir / `50.0%` DS-8**, `KL=0.00122`, `ff=0.00129`.
  - `qwen2.5-1.5b L13` (`2048 → 1536`): **`50.0%` dir (`p=0.0391`) / `16.7%` rip3 (`p=0.0391`) / `50.0%` DS-8**, `KL=0.00059`, `ff=0.00042`.
- *Comparison against bespoke prototype (`results_organs_multifamily.json`):* Our earlier SmolLM2-only hand-factored `Entity × Country` bilinear prototype reached higher single-model Cap-1 ripple (`45.8–58.3%`, `75.0–87.5%` DS-8, `2–4/175` broken, `KL=0.00021`), but failed on 3/4 architectures and relied on bespoke constants—illustrating the exact tradeoff between single-task specialization and frozen cross-family generality.

---

## Pillar 3 — Cross-Capability Co-Implantation, Output-Layer Comparison Foil & Bit-Verified Unlearning

1. **Cross-Capability Co-Implantation & Selective Bit-Exact Unlearning (`results_bio_ripple_sealed.json`):**
   - Co-implanting **Cap-1 Geography (4 rows) + Cap-2 Biology (4 rows)** simultaneously into solo `smollm2-1.7b L9` (`d_ff 8192 → 8200`, Donor A unloaded): both organs fire simultaneously (`Cap-1: 18.8% / 18.8%`, `Cap-2: 41.7% dir / 54.7% rip`).
   - Slicing out Cap-1 (`8200 → 8196`): Cap-1 reverts to `0.0%`, locality restores to **`0 / 175` broken (`100.0%`)**, and Cap-2 outputs are **bit-identical** to solo Cap-2 (`torch.equal == True`, `max |Δlogit| = 0.0`).
   - Slicing out Cap-2 (`8200 → 8196`): Cap-2 reverts to `0.0%`, and Cap-1 is **bit-identical** to solo Cap-1 (`max |Δlogit| = 0.0`).
   - Slicing all (`8192`): **bit-identical** to the original checkpoint (`torch.equal == True` on both `state_dict` and deterministic logits; note exact unlearning is guaranteed by construction for appended rows and reported strictly as a sanity verification).

2. **Output-Layer Comparison Foil (`results_pilot2_gated.json`, `results_pilot3*.json` — "Late-Layer Flashcard" vs. Mid-Layer Organ):**
   - A late-layer (L23) gated side-module achieves `84.4%` 8-way / `62.5%` EM direct recall, **`0.00 pp` locality drop (`0 / 1,876` false fires)** vs MEMIT's `+24.49 pp` drop, and **62.5% (`20/32`) solo 2-step greedy-to-greedy rollout & hot-swap** across 3 families (`Qwen2.5-3B` $f$ + `Falcon3-3B` $g \leftrightarrow g'$ → solo `SmolLM2-1.7B`, replicated on `Qwen1.5B`, `Falcon3`, and `Granite`), **but exhibits `0%` unseen mid-layer ripple**—providing a clean mechanistic contrast with Pillar 2's mid-layer SwiGLU organs.

---

## Honest Limitations & Scope
- **Capability class tested:** Both Cap-1 and Cap-2 are linearly triggerable, single-layer concept/entity-rule updates; extending to arbitrary multi-token algorithmic or attention-routing capabilities (e.g., code semantics, refusal, or style circuits with input-dependent writes) remains future work.
- **Cap-1 locality vs. Cap-2 locality:** On sealed Cap-2 `BIO-RIPPLE`, unrelated 196-fact regression was `0 / 722` across all 4 models; on Cap-1 Geography, unrelated regressions ranged from `9–34` facts (`78.1–94.9%` retention) because Cap-1's broader entity-vs-prose contrast overlapped more closely with `dataset.json` entity prompts.
- **Multi-donor Stage B (`Qwen + Falcon → SmolLM2`):** Under the frozen `N=1,200` concept write bridge, Qwen's `France→Japan` half fired cleanly (`37.5%` dir / `37.5%` rip / `66.7%` DS-8) while Falcon's rectangular `3072→2048` `Italy→Brazil` half remained silent (`0.0%`) due to `N=1,200 < D=3,072` write-bridge underdetermination; scaling the unlabeled concept write pool to `N >= 6,000` is the natural next step.

---

## Repository Layout
- `platonicswap/` — Core modules (`platonic_organ_engine.py`, `gated_side_module.py`, `maps.py`, `systems.py`, `extract_and_baseline.py`, `dataset.py`).
- `experiments/` — Reproducible Pillar 1–4 scripts (`pilot0_diagnostics.py`, `pilot1_fisher_scaling.py`, `pilot1b_break85.py`, `pilot2_payload_memit.py`, `pilot3_split_composition.py`, `pilot_graft_falsify.py`, `pilot_graft_t3_gap*.py`, `pilot_organs_p0.py`, `pilot_organs_multifamily.py`, `platonic_organ_engine.py`, `bio_ripple_sealed.py`, `build_repo_tables.py`).
- `data/` — Benchmark splits (`dataset.json`, `dataset_2hop.json`).
- `results/` — All 21 audited JSON outputs, trial ledgers (`trial_ledger.csv`, `bridge_2hop_ledger.csv`), and [`master_audited_results.md`](results/master_audited_results.md).
- `audits/` — Independent 4-model adversarial council reviews (Opus 5.5 Max, Fable 5.2 Max, Astra 6.2 Ultra, Neon High) and synthesis.
- [`REPRODUCE.md`](REPRODUCE.md) — Exact environment setup, CPU unit tests, and MPS reproduction commands.
