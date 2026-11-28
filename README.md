# PlatonicSwap (`v1.0`)

**Cross-Family Activation Transfer & Native Mid-Layer SwiGLU "Platonic Organs"**

PlatonicSwap investigates and extends Stanford Hazy Research's *Plato's Key* across **5 ungated base LLMs spanning 4 pretraining families, 3 HuggingFace architecture classes, and 3 hidden widths** (`Qwen2.5-3B` D=2048, `Qwen2.5-1.5B` D=1536, `SmolLM2-1.7B` D=2048 [`LlamaForCausalLM`], `Granite-3.1-2B` L=40 D=2048 `rm=0.22` [`GraniteForCausalLM`], and `Falcon3-3B` D=3072 [`LlamaForCausalLM`]; **3 cross-family recipients and 1 intra-family recipient** from donor `Qwen2.5-3B`), all reproducible on a single local 48 GB Apple Silicon workstation.

---

> **Headline.** PlatonicSwap `v1.0` performs **closed-form cross-model compilation of manually specified concept-steering rules** (using donor activations for labeled class pairs, not automatic discovery of a pretrained donor reasoning circuit) into **four appended bias-free SwiGLU rows** (`d_ff -> d_ff + 4`) of a recipient from a different pretraining family. The donor is completely unloaded at inference; there are zero forward hooks, zero attention masks, zero lookup tables, zero recipient gradient steps, and base weights are untouched. The frozen `v1.0` archival engine (`experiments/platonic_organ_engine.py`, `SHA-256: 22ef57c9…`) and its task-blind universal sockets were developed on Cap-1 (geography) and then applied unchanged to a second, disjoint benchmark specified in advance with a single recorded execution (`908.7 s`, no tuning): Cap-2 `BIO-RIPPLE` (`Bird -> Fish`, `Mammal -> Reptile`; 48 held-out species; 408 trials per recipient).
>
> On Cap-2, **all 8 compiled rule-organs (2 rules × 4 recipients) raised preference for unseen downstream attributes above both the unedited recipient and a forced shuffled-read control** (descriptive `8/8` robustness; binomial sign-test `p = 0.0039` one-sided / `0.0078` two-sided *under an independence assumption that is not established across shared donor/rules/templates*, so we lead with species-clustered CIs). Species-clustered 95% CIs separate from base on **4/4 recipients** (`smollm2-1.7b`: `57.8% [49.5, 65.6]` vs. base `12.5% [7.8, 17.7]`; species-level sign-test bound `p <= 2.38e-7` one-sided / `4.77e-7` two-sided). The organs **broke 0 of 722 base-correct facts on an out-of-domain 196-fact probe** (`0/181, 0/192, 0/186, 0/163`; one-sided 95% Clopper-Pearson upper bound `<= 1.8%` per recipient), kept mean general-text KL `<= 0.0012`, and **automatically abstained on 4/4 shuffled channels** (`8/8` across Cap-1 + Cap-2). Slicing either capability out of a Cap-1 + Cap-2 co-implant reproduces the other's solo logits bit-for-bit (`max |Δlogit| = 0.0`), and slicing both restores the base checkpoint identically (**bit-exact modular removal**, guaranteed by construction for appended rows and verified via `torch.equal`).
>
> **Scope & Current Project Status (`v1.0` – `v5.1`).**
> - **What this repository honestly establishes today:** (1) Closed-form, donor-derived concept steering can be installed as appended native rows and affect held-out species and downstream attributes, with stronger first-token preference (**FC**) than greedy generation (**EM**, ripple EM `8.8–22.9%`) and task-dependent specificity; (2) cross-model representations and positional RoPE attention patterns can be aligned usefully (though attention alignment alone does not guarantee copying gains); (3) smaller recipients can exploit donor-computed oracle signals injected at the outlet; (4) **the tested autonomous student-input adapters in `v3`–`v5.1` (`Qwen3-1.7B -> 0.6B`) fail their declared behavioral gates (`STOP_BEFORE_G3` / `Exp 7b` hard stop)**; (5) **Pythia-160m matched-width Zero-SGD standalone circuit transplant (`docs/PYTHIA_V1_STANDALONE_TRANSPLANT.md`)**: using task-guided circuit selection on discovery splits and closed-form generic-code interface fits (zero SGD, zero hooks), 24 frozen donor attention heads from `EleutherAI/pythia-160m` compile into a standalone 15-layer `GPTNeoXForCausalLM` checkpoint of `pythia-160m-seed2@step2000` (148/148 original tensors byte-identical, `0.0` rollback diff), improving untouched sealed `A B ... A -> B` accuracy over `STOCK`, 5 disjoint `RAND` grafts, KL-matched `RAND`, and edge-only cuts across all 3 formats (`n = 256` each; all 12 Holm-adjusted `p <= 3.82e-4`; primary `rs6` `+54.8%` vs `STOCK` / `+64.0%` vs `RAND₅`), verified causally via edge-only cuts and two-foil key rerouting, at an explicit cost of `+13.10%` stored (`+2.91%` non-zero) parameters and `+30.7%` held-out code perplexity (`KL = 0.2414`); (6) **v1.1 retention repair & v2 falsification boundaries (`docs/PYTHIA_V2_RETENTION_AND_LIMITS.md`)**: matching published v1's `Rh (L5->L3)` alignment on a discovery benchmark (`n=256`), an exact closed-form `GELU(z) - GELU(-z) = z` parallel-MLP subtractor (`d=768 err=9.30e-6`, zero hooks, `148/148` frozen tensors, `0.0` rollback; `1.15×` alone) combined with stronger ridge shrinkage (`lam=0.10`) cuts generic-code KL **`2.70×`** (`0.2094 -> 0.0774`, `PPL 1.108×`; direct-`p` removal leaves `85.5%` of KL); however, an adversarial **Recipient-Self-Graft** control (24 distinct native heads) beats cross-seed induction at overlapping KL (`rs6 = 0.6016, nl6 = 0.7227` at `KL = 0.0753` vs. donor `0.5000 / 0.6289` at `KL = 0.0774`); in discovery scratch diagnostics, `pythia-410m -> 160m` induction failed Gate 2 (`0.185`) and locked-gain Gate 3 (`37.3%`), and on low-skill `seed2@step3000` IOI only the hooked donor oracle strongly beat `SELF-GRAFT` (`89.1% / 62.5%` vs `33.3% / 0.0%`), while tested `POST-L11` configurations failed an exact Donor-attention Gate-0 sweep (`fc2 = 0.417, ΔSELF = +0.083` at `KL = 0.091`); and (7) native expansions, storage lanes, and bit-exact rollback are workable engineering mechanisms.
> - **Detailed `v1.0` caveats:** Direct class-label flips are significant vs. shuffled-read on **2/4 recipients** (`smollm2-1.7b` and intra-family `qwen2.5-1.5b`; `falcon3-3b` `p=0.125`, `granite-3.1-2b` `p=0.25`). In-domain hard negatives show `8.3–33.3%` FC preference bleed (`0.0–4.2%` EM). A per-query in-context premise baseline roughly ties the organ on `smollm2-1.7b` ripple FC (`57.8%` vs. `55.7%`) and beats it on ripple EM everywhere (`38.0%` vs. `22.9%` on SmolLM2); the organ beats that premise baseline on the direct class label on 4/4 recipients (by `6–52 pp`). On Cap-1 Geography, the same frozen engine transfers more weakly (unseen ripple `6.2–22.9%` EM, significant vs. shuffled on 3/4 recipients, with `15–34` unrelated facts broken).

All `v1.0` tables and numbers below are generated programmatically from the 24 committed `v1.0` JSON artifacts in [`results/`](results/) via [`experiments/build_repo_tables.py`](experiments/build_repo_tables.py) — see [`results/master_audited_results.md`](results/master_audited_results.md) for the serialized manifest, confidence intervals, per-rule/per-attribute scorecards, and exact McNemar statistics (`v2`–`v5.1` JSONs are documented in `docs/V*.md`).

---

## Sealing, Provenance & Reproducibility Lock

| Artifact | Full SHA-256 | Role & Provenance |
| --- | --- | --- |
| `experiments/platonic_organ_engine.py` | `22ef57c9d196e12f2cd3c9ecc79ec0813803df43d5a6009f5422904948802306` | Frozen `v1.0` archival engine matching `meta.engine_sha256` in canonical `v1.0` JSONs (untouched; see Errata below for `v1.0.1` library fixes in `platonicswap/platonic_organ_engine.py`) |
| `experiments/bio_ripple_sealed.py` | `001d296b4c278ab8b86f31363844b9d65ecc933761b12b02ec0afff9fc910fcb` | Cap-2 runner; internal payload digest `d321fd0255c47cd3…` (self-attested; specified in advance, single recorded execution `908.7 s`, no tuning; not externally pre-registered prior to git init) |
| `results/results_cap1_channel_pushforward.json` | `bcf2dc70997e22c82418699535f9f7c9aa19967618978407170ee73ea047376e` | Canonical `v1.0` Cap-1 run (`meta.engine_sha256` matches above) |
| `results/results_bio_ripple_sealed.json` | `cd6c3c02a0881e04acbcdd4ea2144fc63389ebf61651395a1ac37912ee42cc23` | Canonical `v1.0` Cap-2 single-execution run (`meta.engine_sha256` matches above) |
| `results/socket_manifest.json` | *(manifest file)* | SHA-256 & byte-size index of the 17 gitignored `experiments/cache/*.pt` universal sockets (~5.1 GB) |

---

## Definitions & Statistical Notes
- **FC vs. EM:** **FC** (forced-choice) compares first-token log-probabilities between the new and old candidate answers (`log P(new) > log P(old)`). **EM** checks whether greedy generation starts with an acceptable answer alias at a word boundary. On `granite-3.1-2b`, first-sub-token FC compares `" re"` vs. `" m"` for `reptile` vs. `mammal` (a single-token tokenization proxy).
- **Eligibility:** An item is *eligible* if the unedited `BASE` model prefers the old (true biological) answer under FC.
- **Item-Level vs. Clustered Tests:** Cap-2 evaluates 4 attributes per species (`n = 192` over 48 species) and 2 hop prompts per species (`n = 96`). Item-level McNemar p-values treat items as independent; we lead with **species-clustered 95% bootstrap CIs**, the conservative **species-level sign-test bound**, and the **8/8 rule×recipient organ sign test**.
- **Cap-2 `hop1_heat_lamp` Floor:** The prompt template `"…with a heat lamp because it is"` entails `" cold"` regardless of species, creating a flat `24/48` (`50.0%`) floor on `Mammal -> Reptile` hops for both `BASE` and the organ. **Eligible-subset hops** (removing base-preflipped items) are the primary hop metric.
- **In-Context Premise Hard Negatives:** Hard-negative prompts are never premise-prefixed (`bio_ripple_sealed.py:337–347`), so `IN_CONTEXT_PREMISE` hard-negative metrics equal `BASE` by construction and are reported as `n/a (unprefixed)`.
- **Out-of-Domain Locality Probe:** The 196-fact probe (`data/dataset.json`) tests general factual recall (capitals, countries, elements, authors, etc.) and is out-of-domain for Cap-2 biology (`0/722` broken; one-sided 95% upper bound `<= 1.8%` per recipient). In-domain bleed is measured separately via the 24 Cap-2 hard negatives (`8.3–33.3%` FC / `0.0–4.2%` EM).

---

## Pillar 1 — Zero-QA Geometry, Causal Fisher Tail, & Attention-Knockout Decomposition

1. **Plato's Key Replication & Zero-QA Geometry (`results_baseline.json`, `results_pilot0.json`, `results_pilot1*.json`):**
   - **Like-for-like Plato's Key replication (`20` ordered pairs, `5` models):** Median cross-model retention vs. self-swap is **`70.9%`** (median transfer `58.8%`, self-swap `87.4%`, shuffled `3.6%`), closely matching *Plato's Key*'s `~73%`.
   - **The 84–95% Causal Fisher Tail:** `84.1%` (`smollm2-1.7b`) to `95.0%` (`granite-3.1-2b`) of causal factual Fisher sensitivity lies outside the top-64 prose PCA subspace (`AUC(abs(g·δ)) = 0.820` vs. Euclidean `L2 = 0.631`). Low-rank `k = 64` closed-form Fisher-PLS outperforms PCA-Procrustes by **`1.73–2.01×` raw (`1.85–2.26×` net of shuffle)** across all 3 pairs.
   - **Zero-QA Map Fitting (`N = 10,000` unlabeled states):** Fitting rigid orthogonal $O(d)$ maps with zero QA labels adds **`+0.6` to `+5.8 pp`** over QA-shared Procrustes (overlapping 95% CIs), beats QA-train10rels by **`+25.3` to `+40.6 pp`** on Leave-One-Relation-Out (LORO) (`+8.8` to `+15.6 pp` vs. QA-shared), and outperforms unconstrained Ridge $GL(d)$ on **2 of 3 pairs** (`+8.7 pp` SmolLM2, `+10.4 pp` Granite; `-0.5 pp` on Falcon→SmolLM2) despite lower Euclidean $R^2$. *(Note: swap layers were chosen via labeled validation transfer; "0-QA" refers to map fitting only.)*

2. **Natural Single-Layer vs. Attention-Knockout Regimes (`results_pilot1b_*.json` — Strictly Separated):**

| Donor → Recipient | Natural Single-Layer CF (0 KO) | Matched Natural Self-Swap | Natural Ratio | Earlier-Subject KO CF | Full-Subject KO CF (1x / 5x Powered) | Shuffled Ctrl (1x / 5x) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-3B` → `SmolLM2-1.7B` | **83.2%** (`B_real_cross_ctx`, AD `80.3%`) | 87.9% (`A0_self_single_l9`) | 94.7% | 85.2% | **89.3%** / **88.8%** (AD `83.8%`) | 0.0% / 0.0% |
| `Qwen2.5-3B` → `Granite-3.1-2B` | **52.9%** (`ridge_clamped_1_5`, LORO `57.1%`) | 53.6% (`A0_self_l8`) | 98.8% | 88.2% | **91.5%** / **92.0%** (AD `88.5%`) | 0.0% / 0.0% |
| `Falcon3-3B` (`3072`) → `SmolLM2-1.7B` (`2048`) | **74.6%** (`proc_ls_scale`) | 83.1% (`A0_self_single_l9`) | 89.8% | 79.1% | **87.6%** / **87.2%** | 0.6% / 0.1% |

   - Crossing `>= 85%` requires a **recipient-side attention knockout** (`earlier_subj` or `full_knockout`) that suppresses the residual original-subject route diagnosed in Pilot 0 (`43.6–45.3%` of failures).
   - **T3 Latent-Knowledge Unlocking with Donor-Attributable Content (`results_graft_t3_gap*.json`):** On 21 newly curated `A-right, B-startswith-wrong` obscure entities on SmolLM2 (`n = 38` total with the initial `n = 17`), zero-QA $O(d)$ transport recovers **`76.2–81.0%`** country (`17:0, p=1.53e-5` vs. `0.0%` B-self-roundtrip; `13:0, p=2.44e-4` vs. `shuffled_R`; `16:0, p=3.05e-5` vs. `wrong_entity`), with wrong-entity `e'` tracking `e'` in `81.0%` of cases. Because `10/21` base misses were region-first answers naming the country later and unpatched SmolLM2 already knew `11/21` capitals, `16/21` currencies, and `15/21` languages, we frame T3 as **latent-knowledge unlocking with donor-attributable content**; on the strict blind-spot subset (`n = 8`), `strict_blindspot_gap.unseen_3attr_ripple` is `70.8–75.0%` vs. `33.3%` B-self-roundtrip, and pooled strict per-attribute recovery is **`50.0%` vs. `0.0%` roundtrip** (`8:0, p=0.0078`).

---

## Pillar 2 — Frozen Channel-Pushforward Native SwiGLU Platonic Organs

### Construction (`platonicswap/platonic_organ_engine.py`, `SHA-256: 22ef57c9…`)
Parameterized by **three declared statistical tolerances** (`alpha = 0.002`, `alpha_neg = 0.03`, `delta = 0.02`) plus **three frozen abstention thresholds** (`R²_chan,ev <= 0`, `r_dir <= 0.25`, `a_emp <= 0.15`):
1. **Task-Blind Universal Channels:** Ledoit-Wolf generative read channel $G = \Sigma_{BA}\Sigma_{AA,\text{LW}}^{-1}$ on 300 multi-register unlabeled passages (strict 50/50 residual split for $N_{\text{LW}}$ vs. untouched conformal calibration) + smooth parameter-free Cauchy-weighted Stiefel SVD write bridge on 1,200 task-blind short-context declarative concept tokens + exact architectural residual-multiplier inversion $W_{d,B} = \frac{1}{\mathrm{rm}_B} M_{\text{out}} W_{d,A}$.
2. **Exact 2-Constraint KKT Direction + Conformal Digital Compilation:** Enforces a mandatory frame-counterbalance certificate (`frame_only_auroc == 0.5000`), solves the closed-form 2-constraint KKT QP, sets real-B conformal thresholds (`t1 = Q_{1-alpha}(X_{B,ev} w_c)`, `t2 = t1 + Std`), analytic SiLU sharpness `beta = 0.27846454 / (delta * Std)`, and pure digital BOS-nulled bias-free SwiGLU rows (`2` rows per rule = `4` rows total).
3. **Regime 1 Pretrained Neuron Conjugation (`results_organs_p0.json`):** Conjugating 2,000 stock `Qwen2.5-3B` neurons into `SmolLM2-1.7B` (`mlp_bias=False`) yields `median r(g) = 0.774`, `|Δ|/σ = 0.0425` (`20×` less drift than naive `0.831`), and Gate G0 retrieval `95.8%` top-1 (`100.0%` top-5) on known landmarks / `88.2%` top-1 (`100.0%` top-5) on Gap-17 entities.

### Cap-2 `BIO-RIPPLE` Scorecard (`CHANNEL_PUSHFORWARD_BIASFREE`, `results_bio_ripple_sealed.json`)
*Single recorded execution (`908.7 s`) with frozen engine `22ef57c9…` and Cap-1 universal sockets; 48 held-out species, 408 eval trials/model.*

| Solo Recipient (`mlp_bias=False`, Donor Unloaded) | Direct FC / EM (vs Shuf p) | 4-Attr Ripple FC [Sp. 95% CI] vs BASE | Ripple EM (In-Ctx EM) | Hops FC (Elig FC) / EM | HardNeg FF FC / EM (BASE FC) | Out-of-Domain Broken (all 196) | KL ; Firing (al / unal) | Shuf Auto / Bit-Rem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`smollm2-1.7b` L9** (`2048→2048`) | **41.7% / 8.3%** (`p=3.8e-6`) | **57.8% [49.5, 65.6]** vs 12.5% [7.8, 17.7] | 22.9% (38.0%) | 51.0% (**34.7%**) / 24.0% | 8.3% / 0.0% (0.0%) | **0 / 181** | 0.00107 ; 0.13% / 0.10% | `ABSTAIN` / `True` |
| **`granite-3.1-2b` L12** (`rm=0.22`, fp32) | 6.2% / 4.2% (`p=0.25`) | **34.9% [27.6, 42.7]** vs 15.6% [10.9, 20.8] | 8.8% (25.5%) | 43.8% (**20.6%**) / 18.8% | 12.5% / 0.0% (4.2%) | **0 / 192** | 0.00076 ; 0.17% / 0.64% | `ABSTAIN` / `True` |
| **`falcon3-3b` L9** (`2048→3072`) | 12.5% / 0.0% (`p=0.125`) | **51.6% [45.8, 57.3]** vs 12.0% [8.3, 15.6] | 13.5% (33.3%) | 42.7% (**20.9%**) / 30.2% | 25.0% / 4.2% (0.0%) | **0 / 186** | 0.00071 ; 0.14% / 0.12% | `ABSTAIN` / `True` |
| **`qwen2.5-1.5b` L13** (`2048→1536`, intra-family) | **54.2% / 33.3%** (`p=3.0e-8`) | **52.1% [45.8, 58.3]** vs 15.1% [10.9, 19.8] | 14.1% (40.1%) | 45.8% (**23.5%**) / 18.8% | 33.3% / 4.2% (0.0%) | **0 / 163** | 0.00119 ; 0.16% / 0.08% | `ABSTAIN` / `True` |

- **Per-rule breakdown (Table 4B-iv):** Both `Bird -> Fish` and `Mammal -> Reptile` raise ripple FC above BASE and forced shuffled-read on all 4 recipients (**8/8 rule-organs**); `Bird -> Fish` carries direct FC (`12.5–100.0%`) and hop gains, while `Mammal -> Reptile` direct FC stays `0.0–12.5%` and hops sit at the `24/48` `hop1_heat_lamp` floor.
- **Cross-organ specificity:** Implanting the Cap-1 Geography organ on Cap-2 prompts gives **`0.0%` direct FC/EM on 4/4** and base-level ripple FC (`12.0–15.6%`), while breaking `18–41` of the same 196 facts—confirming that `0/722` breakage is a property of Cap-2's trigger separation, not a universal guarantee of the engine.
- **Invariant accuracy:** `100.0%` FC across all 4 recipients; EM accuracy `85.4–89.6%` (vs. `BASE` `72.9–85.4%`).

### Cap-1 Geography Scorecard (`PUSHFWD_DIGITAL_BIASFREE`, `results_cap1_channel_pushforward.json`)
- **Per-recipient summary (`n = 16` entities/attribute; `SHUFFLED_READ_AUTO == ABSTAIN` on 4/4):**
  - `smollm2-1.7b L9`: `18.8%` Direct / `18.8%` 3-Attr Ripple (`p=0.0039` item-level; entity-level `p >= 0.0625`) / `25.0%` DS-8, `15/175` unrelated broken, `KL=0.00959`. (`2×2` `Prose Read × Concept Write`: `56.2%` dir / `50.0%` rip3 vs. `0.0% / 8.3%` long-paragraph write.)
  - `granite-3.1-2b L12`: `25.0%` Direct / `22.9%` Ripple (`p=0.0010`) / `25.0%` DS-8, `22/183` broken, `KL=0.00159`.
  - `falcon3-3b L9`: `37.5%` Direct / `6.2%` Ripple (`p=0.25`, n.s.) / `50.0%` DS-8, `19/179` broken, `KL=0.00122`.
  - `qwen2.5-1.5b L13`: `50.0%` Direct (`p=0.0391`) / `16.7%` Ripple (`p=0.0391`) / `50.0%` DS-8, `34/155` broken, `KL=0.00059`.
- **Hard-negative floor:** `Kyoto` and `Nagoya` answer `Japan` natively, giving every untouched `BASE` model a `12.5%` (`2/16`) baseline on `hardneg_false_flip_to_rule`; net deltas vs. `BASE` are `+6.2 pp` (`smollm2`, `granite`, `falcon3`) and `+25.0 pp` (`qwen1.5b`). See Table 4A-iv for per-attribute `Capital / Currency / Language` splits.

---

## Pillar 3 — Cross-Capability Co-Implantation, Bit-Exact Modular Removal & Output-Layer Foil

1. **Cross-Capability Co-Implantation (`smollm2-1.7b L9`, `d_ff: 8192 → 8200 → 8196 → 8192`, Table 4B-iii):**
   - Co-implanting **Cap-1 Geo (4 rows) + Cap-2 Bio (4 rows)** simultaneously preserves both organs (`Cap-1`: `18.8% / 18.8%`; `Cap-2`: `41.7%` dir FC / `54.7%` rip FC, vs. `57.8%` solo; `KL = 0.01059 ≈ 0.00959 + 0.00107`).
   - **Slicing Cap-1 (`8200 → 8196`):** Cap-1 returns to `0.0%`, locality restores to `0/175` broken, and Cap-2 logits are **bit-identical** to solo Cap-2 (`max |Δlogit| = 0.0`).
   - **Slicing Cap-2 (`8200 → 8196`):** Cap-2 direct returns to `0.0%` (ripple to base-level `12.0%` FC / `5.2%` EM) and Cap-1 logits are **bit-identical** to solo Cap-1 (`max |Δlogit| = 0.0`).
   - **Slicing both (`8192`):** Restores base checkpoint bit-identically (`torch.equal == True` on `state_dict` and logits).
2. **Output-Layer Comparison Foil (Layer-23 Gated Sidecar vs. Non-Canonical L23 `W_down` Overwrite, Table 2B-foil & Table 3B):**
   - A hooked late-layer (`L23`) gated sidecar achieves `84.4%` 8-way / `62.5%` EM direct recall with `0.00 pp` unrelated-fact drop (`0/1,876` token false fires) vs. `+24.49 pp` for a **non-canonical single-layer L23 closed-form `W_down` overwrite (`α=1.5, λ=10`)** *(not canonical multi-layer MEMIT)*.
   - Full 2-step greedy-to-greedy split-donor composition (`Qwen f + Falcon g/g' → SmolLM2`, `62.5%` EM) succeeds on **`smollm2-1.7b` only** (`6.2%` Qwen-1.5B, `18.8%` Falcon3, `15.6%` Granite). Mid-layer multi-hop ripple was not measured for these output-layer foils.

---

## Honest Limitations & Frozen-Engine Errata

### Scope Limitations
- **Capability class:** Both Cap-1 and Cap-2 are single-layer, linearly triggerable constant-write concept rules. Attention-routing, input-dependent writes, and multi-layer procedural skills are addressed in `v2` (see [`docs/COUNCIL_SYNTHESIS.md`](docs/COUNCIL_SYNTHESIS.md) and [`docs/V1_CLOSEOUT_COUNCIL_R2.md`](docs/V1_CLOSEOUT_COUNCIL_R2.md)).
- **Single-layer zero-SGD ceiling:** Quantitative ceiling analysis (`docs/V1_CLOSEOUT_COUNCIL_R2.md`) shows that `v1.0`'s 1,200-token concept write bridge (`cos = 0.38–0.50`) and general-text-only `t1/t2` bound single-layer cross-family transfer well below `85–95%`.
- **Stage B multi-donor silence (`Falcon3 3072 → SmolLM2 2048`):** In Cap-1 Stage B, Qwen's `France → Japan` half fired (`37.5% / 37.5% / 66.7%` DS-8) while Falcon's `Italy → Brazil` half stayed silent (`0.0%`). *Hypothesis (untested in v1):* the `N = 1,200` concept write pool is underdetermined for `D = 3,072`; scaling to `N >= 6,000` is deferred to `v2`.

### Frozen-Code Errata (Documented Without Modifying Hashed `v1.0` Engine Bytes)
1. **`experiments/platonic_organ_engine.py` docstring vs. code (`SHA-256: 22ef57c9…`):** Line 27 documents `alpha_neg = 0.02`, whereas the executed constant at line 90 (and all canonical `v1.0` JSONs) uses **`alpha_neg = 0.03`**. Lines 32–34 describe a two-sided `t1 = max(general, in-domain)` and `t2 = Q_{alpha_miss}`, whereas lines 963–967 compute `t1` from general text only and `t2 = t1 + Std` (leaving `alpha_neg` in the KKT direction solve only). Line 43 contains a legacy `/tmp/platos_key_lab` usage example. None of these lines were edited in `experiments/platonic_organ_engine.py` so the frozen `v1.0` SHA-256 remains byte-identical to the canonical runs.
2. **Compiler-Guard Defects Identified by Adversarial Audit (Fixed in `platonicswap/platonic_organ_engine.py`, `v1.0.1`):** Three reproducible defects exist in the frozen `v1.0` archival snapshot (`22ef57c9…`) and have been fixed in the live library copy `platonicswap/platonic_organ_engine.py` (verified by `experiments/verify_audit_fixes.py`):
   - **Frame-certificate enforcement (`F19`):** In frozen `v1.0`, `frame_ok` used `|auroc - 0.5| <= 0.15` and was not wired into `abstained` (all 90 serialized historical runs had `frame_only_auroc = 0.5000`, so historical runs were not confounded, but the guard did not reject bad inputs). Patched in `platonicswap/platonic_organ_engine.py` to require `|auroc - 0.5| <= 1e-6` and force `status = "ABSTAIN"`, `weights = None`.
   - **Conformal compiled-gate event (`F20`):** In frozen `v1.0`, `t1` was calibrated on raw `w_c · x > t1` rather than the compiled bias-free gate event `(w_c - t1 * c_B) · x > 0` (where pin `c_B · x` fluctuates), and small-`n` `k_conf` was clipped instead of returning `+inf`/abstaining. Patched in `platonicswap/platonic_organ_engine.py` to calibrate on the exact compiled ratio `z / (c_B · x)` (mapping `c_B · x <= 0` conservatively to `+inf`) and abstain when `(n_ev + 1)(1 - alpha) > n_ev`.
   - **Infeasible collinear QP (`F21`):** In frozen `v1.0`, `solve_2constraint_kkt` silently returned an infeasible fallback vector `[0, 0]` on contradictory collinear constraints. Patched in `platonicswap/platonic_organ_engine.py` to verify feasibility and raise `ValueError` on infeasible constraints.
3. **BOS linear up-null precision & offline tokenizer check:** `apply_exact_bos_null_affine` in `experiments/pilot_organs_p0.py` accepts `target_bos_gate = -20.0` for API compatibility without modifying `W_g`/`b_g` (docstring updated; sealed Cap-2 uses `apply_exact_bos_null_biasfree`). The linear up-null is exact at the fitted `x_BOS` vector (`~4.9e-7`) rather than identically zero across unseen pos-0 tokens (e.g., Cap-2 SmolLM2 `bos_eval60_max_norm = 0.028`). Note also that frozen `experiments/bio_ripple_sealed.py --unit-test` skips local HF tokenizer checks if `experiments/hf_cache` is not downloaded (do not treat `0/5` skipped tokenizers as verified tokenizer checks).

---

## Repository Layout
- `platonicswap/` — Core modules (`platonic_organ_engine.py`, `gated_side_module.py`, `maps.py`, `systems.py`, `extract_and_baseline.py`, `dataset.py`).
- `experiments/` — Reproducible Pillar 1–4 scripts, `run_referee_controls.py`, and `build_repo_tables.py`.
- `data/` — Benchmark datasets (`dataset.json`, `dataset_2hop.json`).
- `results/` — All 24 audited JSON artifacts, `socket_manifest.json`, CSV trial ledgers, and [`master_audited_results.md`](results/master_audited_results.md).
- `audits/` — Round 1 4-model independent council reviews and synthesis.
- `docs/` — 24 full-text reference papers (`papers/*.md`), `bibliography.bib`, [`COUNCIL_SYNTHESIS.md`](docs/COUNCIL_SYNTHESIS.md), and [`V1_CLOSEOUT_COUNCIL_R2.md`](docs/V1_CLOSEOUT_COUNCIL_R2.md).
- [`REPRODUCE.md`](REPRODUCE.md) & [`requirements-lock.txt`](requirements-lock.txt) — Exact setup, CPU verification gate, and MPS reproduction commands.
