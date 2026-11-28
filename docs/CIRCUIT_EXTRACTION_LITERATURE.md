# Circuit Extraction & Cross-Model Transport: Literature Survey for PlatonicSwap

**Prepared:** 2026-11-27  
**Scope:** Mechanistic interpretability, automated circuit extraction, entity binding, sparse transcoders/crosscoders, model diffing, causal abstraction, and zero-SGD cross-model representation transport — synthesized directly against our live `Qwen/Qwen3-1.7B → Qwen/Qwen3-0.6B` (`d_A = 2048 → d_B = 1024`, `28L`, `16Q/8KV`) circuit-grafting bottlenecks.

---

## 1. Executive Synthesis: Why Extraction Literature Matters Directly to PlatonicSwap

Our live `Qwen3-1.7B → Qwen3-0.6B` experiments revealed a sharp structural paradox:
1. **The recipient's `1,024`-dim residual stream *is* readable:** Replacing the whole residual at `B24` with a generic suffix-calibrated or vocab-anchored map from `A28` reaches **`28–29 / 32` (`87.5–90.6%`)** joint unrestricted full-vocabulary hits through four frozen recipient blocks (`B24..B27`), matching intact Donor (`28/32`) vs. Stock (`5/32`).
2. **Yet sparse standalone in-weights grafts (`L0/L11/L19`, `L0..L21`, native/quiet slots) tie or trail `STOCK_B` (`best Δs = 0.0000`),** even when donor-internal clean→bind patching recovers **`70.6–93.6%`** of the donor margin (`0.9–1.0%` for random controls).

Reading the modern mechanistic interpretability literature exposes **three exact extraction failure modes** (beyond transport geometry) that explain how a subgraph can recover `93.6%` under single-axis donor patching while still failing to be a portable, self-sufficient computational organ:

| Blind Spot in Current Extraction | Literature Phenomenon | Core Papers | Remedy for PlatonicSwap |
|---|---|---|---|
| **1. Choke-Point Illusion** | Patching a narrow bottleneck (`L11` or `L0/L11/L19`) flips the donor because the donor's *remaining* `L12..L27` machinery (which carries `86.6%` of the margin!) amplifies it — NOT because the 4–10 groups are self-contained. | **Faithfulness & Completeness** (Wang et al. 2022; Conmy et al. 2023; Goldowsky-Dill et al. 2023); **Hydra Effect & Backup Heads** (McGrath et al. 2023) | **Isolation / Completeness test** (corrupt/optimal-ablate *everything outside* the subgraph inside the donor) + **Hooked Subgraph-Only Sufficiency** inside the recipient *before* compiling weights. |
| **2. Invariant-Half Blindness** | Single-axis Binding-Swap keeps the Rule line (`Rule: Q6`) and Table 2 identical across Clean and Bind, making query-routing and value-readout heads mathematically invisible (`Δ = 0`). | **Multi-Axis Path Patching** (Wang et al. 2022; Goldowsky-Dill et al. 2023); **Causal Scrubbing** (Chan et al. 2022); **Binding IDs** (Feng & Steinhardt 2024) | **Multi-axis contrastive localization**: union of **Binding-Swap** (Table 1) + **Query-Swap** (Rule key) + **Table-2-Swap** (Table 2 values) + **Donor-minus-Recipient gap attribution**. |
| **3. Coarse-Component & MLP Opacity** | Whole KV-group grafting swaps `256` dense polysemantic channels (causing generic KL spikes or requiring aggressive ridge healing that cancels task writes) and cannot isolate sparse **SwiGLU MLP** computations (`L22/L23/L27` write `+5.91` pts in Donor!). | **Transcoders** (Dunefsky et al. 2024); **Cross-Layer Transcoders & Attribution Graphs** (Ameisen, Lindsey et al. 2025); **Sparse Feature Circuits** (Marks et al. 2024); **Sparse Crosscoders / Model Diffing** (Lindsey et al. 2024; Goodfire 2025) | **Feature-level & Transcoder extraction**: factorize MLP/cross-layer computation into sparse interpretable feature edges (`W_enc, W_dec`) and diff Donor vs. Recipient directly via **Crosscoders / Model Diffing**. |

---

## 2. Anthropic Interpretability: From Induction Heads to Attribution Graphs & Crosscoders

### 2.1 Induction Heads & Compositional Circuits (Olsson et al., 2022)
- **Citation:** Olsson, Elhage, Nanda, Joseph, DasSarma, Henighan, Mann, Askell, Bai, Chen, Conerly, Dawn, Drain, Ganguli, Hatfield-Dodds, Hernandez, Johnston, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, & Olah (2022), *"In-context Learning and Induction Heads"*, Transformer Circuits Thread.
- **Core Mechanism:** Two-layer attention composition: a **previous-token head** copies token `A`'s identity onto the following token `B` (K-composition), enabling a downstream **induction head** attending from a later occurrence of `A` to `B` to copy `B`'s value (V-composition).
- **Direct Connection to `Qwen3-1.7B → 0.6B`:** Our `L0 [1,3] → L11 [1,6]` edge is literally a generalized induction/binding circuit (`L0` copies `"Q6"` onto Table-1 `' cable'`, and `L11 h2/h13` attends `0.93` from Table-2 `' cable'` back to Table-1 `' cable'`). Crucially, `Qwen3-0.6B` *already possesses* the exact `16/16` homologous `L11` induction heads — meaning the missing capability is **higher-order compositional binding and downstream value-selection (`L14..L27`)**, not basic `L11` induction.

### 2.2 Scaling Monosemanticity & Superposition (Elhage et al., 2022; Templeton et al., 2024)
- **Citations:**
  - Elhage et al. (2022), *"Toy Models of Superposition"*, Transformer Circuits Thread.
  - Templeton, Conerly, Marcus, Lindsey, Bricken, Chen, Pearce, Citro, Ameisen, Jones, Cunningham, Turner, McDougall, MacDiarmid, Freeman, Sumers, Rees, Batson, Jermyn, Carter, Olah, & Henighan (2024), *"Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"*, Transformer Circuits Thread.
- **Core Insight:** Transformers pack far more features than residual dimensions (`n_features >> d_model`) in nearly orthogonal directions (superposition). Individual neurons and attention heads are **polysemantic**, mixing unrelated generic and task-specific computations.
- **Implication for PlatonicSwap:** Replacing an entire recipient KV group (`256` dims) or MLP block destroys dozens of unrelated superposed host features (forcing `GenKL = 0.04–0.28` unless heavily healed, while isotropic ridge healing `I - H_η` attenuates any donor write correlated with surviving host heads). Feature-level extraction targets the *sparse feature subspace* rather than dense head slots.

### 2.3 Sparse Crosscoders & Model Diffing (Lindsey et al., 2024)
- **Citation:** Lindsey, Templeton, Marcus, Conerly, Batson, & Olah (2024), *"Sparse Crosscoders for Cross-Layer Features and Model Diffing"*, Transformer Circuits Thread.
- **Core Mechanism:** A **Sparse Crosscoder** generalizes SAEs by reading activations from *multiple layers or two distinct models simultaneously* into a **shared sparse latent dictionary**, with separate per-layer / per-model decoder matrices `W_dec^{(A)}, W_dec^{(B)}`:
  $$f(x) = \text{ReLU}\!\left(\sum_{m \in \{A, B\}} W_{\text{enc}}^{(m)} h^{(m)}(x) + b_{\text{enc}}\right), \qquad \hat{h}^{(m)}(x) = W_{\text{dec}}^{(m)} f(x) + b_{\text{dec}}^{(m)}.$$
- **Model Diffing Diagnostic:** Comparing decoder norms `||W_dec^{(A)}[:, i]|| / (||W_dec^{(A)}[:, i]|| + ||W_dec^{(B)}[:, i]||)` cleanly partitions every feature into **shared features** (ratio `≈ 0.5`, with **built-in cross-model feature alignment `W_dec^{(A)}[:, i] ↔ W_dec^{(B)}[:, i]`!**) vs. **donor-exclusive features** (ratio `≈ 1.0`, the exact features `Qwen3-1.7B` has that `Qwen3-0.6B` lacks!).
- **Key Caveat Noted by Anthropic:** Model-exclusive features tend to be denser/more polysemantic ("exclusive-feature polysemanticity") because shared features absorb common variance while exclusive features compete for residual capacity — a key warning if we build donor-minus-recipient feature diffs.
- **Zero-SGD Charter Note:** Training a crosscoder uses unsupervised generic-corpus (`Q_T = 0`) reconstruction (still requires SGD for the *interpreter*, NOT the recipient weights — if strict zero-SGD applies even to discovery tools, closed-form CCA/SVD/NMF or analytical dictionary diffing provides a zero-SGD analog!).

### 2.4 Attribution Graphs, Circuit Tracing, & Cross-Layer Transcoders (Ameisen, Lindsey et al., 2025)
- **Citations:**
  - Ameisen, Lindsey, Pearce, Gurnee, Turner, Chen, Citro, Abrahams, Carter, Hosmer, Marcus, Sklar, Templeton, Bricken, McDougall, Cunningham, Henighan, Jermyn, Jones, Persic, Qi, Thompson, Zimmerman, Rivoire, Conerly, Olah, & Batson (2025), *"Circuit Tracing: Revealing Computational Graphs in Language Models"*, Transformer Circuits Thread.
  - Lindsey et al. (2025), *"On the Biology of a Large Language Model"*, Transformer Circuits Thread.
- **Core Mechanism:**
  1. **Cross-Layer Transcoders (CLTs):** Replace every dense MLP across layers with sparse interpretable features that read from layer `l`'s residual stream and **write directly to *all* downstream layers `l' >= l`** (`\sum_{l' \ge l} W_{\text{dec}}^{l \to l'} f_l`). This eliminates cross-layer MLP superposition and turns multi-layer MLP processing into a linear-in-features replacement model.
  2. **Local Linearization ("Frozen Attention & LayerNorm") & Virtual Weights:** By freezing attention patterns and LayerNorm denominators on a prompt, the replacement model becomes **piecewise linear** from every feature node to every downstream feature node and output logit! Edge weights are exact **virtual weights** `A_{s \to t} = a_s \cdot \frac{\partial a_t}{\partial a_s}` computed via backward Jacobian tracing — capturing **exact multi-hop paths across intermediate layers** (including error nodes for unexplained residual variance!).
  3. **Direct Relevance to `Qwen3`:** Anthropic's **frozen-attention + frozen-RMSNorm virtual-weight Jacobian** is the *exact* mathematical missing piece behind our `L0 → L11 → L19 → L24/25 → logits` gap! In `Qwen3-1.7B`, `L22/L23/L27` MLPs write `+5.91` margin points and `L24/L25` attention writes `+14.45` points; CLT/virtual-weight attribution graphs trace *every* multi-hop edge from `L0/L11` into `L22..L27` without the single-layer choke-point illusion.

---

## 3. Goodfire Interpretability Research: Transcoders, Model Diffing, Steering & Intentional Editing

- **Core Research Program (Goodfire AI, 2024–2026):** Founded by interpretability pioneers from OpenAI and DeepMind; focuses on transitioning from black-box behavioral iteration to **intentional model design** via sparse features, transcoders, model diffing, and causal interventions.
- **Key Technical Contributions Relevant to PlatonicSwap:**
  1. **Frontier SAEs & Reasoning-Feature Discovery (Llama 3, DeepSeek-R1):** Demonstrated that structured reasoning behaviors (backtracking, verification, multi-step lookup) correspond to sparse, causally steerable latent features rather than diffuse untraceable weights.
  2. **Transcoders & Collaborative Circuit Tracing (with Anthropic, DeepMind, EleutherAI, Neuronpedia):** Validated transcoders / CLTs as superior to single-layer SAEs for algorithmic circuit extraction because they directly approximate the **input→output functional transition** of MLP sublayers rather than merely reconstructing static activations.
  3. **Model Diff Amplification (`logits_amp = logits_after + α (logits_after - logits_before)`):** Highlights a powerful contrastive principle for **Donor-minus-Recipient diffing**: amplifying the difference between a stronger model (`1.7B`) and a weaker model (`0.6B`) isolates the *exact* behavioral and internal-state deltas where the two models diverge, preventing extraction from wasting budget on components the recipient *already has* (like `0.6B`'s native `L11` induction heads!).
  4. **Gradient-Attributed Feature Steering & Surgical Editing:** Emphasizes causal intervention validation (steering/editing specific feature directions with minimal collateral KL damage) — aligning directly with our Tail-KL (`p95/p99`) preservation standards.

---

## 4. Core & SOTA Mechanistic Interpretability Circuit Extraction Methods

### 4.1 Path Patching & IOI Completeness (Wang et al., 2022; Goldowsky-Dill et al., 2023)
- **Citations:**
  - Wang, Variengien, Conmy, Shlegeris, & Steinhardt (2022), *"Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"*, ICLR 2023.
  - Goldowsky-Dill, MacLeod, Sato, & Arora (2023), *"Localizing Model Behavior with Path Patching"*, arXiv:2304.05969.
- **Core Idea:** Instead of patching an entire node `v` (which mixes all downstream effects), **Path Patching** patches *only* the edge `u → v` (holding all other paths from `u` fixed at clean/corrupt). Wang et al. established the three gold-standard circuit criteria:
  - **Faithfulness:** `|F(M) - F(C)|` is small when running only circuit `C` (everything outside `C` mean-ablated).
  - **Completeness:** For *every* subset `K ⊆ C`, `|F(C \ K) - F(M \ K)|` is small — **directly ruling out the Choke-Point Illusion and hidden Backup Heads!**
  - **Minimality:** Every node `v ∈ C` is necessary for some subset `K`.

### 4.2 Causal Scrubbing (Chan et al., 2022)
- **Citation:** Chan, Garriga-Alonso, Goldowsky-Dill, Greenblatt, Nitishinskaya, Radhakrishnan, Shlegeris, & Thomas (2022), *"Causal Scrubbing: a method for rigorously testing interpretability hypotheses"*, Alignment Forum / Redwood Research.
- **Core Idea:** Tests a full computational graph hypothesis by recursively resampling inputs to every node from *different* examples that agree on the hypothesized invariant variable, while scrambling everything else. If a circuit hypothesis omits an "invariant half" (like the queried key `Q6` in our Binding-Swap contrast!), Causal Scrubbing immediately exposes the missing path when `Q6` is resampled!

### 4.3 ACDC — Automated Circuit Discovery (Conmy et al., 2023)
- **Citation:** Conmy, Mavor-Parker, Lynch, Heimersheim, & Garriga-Alonso (2023), *"Towards Automated Circuit Discovery for Mechanistic Interpretability"*, NeurIPS 2023 (arXiv:2304.14997).
- **Core Idea:** Iteratively traverses the computational graph in reverse topological order and prunes every edge `u → v` whose replacement with a corrupted activation changes the metric (e.g., KL divergence to intact logits) by less than threshold `τ`.
- **Strength / Weakness:** Exact nonlinear edge pruning, but `O(|E|)` forward passes (`thousands` of passes on 28L models) and sensitive to the corruption distribution and threshold `τ`.

### 4.4 Attribution Patching (`AtP`), `EAP`, `EAP-IG`, & `AtP*` (Nanda 2023; Syed et al. 2023; Hanna et al. 2024; Kramár et al. 2024)
- **Citations:**
  - Syed, Rager, & Conmy (2023), *"Attribution Patching Outperforms Automated Circuit Discovery"* (**EAP**), arXiv:2310.10348.
  - Hanna, Pezzelle, & Belinkov (2024), *"Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms"* (**EAP-IG**), COLM 2024 (arXiv:2403.17806).
  - Kramár, Lieberum, Shah, & Nanda (2024), *"AtP\*: An efficient and scalable method for localizing LLM behaviour to components"* (Google DeepMind), arXiv:2403.00745.
- **Core Idea:**
  - **EAP / AtP:** Linearizes patching score for *every* edge simultaneously in `O(1)` passes:
    $$\text{score}(u \to v) \approx (z_u^{\text{corrupt}} - z_u^{\text{clean}})^\top \nabla_{z_v^{\text{in}}} \mathcal{L}\big|_{\text{clean}}.$$
  - **EAP-IG (Hanna et al.):** Fixes zero-gradient saturation regions (e.g., softmax saturation in QK attention or SwiGLU gates!) using **Integrated Gradients** along the straight-line path `z(α) = z^{\text{corrupt}} + α(z^{\text{clean}} - z^{\text{corrupt}})` (`m = 5` steps), massively improving *faithfulness* (not just node overlap!).
  - **AtP\* (Kramár et al.):** Diagnoses two fatal failure modes of first-order AtP — **(1) QK-attention gradient distortion** (softmax curvature causes false negatives on Q/K paths) and **(2) cross-layer gradient cancellation / GradDrop** — and fixes both with exact attention-block handling and multi-step corrections, achieving near-exact activation-patching accuracy at `O(1)` cost on multi-billion-parameter LLMs.
  - **Note on Zero-SGD Charter:** Using backward gradients *inside the donor* strictly to rank/select which frozen donor weights to extract does **NOT** update any weights (0 SGD steps on weights!), though we must explicitly distinguish analytical gradient-based *localization* (on `disc_rank`) from strict zero-SGD *weight compilation* (`Q_T = 0`, purely generic closed-form linear algebra).

### 4.5 Transcoders & Sparse Feature Circuits (Dunefsky et al., 2024; Marks et al., 2024)
- **Citations:**
  - Dunefsky, Chlenski, & Nanda (2024), *"Transcoders Find Interpretable LLM Feature Circuits"*, NeurIPS 2024 (arXiv:2406.11944).
  - Marks, Rager, Michaud, Belinkov, Bau, & Mueller (2024), *"Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models"*, ICLR 2025 (arXiv:2403.19647).
- **Core Idea:**
  - **Marks et al. (SFC):** Applies linear attribution patching (`AtP` / `EAP-IG`) directly over SAE feature latents (plus SAE error nodes!) across every residual/attn/MLP site, finding sparse *concept-level* circuits that support **surgical model editing (SHIFT)** without damaging unrelated capabilities.
  - **Dunefsky et al. (Transcoders):** Replaces dense MLP `f_{\text{MLP}}(x)` with a wide sparse MLP `W_{\text{dec}} \text{ReLU}(W_{\text{enc}} x + b)`. Because the encoder reads `h_{\text{pre-MLP}}` and the decoder writes `h_{\text{post-MLP}}`, **weights-based input-invariant circuit edges** (`W_{\text{enc}}^{(l')} W_{\text{dec}}^{(l)}`) can be read off directly without dense neuron entanglement!
  - **Direct Path to SwiGLU MLP Extraction in `Qwen3`:** Standard `Qwen3` MLPs are `W_{\text{down}} (\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x)` (`d_{\text{ff}} = 6144 \to 3072`). Native SwiGLU neurons are polysemantic and hard to transplant individually without blowing up `GenKL`; a sparse transcoder decomposition (or sparse top-`k` gated-neuron community factorization) is the exact bridge from attention-only grafts to full **Attn + SwiGLU MLP circuit transfer**!

### 4.6 Optimal Ablation, Subnetwork Probing, & The Hydra Effect (Cao et al. 2021; McGrath et al. 2023; Li & Janson 2024)
- **Citations:**
  - Cao, Sanh, & Rush (2021), *"Low-Complexity Probing via Finding Subnetworks"*, NAACL 2021.
  - McGrath, Rahtz, Kramár, Mikulik, & Legg (2023), *"The Hydra Effect: Emergent Self-repair in Language Model Computations"*, arXiv:2307.15771.
  - Li & Janson (2024), *"Optimal Ablation for Interpretability"*, NeurIPS 2024 (arXiv:2409.09951).
- **Core Insights Directly Explaining Our Live Observations:**
  1. **Hydra Effect / Self-Repair (McGrath et al.):** When we delete `0.6B`'s native heads (or ablate a donor head), downstream components dynamically compensate — which is why single-component ablation underestimates distributed circuits AND why `Z_FAR` can unexpectedly recover `8/32` joint hits when `B19` slots are deleted and healed!
  2. **Optimal Ablation (Li & Janson):** Zero-ablation and mean-ablation push activations off-manifold, causing artificial loss spikes. Replacing an ablated component with the **loss-minimizing constant vector `c^*`** gives a principled, on-manifold baseline — directly paralleling why our `Z_SHR` (unbalanced ablation, `KL = 2.88`) created fake `+3.13` gaps while fair `Z_FAR` exposed the true effect!

---

## 5. Entity Binding & Causal Abstraction (Directly Relevant to Our `S1 → S2` Task!)

### 5.1 "How do Language Models Bind Entities in Context?" (Feng & Steinhardt, ICLR 2024)
- **Citation:** Feng & Steinhardt (2024), *"How do Language Models Bind Entities in Context?"*, ICLR 2024 (arXiv:2310.17191).
- **Core Mechanism — Binding IDs:**
  - On multi-entity key-value binding contexts (`Table 1: Q6 -> cable; ... Table 2: cable = sigma`), sufficiently capable LLMs attach **abstract Binding ID vectors** `b_1, b_2, \dots, b_K` living in a **low-dimensional continuous binding subspace** to both the entity and its bound attribute/key!
  - Binding IDs are **factorized from lexical identity**: shifting a Binding ID vector swaps *which* entity binds to *which* value without changing token identities.
  - **CRITICAL CONNECTION TO `Qwen3-1.7B → 0.6B`:**
    1. **Why generic-corpus linear maps `M, Q` had `R² = 0.78–0.81` yet `DIFF_cos(Clean - Bind) = 0.24–0.34`!** Binding IDs are *contextual, low-variance relational subspaces* that carry tiny variance on generic prose (`G_fit`), whereas high-energy lexical/positional directions dominate isotropic L2 ridge covariance `Σ`! So isotropic ridge `Q = Σ_{BA} (Σ_{AA} + ηI)^{-1}` systematically shrinks the exact **Binding ID subspace** (`λ_i / (λ_i + η) << 1`) while fitting high-variance lexical identity!
    2. **Why `Qwen3-0.6B` gets `93.8%` on `S1-alone` and `93.8%` on `DirS2`, yet `15.6%` on `S1 → S2`:** `0.6B` can bind 1 table (shallow lexical/positional matching), but lacks a clean, high-SNR **2-table compositional Binding-ID transfer** from Table 1 (`L0..L19`) into Table 2's value readout (`L22..L25`)!

### 5.2 Distributed Alignment Search (`DAS`) & `Boundless DAS` (Geiger et al., 2023; Wu et al., 2023)
- **Citations:**
  - Geiger, Wu, Potts, Icard, & Goodman (2024), *"Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"* (**DAS**), CLeaR 2024 (arXiv:2303.02536).
  - Wu, Geiger, Icard, Potts, & Goodman (2023), *"Interpretability at Scale: Identifying Causal Mechanisms in Alpaca"* (**Boundless DAS**), NeurIPS 2023 (arXiv:2305.08809).
- **Core Idea:** Causal variables (like "the resolved intermediate entity `k2`" or "the active Binding ID") do **not** align with individual neurons or single attention heads — they live in **rotated linear subspaces** found via interchange interventions (`h_new = R^T (Π_S R h_src + (I - Π_S) R h_base)`).
- **Implication for PlatonicSwap:** Our `A28 → B24` (`29/32`) and `B21` (`23/32`) whole-state ceilings confirm that the causal variable *is* linearly encoded in a distributed subspace of `Qwen3-1.7B` and readable by `Qwen3-0.6B`. The challenge is that **sparse KV-group extraction assumes head-aligned modularity**, whereas late binding/readout (`L22–L25`) is distributed across both attention heads and SwiGLU MLP subspaces!

---

## 6. Model Stitching, Platonic Representations, & Zero-SGD Transport

- **Model Stitching:** Lenc & Vedaldi (2015), *"Understanding Image Representations by Measuring Their Equivariance and Equivalence"*, CVPR 2015; Bansal, Nakkiran, & Barak (2021), *"Revisiting Model Stitching to Compare Neural Representations"*, NeurIPS 2021 — proved independently trained networks learn linearly compatible intermediate spaces (even across capacities: stronger prefixes can plug into weaker suffixes!).
- **Platonic Representation Hypothesis (PRH):** Huh, Cheung, Wang, & Isola (2024), *"The Platonic Representation Hypothesis"*, ICML 2024 (arXiv:2405.07987) — showed scaling drives representation geometry across models and modalities toward convergence.
- **Plato's Key:** Cheng, Liu, Yau, & Ré (Hazy Research, Nov 2026), *"Plato's Key"* — demonstrated zero-SGD closed-form orthogonal Procrustes + global scale alignment transfers *live activation states* across distinct base LLMs at inference time.
- **Where PlatonicSwap Pushes Beyond Plato's Key:** Plato's Key transfers **live hooked activations** (`Q @ h_A`, which our `A28 → B24` probe confirmed works at `29/32 = 90.6%` on `Qwen3-1.7B → 0.6B`!). **PlatonicSwap** attempts the far harder step: **compiling the donor's *multi-layer circuit weights* permanently into the recipient's fixed-architecture weights (`[STANDALONE IN-WEIGHTS, ZERO HOOKS]`)** so the recipient executes the *computation* itself without the donor present at inference!

---

## 7. Concrete Takeaways for the Circuit-Extraction Theory Council

The literature directly illuminates **four concrete theoretical questions** for our Council to solve next:
1. **Extraction vs. Transport Attribution:** In `exp1_donor_extraction_sanity_mps.py`, if **hooked subgraph-only outputs** fail to beat `STOCK_B` inside `0.6B`, then our extraction missed essential nodes (choke-point / invariant-half / MLP nodes); if hooked subgraph-only *does* beat `STOCK_B`, the bottleneck is purely weight-compilation transport (`J_B != J_A` + ridge shrinkage of the Binding-ID subspace + healer cancellation).
2. **Multi-Axis + Completeness Extraction:** Replace single-axis Binding-Swap ranking with **Multi-Axis (`Binding-Swap ∪ Query-Swap ∪ Table-2-Swap`) + Donor-minus-Recipient diffing + Isolation/Completeness verification** (following Wang et al., Chan et al., and Goodfire's diffing principle).
3. **Binding-ID Subspace Geometry (Feng & Steinhardt 2024):** Generic prose covariance `Σ` buries the low-variance contextual Binding-ID subspace under high-variance lexical modes. How do we construct a **strictly `Q_T = 0`, zero-SGD transport metric** (whitened/spectral, Jacobian pushforward, or generic contextual-binding covariance without task leakage) that preserves contextual relational subspaces instead of shrinking them?
4. **Crossing the Attention-Only Wall to SwiGLU MLPs (`L22/L23/L27`):** Since `L22–L27` writes `86.6%` of the margin (including `+5.91` pts from `L22/L23/L27` MLPs), how can **Transcoder / CLT / Sparse Feature Circuit** ideas (Dunefsky et al., Marks et al., Ameisen et al.) be adapted into a **zero-SGD or analytic low-rank/sparse SwiGLU weight compiler** that keeps `p95/p99` Tail-KL clean?


---

## 8. 2025–2026 Frontier Research Addendum

A targeted 2025–2026 frontier sweep (`transformer-circuits.pub`, `goodfire.ai`, NeurIPS 2025/2026, ICLR 2026, OpenReview, arXiv) surfaced **four decisive developments** that directly mirror and inform our exact `Qwen3-1.7B → Qwen3-0.6B` empirical observations.

### 8.1 Cross-Model Stitching & Weight-Space Programming (2025–2026)

1. ***"Cross-Model Stitching Transfers Answers, Not Computation"*** (ICLR / OpenReview, 2026):
   - **Scope:** Evaluates activation stitching across same-family pairs (`Gemma-2 9B → 2B`, `Qwen2.5 7B → 0.5B`, `Llama-3.1 8B → 3.2 1B`) and cross-family pairs on arithmetic, **symbolic binding**, state tracking, factual recall, and GSM8K.
   - **Core Finding:** Task-supervised linear/affine stitching achieves up to **88.6%** transfer of correct answers from a larger donor into a smaller recipient — **yet controls and ablations prove it transfers *precomputed answers* at late layers, NOT intermediate upstream computation!** The recipient acts mainly as a decoder for a redundantly encoded low-rank answer signal; there is little evidence that step-by-step upstream computational circuits transfer automatically through static activation stitches.
   - **Direct Mirror of Our Live `Qwen3` Results:** This *independently explains* why our hooked `A28 → B24` stitch hit **`29/32` (`90.6%`)** through four frozen blocks (transferring the already-computed answer!), while `A14..A17 → B14..B17` hit **`0/32`** and sparse `L0/L11/L19` weight grafts stalled at `Δs = 0.00`! It defines the exact scientific bar for PlatonicSwap: to achieve genuine **computation transfer** (and compose with a Recipient-Only Skill 3* that the donor does NOT have!), we must solve multi-layer circuit execution rather than late-layer answer decoding.
2. ***"Transferring Linear Features Across Language Models With Model Stitching"*** (`arXiv:2506.06609`, Jun 2025) & ***"Revisiting Model Stitching in the Foundation Model Era"*** (`arXiv:2603.12433`, Mar 2026):
   - Analyzes feature-matching and penultimate-layer stitching losses across modern foundation models, highlighting why penultimate/readout-aligned maps (`Q_voc` / suffix-calibrated maps) dramatically outperform mid-network MSE/ridge maps (`B27`: `28/32` vocab vs `10/32` ridge in our own sweep!).
3. ***"Pre-training interventions, ex post facto: Grafting model beliefs across checkpoints"*** (`arXiv:2609.xxxxx`, Sep 2026) & ***"TrojanMaxxing: One-shotting VLM Backdoors with Weight-Space Programming"*** (Dabholkar et al., `arXiv:2611.27425`, Nov 2026):
   - Demonstrates **training-free / zero-gradient Weight-Space Programming (WSP)** — inspecting native weights directly to insert targeted functional circuits without SGD, validating that direct weight-space circuit compilation is a live frontier paradigm when the target mechanism is cleanly isolated.

### 8.2 Goodfire AI's 2025–2026 Research: Clean Native Neurons, Subspace Blocks, & Sparse MLP Diffs

1. ***"Neural Networks Are Already Clean"*** (Cammarata, McGrath, Tigges et al., Goodfire, Oct 2026) & **Neuron Atlases / Neuron Communities:**
   - **Core Thesis:** Challenges the pessimistic view that superposition renders native MLP neurons hopelessly tangled. With proper upstream/downstream circuit-context microscopy, **~90% of native neurons in modern models (explicitly including Qwen models!)** have clean, legible computational roles — much apparent polysemanticity was an artifact of weak activations and context-free inspection.
   - **Direct Implication for `L16/L18/L22` SwiGLU MLPs in `Qwen3`:** Our Exp 1 just discovered that `L16 MLP` (`+0.55` on T2-Swap), `L18 MLP` (`+0.19`), and `L22 MLP` (`+0.24` across all 3 axes!) are massive causal hubs. Goodfire's thesis implies we may NOT need an expensive external SAE/CLT dictionary to extract them — **sparse native SwiGLU neuron communities** inside `Qwen3-1.7B` (`L16/L18/L22`) may already be clean enough to isolate directly!
2. **Block-Sparse Featurizers (`BSF` — *"Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds"*, Goodfire & collaborators, mid-2026):**
   - Lifts dictionary sparsity from 1D directions to **coordinated multi-dimensional subspace blocks** that fire together on continuous concept manifolds (achieving lower MDL than 1D SAEs).
   - **Direct Relevance to Entity Binding:** Matches **Feng & Steinhardt's (2024) multi-dimensional Binding ID subspace** — binding states are *multi-dimensional blocks*, not isolated 1D directions, explaining why 1D/rank-truncated projections smear relational binding!
3. ***"Transcoder Adapters for Reasoning-Model Diffing"*** (Hu et al., NeurIPS 2026 — comparing `Qwen2.5-Math-7B` vs `DeepSeek-R1-Distill-Qwen-7B` and `Qwen2.5-32B` vs `QwQ-32B`):
   - Runs a **sparse parallel transcoder adapter** alongside frozen base MLPs to approximate *only the MLP difference* (`ΔMLP = MLP_A - MLP_B`).
   - **Key Quantitative Discovery:** Modeling the **functional MLP diff** requires an **order of magnitude fewer active features** than reconstructing full MLPs! For PlatonicSwap, instead of trying to replace `0.6B`'s entire `L16/L22` MLP or graft hundreds of generic neurons, we should target **only the sparse Donor-minus-Recipient functional delta**!
4. **Model Diff Amplification, Parameter Interpretation, & RLFR (Goodfire, 2025–2026):** Reinforces contrastive Donor-vs-Recipient diffing (`logits_amp = logits_A + α (logits_A - logits_B)`) to avoid wasting grafting budget on circuits the recipient already executes natively.

### 8.3 Anthropic & SOTA 2025–2026 Circuit Extraction, Relevance Patching, & Crosscoder Upgrades

1. **Anthropic QK Attributions (*"Tracing Attention Computation Through Feature Interactions"*, Kamath et al., Transformer Circuits, 2025–2026):**
   - Extends CLT attribution graphs beyond MLPs by decomposing attention routing scores as **bilinear feature-to-feature QK interactions** (`q_feature · k_feature`), explaining *why* attention heads route to specific positions — directly relevant to diagnosing why compiled `Q/K` projections work at `L11` (`dup-mass = 0.92`) yet degrade at `L24/L25` (`DIFF_cos = 0.14–0.20`).
2. **`RelP` — Relevance Patching (NeurIPS 2025 / arXiv 2025–2026):**
   - Replaces noisy local gradients in `AtP` with **Layer-wise Relevance Propagation (LRP)** propagation coefficients, preserving `O(1)` cost (`2` forwards + `1` backward) while dramatically improving faithfulness across **deep nonlinear MLPs, LayerNorm/RMSNorm, and residual streams** — directly solving `AtP`'s blind spot on `Qwen3`'s `L16/L18/L22` SwiGLU MLPs!
3. **`CircuitLasso` (2026):**
   - Formulates sparse circuit discovery as **L1-regularized sparse linear regression** over component/feature interventions, recovering minimal faithful subgraphs at a fraction of combinatorial search cost.
4. ***"Demystifying Variance in Circuit Discovery"*** (arXiv, Jun 2026 — also introducing **CEAP** / Conductance-based EAP):
   - Documents three distinct variance sources in `EAP-IG`: **resampling variance**, **rephrasing variance** (circuits shift sharply across prompt wordings!), and **sample-wise unfaithfulness variance**.
   - **Direct Implication:** Single-template circuit localization overfits template-specific shortcuts; extracting a portable organ requires **multi-axis AND multi-template** contrastive discovery so only the wording-invariant core circuit is transplanted!
5. **2026 Crosscoder & CLT Algorithmic Upgrades:**
   - **BatchTopK + Latent Scaling Crosscoders** (*"Overcoming Sparsity Artifacts in Crosscoders"*, Minder, Nanda et al., 2025): Fixes L1-penalty shrinkage artifacts that previously misclassified shared vs. model-exclusive features.
   - **Dedicated Feature Crosscoders (`DFCs`)** (*"Cross-Architecture Model Diffing with Crosscoders"*, Jiralerspong & Bricken, Feb 2026): Separates shared vs. strictly model-exclusive features cleanly even across distinct model families (Llama vs. Qwen).
   - **Delta-Crosscoder** (Kassem, Jiralerspong et al., Feb 2026): Combines BatchTopK with a delta reconstruction loss to isolate narrow, localized capability differences without drowning in shared background variance.
   - **CLT-Forge** (Draye et al., Mar 2026): Open-source scalable Cross-Layer Transcoder training & tracing toolkit.

---

### 8.4 Updated Strategic Synthesis for the Theory Council

Combining Sections 1–8 with our live `Qwen3-1.7B → 0.6B` measurements sharpens the Council's exact design challenge:
1. **Heed the 2026 *"Answers, Not Computation"* Warning:** Hooked `A28 → B24` (`29/32`) transfers an *already-computed answer*; genuine **weight-compiled computation transfer** must beat `STOCK_B` on BOTH `S1 → S2` AND Recipient-Only `S1 → S3*` (where the Donor has `0/32` and therefore *cannot* supply a precomputed answer!).
2. **Exploit Goodfire's *"Already Clean"* + *Transcoder Adapter Diff* Insight on `L16/L18/L22` MLPs:** Instead of ignoring SwiGLU MLPs or grafting dense blocks, isolate the **sparse Donor-minus-Recipient neuron/block communities** in `L16/L18/L22` MLPs alongside our multi-axis attention heads (`L11 g1/g4`, `L16 g2`, `L17 g4`, `L19 g2`, `L24/L25`).
3. **Use `RelP` / Multi-Axis / Multi-Template Extraction + Isolation Completeness:** Eliminate both the **Choke-Point Illusion** and **Rephrasing/Single-Axis Variance** before spending a single weight edit!
4. **Match Multi-Dimensional Subspaces (BSF / Binding IDs) Across Nonlinear Layers:** Address why isotropic L2 ridge smears low-variance relational Binding-ID subspaces (`DIFF_cos = 0.24–0.34`) and how cross-layer pushforward alignment + non-cancelling healing can preserve them at `p95/p99` Tail-KL bounds.
