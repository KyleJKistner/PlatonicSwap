# Want-to-Read & Pending Full-Text Queue

This file tracks relevant literature for **PlatonicSwap** (v1 positioning and v2 higher-level mechanism transfer) that our agents **could not automatically obtain in full-paper form** (OpenReview-only, preprint not yet released, or truncated due to size), plus a separate optional queue for v2 expansion.

## 1. Could Not Automatically Obtain / Full Paper Pending Release

| # | Title | Authors / Venue | Link | Why Not Obtained | Why It Matters to PlatonicSwap |
|---:|---|---|---|---|---|
| 1 | **Transplanting Knowledge: A Study on Layer-Specific Grafting in LLMs** | Bastien Zimmermann, Matthieu Boussard (craft ai) — AAAI 2025 Workshop (CoLoRAI) | https://openreview.net/search?term=Transplanting+Knowledge+Layer-Specific+Grafting | OpenReview-only (no arXiv ID indexed); requires direct OpenReview PDF download & conversion. | Grafts task-specialized layers between models using lightweight identity-initialized linear residual-stream translators — direct precedent for multi-layer grafting in v2. |
| 2 | **Plato's Key (Full Technical Paper & Official Code)** | Vivien Cheng, Jerry Liu, Morris Yau, Christopher Ré (Stanford Hazy Research, Nov 2026) | https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key | Only the blog post is currently public (archived in [`papers/cheng_2026_platos_key_hazy_research.md`](papers/cheng_2026_platos_key_hazy_research.md)); full arXiv paper, proofs, and code release are not yet published. | Direct parent work; full paper will clarify exact Wasserstein Procrustes hyperparameters, layer-sweep protocol, and dark-memory evaluation details. |
| 3 | **Chantless (Full Technical Paper / Formal Ablations)** | Abhishek Harshvardhan Mishra (`@tokenbender` / Deus Experiments, Nov 2026) | https://deusexperiments.com/research/how-do-models-lose-weight-without-crying/ | Blog post and GitHub README are archived in [`papers/tokenbender_2026_chantless.md`](papers/tokenbender_2026_chantless.md); no formal arXiv paper / full mathematical writeup published yet. | Complementary subtractive single-model capability harvester; useful upstream of PlatonicSwap v2 for isolating donor subnetworks before cross-architecture splicing. |
| 4 | **The Geometry of Semantic Space (Full Appendix beyond 220 KB)** | Liang (arXiv:2607.17146, Jul 2026) | https://arxiv.org/abs/2607.17146 | Main body archived in [`papers/liang_2026_geometry_of_semantic_space.md`](papers/liang_2026_geometry_of_semantic_space.md), truncated at 220,000 chars to keep git lean. | Continuous geometric flow formulation across 5 transformer families; grab full PDF only if v2 needs the extended appendix proofs. |

## 2. Optional v2 Expansion Queue (Not Yet Ingested)

Adjacent foundational or venue-only works not needed for v1 positioning, queued for v2 (multi-layer circuits, permutation matching, hypernetwork adapters):

| # | Title | Authors / Venue | Why Queue for v2 |
|---:|---|---|---|
| 1 | **Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models** | ICLR 2025 (OpenReview) | Direct layer-swapping evidence for separating task/reasoning layers from language-specific layers. |
| 2 | **Git Re-Basin: Merging Models modulo Permutation Symmetries** | Ainsworth, Hayase, Srinivasa (ICLR 2023) | Classic activation/weight permutation matching foundation underlying modern cross-architecture OT merging. |
| 3 | **Model Stitching: Looking for Conserved Representations** | Bansal, Nakkiran, Barak (NeurIPS 2021) | Foundational linear stitching test for functional interchangeability across architectures. |
| 4 | **FuseLLM / FuseChat: Knowledge Fusion of Large Language Models** | Wan et al. (ICLR 2024 / 2025) | Probabilistic distribution-level cross-architecture knowledge fusion baseline (requires target training). |
| 5 | **Text-to-LoRA: Instant Transformer Adaption** | Sakana AI (2025) | Hypernetwork generation of task LoRAs — relevant if v2 compiles skills from text or donor probes into native weights. |

## How to Resolve Items in This Queue

1. Download the OpenReview / arXiv PDF (or wait for preprint release for items 2–3).
2. Extract clean text via `pdftotext <file.pdf> -` and save as `docs/papers/<slug>.md` with standard YAML frontmatter (`title`, `authors`, `date`, `cluster`, `tags`, `status: FULL_TEXT`).
3. Update `docs/README.md`, `docs/bibliography.bib`, and `docs/manifest.json`, then remove the resolved entry from Section 1 above.
