# PlatonicSwap Literature Archive (`docs/`)

A structured, plain-text / Markdown research archive designed to be **instantly searchable by coding & research agents** (`grep`, `git grep`, `rg`, or direct file reads). Currently holds **24 indexed works** (**24 full-text extracted**, **0 abstract-only**), a **Round 1 6-seat literature synthesis** ([`COUNCIL_SYNTHESIS.md`](COUNCIL_SYNTHESIS.md)), a **Round 2 5-seat v1 closeout & 85–95% ceiling verdict** ([`V1_CLOSEOUT_COUNCIL_R2.md`](V1_CLOSEOUT_COUNCIL_R2.md)), **4 unobtainable / pending-release items** and optional v2 reads tracked in [`WANT_TO_READ.md`](WANT_TO_READ.md), verified citations in [`bibliography.bib`](bibliography.bib), and a machine-readable catalog in [`manifest.json`](manifest.json).

## Quick Agent Search Recipes

```bash
# Search across all papers for a specific mechanism, metric, or model (portable grep or git grep)
grep -rnEi "procrustes|optimal transport|swiglu|abstention" docs/papers/
git grep -niE "relative representations|lora-align|crosscoder" -- docs/papers/

# Filter papers by cluster or tag using YAML frontmatter
grep -rnE "^(cluster|tags|status):" docs/papers/

# Find v2-relevant notes on multi-layer / attention / LoRA transfer
grep -rn "Relevance to PlatonicSwap v2:" docs/papers/

# Or with ripgrep if installed:
# rg -i "procrustes|optimal transport|swiglu|abstention" docs/papers/
```

## Master Paper Index

| Paper | Date | ID | Cluster | Status | Size | File |
|---|---|---|---|---|---:|---|
| **Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts** | 2025-01-02 | [2501.02009](https://arxiv.org/abs/2501.02009) | `activation_steering_transfer` | `FULL_TEXT` | 77.7 KB | [`papers/huang_2025_platonic_concept_transferability.md`](papers/huang_2025_platonic_concept_transferability.md) |
| **Activation Space Interventions Can Be Transferred Between Large Language Models** | 2025-03-06 | [2503.04429](https://arxiv.org/abs/2503.04429) | `activation_steering_transfer` | `FULL_TEXT` | 129.4 KB | [`papers/oozeer_2025_activation_intervention_transfer.md`](papers/oozeer_2025_activation_intervention_transfer.md) |
| **Cross-Architecture Steering Transfer in Language Models: A Systematic Empirical Study** | 2026-05-26 | [2608.05164](https://arxiv.org/abs/2608.05164) | `activation_steering_transfer` | `FULL_TEXT` | 100.6 KB | [`papers/agarwal_2026_cross_architecture_steering_transfer.md`](papers/agarwal_2026_cross_architecture_steering_transfer.md) |
| **Rotate2Think: Geometric Priming via Orthogonal Rotation to Improve Language Model Reasoning** | 2026-06-02 | [2606.09873](https://arxiv.org/abs/2606.09873) | `activation_steering_transfer` | `FULL_TEXT` | 49.5 KB | [`papers/sharma_2026_rotate2think_orthogonal_priming.md`](papers/sharma_2026_rotate2think_orthogonal_priming.md) |
| **Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse** | 2026-08-04 | [2608.03893](https://arxiv.org/abs/2608.03893) | `activation_steering_transfer` | `FULL_TEXT` | 73.8 KB | [`papers/heo_2026_cross_model_kv_cache_transfer.md`](papers/heo_2026_cross_model_kv_cache_transfer.md) |
| **Architecture-Dependent Causal Transfer of Activation States Across Large Language Models** | 2026-08-17 | [2608.16347](https://arxiv.org/abs/2608.16347) | `activation_steering_transfer` | `FULL_TEXT` | 51.8 KB | [`papers/piepereit_2026_architecture_dependent_causal_transfer.md`](papers/piepereit_2026_architecture_dependent_causal_transfer.md) |
| **Chantless: How Do Models Lose Weight Without Crying? (Training-Free Capability Extraction)** | 2026-11-26 | [web](https://deusexperiments.com/research/how-do-models-lose-weight-without-crying/) | `capability_extraction_complementary` | `FULL_TEXT` | 49.4 KB | [`papers/tokenbender_2026_chantless.md`](papers/tokenbender_2026_chantless.md) |
| **Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT** | 2025-09-30 | [2510.00296](https://arxiv.org/abs/2510.00296) | `cross_model_probing` | `FULL_TEXT` | 99.8 KB | [`papers/barshalom_2025_act_vit_activation_tensors.md`](papers/barshalom_2025_act_vit_activation_tensors.md) |
| **Building Production-Ready Probes For Gemini** | 2026-01-16 | [2601.11516](https://arxiv.org/abs/2601.11516) | `cross_model_probing` | `FULL_TEXT` | 126.5 KB | [`papers/kramar_2026_production_probes_gemini.md`](papers/kramar_2026_production_probes_gemini.md) |
| **Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs** | 2026-02-12 | [2602.11729](https://arxiv.org/abs/2602.11729) | `cross_model_probing` | `FULL_TEXT` | 149.7 KB | [`papers/crosscoders_2026_cross_architecture_model_diffing.md`](papers/crosscoders_2026_cross_architecture_model_diffing.md) |
| **Do Activation Monitors Survive Model Updates? Benchmarking, Predicting, and Repairing Activation-Monitor Staleness** | 2026-06-14 | [2606.15980](https://arxiv.org/abs/2606.15980) | `cross_model_probing` | `FULL_TEXT` | 48.7 KB | [`papers/duan_2026_monitor_staleness.md`](papers/duan_2026_monitor_staleness.md) |
| **Relative Representations Enable Portable White-Box Monitoring Across Language Models** | 2026-10-20 | [2610.28366](https://arxiv.org/abs/2610.28366) | `cross_model_probing` | `FULL_TEXT` | 206.9 KB | [`papers/bakker_2026_relative_representations_monitoring.md`](papers/bakker_2026_relative_representations_monitoring.md) |
| **Relative representations enable zero-shot latent space communication** | 2022-09-30 | [2209.15430](https://arxiv.org/abs/2209.15430) | `foundations_platonic` | `FULL_TEXT` | 94.8 KB | [`papers/moschella_2023_relative_representations.md`](papers/moschella_2023_relative_representations.md) |
| **The Platonic Representation Hypothesis** | 2024-05-13 | [2405.07987](https://arxiv.org/abs/2405.07987) | `foundations_platonic` | `FULL_TEXT` | 123.3 KB | [`papers/huh_2024_platonic_representation_hypothesis.md`](papers/huh_2024_platonic_representation_hypothesis.md) |
| **The Indra Representation Hypothesis for Multimodal Alignment** | 2026-04-06 | [2604.04496](https://arxiv.org/abs/2604.04496) | `foundations_platonic` | `FULL_TEXT` | 66.9 KB | [`papers/lu_2025_indra_representation_hypothesis.md`](papers/lu_2025_indra_representation_hypothesis.md) |
| **The Geometry of Semantic Space: A Continuous Geometric Framework for the Transformer Architecture** | 2026-07-19 | [2607.17146](https://arxiv.org/abs/2607.17146) | `foundations_platonic` | `FULL_TEXT` | 217.8 KB | [`papers/liang_2026_geometry_of_semantic_space.md`](papers/liang_2026_geometry_of_semantic_space.md) |
| **Plato's Key: Cross-Model Representation Transfer via Closed-Form Rotation and Scaling** | 2026-11-25 | [web](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key) | `foundations_platonic` | `FULL_TEXT` | 13.0 KB | [`papers/cheng_2026_platos_key_hazy_research.md`](papers/cheng_2026_platos_key_hazy_research.md) |
| **Locating and Editing Factual Associations in GPT** | 2022-02-10 | [2202.05262](https://arxiv.org/abs/2202.05262) | `model_editing_baselines` | `FULL_TEXT` | 97.9 KB | [`papers/meng_2022_rome.md`](papers/meng_2022_rome.md) |
| **Mass-Editing Memory in a Transformer** | 2022-10-13 | [2210.07229](https://arxiv.org/abs/2210.07229) | `model_editing_baselines` | `FULL_TEXT` | 70.9 KB | [`papers/meng_2023_memit.md`](papers/meng_2023_memit.md) |
| **LoRA-X: Bridging Foundation Models with Training-Free Cross-Model Adaptation** | 2025-01-27 | [2501.16559](https://arxiv.org/abs/2501.16559) | `weight_adapter_transfer` | `FULL_TEXT` | 70.8 KB | [`papers/lorax_2025_training_free_cross_model_adaptation.md`](papers/lorax_2025_training_free_cross_model_adaptation.md) |
| **Knowledge Fusion of Large Language Models Via Modular SkillPacks** | 2025-05-24 | [2505.18502](https://arxiv.org/abs/2505.18502) | `weight_adapter_transfer` | `FULL_TEXT` | 102.2 KB | [`papers/graftllm_2025_modular_skillpacks.md`](papers/graftllm_2025_modular_skillpacks.md) |
| **Cross-LoRA: A Data-Free LoRA Transfer Framework across Heterogeneous LLMs** | 2025-08-07 | [2508.05232](https://arxiv.org/abs/2508.05232) | `weight_adapter_transfer` | `FULL_TEXT` | 51.9 KB | [`papers/xia_2025_cross_lora_heterogeneous_transfer.md`](papers/xia_2025_cross_lora_heterogeneous_transfer.md) |
| **Transport and Merge: Cross-Architecture Merging for Large Language Models** | 2026-02-05 | [2602.05495](https://arxiv.org/abs/2602.05495) | `weight_adapter_transfer` | `FULL_TEXT` | 83.0 KB | [`papers/cui_2026_transport_and_merge_cross_arch.md`](papers/cui_2026_transport_and_merge_cross_arch.md) |
| **Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion** | 2026-04-07 | [2604.05688](https://arxiv.org/abs/2604.05688) | `weight_adapter_transfer` | `FULL_TEXT` | 61.7 KB | [`papers/attention_editing_2026.md`](papers/attention_editing_2026.md) |

## Positioning Summary: PlatonicSwap vs. Prior Art (Council-Audited)

- **Runtime Activation / Steering / KV Transfer** (`Plato's Key`, `Huang 2025`, `Oozeer 2025`, `Agarwal 2026`, `Piepereit 2026`, `Heo 2026`, `Rotate2Think 2026`): Require runtime hooks, steering vectors, or KV mappers (`Oozeer 2025` also explores an in-weights trained autoencoder safety switch at ~0.32% params). PlatonicSwap v1 compiles zero-recipient-SGD, bias-free native SwiGLU rows (`d_ff -> d_ff + k`) with verified structural reversibility.

- **Cross-Model Monitoring & Diffing** (`Bakker & Louizos 2026`, `Crosscoders 2026`, `Kramár 2026`, `Duan 2026`, `ACT-ViT 2025`): `Bakker & Louizos` transfer read-only probes via relative angular anchors (including global all-layer `1B -> 70B`); `Crosscoders` discovers shared vs. exclusive features across architectures and demonstrates causal persona steering (§3.1) via shared dictionary atoms. Neither compiles donor-free standalone native SwiGLU organs.

- **Cross-Architecture Weight/Adapter Fusion** (`Cross-LoRA 2025`, `LoRA-X 2025`, `Transport & Merge 2026`, `GraftLLM 2025/ICLR 2026`): `Cross-LoRA` and `LoRA-X` project LoRA updates using weight-only SVD subspaces (`LoRA-X` is same-family only; see [`COUNCIL_SYNTHESIS.md`](COUNCIL_SYNTHESIS.md) for the weight-only alignment collapse proof). `Transport & Merge` infers activation-OT correspondences from ~2k samples and interpolates in place into the top-k=128 most-active target neurons (optionally followed by recipient fine-tuning). `GraftLLM` distills via target SFT+DPO into routed SkillPacks. PlatonicSwap v1 is append-only (`d_ff -> d_ff + k`, `mlp_bias=False`, base weights untouched), gated per rule with an in-distribution conformal false-fire bound (`α=0.002`), self-abstaining under shuffled correspondence, and evaluated against shuffled and wrong-entity controls (note: requires a 300-passage unlabeled paired corpus, and cross-family Cap-1 unseen ripple is 6.2–22.9% across 4 recipients).

- **Direct Single-Model Editing** (`ROME 2022`, `MEMIT 2023`): Single-model in-place `W_out` edits with recipient-side value optimization (`MEMIT` spreads across mid-layers with `λ=15,000`); no cross-model transfer. Our L23 closed-form `W_down` overwrite foil is non-canonical (single layer, `λ=10`), and we make no CounterFact claims.

- **Subtractive Single-Model Capability Extraction** (`Chantless 2026`): Prunes/masks a *single* model down to a specialist without cross-architecture transfer — complementary to PlatonicSwap v2 as an upstream donor-harvesting stage.
