# PlatonicSwap — Master Programmatically Audited Results (`v1.0`)

> **Auto-generated** by `experiments/build_repo_tables.py` at `2026-11-26 13:14:29` directly from JSON artifacts in `results/` (zero hand-typed numbers; strict column-count and no-bare-pipe assertions enforced).

### Artifact Manifest (24/24 present)
| File | Status | Size | Modified | SHA-256 (12) |
| --- | --- | --- | --- | --- |
| `results_baseline.json` | AUDITED | 1360 KB | 2026-11-26 13:09:52 | `894ecd527e98` |
| `results_referee_controls.json` | AUDITED | 452 KB | 2026-11-26 13:09:52 | `7055c5d86eb1` |
| `results_systems.json` | AUDITED | 5662 KB | 2026-11-26 13:09:52 | `8ea3fae3f3e7` |
| `results_pilot0.json` | AUDITED | 220 KB | 2026-11-26 09:07:16 | `e854ee31fe38` |
| `results_pilot1.json` | AUDITED | 70 KB | 2026-11-26 09:07:16 | `1162e3e485e3` |
| `results_pilot1_qwen2.5-3b_to_granite-3.1-2b.json` | AUDITED | 70 KB | 2026-11-26 09:07:16 | `3be8102838da` |
| `results_pilot1_falcon3-3b_to_smollm2-1.7b.json` | AUDITED | 70 KB | 2026-11-26 09:07:16 | `e7aae20e5c7c` |
| `results_pilot1b_qwen2.5-3b_to_smollm2-1.7b.json` | AUDITED | 63 KB | 2026-11-26 09:07:16 | `615f3c8a0210` |
| `results_pilot1b_qwen2.5-3b_to_granite-3.1-2b.json` | AUDITED | 76 KB | 2026-11-26 09:07:16 | `17b726a30867` |
| `results_pilot1b_falcon3-3b_to_smollm2-1.7b.json` | AUDITED | 61 KB | 2026-11-26 09:07:16 | `4c801c269186` |
| `results_organs_p0.json` | AUDITED | 57 KB | 2026-11-26 09:07:16 | `76600948ea10` |
| `results_organs_multifamily.json` | AUDITED | 118 KB | 2026-11-26 12:13:35 | `87aa28879370` |
| `results_pilot2_fullrank.json` | AUDITED | 457 KB | 2026-11-26 09:07:16 | `3f00e6eddcd0` |
| `results_pilot2_gated.json` | AUDITED | 16 KB | 2026-11-26 09:07:16 | `24ccc69e0cb4` |
| `results_pilot3.json` | AUDITED | 81 KB | 2026-11-26 09:07:16 | `dbdebe4af48e` |
| `results_pilot3_qwen2.5-1.5b.json` | AUDITED | 62 KB | 2026-11-26 09:07:16 | `21fc13a9de82` |
| `results_pilot3_granite-3.1-2b.json` | AUDITED | 62 KB | 2026-11-26 09:07:16 | `a565ae98228a` |
| `results_pilot3_falcon3-3b.json` | AUDITED | 62 KB | 2026-11-26 09:07:16 | `98bfcb22f407` |
| `results_graft_falsify.json` | AUDITED | 170 KB | 2026-11-26 09:07:16 | `112f55c05a77` |
| `results_graft_t3_gap.json` | AUDITED | 25 KB | 2026-11-26 09:07:16 | `edbd378112e0` |
| `results_graft_t3_gap_scaled.json` | AUDITED | 80 KB | 2026-11-26 12:13:35 | `b074324cd458` |
| `results_graft_t3_gap_scaled_granite.json` | AUDITED | 66 KB | 2026-11-26 12:13:35 | `d442da0713e3` |
| `results_cap1_channel_pushforward.json` | AUDITED | 128 KB | 2026-11-26 12:13:35 | `bcf2dc70997e` |
| `results_bio_ripple_sealed.json` | AUDITED | 580 KB | 2026-11-26 12:13:35 | `cd6c3c02a088` |

## Pillar 1: Zero-QA Geometry, $N \gg D$ Scaling, $O(d)$ vs. $GL(d)$, and Attention-Knockout Decomposition
### Table 1.0 — Plato's Key 20-Pair Protocol Replication (`results_baseline.json`)
| Metric (20 Ordered Pairs, 5 Models) | Value |
| --- | --- |
| Median Self-Swap Accuracy at $l_B$ (20 pairs) | 87.4% |
| Median Cross-Model Procrustes Transfer (all 20 pairs) | 58.8% |
| Median Retention vs. Self-Swap (cf. Plato's Key ~73%) | 70.9% |
| Median Shuffled / Random Control | 3.6% / 3.9% |

### Table 1A — Pilot 0 Error Taxonomy, Data-Starvation Curve & Causal-Fisher Subspace (`results_pilot0.json`)
| Pair | Overall Acc | Answer-Disjoint | Base Subj Survived (of fails) | Right Type Wrong Ent | Off-Manifold Other | Garbled/Empty |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | 110/149 (73.8%) | 45/66 (68.2%) | 17/39 (43.6%) | 4/39 (10.3%) | 18/39 (46.2%) | 0 |
| `falcon3-3b->smollm2-1.7b` | 124/177 (70.1%) | 44/69 (63.8%) | 24/53 (45.3%) | 5/53 (9.4%) | 24/53 (45.3%) | 0 |
| `qwen2.5-3b->granite-3.1-2b` | 59/153 (38.6%) | 19/67 (28.4%) | 83/94 (88.3%) | 5/94 (5.3%) | 6/94 (6.4%) | 0 |

| Pair | ProcGlobal Learning Curve (All-Train Pool) | Pilot 0d Causal-Fisher Diagnostic |
| --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | N=147: 53.2% (AD 38.4%, gap +39.0pp) -> N=293: 67.6% (AD 54.0%, gap +19.9pp) -> N=587: 75.8% (AD 68.2%, gap +4.2pp) | AUC `abs(g·δ)`=0.8196 vs L2=0.6308; Fisher-in-PCA64=15.9% (PR=48.2) |
| `falcon3-3b->smollm2-1.7b` | N=147: 49.1% (AD 38.2%, gap +38.4pp) -> N=293: 61.2% (AD 50.2%, gap +21.9pp) -> N=587: 69.5% (AD 62.3%, gap +9.7pp) | — |
| `qwen2.5-3b->granite-3.1-2b` | N=147: 22.4% (AD 10.9%, gap +22.6pp) -> N=293: 34.2% (AD 20.9%, gap +15.8pp) -> N=587: 41.2% (AD 25.4%, gap +5.5pp) | AUC `abs(g·δ)`=0.6659 vs L2=0.4327; Fisher-in-PCA64=5.0% (PR=2.2) |

*Pilot 0c (Dark slice beyond GlobalFV, 20 pairs):* `154/283` remain wrong after GlobalFV; matched key swap = `9/154 (5.8%)` vs shuffled = `5/154 (3.2%)` (strictly answer-disjoint `n=62`: `1/62 (1.6%)` vs `1/62 (1.6%)`).

### Table 1B — Pilot 1 Zero-QA Unlabeled $N \gg D$ Scaling, $O(d)$ vs. $GL(d)$, and Rank-$k$ Fisher-PLS
| Pair | Map (0-QA unless labeled QA) | CF-Flip Test (95% CI) | Shuf | Answer-Disjoint | LORO 5 Unseen Rels (95% CI) | Geom $R^2$ | Unlab KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | QA `qa_shared_procrustes_N429` | 72.5% [64.3, 79.8] (n=149) | 0.7% | 65.1% (n=66) | 62.9% [52.8, 73.0] (n=89) | E=0.062 / F=0.288 | 0.2028 |
| `qwen2.5-3b->smollm2-1.7b` | QA `qa_train10rels_procrustes_N415` | 68.5% [60.1, 76.2] (n=149) | 0.0% | 56.1% (n=66) | 46.1% [36.0, 56.2] (n=89) | E=-0.157 / F=0.131 | 0.1885 |
| `qwen2.5-3b->smollm2-1.7b` | Unlab N=500 `procrustes_global` | 51.0% [41.5, 59.9] (n=149) | 1.3% | 45.5% (n=66) | 42.7% [32.6, 52.8] (n=89) | E=-0.704 / F=-0.177 | 0.0553 |
| `qwen2.5-3b->smollm2-1.7b` | Unlab N=2000 `procrustes_global` | 69.8% [61.6, 77.3] (n=149) | 0.7% | 63.6% (n=66) | 64.0% [53.9, 74.2] (n=89) | E=-0.376 / F=0.044 | 0.0386 |
| `qwen2.5-3b->smollm2-1.7b` | Unlab N=10000 `procrustes_global` | 76.5% [69.4, 83.0] (n=149) | 0.0% | 72.7% (n=66) | 73.0% [64.0, 82.0] (n=89) | E=-0.206 / F=0.136 | 0.0348 |
| `qwen2.5-3b->smollm2-1.7b` | Unlab N=10000 `fisher_orth_Od_multi` | 71.8% [63.5, 79.4] (n=149) | 0.0% | 66.7% (n=66) | 67.4% [57.3, 76.4] (n=89) | E=-0.505 / F=-0.781 | 0.0600 |
| `qwen2.5-3b->smollm2-1.7b` | Unlab N=10000 `ridge_gl_raw` | 67.8% [59.0, 76.0] (n=149) | 0.0% | 60.6% (n=66) | 60.7% [50.6, 70.8] (n=89) | E=0.158 / F=0.378 | 0.0308 |
| `qwen2.5-3b->smollm2-1.7b` | Unlab N=10000 `ridge_gl_rescaled` | 69.8% [61.7, 77.5] (n=149) | 0.0% | 57.6% (n=66) | 65.2% [55.1, 75.3] (n=89) | E=0.047 / F=0.243 | 0.0301 |
| `qwen2.5-3b->smollm2-1.7b` | Unlab N=10000 `zca_whitened_procrustes` | 71.8% [63.6, 79.4] (n=149) | 0.0% | 60.6% (n=66) | 59.6% [49.4, 69.7] (n=89) | E=-0.322 / F=0.076 | 0.0307 |
| `qwen2.5-3b->smollm2-1.7b` | Unlab N=10000 `pullback_zca_fisher_multi` | 70.5% [62.2, 78.2] (n=149) | 0.0% | 59.1% (n=66) | 60.7% [50.6, 70.8] (n=89) | E=-0.335 / F=0.122 | 0.0315 |
| `qwen2.5-3b->granite-3.1-2b` | QA `qa_shared_procrustes_N456` | 38.6% [30.2, 47.5] (n=153) | 0.0% | 28.4% (n=67) | 30.8% [22.0, 40.7] (n=91) | E=-0.040 / F=0.285 | 0.2164 |
| `qwen2.5-3b->granite-3.1-2b` | QA `qa_train10rels_procrustes_N415` | 33.3% [26.1, 41.2] (n=153) | 0.0% | 25.4% (n=67) | 14.3% [7.7, 22.0] (n=91) | E=-0.247 / F=0.080 | 0.2716 |
| `qwen2.5-3b->granite-3.1-2b` | Unlab N=500 `procrustes_global` | 20.3% [14.0, 26.5] (n=153) | 0.0% | 22.4% (n=67) | 15.4% [8.8, 23.1] (n=91) | E=-0.671 / F=-0.133 | 0.0588 |
| `qwen2.5-3b->granite-3.1-2b` | Unlab N=2000 `procrustes_global` | 36.6% [28.7, 45.0] (n=153) | 0.0% | 32.8% (n=67) | 35.2% [26.4, 45.1] (n=91) | E=-0.439 / F=0.024 | 0.0418 |
| `qwen2.5-3b->granite-3.1-2b` | Unlab N=10000 `procrustes_global` | 44.4% [36.3, 53.4] (n=153) | 0.0% | 34.3% (n=67) | 39.6% [30.7, 50.5] (n=91) | E=-0.289 / F=0.063 | 0.0391 |
| `qwen2.5-3b->granite-3.1-2b` | Unlab N=10000 `fisher_orth_Od_multi` | 36.6% [29.7, 44.2] (n=153) | 0.0% | 40.3% (n=67) | 44.0% [34.1, 54.9] (n=91) | E=-0.620 / F=-1.882 | 0.0759 |
| `qwen2.5-3b->granite-3.1-2b` | Unlab N=10000 `ridge_gl_raw` | 34.0% [26.4, 41.8] (n=153) | 0.7% | 29.8% (n=67) | 29.7% [20.9, 39.6] (n=91) | E=0.162 / F=0.386 | 0.0329 |
| `qwen2.5-3b->granite-3.1-2b` | Unlab N=10000 `ridge_gl_rescaled` | 35.3% [27.3, 43.5] (n=153) | 0.0% | 28.4% (n=67) | 34.1% [24.2, 44.0] (n=91) | E=-0.005 / F=0.050 | 0.0344 |
| `qwen2.5-3b->granite-3.1-2b` | Unlab N=10000 `zca_whitened_procrustes` | 34.0% [26.5, 41.9] (n=153) | 0.0% | 25.4% (n=67) | 34.1% [25.3, 44.0] (n=91) | E=-1.018 / F=-0.551 | 0.0346 |
| `qwen2.5-3b->granite-3.1-2b` | Unlab N=10000 `pullback_zca_fisher_multi` | 32.7% [25.3, 40.4] (n=153) | 0.7% | 25.4% (n=67) | 34.1% [24.2, 44.0] (n=91) | E=-1.050 / F=-0.333 | 0.0340 |
| `falcon3-3b->smollm2-1.7b` | QA `qa_shared_procrustes_N499` | 69.5% [62.4, 76.3] (n=177) | 1.1% | 62.3% (n=69) | 64.6% [54.2, 74.0] (n=96) | E=0.218 / F=0.391 | 0.1257 |
| `falcon3-3b->smollm2-1.7b` | QA `qa_train10rels_procrustes_N415` | 58.2% [50.6, 65.4] (n=177) | 1.7% | 52.2% (n=69) | 39.6% [30.2, 49.0] (n=96) | E=0.013 / F=0.233 | 0.1350 |
| `falcon3-3b->smollm2-1.7b` | Unlab N=500 `procrustes_global` | 44.6% [35.9, 52.7] (n=177) | 0.6% | 39.1% (n=69) | 33.3% [24.0, 43.8] (n=96) | E=-0.597 / F=-0.142 | 0.0598 |
| `falcon3-3b->smollm2-1.7b` | Unlab N=2000 `procrustes_global` | 58.8% [50.6, 66.3] (n=177) | 0.0% | 56.5% (n=69) | 66.7% [57.3, 76.0] (n=96) | E=-0.353 / F=0.070 | 0.0418 |
| `falcon3-3b->smollm2-1.7b` | Unlab N=10000 `procrustes_global` | 70.1% [62.7, 76.8] (n=177) | 0.6% | 71.0% (n=69) | 80.2% [71.9, 87.5] (n=96) | E=-0.185 / F=0.151 | 0.0387 |
| `falcon3-3b->smollm2-1.7b` | Unlab N=10000 `fisher_orth_Od_multi` | 67.8% [60.4, 74.8] (n=177) | 0.6% | 68.1% (n=69) | 77.1% [67.7, 85.4] (n=96) | E=-0.503 / F=-1.207 | 0.0654 |
| `falcon3-3b->smollm2-1.7b` | Unlab N=10000 `ridge_gl_raw` | 70.6% [63.7, 77.3] (n=177) | 0.6% | 68.1% (n=69) | 71.9% [62.5, 81.2] (n=96) | E=0.243 / F=0.437 | 0.0334 |
| `falcon3-3b->smollm2-1.7b` | Unlab N=10000 `ridge_gl_rescaled` | 70.6% [63.9, 77.0] (n=177) | 0.6% | 66.7% (n=69) | 78.1% [69.8, 86.5] (n=96) | E=0.107 / F=0.250 | 0.0341 |
| `falcon3-3b->smollm2-1.7b` | Unlab N=10000 `zca_whitened_procrustes` | 71.2% [64.0, 77.5] (n=177) | 0.6% | 69.6% (n=69) | 76.0% [66.7, 84.4] (n=96) | E=-0.626 / F=-0.098 | 0.0357 |
| `falcon3-3b->smollm2-1.7b` | Unlab N=10000 `pullback_zca_fisher_multi` | 71.8% [64.6, 78.2] (n=177) | 0.6% | 69.6% (n=69) | 75.0% [65.6, 83.3] (n=96) | E=-0.671 / F=0.003 | 0.0353 |

*Rank-$k$ Unlabeled Compression (0 QA labels):*
| Pair | Rank | PCA-Procrustes | PLS | Closed-Form Fisher-PLS | Fisher-PLS / PCA Ratio |
| --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | k=32 | 7.4% (net +4.0pp) | 10.1% (net +6.0pp) | 11.4% (net +8.7pp) | raw 1.55x, net 2.16x |
| `qwen2.5-3b->smollm2-1.7b` | k=64 | 16.8% (net +12.8pp) | 22.8% (net +19.5pp) | 30.2% (net +28.9pp) | raw 1.80x, net 2.26x |
| `qwen2.5-3b->granite-3.1-2b` | k=32 | 2.6% (net +2.0pp) | 4.6% (net +4.6pp) | 8.5% (net +7.8pp) | raw 3.26x, net 4.00x |
| `qwen2.5-3b->granite-3.1-2b` | k=64 | 7.8% (net +7.8pp) | 13.1% (net +13.1pp) | 15.7% (net +14.4pp) | raw 2.00x, net 1.83x |
| `falcon3-3b->smollm2-1.7b` | k=32 | 6.2% (net +2.8pp) | 11.3% (net +7.3pp) | 14.7% (net +11.3pp) | raw 2.37x, net 4.01x |
| `falcon3-3b->smollm2-1.7b` | k=64 | 14.7% (net +10.7pp) | 20.3% (net +19.8pp) | 25.4% (net +23.7pp) | raw 1.73x, net 2.21x |

### Table 1C — Pilot 1B: STRICT Separation of Natural Unmasked vs. Attention-Knockout Regimes
*Note: Transport Efficiency is the ratio of cross-family transfer to matched native self-swap at the same layer and mask configuration (not a strict 100% upper bound; values >100% occur when the cross-family map slightly exceeds the recipient's own single-layer/band self-swap).* *Note on layer choice: swap layers were chosen via labeled validation transfer; '0-QA' refers to map fitting only.*

#### 1C-i. Natural Unmasked Regime (`mask_mode == none`)
| Pair | Config (Unmasked) | CF-Flip 1x (95% CI) | CF-Flip 5x Powered (95% CI) | AD / LORO-5 | Shuf 1x / 5x | Matched Native Ceiling | Transport Eff (1x / 5x) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `proc_var_matched` (1L, no mask) | 76.5% [69.4, 83.0] (n=149) | — | 72.7% / 71.9% | 0.0% | 87.9% (`A0_self_single_l9`) | 87.0% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `proc_ls_scale` (1L, no mask) | 75.2% [67.2, 82.3] (n=149) | — | 74.2% / 68.5% | 0.0% | 87.9% (`A0_self_single_l9`) | 85.5% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `ridge_sigma_raw` (1L, no mask) | 69.1% [60.7, 77.0] (n=149) | — | 62.1% / 60.7% | 0.0% | 87.9% (`A0_self_single_l9`) | 78.6% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `ridge_sigma_rescaled` (1L, no mask) | 69.8% [61.7, 77.5] (n=149) | — | 56.1% / 65.2% | 0.0% | 87.9% (`A0_self_single_l9`) | 79.4% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `ridge_polar_I` (1L, no mask) | 75.8% [68.3, 82.6] (n=149) | — | 74.2% / 66.3% | 0.0% | 87.9% (`A0_self_single_l9`) | 86.3% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `ridge_clamped_1_5` [best] (1L, no mask) | 78.5% [71.2, 84.8] (n=149) | — | 75.8% / 74.2% | 0.0% | 87.9% (`A0_self_single_l9`) | 89.3% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `ridge_top16_clamped` (1L, no mask) | 75.2% [67.8, 81.8] (n=149) | — | 72.7% / 69.7% | 0.0% | 87.9% (`A0_self_single_l9`) | 85.5% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `F_proc_base_norm_rescale` (1L, no mask) | 77.8% [70.7, 84.2] (n=149) | — | 75.8% / 73.0% | 0.0% | 87.9% (`A0_self_single_l9`) | 88.5% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `F_spherical_procrustes` (1L, no mask) | 76.5% [68.9, 83.1] (n=149) | — | 69.7% / 73.0% | 0.0% | 87.9% (`A0_self_single_l9`) | 87.0% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `F_entity_transplant_r256` (1L, no mask) | 51.7% [42.2, 61.2] (n=149) | — | 48.5% / 53.9% | 0.0% | 87.9% (`A0_self_single_l9`) | 58.8% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `F_entity_transplant_r1024` (1L, no mask) | 73.2% [65.0, 80.2] (n=149) | — | 65.1% / 68.5% | 0.0% | 87.9% (`A0_self_single_l9`) | 83.2% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A0_single_unlab10k` (no mask) | 76.5% [69.4, 83.0] (n=149) | 79.0% [74.1, 83.9] (n=725), AD=74.9% | 72.7% / 71.9% | 0.0% / 0.4% | 87.9% (`A0_self_single_l9`) | 87.0% / 88.3% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A1_band_main_none` (no mask) | 75.2% [67.6, 82.3] (n=149) | — | 72.7% / 76.4% | 0.0% | 87.9% (`A0_self_single_l9`) | 85.5% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_raw_single_A0` (no mask) | 81.9% [75.3, 87.9] (n=149) | — | 75.8% / 77.5% | 0.0% | 87.9% (`A0_self_single_l9`) | 93.1% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_m3_avg_single_A0` (no mask) | 81.9% [75.2, 87.8] (n=149) | — | 77.3% / 77.5% | 0.0% | 87.9% (`A0_self_single_l9`) | 93.1% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_bet_01wit_single_A0` (no mask) | 81.2% [74.4, 87.3] (n=149) | — | 75.8% / 76.4% | 0.0% | 87.9% (`A0_self_single_l9`) | 92.4% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_cross_ctx_single_A0` (no mask) | 83.2% [76.6, 89.0] (n=149) | — | 80.3% / 75.3% | 0.0% | 87.9% (`A0_self_single_l9`) | 94.7% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `proc_var_matched` (1L, no mask) | 43.8% [35.9, 52.3] (n=153) | — | 32.8% / 39.6% | 0.0% | 53.6% (`A0_self_l8`) | 81.7% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `proc_ls_scale` (1L, no mask) | 43.1% [35.6, 51.4] (n=153) | — | 34.3% / 40.7% | 0.0% | 53.6% (`A0_self_l8`) | 80.5% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `ridge_sigma_raw` (1L, no mask) | 34.0% [26.6, 41.7] (n=153) | — | 29.8% / 29.7% | 0.7% | 53.6% (`A0_self_l8`) | 63.4% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `ridge_sigma_rescaled` (1L, no mask) | 35.9% [28.2, 44.2] (n=153) | — | 28.4% / 34.1% | 0.0% | 53.6% (`A0_self_l8`) | 67.1% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `ridge_polar_I` (1L, no mask) | 50.3% [42.2, 58.4] (n=153) | — | 40.3% / 51.6% | 1.3% | 53.6% (`A0_self_l8`) | 93.9% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `ridge_clamped_1_5` [best] (1L, no mask) | 52.9% [44.9, 60.9] (n=153) | — | 41.8% / 57.1% | 0.0% | 53.6% (`A0_self_l8`) | 98.8% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `ridge_top16_clamped` (1L, no mask) | 47.7% [39.6, 55.7] (n=153) | — | 38.8% / 46.2% | 0.7% | 53.6% (`A0_self_l8`) | 89.0% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `F_proc_base_norm_rescale` (1L, no mask) | 41.8% [34.1, 50.3] (n=153) | — | 31.3% / 41.8% | 0.0% | 53.6% (`A0_self_l8`) | 78.1% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `F_spherical_procrustes` (1L, no mask) | 39.9% [32.7, 47.9] (n=153) | — | 29.8% / 41.8% | 0.0% | 53.6% (`A0_self_l8`) | 74.4% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `F_entity_transplant_r256` (1L, no mask) | 26.8% [20.0, 33.8] (n=153) | — | 25.4% / 24.2% | 0.0% | 53.6% (`A0_self_l8`) | 50.0% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `F_entity_transplant_r1024` (1L, no mask) | 41.2% [33.5, 49.3] (n=153) | — | 32.8% / 42.9% | 0.0% | 53.6% (`A0_self_l8`) | 76.8% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A0_single_unlab10k` (no mask) | 43.8% [35.9, 52.3] (n=153) | 47.5% [40.5, 54.0] (n=750), AD=41.4% | 32.8% / 39.6% | 0.0% / 0.3% | 53.6% (`A0_self_l8`) | 81.7% / 89.2% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A1_band_main_none` (no mask) | 35.9% [28.5, 43.8] (n=153) | — | 34.3% / 39.6% | 0.0% | 52.9% (`A1_self_band3_7_8_9`) | 67.9% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_raw_single_A0` (no mask) | 43.1% [35.3, 51.3] (n=153) | — | 35.8% / 46.2% | 0.0% | 53.6% (`A0_self_l8`) | 80.5% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_m3_avg_single_A0` (no mask) | 42.5% [34.2, 50.7] (n=153) | — | 32.8% / 44.0% | 0.0% | 53.6% (`A0_self_l8`) | 79.3% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_bet_01wit_single_A0` (no mask) | 42.5% [34.4, 50.7] (n=153) | — | 34.3% / 44.0% | 0.0% | 53.6% (`A0_self_l8`) | 79.3% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_cross_ctx_single_A0` (no mask) | 43.1% [35.1, 51.3] (n=153) | — | 32.8% / 42.9% | 0.0% | 53.6% (`A0_self_l8`) | 80.5% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `proc_var_matched` (1L, no mask) | 70.6% [63.3, 77.1] (n=177) | — | 72.5% / 80.2% | 0.6% | 83.0% (`A0_self_single_l9`) | 85.0% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `proc_ls_scale` [best] (1L, no mask) | 71.2% [64.0, 78.0] (n=177) | — | 73.9% / 82.3% | 0.6% | 83.0% (`A0_self_single_l9`) | 85.7% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `ridge_sigma_raw` (1L, no mask) | 70.1% [63.3, 76.7] (n=177) | — | 66.7% / 72.9% | 0.6% | 83.0% (`A0_self_single_l9`) | 84.4% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `ridge_sigma_rescaled` (1L, no mask) | 71.2% [64.5, 77.7] (n=177) | — | 69.6% / 80.2% | 0.6% | 83.0% (`A0_self_single_l9`) | 85.7% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `ridge_polar_I` (1L, no mask) | 68.9% [61.4, 75.8] (n=177) | — | 66.7% / 74.0% | 0.6% | 83.0% (`A0_self_single_l9`) | 83.0% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `ridge_clamped_1_5` (1L, no mask) | 70.1% [62.9, 76.8] (n=177) | — | 71.0% / 77.1% | 0.6% | 83.0% (`A0_self_single_l9`) | 84.4% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `ridge_top16_clamped` (1L, no mask) | 68.4% [60.6, 75.4] (n=177) | — | 66.7% / 75.0% | 0.6% | 83.0% (`A0_self_single_l9`) | 82.3% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `F_proc_base_norm_rescale` (1L, no mask) | 70.1% [62.9, 76.9] (n=177) | — | 69.6% / 80.2% | 0.6% | 83.0% (`A0_self_single_l9`) | 84.4% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `F_spherical_procrustes` (1L, no mask) | 69.5% [62.2, 76.3] (n=177) | — | 69.6% / 81.2% | 0.6% | 83.0% (`A0_self_single_l9`) | 83.7% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `F_entity_transplant_r256` (1L, no mask) | 53.1% [45.2, 61.3] (n=177) | — | 55.1% / 59.4% | 0.0% | 83.0% (`A0_self_single_l9`) | 63.9% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `F_entity_transplant_r1024` (1L, no mask) | 68.9% [61.6, 75.7] (n=177) | — | 68.1% / 79.2% | 0.6% | 83.0% (`A0_self_single_l9`) | 83.0% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A0_single_unlab10k` (no mask) | 70.6% [63.3, 77.1] (n=177) | 71.7% [66.8, 76.6] (n=880), AD=71.6% | 72.5% / 80.2% | 0.6% / 0.4% | 83.0% (`A0_self_single_l9`) | 85.0% / 84.0% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A1_band_main_none` (no mask) | 67.8% [60.2, 75.0] (n=177) | — | 72.5% / 81.2% | 0.6% | 82.5% (`A1_self_band3_8_9_10`) | 82.2% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_raw_single_A0` (no mask) | 74.6% [68.0, 80.9] (n=177) | — | 73.9% / 82.3% | 0.6% | 83.0% (`A0_self_single_l9`) | 89.8% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_m3_avg_single_A0` (no mask) | 71.2% [64.2, 77.7] (n=177) | — | 66.7% / 79.2% | 0.6% | 83.0% (`A0_self_single_l9`) | 85.7% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_bet_01wit_single_A0` (no mask) | 72.3% [65.5, 78.9] (n=177) | — | 68.1% / 79.2% | 0.6% | 83.0% (`A0_self_single_l9`) | 87.1% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_cross_ctx_single_A0` (no mask) | 72.3% [65.5, 78.8] (n=177) | — | 68.1% / 82.3% | 0.6% | 83.0% (`A0_self_single_l9`) | 87.1% |

#### 1C-ii. Earlier-Subject / Full-Subject Attention-Knockout Regime (`mask_mode != none`)
| Pair | Config (Attention Knockout) | CF-Flip 1x (95% CI) | CF-Flip 5x Powered (95% CI) | AD / LORO-5 | Shuf 1x / 5x | Matched Native Ceiling | Transport Eff (1x / 5x) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A2_single_full_ko` (mask=full_knockout) | 85.9% [79.8, 91.2] (n=149) | 87.3% [83.9, 90.4] (n=725), AD=82.0% | 81.8% / 82.0% | 0.0% / 0.5% | 93.3% (`A2_self_l9_full_ko`) | 92.1% / 91.9% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A3_band_main_earlier_ko` (mask=earlier_subj) | 85.2% [79.0, 90.7] (n=149) | 86.2% [82.6, 89.5] (n=725), AD=82.0% | 81.8% / 83.2% | 0.0% / 0.5% | 94.6% (`A3_self_band3_earlier_ko`) | 90.1% / 90.5% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A3_band_main_full_ko` (mask=full_knockout) | 85.2% [78.6, 91.2] (n=149) | 85.4% [82.2, 88.4] (n=725), AD=74.6% | 72.7% / 82.0% | 0.0% / 0.5% | 96.0% (`A3_self_band3_full_ko`) | 88.8% / 89.8% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A3_band_main_nonsubj_ctrl` (mask=nonsubj_ctrl) | 74.5% [67.1, 81.5] (n=149) | — | 74.2% / 75.3% | 0.0% | 87.2% (`A3_self_band3_nonsubj_ctrl`) | 85.4% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_raw_band_A3_full_ko` (mask=full_knockout) | 89.3% [83.7, 94.0] (n=149) | 87.3% [84.7, 89.8] (n=725), AD=77.4% | 80.3% / 86.5% | 0.0% / 0.5% | 96.0% (`A3_self_band3_full_ko`) | 93.0% / 91.9% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_m3_avg_band_A3_full_ko` (mask=full_knockout) | 86.6% [80.5, 92.0] (n=149) | 87.2% [84.5, 89.7] (n=725), AD=78.0% | 77.3% / 84.3% | 0.0% / 0.5% | 96.0% (`A3_self_band3_full_ko`) | 90.2% / 91.7% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_bet_01wit_band_A3_full_ko` (mask=full_knockout) | 88.6% [82.9, 93.7] (n=149) | 87.2% [84.5, 89.7] (n=725), AD=78.0% | 80.3% / 84.3% | 0.0% / 0.5% | 96.0% (`A3_self_band3_full_ko`) | 92.3% / 91.7% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_cross_ctx_band_A3_full_ko` (mask=full_knockout) | 88.6% [82.9, 93.5] (n=149) | 88.8% [86.3, 91.1] (n=725), AD=83.8% | 81.8% / 83.2% | 0.0% / 0.5% | 96.0% (`A3_self_band3_full_ko`) | 92.3% / 93.5% |
| `qwen2.5-3b->smollm2-1.7b` | LeverC `C_woodbury_k16_full_ko` (mask=full_knockout) | 89.3% [83.7, 93.8] (n=149) | 87.9% [85.1, 90.5] (n=725), AD=79.5% | 80.3% / 80.9% | 0.0% / 0.5% | 96.0% (`A3_self_band3_full_ko`) | 93.0% / 92.5% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A2_single_full_ko` (mask=full_knockout) | 91.5% [86.6, 95.7] (n=153) | 92.0% [89.5, 94.3] (n=750), AD=88.5% | 88.1% / 81.3% | 0.0% / 0.7% | 93.5% (`A2_self_l8_full_ko`) | 97.9% / 99.4% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A3_band_main_earlier_ko` (mask=earlier_subj) | 88.2% [82.8, 93.3] (n=153) | 87.5% [84.1, 90.5] (n=750), AD=84.0% | 86.6% / 81.3% | 0.0% / 0.4% | 83.7% (`A3_self_band3_12_14_full_ko`) | 105.5% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A3_band_main_full_ko` (mask=full_knockout) | 89.5% [84.0, 94.3] (n=153) | 89.5% [86.4, 92.2] (n=750), AD=83.4% | 83.6% / 81.3% | 0.0% / 0.4% | 83.7% (`A3_self_band3_12_14_full_ko`) | 107.0% / 107.2% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A3_band_main_nonsubj_ctrl` (mask=nonsubj_ctrl) | 35.3% [28.2, 43.0] (n=153) | — | 32.8% / 39.6% | 0.0% | 41.8% (`A0_self_l16`) | 84.4% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A2_l16_full_ko` (mask=full_knockout) | 83.7% [77.1, 89.5] (n=153) | 83.9% [80.1, 87.4] (n=750), AD=74.6% | 73.1% / 71.4% | 0.0% / 0.3% | 77.8% (`A2_self_l16_full_ko`) | 107.6% / 108.3% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_raw_band_A3_full_ko` (mask=full_knockout) | 88.2% [82.8, 93.2] (n=153) | 89.5% [86.7, 92.1] (n=750), AD=83.4% | 80.6% / 84.6% | 0.0% / 0.8% | 83.7% (`A3_self_band3_12_14_full_ko`) | 105.5% / 107.2% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_m3_avg_band_A3_full_ko` (mask=full_knockout) | 87.6% [82.1, 92.7] (n=153) | 89.5% [86.7, 92.2] (n=750), AD=84.9% | 82.1% / 84.6% | 0.0% / 0.8% | 83.7% (`A3_self_band3_12_14_full_ko`) | 104.7% / 107.2% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_bet_01wit_band_A3_full_ko` (mask=full_knockout) | 88.9% [83.5, 93.8] (n=153) | 89.9% [87.2, 92.4] (n=750), AD=85.2% | 83.6% / 84.6% | 0.0% / 0.8% | 83.7% (`A3_self_band3_12_14_full_ko`) | 106.2% / 107.7% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_cross_ctx_band_A3_full_ko` (mask=full_knockout) | 89.5% [84.3, 94.4] (n=153) | 89.6% [86.9, 92.2] (n=750), AD=85.2% | 85.1% / 84.6% | 0.0% / 0.8% | 83.7% (`A3_self_band3_12_14_full_ko`) | 107.0% / 107.3% |
| `qwen2.5-3b->granite-3.1-2b` | LeverC `C_woodbury_k16_full_ko` (mask=full_knockout) | 88.9% [83.0, 93.9] (n=153) | 88.4% [85.2, 91.4] (n=750), AD=80.8% | 80.6% / 84.6% | 0.0% / 0.5% | 83.7% (`A3_self_band3_12_14_full_ko`) | 106.2% / 105.9% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A2_single_full_ko` (mask=full_knockout) | 81.4% [75.3, 86.9] (n=177) | 82.8% [79.6, 85.7] (n=880), AD=79.8% | 78.3% / 84.4% | 0.6% / 0.8% | 92.1% (`A2_self_l9_full_ko`) | 88.3% / 87.9% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A3_band_main_earlier_ko` (mask=earlier_subj) | 79.1% [72.6, 84.9] (n=177) | 80.5% [76.8, 83.9] (n=880), AD=81.6% | 81.2% / 83.3% | 0.6% / 0.7% | 93.2% (`A3_self_band3_earlier_ko`) | 84.9% / 85.5% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A3_band_main_full_ko` (mask=full_knockout) | 81.4% [75.2, 87.1] (n=177) | 82.8% [79.6, 85.7] (n=880), AD=81.3% | 81.2% / 83.3% | 0.6% / 0.8% | 94.9% (`A3_self_band3_full_ko`) | 85.7% / 88.0% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A3_band_main_nonsubj_ctrl` (mask=nonsubj_ctrl) | 67.2% [59.6, 74.6] (n=177) | — | 72.5% / 80.2% | 0.6% | 82.5% (`A1_self_band3_8_9_10`) | 81.5% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_raw_band_A3_full_ko` (mask=full_knockout) | 85.3% [79.6, 90.4] (n=177) | 85.8% [83.0, 88.3] (n=880), AD=84.5% | 84.1% / 88.5% | 0.6% / 0.4% | 94.9% (`A3_self_band3_full_ko`) | 89.9% / 91.2% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_m3_avg_band_A3_full_ko` (mask=full_knockout) | 87.6% [82.2, 92.6] (n=177) | 86.9% [84.4, 89.2] (n=880), AD=84.8% | 85.5% / 90.6% | 0.6% / 0.4% | 94.9% (`A3_self_band3_full_ko`) | 92.3% / 92.4% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_bet_01wit_band_A3_full_ko` (mask=full_knockout) | 87.0% [81.6, 92.0] (n=177) | 86.8% [84.3, 89.1] (n=880), AD=84.8% | 85.5% / 89.6% | 0.6% / 0.4% | 94.9% (`A3_self_band3_full_ko`) | 91.7% / 92.3% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_cross_ctx_band_A3_full_ko` (mask=full_knockout) | 87.0% [81.5, 91.9] (n=177) | 87.2% [84.9, 89.3] (n=880), AD=86.3% | 87.0% / 90.6% | 0.6% / 0.7% | 94.9% (`A3_self_band3_full_ko`) | 91.7% / 92.6% |

## Pillar 2: Platonic Organs — Native SwiGLU Conjugation & Earlier Prototype Comparison
### Table 2A — P-ORGAN-0 Stage 0: Channel Census, Regime 1 (2,000-Neuron Conjugation) & Gate G0 (`results_organs_p0.json`)
| Diagnostic | Metric | Value |
| --- | --- | --- |
| Channel Census (`qwen2.5-3b L17 -> smollm2-1.7b L9`) | Held-out Read $R^2$ / Write $R^2$ (filtered vs raw) | 0.6207 / 0.6467 (raw: -3.0343) |
| Massive-Activation Outlier Filter | Dropped tokens; max norm before->after | 8/10559; 3302.9 -> 109.1 |
| Split-Half Socket Stability | Top-256 PC mean / median cos (`>=0.80`) | 0.8427 / 0.8298 (200/256) |
| Constant Pin $c_B^\top x_B \approx 1$ | $\sigma(c_A^\top x_A)$ / $\sigma(c_B^\top x_B)$ / pin err | 0.01115 / 0.00698 / 2.02e-15 |
| Gate G0 (1000-distractor Read Retrieval) | Known 24 LM Top-1 (Top-5) / Gap-17 Top-1 (Top-5) | 95.8% (100.0%) / 88.2% (100.0%); ratio 0.921 |

*Regime 1: 2,000-Neuron Stock MLP Conjugation (`n_eval_tokens = 4,104`):*
| Conjugation Arm | Median $r(g)$ | Frac $r(g)\ge 0.70$ | Median $r(a)$ | Median $\lvert\Delta\rvert/\sigma$ | BOS Max $\lvert y\rvert$ | Channel Model $R^2$ |
| --- | --- | --- | --- | --- | --- | --- |
| `V0_naive_biasfree` | 0.8192 | 99.9% | 0.5072 | 0.8314 | 0.005354 | 0.9756 |
| `V1_affine_bosnull` | 0.8192 | 99.9% | 0.6889 | 0.0459 | 0.000000 | 0.9756 |
| `V1_gamma_cpin_biasfree` | 0.7743 | 98.7% | 0.6116 | 0.0425 | 0.000000 | 0.8690 |
| `V1_ridge_affine_bosnull` | 0.9138 | 100.0% | 0.8484 | 0.0074 | 0.000000 | 0.8170 |

### Table 2B — Earlier Bespoke Prototype Across 4 Recipients (`results_organs_multifamily.json` Stage A)
*Note: This earlier hand-factored bilinear prototype (`Entity × Country`) succeeded on `smollm2-1.7b` only and collapsed to `0.0%` on Granite, Falcon3, and Qwen-1.5B, motivating the frozen Channel-Pushforward engine in Pillar 4.*

| Recipient | Arm | Direct EM | Unseen 3-Attr Ripple | Unseen DS-8 | Protected Ret. | Unrelated Loc (Broken) | WikiText KL, ΔPPL | Bit-Exact Removal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` (L9) | `PURE_DONOR_BIASFREE` | 37.5% | 45.8% | 75.0% | 93.8% | 2/175 (98.9%) | KL=0.000262, ΔPPL=+0.0148 | 0.0 |
| `smollm2-1.7b` (L9) | `PURE_DONOR_AFFINE` | 37.5% | 45.8% | 75.0% | 93.8% | 3/175 (98.3%) | KL=0.000704, ΔPPL=+0.0353 | 0.0 |
| `smollm2-1.7b` (L9) | `PURE_DONOR_CONST_UP_BIASFREE` | 25.0% | 29.2% | 62.5% | 100.0% | 0/175 (100.0%) | KL=0.003146, ΔPPL=+0.0413 | 0.0 |
| `smollm2-1.7b` (L9) | `PURE_DONOR_CONST_UP_AFFINE` | 37.5% | 45.8% | 75.0% | 81.2% | 3/175 (98.3%) | KL=0.036897, ΔPPL=+0.7829 | 0.0 |
| `smollm2-1.7b` (L9) | `PACKAGED_BIASFREE` | 37.5% | 58.3% | 87.5% | 93.8% | 4/175 (97.7%) | KL=0.000210, ΔPPL=-0.0043 | 0.0 |
| `smollm2-1.7b` (L9) | `PACKAGED_AFFINE` | 50.0% | 58.3% | 87.5% | 93.8% | 6/175 (96.6%) | KL=0.000474, ΔPPL=+0.0153 | 0.0 |
| `smollm2-1.7b` (L9) | `SHUFFLED_R` | 0.0% | 0.0% | 0.0% | 93.8% | 0/175 (100.0%) | KL=0.001987, ΔPPL=+0.0217 | 0.0 |
| `smollm2-1.7b` (L9) | `NATIVE_B_TWIN_ORACLE` | 87.5% | 85.4% | 75.0% | 81.2% | 0/175 (100.0%) | KL=0.000033, ΔPPL=+0.0000 | 0.0 |
| `granite-3.1-2b` (L8) | `PURE_DONOR_BIASFREE` | 0.0% | 0.0% | 0.0% | 62.5% | 1/183 (99.5%) | KL=0.000267, ΔPPL=+0.0040 | 0.0 |
| `granite-3.1-2b` (L8) | `PURE_DONOR_AFFINE` | 0.0% | 0.0% | 0.0% | 62.5% | 1/183 (99.5%) | KL=0.000255, ΔPPL=+0.0041 | 0.0 |
| `granite-3.1-2b` (L8) | `SMOL_PACKAGED_ON_GRANITE_BIASFREE` | 0.0% | 0.0% | 0.0% | 62.5% | 0/183 (100.0%) | KL=0.000180, ΔPPL=+0.0003 | 0.0 |
| `granite-3.1-2b` (L8) | `SMOL_PACKAGED_ON_GRANITE_AFFINE` | 0.0% | 0.0% | 0.0% | 62.5% | 0/183 (100.0%) | KL=0.000174, ΔPPL=+0.0004 | 0.0 |
| `granite-3.1-2b` (L8) | `SHUFFLED_R` | 0.0% | 0.0% | 0.0% | 56.2% | 0/183 (100.0%) | KL=0.000669, ΔPPL=-0.0057 | 0.0 |
| `falcon3-3b` (L9) | `PURE_DONOR_BIASFREE` | 0.0% | 0.0% | 0.0% | 93.8% | 0/177 (100.0%) | KL=0.000147, ΔPPL=-0.0004 | 0.0 |
| `falcon3-3b` (L9) | `PURE_DONOR_AFFINE` | 0.0% | 0.0% | 0.0% | 93.8% | 0/177 (100.0%) | KL=0.000185, ΔPPL=-0.0062 | 0.0 |
| `falcon3-3b` (L9) | `SHUFFLED_R` | 0.0% | 0.0% | 0.0% | 81.2% | 1/177 (99.4%) | KL=0.000178, ΔPPL=-0.0089 | 0.0 |
| `qwen2.5-1.5b` (L13) | `PURE_DONOR_BIASFREE` | 0.0% | 0.0% | 0.0% | 93.8% | 6/155 (96.1%) | KL=0.000159, ΔPPL=+0.0036 | 0.0 |
| `qwen2.5-1.5b` (L13) | `PURE_DONOR_AFFINE` | 0.0% | 0.0% | 0.0% | 93.8% | 4/155 (97.4%) | KL=0.000160, ΔPPL=+0.0028 | 0.0 |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_R` | 0.0% | 0.0% | 0.0% | 100.0% | 0/155 (100.0%) | KL=0.000690, ΔPPL=-0.0333 | 0.0 |

#### Pillar 2 Comparison Foils — Output-Layer (L23) Non-Canonical `W_down` Overwrite vs. Non-Destructive Gated Sidecar (`qwen2.5-3b -> smollm2-1.7b`)
*Note: The `W_down` overwrite foil is a non-canonical single-layer L23 closed-form edit (`α=1.5, λ=10`), NOT canonical multi-layer MEMIT (`meng_2023_memit`, `R={3..8}, λ=15,000`). Mid-layer multi-hop ripple was not measured for these output-layer foils.*

| Method | Deployment Regime | Private Table `f` & Deranged `f'` (96 unseen paraphrases) | 196-Fact Unrelated Retention Drop | Mid-Layer Multi-Hop Ripple |
| --- | --- | --- | --- | --- |
| Live Dual-Adjoint Crossing (`L35->L23`) | Donor alive | f=93.8%/64.6% EM; f'=82.3%/66.7% EM (on old f: 5.2%) | n/a (live) | not measured |
| Non-canonical single-layer L23 closed-form `W_down` overwrite (`α=1.5, λ=10`) | Donor unloaded, 0 SGD | f=47.9%/27.1% EM; f'=51.0%/47.9% EM | **+24.49 pp** | not measured |
| Non-Destructive Gated Sidecar (`L23`, frozen `MLP_orig`) | Donor unloaded, 0 SGD | f=84.4%/62.5% EM; f'=70.8%/60.4% EM (on old f: 5.2%) | **+0.00 pp** (FPR=0.00%, 0/1876 toks) | not measured |

## Pillar 3: Multi-Donor Composition, Modular Removal & Latent-Knowledge Unlocking (T3)
### Table 3A — Earlier Prototype Native Multi-Donor Composition (`Qwen2.5-3B` + `Falcon3-3B` -> Solo `SmolLM2-1.7B`) (`results_organs_multifamily.json` Stage B)
| Mode | State | $d_{\text{ff}}$ | France->Japan (Qwen Donor) | Italy->Brazil (Falcon Donor) | 175 Unrelated Loc | WikiText KL | Max $\lvert\Delta\text{logit}\rvert$ vs Base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `biasfree` | `JOINT_QWEN_PLUS_FALCON` | 8196 | dir=37.5%, rip3=50.0% | dir=0.0%, rip3=0.0% | 0/175 (100.0%) | 0.000141 | — |
| `biasfree` | `SELECTIVE_UNLEARN_QWEN` | 8194 | dir=0.0%, rip3=0.0% | dir=0.0%, rip3=0.0% | 0/175 (100.0%) | 0.000054 | — |
| `biasfree` | `SELECTIVE_UNLEARN_FALCON` | 8194 | dir=37.5%, rip3=50.0% | dir=0.0%, rip3=0.0% | 0/175 (100.0%) | 0.000120 | — |
| `biasfree` | `UNLEARN_ALL` | 8192 | restored to base | restored to base | 0/175 (100.0%) | 0.000000 | 0.0 |
| `affine` | `JOINT_QWEN_PLUS_FALCON` | 8196 | dir=37.5%, rip3=50.0% | dir=0.0%, rip3=0.0% | 0/175 (100.0%) | 0.000312 | — |
| `affine` | `SELECTIVE_UNLEARN_QWEN` | 8194 | dir=0.0%, rip3=0.0% | dir=0.0%, rip3=0.0% | 0/175 (100.0%) | 0.000049 | — |
| `affine` | `SELECTIVE_UNLEARN_FALCON` | 8194 | dir=37.5%, rip3=50.0% | dir=0.0%, rip3=0.0% | 0/175 (100.0%) | 0.000294 | — |
| `affine` | `UNLEARN_ALL` | 8192 | restored to base | restored to base | 0/175 (100.0%) | 0.000000 | 0.0 |

### Table 3B — Output-Layer 3-Family Split-Donor Composition (`results_pilot3*.json`)
*Note: Full 2-step greedy EM composition (`62.5%`) succeeds on `smollm2-1.7b` only; on the other 3 recipients 2-step greedy EM is `6.2%` (`qwen2.5-1.5b`), `18.8%` (`falcon3-3b`), and `15.6%` (`granite-3.1-2b`).*

| Recipient | Condition (Both Donors Unloaded) | Hop-1 `f` (8w / EM) | Hop-2 `g` (8w / EM) | 2-Step 8-Way (alt) | 2-Step Greedy EM (alt) | 196-Fact Loc Drop | Test FPR (tok / pr) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` | Non-canonical L23 `W_down` `f + g` | 40.6% / 6.2% | 33.3% / 0.0% | 21.9% (alt 12.5%) | 0.0% | +45.92pp | n/a |
| `smollm2-1.7b` | Gated `f + g` | 84.4% / 62.5% | 79.2% / 87.5% | 59.4% (alt 6.2%) | 62.5% (alt 6.2%) | +0.00pp | 0.00% (0.0% pr) |
| `smollm2-1.7b` | Gated `f + g'` (Hot-Swap) | 84.4% / 62.5% | 79.2% / 79.2% | 59.4% (alt 12.5%) | 62.5% (alt 12.5%) | +0.00pp | 0.00% (0.0% pr) |
| `smollm2-1.7b` | Ablation: Donor A only | 84.4% / 62.5% | 4.2% / 0.0% | 9.4% (alt 18.8%) | 6.2% (alt 12.5%) | +0.00pp | 0.00% (0.0% pr) |
| `smollm2-1.7b` | Ablation: Donor C only | 11.5% / 0.0% | 79.2% / 87.5% | 12.5% (alt 12.5%) | 12.5% (alt 12.5%) | +0.00pp | 0.00% (0.0% pr) |
| `smollm2-1.7b` | Control: Random Dual Map | 7.3% / 0.0% | 12.5% / 0.0% | 9.4% (alt 12.5%) | 0.0% (alt 3.1%) | +0.00pp | 0.00% (0.0% pr) |
| `qwen2.5-1.5b` | Gated `f + g` | 61.5% / 45.8% | 20.8% / 16.7% | 12.5% (alt 12.5%) | 6.2% (alt 12.5%) | +0.00pp | 0.00% (0.0% pr) |
| `qwen2.5-1.5b` | Gated `f + g'` (Hot-Swap) | 61.5% / 45.8% | 20.8% / 20.8% | 12.5% (alt 12.5%) | 12.5% (alt 12.5%) | +0.00pp | 0.00% (0.0% pr) |
| `qwen2.5-1.5b` | Ablation: Donor A only | 61.5% / 45.8% | 8.3% / 0.0% | 12.5% (alt 12.5%) | 0.0% (alt 0.0%) | +0.00pp | 0.00% (0.0% pr) |
| `qwen2.5-1.5b` | Ablation: Donor C only | 10.4% / 0.0% | 20.8% / 16.7% | 9.4% (alt 9.4%) | 12.5% (alt 9.4%) | +0.00pp | 0.00% (0.0% pr) |
| `qwen2.5-1.5b` | Control: Random Dual Map | 4.2% / 0.0% | 4.2% / 0.0% | 12.5% (alt 6.2%) | 0.0% (alt 0.0%) | +0.00pp | 0.00% (0.0% pr) |
| `falcon3-3b` | Gated `f + g` | 68.8% / 47.9% | 91.7% / 91.7% | 28.1% (alt 12.5%) | 18.8% (alt 12.5%) | +0.00pp | 0.00% (0.0% pr) |
| `falcon3-3b` | Gated `f + g'` (Hot-Swap) | 66.7% / 47.9% | 91.7% / 91.7% | 21.9% (alt 6.2%) | 21.9% (alt 9.4%) | +0.00pp | 0.00% (0.0% pr) |
| `falcon3-3b` | Ablation: Donor A only | 66.7% / 50.0% | 20.8% / 4.2% | 15.6% (alt 15.6%) | 0.0% (alt 0.0%) | +0.00pp | 0.00% (0.0% pr) |
| `falcon3-3b` | Ablation: Donor C only | 11.5% / 0.0% | 91.7% / 91.7% | 15.6% (alt 9.4%) | 12.5% (alt 9.4%) | +0.00pp | 0.00% (0.0% pr) |
| `falcon3-3b` | Control: Random Dual Map | 11.5% / 0.0% | 12.5% / 0.0% | 12.5% (alt 9.4%) | 0.0% (alt 0.0%) | +0.00pp | 0.00% (0.0% pr) |
| `granite-3.1-2b` | Gated `f + g` | 21.9% / 0.0% | 79.2% / 50.0% | 15.6% (alt 15.6%) | 15.6% (alt 6.2%) | +0.00pp | 0.00% (0.0% pr) |
| `granite-3.1-2b` | Gated `f + g'` (Hot-Swap) | 21.9% / 0.0% | 83.3% / 50.0% | 15.6% (alt 15.6%) | 12.5% (alt 6.2%) | +0.00pp | 0.00% (0.0% pr) |
| `granite-3.1-2b` | Ablation: Donor A only | 21.9% / 0.0% | 8.3% / 0.0% | 12.5% (alt 12.5%) | 6.2% (alt 9.4%) | +0.00pp | 0.00% (0.0% pr) |
| `granite-3.1-2b` | Ablation: Donor C only | 14.6% / 0.0% | 79.2% / 50.0% | 12.5% (alt 18.8%) | 9.4% (alt 0.0%) | +0.00pp | 0.00% (0.0% pr) |
| `granite-3.1-2b` | Control: Random Dual Map | 13.5% / 0.0% | 0.0% / 0.0% | 12.5% (alt 12.5%) | 3.1% (alt 0.0%) | +0.00pp | 0.00% (0.0% pr) |

### Table 3C — Grafting Falsification Pilot: T1 Forced-Gate Ripple & Retention + T2 Key Selectivity (`results_graft_falsify.json`)
| T1 Vector Family | Direct Country EM | Unseen 3-Attr Ripple | Joint 4/4 Flip | Protected Retention | Country ΔLogOdds |
| --- | --- | --- | --- | --- | --- |
| `native_B_noun_diff` (`lA=20->lB=11, s=1.0`) | 70.8% | 62.5% | 54.2% | 88.4% | +10.46 |
| `socket_noun_diff` (`lA=20->lB=11, s=1.0`) | 62.5% | 58.3% | 37.5% | 93.0% | +8.64 |
| `city_country_sibling_native_B` (`lA=20->lB=11, s=1.0`) | 83.3% | 66.7% | 66.7% | 69.8% | +12.63 |
| `city_country_sibling_socket` (`lA=20->lB=11, s=1.0`) | 75.0% | 65.3% | 62.5% | 79.1% | +10.92 |
| `wrong_country_control` (`lA=20->lB=11, s=1.0`) | 0.0% | 2.8% | 0.0% | 90.7% | +4.80 |
| `norm_matched_random_dir` (`lA=20->lB=11, s=1.0`) | 0.0% | 0.0% | 0.0% | 100.0% | +0.21 |
| `shuffled_alignment_control` (`lA=20->lB=11, s=1.0`) | 0.0% | 0.0% | 0.0% | 100.0% | +0.14 |

*T2 Key Selectivity & Gate 3 Implicit Constant Stability:*
| Key Family | Mean Margin (min pos - max hneg) | Neon Margin (p10 - p99) | Recall@0.1%FPR | Mean $d'^2$ |
| --- | --- | --- | --- | --- |
| `donor_Pk_raw_cos` (`lB=9, k=2048`) | -0.0025 | -0.0574 | 65.5% | 28.6 |
| `donor_Pk_whitened` (`lB=9, k=2048`) | +0.0004 | -0.0157 | 56.5% | 41.1 |
| `donor_Pk_whitened_r2` (`lB=9, k=2048`) | -0.0002 | -0.0258 | 67.9% | 35.6 |
| `native_B_mvdr` (`lB=9, k=2048`) | +0.5046 | +0.5548 | 100.0% | 326.1 |

*Gate 3 (`lB=9`):* non-BOS $\delta_{99}=0.204$, **with-BOS $\delta_{99}=1.553$** (kills naive zero-bias SwiGLU without explicit BOS null).

### Table 3D — Pillar 3c: T3 Latent-Knowledge Unlocking with Donor-Attributable Content (`results_graft_t3_gap*.json`)
*Note: In the scaled SmolLM2 cohort (`n=21`), 10/21 base misses were region-first answers naming the country later in the completion, and unpatched SmolLM2 already knew 11/21 capitals, 16/21 currencies, and 15/21 languages. On the strict blind-spot subset (`n=8`), B-self-roundtrip scores `33.3%` on `strict_blindspot_gap.unseen_3attr_ripple` (`0.0%` on the pooled strict per-attribute metric where matched transport scores `50.0%`, `8:0, p=0.0078`).*


#### Original T3 (`qwen2.5-3b->smollm2-1.7b`, n=17) (`results_graft_t3_gap.json`)
*Screened `292` -> `n_gap=17` (`obscure=12`, `dataset=5`).*
| Condition | All-Gap Country | All-Gap 4-Attr Mean | Strict Attr-Gap 4-Attr Mean | Alt-Entity Tracking |
| --- | --- | --- | --- | --- |
| `p1_10k_procrustes_matched` | 82.3% (n=17) | 92.7% | 62.5% (cap n=2, cur n=2, lng n=1) | — |
| `p1_10k_fisher_orth_matched` | 70.6% (n=17) | 92.7% | 87.5% (cap n=2, cur n=2, lng n=1) | — |
| `ctrl_diff_gap_entity_from_A` | 11.8% (n=17) | 20.6% | 0.0% (cap n=2, cur n=2, lng n=1) | tracks e': ctry=76.5%, 4attr=70.6% |
| `ctrl_B_self_roundtrip` | 0.0% (n=17) | 82.3% | 0.0% (cap n=2, cur n=2, lng n=1) | — |
| `ctrl_norm_matched_random_rot` | 41.2% (n=17) | 63.2% | 25.0% (cap n=2, cur n=2, lng n=1) | — |

#### Scaled T3 (`qwen2.5-3b->smollm2-1.7b`) (`results_graft_t3_gap_scaled.json`)
*Screened `265` obscure entities across `9` countries -> `n_gap=21` `A-right, B-startswith-wrong` (`strict_blindspot=8`).*
| Condition | Country (Wilson 95% CI) | 4-Attr CCL (Entity Boot CI) | Strict Blindspot (ctry / 3-rip) | Strict Per-Attr Gap Recovery | Protected Ret. | Tracks `e'` |
| --- | --- | --- | --- | --- | --- | --- |
| `p1_10k_procrustes_matched` | 76.2% [54.9, 89.4] (16/21) | 77.4% [65.5, 88.1] | 62.5% / 70.8% (n=8) | cap=44.4%(n=9), cur=100.0%(n=4), lng=0.0%(n=3) | 50.0% (1/2) | — |
| `p1_10k_fisher_orth_matched` | 81.0% [60.0, 92.3] (17/21) | 81.0% [69.0, 91.7] | 62.5% / 75.0% (n=8) | cap=33.3%(n=9), cur=100.0%(n=4), lng=66.7%(n=3) | 50.0% (1/2) | — |
| `ctrl_B_self_roundtrip` | 0.0% [0.0, 15.5] (0/21) | 50.0% [39.3, 60.7] | 0.0% / 33.3% (n=8) | cap=0.0%(n=9), cur=0.0%(n=4), lng=0.0%(n=3) | 100.0% (2/2) | — |
| `ctrl_shuffled_R` | 19.1% [7.7, 40.0] (4/21) | 33.3% [17.9, 50.0] | 0.0% / 16.7% (n=8) | cap=33.3%(n=9), cur=0.0%(n=4), lng=0.0%(n=3) | 50.0% (1/2) | — |
| `ctrl_wrong_entity_e_prime` | 4.8% [0.9, 22.7] (1/21) | 5.9% [0.0, 14.3] | 0.0% / 0.0% (n=8) | cap=11.1%(n=9), cur=0.0%(n=4), lng=0.0%(n=3) | 0.0% (0/2) | ctry=81.0%, 3rip=71.4% |

*Exact Two-Sided McNemar Tests:*
| Comparison | Country (n10:n01) | p-value | Pooled Strict 3-Attr Ripple | p-value |
| --- | --- | --- | --- | --- |
| `p1_10k_procrustes_matched` vs `ctrl_B_self_roundtrip` | 76.2% vs 0.0% (16:0) | 3.05e-05 | 50.0% vs 0.0% (8:0) | 0.0078 |
| `p1_10k_procrustes_matched` vs `ctrl_shuffled_R` | 76.2% vs 19.1% (12:0) | 0.0005 | 50.0% vs 18.8% (7:2) | 0.1797 |
| `p1_10k_procrustes_matched` vs `ctrl_wrong_entity_e_prime` | 76.2% vs 4.8% (15:0) | 6.10e-05 | 50.0% vs 6.2% (8:1) | 0.0391 |
| `p1_10k_fisher_orth_matched` vs `ctrl_B_self_roundtrip` | 81.0% vs 0.0% (17:0) | 1.53e-05 | 56.2% vs 0.0% (9:0) | 0.0039 |
| `p1_10k_fisher_orth_matched` vs `ctrl_shuffled_R` | 81.0% vs 19.1% (13:0) | 0.0002 | 56.2% vs 18.8% (8:2) | 0.1094 |
| `p1_10k_fisher_orth_matched` vs `ctrl_wrong_entity_e_prime` | 81.0% vs 4.8% (16:0) | 3.05e-05 | 56.2% vs 6.2% (9:1) | 0.0215 |

#### Scaled T3 (`qwen2.5-3b->granite-3.1-2b`) (`results_graft_t3_gap_scaled_granite.json`)
*Screened `265` obscure entities across `6` countries -> `n_gap=8` `A-right, B-startswith-wrong` (`strict_blindspot=5`).*
| Condition | Country (Wilson 95% CI) | 4-Attr CCL (Entity Boot CI) | Strict Blindspot (ctry / 3-rip) | Strict Per-Attr Gap Recovery | Protected Ret. | Tracks `e'` |
| --- | --- | --- | --- | --- | --- | --- |
| `p1_10k_procrustes_matched` | 37.5% [13.7, 69.4] (3/8) | 46.9% [28.1, 65.6] | 40.0% / 46.7% (n=5) | cap=33.3%(n=3), cur=50.0%(n=4), lng=0.0%(n=2) | 100.0% (2/2) | — |
| `p1_10k_fisher_orth_matched` | 50.0% [21.5, 78.5] (4/8) | 46.9% [28.1, 62.5] | 60.0% / 33.3% (n=5) | cap=33.3%(n=3), cur=25.0%(n=4), lng=0.0%(n=2) | 100.0% (2/2) | — |
| `ctrl_B_self_roundtrip` | 0.0% [0.0, 32.4] (0/8) | 37.5% [18.8, 56.2] | 0.0% / 26.7% (n=5) | cap=0.0%(n=3), cur=0.0%(n=4), lng=0.0%(n=2) | 100.0% (2/2) | — |
| `ctrl_shuffled_R` | 12.5% [2.2, 47.1] (1/8) | 31.2% [12.5, 56.2] | 0.0% / 20.0% (n=5) | cap=0.0%(n=3), cur=25.0%(n=4), lng=0.0%(n=2) | 100.0% (2/2) | — |
| `ctrl_wrong_entity_e_prime` | 0.0% [0.0, 32.4] (0/8) | 18.8% [6.2, 37.5] | 0.0% / 13.3% (n=5) | cap=0.0%(n=3), cur=0.0%(n=4), lng=0.0%(n=2) | 100.0% (2/2) | ctry=75.0%, 3rip=41.7% |

*Exact Two-Sided McNemar Tests:*
| Comparison | Country (n10:n01) | p-value | Pooled Strict 3-Attr Ripple | p-value |
| --- | --- | --- | --- | --- |
| `p1_10k_procrustes_matched` vs `ctrl_B_self_roundtrip` | 37.5% vs 0.0% (3:0) | 0.2500 | 33.3% vs 0.0% (3:0) | 0.2500 |
| `p1_10k_procrustes_matched` vs `ctrl_shuffled_R` | 37.5% vs 12.5% (3:1) | 0.6250 | 33.3% vs 11.1% (2:0) | 0.5000 |
| `p1_10k_procrustes_matched` vs `ctrl_wrong_entity_e_prime` | 37.5% vs 0.0% (3:0) | 0.2500 | 33.3% vs 0.0% (3:0) | 0.2500 |
| `p1_10k_fisher_orth_matched` vs `ctrl_B_self_roundtrip` | 50.0% vs 0.0% (4:0) | 0.1250 | 22.2% vs 0.0% (2:0) | 0.5000 |
| `p1_10k_fisher_orth_matched` vs `ctrl_shuffled_R` | 50.0% vs 12.5% (4:1) | 0.3750 | 22.2% vs 11.1% (1:0) | 1.0000 |
| `p1_10k_fisher_orth_matched` vs `ctrl_wrong_entity_e_prime` | 50.0% vs 0.0% (4:0) | 0.1250 | 22.2% vs 0.0% (2:0) | 0.5000 |

## Pillar 4: Frozen Channel-Pushforward Engine (Cap-1 Geography & Single-Execution Cap-2 `BIO-RIPPLE`)
### Table 4A — Cap-1 Multi-Family Channel-Pushforward (`results_cap1_channel_pushforward.json`)
*Frozen Engine SHA-256:* `22ef57c9d196e12f` ; *Declared Tolerances:* `alpha=0.002, alpha_neg=0.03, delta=0.02` (plus frozen abstention gates `R^2<=0, r<=0.25, a<=0.15`) ; *Targets:* `smollm2, granite, falcon3, qwen1.5b`.

*Note on `HardNeg (JP/BR)`: The 16 hard-negative prompts include `Kyoto` and `Nagoya`, whose true answer is `Japan`, producing a `12.5%` (2/16) BASE floor on every recipient. Delta vs. BASE is shown explicitly.*

| Recipient | Arm | Status | $d_{\text{ff}}$ | Direct EM (n=16) | Unseen 3-Attr Ripple (n=48) | DS-8 | Prot Ret. | HardNeg JP/BR (Δ vs 12.5% BASE) | 196-Fact Unrel Loc (Broken) | KL / ΔPPL | Firing Rate (al, un) | Bit-Rem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` (L9) | `BASE` | unedited | — | 0.0% | 0.0% | 0.0% | 93.8% | 12.5% (+0.0pp) | 0/— (100.0%) | 0.00000 / PPL=26.100 | — | — |
| `smollm2-1.7b` (L9) | `PUSHFWD_DIGITAL_BIASFREE` | COMPILED | 8196 | 18.8% (p=0.2500) | 18.8% (p=0.0039) | 25.0% | 100.0% | 18.8% (+6.2pp) | 15/175 (91.4%) | 0.00959 / +0.2883 | al=0.00136, un=0.00118 | True |
| `smollm2-1.7b` (L9) | `PUSHFWD_DIGITAL_AFFINE` | COMPILED | 8196 | 18.8% (p=0.2500) | 14.6% (p=0.0156) | 25.0% | 100.0% | 12.5% (+0.0pp) | 9/175 (94.9%) | 0.00394 / +0.1248 | al=0.00136, un=0.00118 | True |
| `smollm2-1.7b` (L9) | `SHUFFLED_READ_AUTO` | **ABSTAIN** | — | — | abstained (France->Japan: r=0.30 a=0.18, Italy->Brazil: r=0.20 a=0.13) | — | — | — | — | — | — | — |
| `smollm2-1.7b` (L9) | `SHUFFLED_READ_FORCED` | FORCED_FROM_ABSTAIN | 8196 | 0.0% | 0.0% | 0.0% | 93.8% | 12.5% (+0.0pp) | 7/175 (96.0%) | 0.00514 / +0.1082 | al=0.00060, un=0.00098 | True |
| `smollm2-1.7b` (L9) | `SHUFFLED_WRITE` | COMPILED | 8196 | 0.0% (p=1.0000) | 0.0% (p=1.0000) | 0.0% | 100.0% | 12.5% (+0.0pp) | 6/175 (96.6%) | 0.00581 / +0.1205 | al=0.00136, un=0.00118 | True |
| `smollm2-1.7b` (L9) | `NATIVE_B_ORACLE` | COMPILED | 8196 | 62.5% (p=0.0020) | 41.7% (p=1.91e-06) | 87.5% | 81.2% | 100.0% (+87.5pp) | 133/175 (24.0%) | 0.00915 / +0.1629 | al=0.00068, un=0.00098 | True |
| `granite-3.1-2b` (L12) | `BASE` | unedited | — | 0.0% | 0.0% | 0.0% | 56.2% | 12.5% (+0.0pp) | 0/— (100.0%) | 0.00000 / PPL=14.681 | — | — |
| `granite-3.1-2b` (L12) | `PUSHFWD_DIGITAL_BIASFREE` | COMPILED | 8196 | 25.0% (p=0.1250) | 22.9% (p=0.0010) | 25.0% | 87.5% | 18.8% (+6.2pp) | 22/183 (88.0%) | 0.00159 / +0.0267 | al=0.00000, un=0.00075 | True |
| `granite-3.1-2b` (L12) | `PUSHFWD_DIGITAL_AFFINE` | COMPILED | 8196 | 25.0% (p=0.1250) | 14.6% (p=0.0156) | 25.0% | 81.2% | 18.8% (+6.2pp) | 19/183 (89.6%) | 0.00117 / +0.0204 | al=0.00000, un=0.00063 | True |
| `granite-3.1-2b` (L12) | `SHUFFLED_READ_AUTO` | **ABSTAIN** | — | — | abstained (France->Japan: r=0.27 a=0.17, Italy->Brazil: r=0.23 a=0.15) | — | — | — | — | — | — | — |
| `granite-3.1-2b` (L12) | `SHUFFLED_READ_FORCED` | FORCED_FROM_ABSTAIN | 8196 | 0.0% | 0.0% | 0.0% | 56.2% | 12.5% (+0.0pp) | 13/183 (92.9%) | 0.00051 / +0.0103 | al=0.00068, un=0.00239 | True |
| `granite-3.1-2b` (L12) | `SHUFFLED_WRITE` | COMPILED | 8196 | 0.0% (p=1.0000) | 0.0% (p=1.0000) | 0.0% | 56.2% | 12.5% (+0.0pp) | 2/183 (98.9%) | 0.00106 / +0.0161 | al=0.00000, un=0.00075 | True |
| `falcon3-3b` (L9) | `BASE` | unedited | — | 0.0% | 0.0% | 0.0% | 87.5% | 12.5% (+0.0pp) | 0/— (100.0%) | 0.00000 / PPL=24.988 | — | — |
| `falcon3-3b` (L9) | `PUSHFWD_DIGITAL_BIASFREE` | COMPILED | 9220 | 37.5% (p=0.1250) | 6.2% (p=0.2500) | 50.0% | 81.2% | 18.8% (+6.2pp) | 17/177 (90.4%) | 0.00122 / +0.0532 | al=0.00129, un=0.01071 | True |
| `falcon3-3b` (L9) | `PUSHFWD_DIGITAL_AFFINE` | COMPILED | 9220 | 25.0% (p=0.3750) | 4.2% (p=0.5000) | 37.5% | 81.2% | 18.8% (+6.2pp) | 12/177 (93.2%) | 0.00112 / +0.0490 | al=0.00120, un=0.00000 | True |
| `falcon3-3b` (L9) | `SHUFFLED_READ_AUTO` | **ABSTAIN** | — | — | abstained (France->Japan: r=0.23 a=0.11, Italy->Brazil: r=0.21 a=0.10) | — | — | — | — | — | — | — |
| `falcon3-3b` (L9) | `SHUFFLED_READ_FORCED` | FORCED_FROM_ABSTAIN | 9220 | 6.2% | 0.0% | 0.0% | 87.5% | 18.8% (+6.2pp) | 16/177 (91.0%) | 0.00061 / +0.0043 | al=0.00069, un=0.00139 | True |
| `falcon3-3b` (L9) | `SHUFFLED_WRITE` | COMPILED | 9220 | 0.0% (p=1.0000) | 0.0% (p=1.0000) | 0.0% | 87.5% | 12.5% (+0.0pp) | 2/177 (98.9%) | 0.00048 / +0.0032 | al=0.00129, un=0.01071 | True |
| `qwen2.5-1.5b` (L13) | `BASE` | unedited | — | 0.0% | 0.0% | 0.0% | 100.0% | 12.5% (+0.0pp) | 0/— (100.0%) | 0.00000 / PPL=28.251 | — | — |
| `qwen2.5-1.5b` (L13) | `PUSHFWD_DIGITAL_BIASFREE` | COMPILED | 8964 | 50.0% (p=0.0391) | 16.7% (p=0.0391) | 50.0% | 100.0% | 37.5% (+25.0pp) | 34/155 (78.1%) | 0.00059 / +0.0351 | al=0.00042, un=0.00084 | True |
| `qwen2.5-1.5b` (L13) | `PUSHFWD_DIGITAL_AFFINE` | COMPILED | 8964 | 50.0% (p=0.0391) | 16.7% (p=0.0391) | 37.5% | 100.0% | 37.5% (+25.0pp) | 30/155 (80.7%) | 0.00050 / +0.0266 | al=0.00034, un=0.00063 | True |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_READ_AUTO` | **ABSTAIN** | — | — | abstained (France->Japan: r=0.25 a=0.17, Italy->Brazil: r=0.23 a=0.16) | — | — | — | — | — | — | — |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_READ_FORCED` | FORCED_FROM_ABSTAIN | 8964 | 6.2% | 2.1% | 0.0% | 100.0% | 12.5% (+0.0pp) | 5/155 (96.8%) | 0.00147 / +0.0029 | al=0.00051, un=0.00105 | True |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_WRITE` | COMPILED | 8964 | 0.0% (p=1.0000) | 0.0% (p=1.0000) | 0.0% | 100.0% | 12.5% (+0.0pp) | 3/155 (98.1%) | 0.00034 / -0.0067 | al=0.00042, un=0.00084 | True |

#### Table 4A-iv — Cap-1 Per-Attribute Scorecard (`n=16` entities per attribute)
*Note: Because the authored geographic write vector combines `country_dst - country_src` and `capital_dst - capital_src`, **Currency** and **Language** are the cleanest tests of propagation beyond authored vector terms. Note also that pooled 3-attribute p-values (`n=48`) treat 3 prompts per entity as independent; entity-level p-values over `n=16` entities are `>= 0.0625` on SmolLM2.*

| Recipient | Arm | Direct Country (n=16) | Capital (n=16) | Currency (n=16) | Language (n=16) | 3-Attr Ripple Mean | Joint 4/4 EM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` (L9) | `PUSHFWD_DIGITAL_BIASFREE` | 18.8% | 18.8% | 18.8% | 18.8% | 18.8% | 12.5% |
| `smollm2-1.7b` (L9) | `PUSHFWD_DIGITAL_AFFINE` | 18.8% | 25.0% | 12.5% | 6.2% | 14.6% | 0.0% |
| `smollm2-1.7b` (L9) | `SHUFFLED_READ_FORCED` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `granite-3.1-2b` (L12) | `PUSHFWD_DIGITAL_BIASFREE` | 25.0% | 25.0% | 25.0% | 18.8% | 22.9% | 12.5% |
| `granite-3.1-2b` (L12) | `PUSHFWD_DIGITAL_AFFINE` | 25.0% | 12.5% | 18.8% | 12.5% | 14.6% | 6.2% |
| `granite-3.1-2b` (L12) | `SHUFFLED_READ_FORCED` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `falcon3-3b` (L9) | `PUSHFWD_DIGITAL_BIASFREE` | 37.5% | 12.5% | 6.2% | 0.0% | 6.2% | 0.0% |
| `falcon3-3b` (L9) | `PUSHFWD_DIGITAL_AFFINE` | 25.0% | 6.2% | 6.2% | 0.0% | 4.2% | 0.0% |
| `falcon3-3b` (L9) | `SHUFFLED_READ_FORCED` | 6.2% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `qwen2.5-1.5b` (L13) | `PUSHFWD_DIGITAL_BIASFREE` | 50.0% | 0.0% | 50.0% | 0.0% | 16.7% | 0.0% |
| `qwen2.5-1.5b` (L13) | `PUSHFWD_DIGITAL_AFFINE` | 50.0% | 0.0% | 50.0% | 0.0% | 16.7% | 0.0% |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_READ_FORCED` | 6.2% | 0.0% | 6.2% | 0.0% | 2.1% | 0.0% |

#### Table 4A-ii — $2 \times 2$ Calibration Ablation (Universal vs. Prose Read $\times$ Concept vs. Prose Write, Solo `smollm2-1.7b` L9)
| 2×2 Cell (`smollm2-1.7b` L9) | Direct EM | Unseen 3-Attr Ripple | DS-8 | Prot Ret. | HardNeg JP/BR | 196-Fact Loc | KL / ΔPPL | Conf. ff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Universal Read × Concept Write (Primary) | 18.8% | 18.8% | 25.0% | 100.0% | 18.8% | 15/175 (91.4%) | 0.00959 / +0.2883 | 0.00136 |
| Universal Read × Prose Write | 0.0% | 8.3% | 12.5% | 100.0% | 12.5% | 17/175 (90.3%) | 0.00879 / +0.1846 | 0.00136 |
| Prose Read × Concept Write | 56.2% | 50.0% | 37.5% | 93.8% | 12.5% | 35/175 (80.0%) | 0.01402 / +0.3578 | 0.00097 |
| Prose Read × Prose Write | 6.2% | 18.8% | 12.5% | 93.8% | 12.5% | 28/175 (84.0%) | 0.01507 / +0.4078 | 0.00097 |

#### Table 4A-iii — Cap-1 Stage B Multi-Donor Composition (`Qwen2.5-3B` Fr→Jp + `Falcon3-3B` It→Br → Solo `smollm2-1.7b` L9)
| State | $d_{\text{ff}}$ | France→Japan (Qwen Donor) | Italy→Brazil (Falcon Donor) | 196-Fact Loc | KL / ΔPPL | Bit-Exact Removal |
| --- | --- | --- | --- | --- | --- | --- |
| `JOINT_QWEN_PLUS_FALCON` | 8196 | dir=37.5%, rip3=37.5%, ds=66.7% | dir=0.0%, rip3=0.0%, ds=0.0% | 13/175 (92.6%) | 0.00170 / +0.0122 | — |
| `SELECTIVE_UNLEARN_QWEN` | 8194 | dir=0.0%, rip3=0.0%, ds=0.0% | dir=0.0%, rip3=0.0%, ds=0.0% | 3/175 (98.3%) | 0.00061 / -0.0090 | — |
| `SELECTIVE_UNLEARN_FALCON` | 8194 | dir=37.5%, rip3=37.5%, ds=66.7% | dir=0.0%, rip3=0.0%, ds=0.0% | 11/175 (93.7%) | 0.00112 / +0.0222 | — |
| `UNLEARN_ALL` | 8192 | restored to base | restored to base | 100.0% | 0.00000 | sd=True, logits=True, maxΔ=0.0 |

### Table 4B — Single-Execution Capability 2 Benchmark: `BIO-RIPPLE` (`Bird->Fish`, `Mammal->Reptile`, `results_bio_ripple_sealed.json`)
*Frozen Engine SHA-256:* `22ef57c9d196e12f` ; *Cap-2 Benchmark SHA-256 (self-attested, single recorded execution):* `d321fd0255c47cd3` ; *Declared Tolerances:* `alpha=0.002, alpha_neg=0.03, delta=0.02` ; *Donor Frame-Only AUROC:* `{'Bird->Fish': 0.5, 'Mammal->Reptile': 0.5}`.

*Definitions:* **FC** = first-token log-probability preference (`new > old`); **EM** = greedy-generation alias match; **elig** = subset of items where BASE prefers the old answer. **Hops caveat:** `hop1_heat_lamp` (`"...with a heat lamp because it is"`) entails `cold` regardless of species, creating a flat `24/48` floor on Mammal→Reptile; use the **eligible-subset hops** as the primary hop metric. **In-context premise baseline caveat:** Hard-negative prompts are never premise-prefixed (`bio_ripple_sealed.py:337-347`), so `IN_CONTEXT_PREMISE` HardNeg equals BASE by construction and is marked `n/a (unprefixed)`. **Granite firing caveat:** Granite's aligned general-text firing rate is `0.00170` (`<= 0.002`), while its unaligned-split firing rate is `0.00641` (`0.641%`); both are shown below.

| Recipient | Arm | Status / $d_{\text{ff}}$ | Direct (FC / EM) | 4-Attr Ripple (FC / EM) [Species 95% CI] | 2-Sent Hops (FC / EM) | Invariant Acc (FC/EM) | HardNeg (FF FC/EM; NT-Ret FC/EM) | Broken / Base-Correct (all 196); KL; Firing; Bit-Rem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` (L9) | `BASE` | elig: dir=48/48, rip=168/192, hop=72/96 | 0.0% / 0.0% | 12.5% / 6.2% [7.8, 17.7] | 25.0% / 0.0% | 100.0% / 72.9% | FF 0.0%/0.0%; NT-ret 100.0%/47.1% | base=181/196; PPL=26.100 |
| `smollm2-1.7b` (L9) | `IN_CONTEXT_PREMISE` | premise baseline | 6.2% / 6.2% (elig 6.2%/6.2%) | 55.7% / 38.0% (elig 50.0%/33.9%) [51.6, 59.9] | 32.3% / 15.6% (elig 9.7%/16.7%) | 100.0% / 66.7% | n/a (unprefixed) | unedited B |
| `smollm2-1.7b` (L9) | `CHANNEL_PUSHFORWARD_BIASFREE` | COMPILED (d_ff=8196) | 41.7% / 8.3% (elig 41.7%/8.3%) | 57.8% / 22.9% (elig 51.8%/19.1%) [49.5, 65.6] | 51.0% / 24.0% (elig 34.7%/31.9%) | 100.0% / 89.6% | FF 8.3%/0.0%; NT-ret 88.2%/47.1% | broken=0/181; KL=0.00107; ff=0.00128 (un=0.00098); bit_rem=True |
| `smollm2-1.7b` (L9) | `SHUFFLED_R_AUTO` | **ABSTAIN** | — | — | — | — | — | auto-abstained (0 writes) |
| `smollm2-1.7b` (L9) | `SHUFFLED_R_FORCED` | FORCED_FROM_ABSTAIN (d_ff=8196) | 2.1% / 0.0% (elig 2.1%/0.0%) | 15.6% / 7.3% (elig 3.6%/1.8%) [9.9, 22.4] | 29.2% / 4.2% (elig 5.6%/5.6%) | 100.0% / 77.1% | FF 0.0%/0.0%; NT-ret 100.0%/47.1% | broken=3/181; KL=0.00292; ff=0.00119 (un=0.00333); bit_rem=True |
| `smollm2-1.7b` (L9) | `CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | COMPILED (d_ff=8196) | 0.0% / 0.0% (elig 0.0%/0.0%) | 12.0% / 5.2% (elig 0.6%/0.6%) [7.8, 16.7] | 25.0% / 0.0% (elig 0.0%/0.0%) | 100.0% / 72.9% | FF 0.0%/0.0%; NT-ret 100.0%/47.1% | broken=18/181; KL=0.00959; ff=0.00136 (un=0.00118); bit_rem=True |
| `smollm2-1.7b` (L9) | `NATIVE_B_ORACLE` | COMPILED (d_ff=8196) | 100.0% / 47.9% (elig 100.0%/47.9%) | 94.3% / 41.1% (elig 93.5%/38.7%) [91.1, 96.9] | 71.9% / 46.9% (elig 62.5%/62.5%) | 100.0% / 83.3% | FF 75.0%/50.0%; NT-ret 35.3%/35.3% | broken=42/181; KL=0.00580; ff=0.00230 (un=0.00549); bit_rem=True |
| `granite-3.1-2b` (L12) | `BASE` | elig: dir=48/48, rip=162/192, hop=68/96 | 0.0% / 0.0% | 15.6% / 4.2% [10.9, 20.8] | 29.2% / 4.2% | 100.0% / 83.3% | FF 4.2%/0.0%; NT-ret 94.1%/41.2% | base=192/196; PPL=14.681 |
| `granite-3.1-2b` (L12) | `IN_CONTEXT_PREMISE` | premise baseline | 0.0% / 0.0% (elig 0.0%/0.0%) | 56.8% / 40.6% (elig 51.2%/42.0%) [51.6, 62.0] | 28.1% / 20.8% (elig 4.4%/7.3%) | 100.0% / 54.2% | n/a (unprefixed) | unedited B |
| `granite-3.1-2b` (L12) | `CHANNEL_PUSHFORWARD_BIASFREE` | COMPILED (d_ff=8196) | 6.2% / 4.2% (elig 6.2%/4.2%) | 34.9% / 8.8% (elig 24.1%/4.9%) [27.6, 42.7] | 43.8% / 18.8% (elig 20.6%/20.6%) | 100.0% / 85.4% | FF 12.5%/0.0%; NT-ret 82.3%/41.2% | broken=0/192; KL=0.00076; ff=0.00170 (un=0.00641); bit_rem=True |
| `granite-3.1-2b` (L12) | `SHUFFLED_R_AUTO` | **ABSTAIN** | — | — | — | — | — | auto-abstained (0 writes) |
| `granite-3.1-2b` (L12) | `SHUFFLED_R_FORCED` | FORCED_FROM_ABSTAIN (d_ff=8196) | 0.0% / 0.0% (elig 0.0%/0.0%) | 14.1% / 2.6% (elig 0.0%/0.0%) [9.9, 18.8] | 28.1% / 3.1% (elig 0.0%/0.0%) | 100.0% / 89.6% | FF 4.2%/0.0%; NT-ret 94.1%/41.2% | broken=0/192; KL=0.00033; ff=0.00128 (un=0.00453); bit_rem=True |
| `granite-3.1-2b` (L12) | `CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | COMPILED (d_ff=8196) | 0.0% / 0.0% (elig 0.0%/0.0%) | 15.6% / 4.2% (elig 0.0%/0.0%) [10.9, 20.8] | 29.2% / 4.2% (elig 0.0%/0.0%) | 100.0% / 83.3% | FF 4.2%/0.0%; NT-ret 94.1%/41.2% | broken=27/192; KL=0.00159; ff=0.00000 (un=0.00075); bit_rem=True |
| `granite-3.1-2b` (L12) | `NATIVE_B_ORACLE` | COMPILED (d_ff=8196) | 68.8% / 18.8% (elig 68.8%/18.8%) | 84.9% / 24.5% (elig 83.3%/24.7%) [80.7, 88.5] | 51.0% / 18.8% (elig 35.3%/26.5%) | 100.0% / 70.8% | FF 37.5%/0.0%; NT-ret 88.2%/5.9% | broken=43/192; KL=0.01063; ff=0.00213 (un=0.00767); bit_rem=True |
| `falcon3-3b` (L9) | `BASE` | elig: dir=48/48, rip=169/192, hop=67/96 | 0.0% / 0.0% | 12.0% / 6.2% [8.3, 15.6] | 30.2% / 15.6% | 100.0% / 85.4% | FF 0.0%/0.0%; NT-ret 100.0%/35.3% | base=186/196; PPL=24.988 |
| `falcon3-3b` (L9) | `IN_CONTEXT_PREMISE` | premise baseline | 2.1% / 2.1% (elig 2.1%/2.1%) | 64.1% / 49.5% (elig 60.4%/47.3%) [59.4, 69.3] | 50.0% / 71.9% (elig 43.3%/59.7%) | 100.0% / 54.2% | n/a (unprefixed) | unedited B |
| `falcon3-3b` (L9) | `CHANNEL_PUSHFORWARD_BIASFREE` | COMPILED (d_ff=9220) | 12.5% / 0.0% (elig 12.5%/0.0%) | 51.6% / 13.5% (elig 45.0%/7.1%) [45.8, 57.3] | 42.7% / 30.2% (elig 20.9%/20.9%) | 100.0% / 87.5% | FF 25.0%/4.2%; NT-ret 70.6%/17.6% | broken=0/186; KL=0.00071; ff=0.00137 (un=0.00119); bit_rem=True |
| `falcon3-3b` (L9) | `SHUFFLED_R_AUTO` | **ABSTAIN** | — | — | — | — | — | auto-abstained (0 writes) |
| `falcon3-3b` (L9) | `SHUFFLED_R_FORCED` | FORCED_FROM_ABSTAIN (d_ff=9220) | 4.2% / 0.0% (elig 4.2%/0.0%) | 16.2% / 6.2% (elig 4.7%/0.6%) [11.5, 20.8] | 32.3% / 17.7% (elig 3.0%/3.0%) | 100.0% / 85.4% | FF 0.0%/0.0%; NT-ret 100.0%/35.3% | broken=3/186; KL=0.00056; ff=0.00172 (un=0.01448); bit_rem=True |
| `falcon3-3b` (L9) | `CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | COMPILED (d_ff=9220) | 0.0% / 0.0% (elig 0.0%/0.0%) | 12.0% / 6.2% (elig 0.6%/0.6%) [8.3, 15.6] | 29.2% / 15.6% (elig 0.0%/0.0%) | 100.0% / 85.4% | FF 0.0%/0.0%; NT-ret 100.0%/35.3% | broken=23/186; KL=0.00122; ff=0.00129 (un=0.01071); bit_rem=True |
| `falcon3-3b` (L9) | `NATIVE_B_ORACLE` | COMPILED (d_ff=9220) | 100.0% / 41.7% (elig 100.0%/41.7%) | 87.5% / 42.2% (elig 85.8%/39.1%) [83.9, 91.1] | 80.2% / 60.4% (elig 73.1%/59.7%) | 100.0% / 39.6% | FF 79.2%/41.7%; NT-ret 29.4%/23.5% | broken=48/186; KL=0.00113; ff=0.00137 (un=0.01468); bit_rem=True |
| `qwen2.5-1.5b` (L13) | `BASE` | elig: dir=48/48, rip=163/192, hop=68/96 | 0.0% / 0.0% | 15.1% / 5.7% [10.9, 19.8] | 29.2% / 2.1% | 100.0% / 81.2% | FF 0.0%/0.0%; NT-ret 100.0%/47.1% | base=163/196; PPL=28.251 |
| `qwen2.5-1.5b` (L13) | `IN_CONTEXT_PREMISE` | premise baseline | 2.1% / 2.1% (elig 2.1%/2.1%) | 60.4% / 37.0% (elig 55.2%/35.0%) [55.7, 65.1] | 45.8% / 41.7% (elig 23.5%/23.5%) | 100.0% / 58.3% | n/a (unprefixed) | unedited B |
| `qwen2.5-1.5b` (L13) | `CHANNEL_PUSHFORWARD_BIASFREE` | COMPILED (d_ff=8964) | 54.2% / 33.3% (elig 54.2%/33.3%) | 52.1% / 14.1% (elig 43.6%/9.8%) [45.8, 58.3] | 45.8% / 18.8% (elig 23.5%/13.2%) | 100.0% / 85.4% | FF 33.3%/4.2%; NT-ret 70.6%/41.2% | broken=0/163; KL=0.00119; ff=0.00161 (un=0.00084); bit_rem=True |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_R_AUTO` | **ABSTAIN** | — | — | — | — | — | auto-abstained (0 writes) |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_R_FORCED` | FORCED_FROM_ABSTAIN (d_ff=8964) | 0.0% / 0.0% (elig 0.0%/0.0%) | 18.2% / 8.8% (elig 3.7%/3.7%) [14.1, 22.9] | 29.2% / 3.1% (elig 0.0%/0.0%) | 100.0% / 83.3% | FF 0.0%/0.0%; NT-ret 100.0%/47.1% | broken=1/163; KL=0.00046; ff=0.00102 (un=0.00649); bit_rem=True |
| `qwen2.5-1.5b` (L13) | `CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | COMPILED (d_ff=8964) | 0.0% / 0.0% (elig 0.0%/0.0%) | 15.1% / 6.2% (elig 0.0%/0.6%) [10.9, 19.8] | 31.2% / 3.1% (elig 2.9%/1.5%) | 100.0% / 79.2% | FF 0.0%/0.0%; NT-ret 100.0%/47.1% | broken=41/163; KL=0.00059; ff=0.00042 (un=0.00084); bit_rem=True |
| `qwen2.5-1.5b` (L13) | `NATIVE_B_ORACLE` | COMPILED (d_ff=8964) | 100.0% / 27.1% (elig 100.0%/27.1%) | 89.1% / 17.7% (elig 89.6%/17.2%) [85.9, 92.7] | 50.0% / 15.6% (elig 29.4%/19.1%) | 100.0% / 79.2% | FF 83.3%/8.3%; NT-ret 23.5%/11.8% | broken=24/163; KL=0.00102; ff=0.00161 (un=0.00293); bit_rem=True |

#### Table 4B-iv — Cap-2 Per-Rule Breakdown & Clustered Evidence (`CHANNEL_PUSHFORWARD_BIASFREE`)
*Clustered statistics:* **8/8 compiled rule-organs** beat both BASE and forced shuffled-read on unseen 4-attribute ripple FC (exact one-sided sign test `p = 0.0039`). On `smollm2-1.7b`, `87:0` item-level ripple FC improvement across 48 species (4 attributes/species, 0 regressions) guarantees at least `22/48` species improved and 0 worsened (conservative species-level sign-test bound `p <= 2.38e-07`).

| Recipient | Rule (24 unseen species) | Direct FC / EM | 4-Attr Ripple FC [Sp. 95% CI] / EM | Ripple FC Controls | 2-Sent Hops FC / EM |
| --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` (L9) | `Bird->Fish` | 70.8% / 8.3% (BASE 0.0%) | 67.7% [60.4, 75.0] / 18.8% | BASE 11.5% ; Shuf 16.7% | 52.1% / 47.9% (BASE 0.0%) |
| `smollm2-1.7b` (L9) | `Mammal->Reptile` | 12.5% / 8.3% (BASE 0.0%) | 47.9% [35.4, 60.4] / 27.1% | BASE 13.5% ; Shuf 14.6% | 50.0% / 0.0% (BASE 50.0%) |
| `granite-3.1-2b` (L12) | `Bird->Fish` | 12.5% / 8.3% (BASE 0.0%) | 32.3% [26.0, 39.6] / 10.4% | BASE 18.8% ; Shuf 18.8% | 37.5% / 37.5% (BASE 8.3%) |
| `granite-3.1-2b` (L12) | `Mammal->Reptile` | 0.0% / 0.0% (BASE 0.0%) | 37.5% [24.0, 51.0] / 7.3% | BASE 12.5% ; Shuf 9.4% | 50.0% / 0.0% (BASE 50.0%) |
| `falcon3-3b` (L9) | `Bird->Fish` | 20.8% / 0.0% (BASE 0.0%) | 44.8% [36.5, 53.1] / 5.2% | BASE 6.2% ; Shuf 12.5% | 45.8% / 45.8% (BASE 16.7%) |
| `falcon3-3b` (L9) | `Mammal->Reptile` | 4.2% / 0.0% (BASE 0.0%) | 58.3% [51.0, 65.6] / 21.9% | BASE 17.7% ; Shuf 19.8% | 39.6% / 14.6% (BASE 43.8%) |
| `qwen2.5-1.5b` (L13) | `Bird->Fish` | 100.0% / 66.7% (BASE 0.0%) | 57.3% [52.1, 62.5] / 6.2% | BASE 10.4% ; Shuf 10.4% | 41.7% / 25.0% (BASE 8.3%) |
| `qwen2.5-1.5b` (L13) | `Mammal->Reptile` | 8.3% / 0.0% (BASE 0.0%) | 46.9% [35.4, 58.3] / 21.9% | BASE 19.8% ; Shuf 26.0% | 50.0% / 12.5% (BASE 50.0%) |

#### Table 4B-ii — Cap-2 Exact Two-Sided Item-Level McNemar p-Values
*Note: Item-level McNemar pools 4 attributes per species (`n=192` over 48 species) and 2 hop prompts per species (`n=96`); treat species-clustered CIs and sign-test bounds in Table 4B-iv as primary.*

| Recipient | Comparison | Direct (n10:n01, p) | 4-Attr Ripple All (n10:n01, p) | 4-Attr Ripple Elig FC | 2-Sent Hops All (n10:n01, p) |
| --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_BASE` | FC 20:0 (p=1.91e-06); EM 4:0 (p=0.1250) | FC 87:0 (p=1.29e-26); EM 32:0 (p=4.66e-10) | FC 87:0 (p=1.29e-26) | FC 25:0 (p=5.96e-08); EM 23:0 (p=2.38e-07) |
| `smollm2-1.7b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_SHUFFLED_R_FORCED` | FC 19:0 (p=3.81e-06); EM 4:0 (p=0.1250) | FC 81:0 (p=8.27e-25); EM 31:1 (p=1.54e-08) | FC 81:0 (p=8.27e-25) | FC 21:0 (p=9.54e-07); EM 19:0 (p=3.81e-06) |
| `smollm2-1.7b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | FC 20:0 (p=1.91e-06); EM 4:0 (p=0.1250) | FC 88:0 (p=6.46e-27); EM 34:0 (p=1.16e-10) | FC 86:0 (p=2.58e-26) | FC 25:0 (p=5.96e-08); EM 23:0 (p=2.38e-07) |
| `smollm2-1.7b` | `NATIVE_B_ORACLE_vs_BASE` | FC 48:0 (p=7.11e-15); EM 23:0 (p=2.38e-07) | FC 157:0 (p=1.09e-47); EM 67:0 (p=1.36e-20) | FC 157:0 (p=1.09e-47) | FC 45:0 (p=5.68e-14); EM 45:0 (p=5.68e-14) |
| `granite-3.1-2b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_BASE` | FC 3:0 (p=0.2500); EM 2:0 (p=0.5000) | FC 39:2 (p=7.84e-10); EM 12:3 (p=0.0352) | FC 39:0 (p=3.64e-12) | FC 14:0 (p=0.0001); EM 14:0 (p=0.0001) |
| `granite-3.1-2b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_SHUFFLED_R_FORCED` | FC 3:0 (p=0.2500); EM 2:0 (p=0.5000) | FC 40:0 (p=1.82e-12); EM 13:1 (p=0.0018) | FC 39:0 (p=3.64e-12) | FC 15:0 (p=6.10e-05); EM 15:0 (p=6.10e-05) |
| `granite-3.1-2b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | FC 3:0 (p=0.2500); EM 2:0 (p=0.5000) | FC 39:2 (p=7.84e-10); EM 12:3 (p=0.0352) | FC 39:0 (p=3.64e-12) | FC 14:0 (p=0.0001); EM 14:0 (p=0.0001) |
| `granite-3.1-2b` | `NATIVE_B_ORACLE_vs_BASE` | FC 33:0 (p=2.33e-10); EM 9:0 (p=0.0039) | FC 135:2 (p=1.09e-37); EM 42:3 (p=8.65e-10) | FC 135:0 (p=4.59e-41) | FC 24:3 (p=4.92e-05); EM 18:4 (p=0.0043) |
| `falcon3-3b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_BASE` | FC 6:0 (p=0.0312); EM 0:0 (p=1.0000) | FC 76:0 (p=2.65e-23); EM 14:0 (p=0.0001) | FC 76:0 (p=2.65e-23) | FC 14:2 (p=0.0042); EM 14:0 (p=0.0001) |
| `falcon3-3b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_SHUFFLED_R_FORCED` | FC 4:0 (p=0.1250); EM 0:0 (p=1.0000) | FC 68:0 (p=6.78e-21); EM 14:0 (p=0.0001) | FC 68:0 (p=6.78e-21) | FC 12:2 (p=0.0129); EM 12:0 (p=0.0005) |
| `falcon3-3b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | FC 6:0 (p=0.0312); EM 0:0 (p=1.0000) | FC 76:0 (p=2.65e-23); EM 14:0 (p=0.0001) | FC 75:0 (p=5.29e-23) | FC 15:2 (p=0.0023); EM 14:0 (p=0.0001) |
| `falcon3-3b` | `NATIVE_B_ORACLE_vs_BASE` | FC 48:0 (p=7.11e-15); EM 20:0 (p=1.91e-06) | FC 145:0 (p=4.48e-44); EM 69:0 (p=3.39e-21) | FC 145:0 (p=4.48e-44) | FC 49:1 (p=9.06e-14); EM 44:1 (p=2.61e-12) |
| `qwen2.5-1.5b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_BASE` | FC 26:0 (p=2.98e-08); EM 16:0 (p=3.05e-05) | FC 71:0 (p=8.47e-22); EM 17:1 (p=0.0001) | FC 71:0 (p=8.47e-22) | FC 16:0 (p=3.05e-05); EM 16:0 (p=3.05e-05) |
| `qwen2.5-1.5b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_SHUFFLED_R_FORCED` | FC 26:0 (p=2.98e-08); EM 16:0 (p=3.05e-05) | FC 66:1 (p=9.22e-19); EM 12:2 (p=0.0129) | FC 66:1 (p=9.22e-19) | FC 16:0 (p=3.05e-05); EM 15:0 (p=6.10e-05) |
| `qwen2.5-1.5b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | FC 26:0 (p=2.98e-08); EM 16:0 (p=3.05e-05) | FC 71:0 (p=8.47e-22); EM 16:1 (p=0.0003) | FC 71:0 (p=8.47e-22) | FC 14:0 (p=0.0001); EM 15:0 (p=6.10e-05) |
| `qwen2.5-1.5b` | `NATIVE_B_ORACLE_vs_BASE` | FC 48:0 (p=7.11e-15); EM 13:0 (p=0.0002) | FC 146:4 (p=2.92e-38); EM 28:5 (p=6.62e-05) | FC 146:0 (p=2.24e-44) | FC 20:0 (p=1.91e-06); EM 14:1 (p=0.0010) |

#### Table 4B-iii — Cross-Capability Co-Implantation & Bit-Exact Modular Removal on Solo `smollm2-1.7b` L9 (`Cap-1 Geo + Cap-2 Bio`: $d_{\text{ff}}$ `8192 → 8200 → 8196 → 8192`)
| State | $d_{\text{ff}}$ | Cap-1 Geography (EM) | Cap-2 BIO-RIPPLE (FC/EM) | 196-Fact Unrel Loc | KL / ΔPPL | Bit-Exact Modular Removal Certificate |
| --- | --- | --- | --- | --- | --- | --- |
| `JOINT_CAP1_PLUS_CAP2` | 8200 | dir=18.8%, rip3=18.8%, ds8=25.0% | dir=41.7%/8.3%, rip=54.7%/20.3%, hop=51.0%/24.0% | 15/175 (91.4%) | 0.01059 / +0.3187 | — |
| `SLICE_OUT_CAP1_KEEP_CAP2` | 8196 | dir=0.0%, rip3=0.0%, ds8=0.0% | dir=41.7%/8.3%, rip=57.8%/22.9%, hop=51.0%/24.0% | 0/175 (100.0%) | 0.00107 / +0.0324 | ==solo_cap2: True (maxΔ=0.0) |
| `SLICE_OUT_CAP2_KEEP_CAP1` | 8196 | dir=18.8%, rip3=18.8%, ds8=25.0% | dir=0.0%/0.0%, rip=12.0%/5.2%, hop=25.0%/0.0% | 15/175 (91.4%) | 0.00959 / +0.2883 | ==solo_cap1: True (maxΔ=0.0) |
| `SLICE_ALL_RESTORE_BASE` | 8192 | restored to base | restored to base | 0/175 (100.0%) | 0.00000 | cap1_eq=True, cap2_eq=True, maxΔ=0.0 |

