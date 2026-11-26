# PlatonicSwap — Master Programmatically Audited Results

> **Auto-generated** by `build_repo_tables.py` at `2026-11-26 12:12:12` directly from JSON artifacts in `/tmp/platos_key_lab/` (zero hand-typed numbers).

### Artifact Manifest (21/21 present)
| File | Status | Size | Modified | SHA-256 (12) |
| --- | --- | --- | --- | --- |
| `results_pilot0.json` | AUDITED | 220 KB | 2026-11-25 20:40:31 | `e854ee31fe38` |
| `results_pilot1.json` | AUDITED | 70 KB | 2026-11-25 20:53:27 | `1162e3e485e3` |
| `results_pilot1_qwen2.5-3b_to_granite-3.1-2b.json` | AUDITED | 70 KB | 2026-11-25 21:42:15 | `3be8102838da` |
| `results_pilot1_falcon3-3b_to_smollm2-1.7b.json` | AUDITED | 70 KB | 2026-11-25 22:13:14 | `e7aae20e5c7c` |
| `results_pilot1b_qwen2.5-3b_to_smollm2-1.7b.json` | AUDITED | 63 KB | 2026-11-25 22:38:57 | `615f3c8a0210` |
| `results_pilot1b_qwen2.5-3b_to_granite-3.1-2b.json` | AUDITED | 76 KB | 2026-11-25 23:29:48 | `17b726a30867` |
| `results_pilot1b_falcon3-3b_to_smollm2-1.7b.json` | AUDITED | 61 KB | 2026-11-25 23:39:28 | `4c801c269186` |
| `results_organs_p0.json` | AUDITED | 57 KB | 2026-11-26 08:26:08 | `76600948ea10` |
| `results_organs_multifamily.json` | AUDITED | 118 KB | 2026-11-26 09:19:41 | `87aa28879370` |
| `results_pilot2_fullrank.json` | AUDITED | 457 KB | 2026-11-25 21:19:21 | `3f00e6eddcd0` |
| `results_pilot2_gated.json` | AUDITED | 16 KB | 2026-11-25 21:43:34 | `24ccc69e0cb4` |
| `results_pilot3.json` | AUDITED | 81 KB | 2026-11-25 22:16:39 | `dbdebe4af48e` |
| `results_pilot3_qwen2.5-1.5b.json` | AUDITED | 62 KB | 2026-11-25 22:18:28 | `21fc13a9de82` |
| `results_pilot3_granite-3.1-2b.json` | AUDITED | 62 KB | 2026-11-25 22:26:36 | `a565ae98228a` |
| `results_pilot3_falcon3-3b.json` | AUDITED | 62 KB | 2026-11-25 22:21:41 | `98bfcb22f407` |
| `results_graft_falsify.json` | AUDITED | 170 KB | 2026-11-25 21:56:53 | `112f55c05a77` |
| `results_graft_t3_gap.json` | AUDITED | 25 KB | 2026-11-25 22:27:15 | `edbd378112e0` |
| `results_graft_t3_gap_scaled.json` | AUDITED | 80 KB | 2026-11-26 09:01:09 | `b074324cd458` |
| `results_graft_t3_gap_scaled_granite.json` | AUDITED | 66 KB | 2026-11-26 09:01:50 | `d442da0713e3` |
| `results_cap1_channel_pushforward.json` | AUDITED | 128 KB | 2026-11-26 11:54:02 | `bcf2dc70997e` |
| `results_bio_ripple_sealed.json` | AUDITED | 580 KB | 2026-11-26 12:11:46 | `cd6c3c02a088` |

## Pillar 1: Zero-QA Geometry, $N \gg D$ Scaling, $O(d)$ vs. $GL(d)$, and Breaking 85%+
### Table 1A — Pilot 0 Error Taxonomy, Data-Starvation Curve & Causal-Fisher Subspace (`results_pilot0.json`)
| Pair | Overall Acc | Answer-Disjoint | Base Subj Survived (of fails) | Right Type Wrong Ent | Off-Manifold Other | Garbled/Empty |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | 110/149 (73.8%) | 45/66 (68.2%) | 17/39 (43.6%) | 4/39 (10.3%) | 18/39 (46.2%) | 0 |
| `falcon3-3b->smollm2-1.7b` | 124/177 (70.1%) | 44/69 (63.8%) | 24/53 (45.3%) | 5/53 (9.4%) | 24/53 (45.3%) | 0 |
| `qwen2.5-3b->granite-3.1-2b` | 59/153 (38.6%) | 19/67 (28.4%) | 83/94 (88.3%) | 5/94 (5.3%) | 6/94 (6.4%) | 0 |

| Pair | ProcGlobal Learning Curve (All-Train Pool) | Pilot 0d Causal-Fisher Diagnostic |
| --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | N=147: 53.2% (AD 38.4%, gap +39.0pp) -> N=293: 67.6% (AD 54.0%, gap +19.9pp) -> N=587: 75.8% (AD 68.2%, gap +4.2pp) | AUC `|g·δ|`=0.8196 vs L2=0.6308; Fisher-in-PCA64=15.9% (PR=48.2) |
| `falcon3-3b->smollm2-1.7b` | N=147: 49.1% (AD 38.2%, gap +38.4pp) -> N=293: 61.2% (AD 50.2%, gap +21.9pp) -> N=587: 69.5% (AD 62.3%, gap +9.7pp) | — |
| `qwen2.5-3b->granite-3.1-2b` | N=147: 22.4% (AD 10.9%, gap +22.6pp) -> N=293: 34.2% (AD 20.9%, gap +15.8pp) -> N=587: 41.2% (AD 25.4%, gap +5.5pp) | AUC `|g·δ|`=0.6659 vs L2=0.4327; Fisher-in-PCA64=5.0% (PR=2.2) |

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
| Pair | Rank | PCA-Procrustes | PLS | Closed-Form Fisher-PLS |
| --- | --- | --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | k=32 | 7.4% (net +4.0pp) | 10.1% (net +6.0pp) | 11.4% (net +8.7pp) |
| `qwen2.5-3b->smollm2-1.7b` | k=64 | 16.8% (net +12.8pp) | 22.8% (net +19.5pp) | 30.2% (net +28.9pp) |
| `qwen2.5-3b->granite-3.1-2b` | k=32 | 2.6% (net +2.0pp) | 4.6% (net +4.6pp) | 8.5% (net +7.8pp) |
| `qwen2.5-3b->granite-3.1-2b` | k=64 | 7.8% (net +7.8pp) | 13.1% (net +13.1pp) | 15.7% (net +14.4pp) |
| `falcon3-3b->smollm2-1.7b` | k=32 | 6.2% (net +2.8pp) | 11.3% (net +7.3pp) | 14.7% (net +11.3pp) |
| `falcon3-3b->smollm2-1.7b` | k=64 | 14.7% (net +10.7pp) | 20.3% (net +19.8pp) | 25.4% (net +23.7pp) |

### Table 1C — Pilot 1B (Breaking 85%+): STRICT Separation of Natural Unmasked vs. Attention-Knockout Regimes
#### 1C-i. Natural Unmasked Regime (`mask_mode == none`)
| Pair | Config (Unmasked) | CF-Flip 1x (95% CI) | CF-Flip 5x Powered (95% CI) | AD / LORO-5 | Shuf 1x / 5x | Native Ceiling | Transport Eff (1x / 5x) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `proc_ls_scale` (1L, no mask) | 75.2% [67.2, 82.3] (n=149) | — | 74.2% / 68.5% | 0.0% | 87.9% | 85.5% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `proc_var_matched` (1L, no mask) | 76.5% [69.4, 83.0] (n=149) | — | 72.7% / 71.9% | 0.0% | 87.9% | 87.0% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `ridge_polar_I` (1L, no mask) | 75.8% [68.3, 82.6] (n=149) | — | 74.2% / 66.3% | 0.0% | 87.9% | 86.3% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `ridge_sigma_rescaled` (1L, no mask) | 69.8% [61.7, 77.5] (n=149) | — | 56.1% / 65.2% | 0.0% | 87.9% | 79.4% |
| `qwen2.5-3b->smollm2-1.7b` | FreeGeom `F_spherical_procrustes` (1L, no mask) | 76.5% [68.9, 83.1] (n=149) | — | 69.7% / 73.0% | 0.0% | 87.9% | 87.0% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A0_single_unlab10k` (no mask) | 76.5% [69.4, 83.0] (n=149) | 79.0% [74.1, 83.9] (n=725) | 72.7% / 71.9% | 0.0% / 0.4% | 87.9% | 87.0% / 88.3% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A1_band_main_none` (no mask) | 75.2% [67.6, 82.3] (n=149) | — | 72.7% / 76.4% | 0.0% | 87.9% | 85.5% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_raw_single_A0` (no mask) | 81.9% [75.3, 87.9] (n=149) | — | 75.8% / 77.5% | 0.0% | 87.9% | 93.1% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_m3_avg_single_A0` (no mask) | 81.9% [75.2, 87.8] (n=149) | — | 77.3% / 77.5% | 0.0% | 87.9% | 93.1% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_bet_01wit_single_A0` (no mask) | 81.2% [74.4, 87.3] (n=149) | — | 75.8% / 76.4% | 0.0% | 87.9% | 92.4% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_cross_ctx_single_A0` (no mask) | 83.2% [76.6, 89.0] (n=149) | — | 80.3% / 75.3% | 0.0% | 87.9% | 94.7% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `proc_ls_scale` (1L, no mask) | 43.1% [35.6, 51.4] (n=153) | — | 34.3% / 40.7% | 0.0% | — | 80.5% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `proc_var_matched` (1L, no mask) | 43.8% [35.9, 52.3] (n=153) | — | 32.8% / 39.6% | 0.0% | — | 81.7% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `ridge_polar_I` (1L, no mask) | 50.3% [42.2, 58.4] (n=153) | — | 40.3% / 51.6% | 1.3% | — | 93.9% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `ridge_sigma_rescaled` (1L, no mask) | 35.9% [28.2, 44.2] (n=153) | — | 28.4% / 34.1% | 0.0% | — | 67.1% |
| `qwen2.5-3b->granite-3.1-2b` | FreeGeom `F_spherical_procrustes` (1L, no mask) | 39.9% [32.7, 47.9] (n=153) | — | 29.8% / 41.8% | 0.0% | — | 74.4% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A0_single_unlab10k` (no mask) | 43.8% [35.9, 52.3] (n=153) | 47.5% [40.5, 54.0] (n=750) | 32.8% / 39.6% | 0.0% / 0.3% | — | 81.7% / 89.2% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A1_band_main_none` (no mask) | 35.9% [28.5, 43.8] (n=153) | — | 34.3% / 39.6% | 0.0% | 52.9% | — |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_raw_single_A0` (no mask) | 43.1% [35.3, 51.3] (n=153) | — | 35.8% / 46.2% | 0.0% | — | 80.5% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_m3_avg_single_A0` (no mask) | 42.5% [34.2, 50.7] (n=153) | — | 32.8% / 44.0% | 0.0% | — | 79.3% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_bet_01wit_single_A0` (no mask) | 42.5% [34.4, 50.7] (n=153) | — | 34.3% / 44.0% | 0.0% | — | 79.3% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_cross_ctx_single_A0` (no mask) | 43.1% [35.1, 51.3] (n=153) | — | 32.8% / 42.9% | 0.0% | — | 80.5% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `proc_ls_scale` (1L, no mask) | 71.2% [64.0, 78.0] (n=177) | — | 73.9% / 82.3% | 0.6% | 83.0% | 85.7% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `proc_var_matched` (1L, no mask) | 70.6% [63.3, 77.1] (n=177) | — | 72.5% / 80.2% | 0.6% | 83.0% | 85.0% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `ridge_polar_I` (1L, no mask) | 68.9% [61.4, 75.8] (n=177) | — | 66.7% / 74.0% | 0.6% | 83.0% | 83.0% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `ridge_sigma_rescaled` (1L, no mask) | 71.2% [64.5, 77.7] (n=177) | — | 69.6% / 80.2% | 0.6% | 83.0% | 85.7% |
| `falcon3-3b->smollm2-1.7b` | FreeGeom `F_spherical_procrustes` (1L, no mask) | 69.5% [62.2, 76.3] (n=177) | — | 69.6% / 81.2% | 0.6% | 83.0% | 83.7% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A0_single_unlab10k` (no mask) | 70.6% [63.3, 77.1] (n=177) | 71.7% [66.8, 76.6] (n=880) | 72.5% / 80.2% | 0.6% / 0.4% | 83.0% | 85.0% / 84.0% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A1_band_main_none` (no mask) | 67.8% [60.2, 75.0] (n=177) | — | 72.5% / 81.2% | 0.6% | 82.5% | 82.2% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_raw_single_A0` (no mask) | 74.6% [68.0, 80.9] (n=177) | — | 73.9% / 82.3% | 0.6% | 83.0% | 89.8% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_m3_avg_single_A0` (no mask) | 71.2% [64.2, 77.7] (n=177) | — | 66.7% / 79.2% | 0.6% | 83.0% | 85.7% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_bet_01wit_single_A0` (no mask) | 72.3% [65.5, 78.9] (n=177) | — | 68.1% / 79.2% | 0.6% | 83.0% | 87.1% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_cross_ctx_single_A0` (no mask) | 72.3% [65.5, 78.8] (n=177) | — | 68.1% / 82.3% | 0.6% | 83.0% | 87.1% |

#### 1C-ii. Earlier-Subject / Full-Subject Attention-Knockout Regime (`mask_mode != none`)
| Pair | Config (Attention Knockout) | CF-Flip 1x (95% CI) | CF-Flip 5x Powered (95% CI) | AD / LORO-5 | Shuf 1x / 5x | Native Ceiling | Transport Eff (1x / 5x) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A2_single_full_ko` (mask=full_knockout) | 85.9% [79.8, 91.2] (n=149) | 87.3% [83.9, 90.4] (n=725) | 81.8% / 82.0% | 0.0% / 0.5% | 96.0% | 92.1% / 91.9% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A3_band_main_earlier_ko` (mask=earlier_subj) | 85.2% [79.0, 90.7] (n=149) | 86.2% [82.6, 89.5] (n=725) | 81.8% / 83.2% | 0.0% / 0.5% | 96.0% | 90.1% / 90.5% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A3_band_main_full_ko` (mask=full_knockout) | 85.2% [78.6, 91.2] (n=149) | 85.4% [82.2, 88.4] (n=725) | 72.7% / 82.0% | 0.0% / 0.5% | 96.0% | 88.8% / 89.8% |
| `qwen2.5-3b->smollm2-1.7b` | LeverA `A3_band_main_nonsubj_ctrl` (mask=nonsubj_ctrl) | 74.5% [67.1, 81.5] (n=149) | — | 74.2% / 75.3% | 0.0% | 96.0% | 85.4% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_raw_band_A3_full_ko` (mask=full_knockout) | 89.3% [83.7, 94.0] (n=149) | 87.3% [84.7, 89.8] (n=725) | 80.3% / 86.5% | 0.0% / 0.5% | 96.0% | 93.0% / 91.9% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_m3_avg_band_A3_full_ko` (mask=full_knockout) | 86.6% [80.5, 92.0] (n=149) | 87.2% [84.5, 89.7] (n=725) | 77.3% / 84.3% | 0.0% / 0.5% | 96.0% | 90.2% / 91.7% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_bet_01wit_band_A3_full_ko` (mask=full_knockout) | 88.6% [82.9, 93.7] (n=149) | 87.2% [84.5, 89.7] (n=725) | 80.3% / 84.3% | 0.0% / 0.5% | 96.0% | 92.3% / 91.7% |
| `qwen2.5-3b->smollm2-1.7b` | LeverB `B_real_cross_ctx_band_A3_full_ko` (mask=full_knockout) | 88.6% [82.9, 93.5] (n=149) | 88.8% [86.3, 91.1] (n=725) | 81.8% / 83.2% | 0.0% / 0.5% | 96.0% | 92.3% / 93.5% |
| `qwen2.5-3b->smollm2-1.7b` | LeverC `C_woodbury_k16_full_ko` (mask=full_knockout) | 89.3% [83.7, 93.8] (n=149) | 87.9% [85.1, 90.5] (n=725) | 80.3% / 80.9% | 0.0% / 0.5% | 96.0% | 93.0% / 92.5% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A2_single_full_ko` (mask=full_knockout) | 91.5% [86.6, 95.7] (n=153) | 92.0% [89.5, 94.3] (n=750) | 88.1% / 81.3% | 0.0% / 0.7% | 91.5% | 97.9% / 99.4% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A3_band_main_earlier_ko` (mask=earlier_subj) | 88.2% [82.8, 93.3] (n=153) | 87.5% [84.1, 90.5] (n=750) | 86.6% / 81.3% | 0.0% / 0.4% | 91.5% | — |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A3_band_main_full_ko` (mask=full_knockout) | 89.5% [84.0, 94.3] (n=153) | 89.5% [86.4, 92.2] (n=750) | 83.6% / 81.3% | 0.0% / 0.4% | 91.5% | 107.0% / 107.2% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A3_band_main_nonsubj_ctrl` (mask=nonsubj_ctrl) | 35.3% [28.2, 43.0] (n=153) | — | 32.8% / 39.6% | 0.0% | 91.5% | 84.4% |
| `qwen2.5-3b->granite-3.1-2b` | LeverA `A2_l16_full_ko` (mask=full_knockout) | 83.7% [77.1, 89.5] (n=153) | 83.9% [80.1, 87.4] (n=750) | 73.1% / 71.4% | 0.0% / 0.3% | 91.5% | 107.6% / 108.3% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_raw_band_A3_full_ko` (mask=full_knockout) | 88.2% [82.8, 93.2] (n=153) | 89.5% [86.7, 92.1] (n=750) | 80.6% / 84.6% | 0.0% / 0.8% | 91.5% | 105.5% / 107.2% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_m3_avg_band_A3_full_ko` (mask=full_knockout) | 87.6% [82.1, 92.7] (n=153) | 89.5% [86.7, 92.2] (n=750) | 82.1% / 84.6% | 0.0% / 0.8% | 91.5% | 104.7% / 107.2% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_bet_01wit_band_A3_full_ko` (mask=full_knockout) | 88.9% [83.5, 93.8] (n=153) | 89.9% [87.2, 92.4] (n=750) | 83.6% / 84.6% | 0.0% / 0.8% | 91.5% | 106.2% / 107.7% |
| `qwen2.5-3b->granite-3.1-2b` | LeverB `B_real_cross_ctx_band_A3_full_ko` (mask=full_knockout) | 89.5% [84.3, 94.4] (n=153) | 89.6% [86.9, 92.2] (n=750) | 85.1% / 84.6% | 0.0% / 0.8% | 91.5% | 107.0% / 107.3% |
| `qwen2.5-3b->granite-3.1-2b` | LeverC `C_woodbury_k16_full_ko` (mask=full_knockout) | 88.9% [83.0, 93.9] (n=153) | 88.4% [85.2, 91.4] (n=750) | 80.6% / 84.6% | 0.0% / 0.5% | 91.5% | 106.2% / 105.9% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A2_single_full_ko` (mask=full_knockout) | 81.4% [75.3, 86.9] (n=177) | 82.8% [79.6, 85.7] (n=880) | 78.3% / 84.4% | 0.6% / 0.8% | 94.9% | 88.3% / 87.9% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A3_band_main_earlier_ko` (mask=earlier_subj) | 79.1% [72.6, 84.9] (n=177) | 80.5% [76.8, 83.9] (n=880) | 81.2% / 83.3% | 0.6% / 0.7% | 94.9% | 84.9% / 85.5% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A3_band_main_full_ko` (mask=full_knockout) | 81.4% [75.2, 87.1] (n=177) | 82.8% [79.6, 85.7] (n=880) | 81.2% / 83.3% | 0.6% / 0.8% | 94.9% | 85.7% / 88.0% |
| `falcon3-3b->smollm2-1.7b` | LeverA `A3_band_main_nonsubj_ctrl` (mask=nonsubj_ctrl) | 67.2% [59.6, 74.6] (n=177) | — | 72.5% / 80.2% | 0.6% | 94.9% | 81.5% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_raw_band_A3_full_ko` (mask=full_knockout) | 85.3% [79.6, 90.4] (n=177) | 85.8% [83.0, 88.3] (n=880) | 84.1% / 88.5% | 0.6% / 0.4% | 94.9% | 89.9% / 91.2% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_m3_avg_band_A3_full_ko` (mask=full_knockout) | 87.6% [82.2, 92.6] (n=177) | 86.9% [84.4, 89.2] (n=880) | 85.5% / 90.6% | 0.6% / 0.4% | 94.9% | 92.3% / 92.4% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_bet_01wit_band_A3_full_ko` (mask=full_knockout) | 87.0% [81.6, 92.0] (n=177) | 86.8% [84.3, 89.1] (n=880) | 85.5% / 89.6% | 0.6% / 0.4% | 94.9% | 91.7% / 92.3% |
| `falcon3-3b->smollm2-1.7b` | LeverB `B_real_cross_ctx_band_A3_full_ko` (mask=full_knockout) | 87.0% [81.5, 91.9] (n=177) | 87.2% [84.9, 89.3] (n=880) | 87.0% / 90.6% | 0.6% / 0.7% | 94.9% | 91.7% / 92.6% |

## Pillar 2: Platonic Organs — Native SwiGLU Conjugation & Write-Once Implant-Anywhere
### Table 2A — P-ORGAN-0 Stage 0: Channel Census, Regime 1 (2,000-Neuron Conjugation) & Gate G0 (`results_organs_p0.json`)
| Diagnostic | Metric | Value |
| --- | --- | --- |
| Channel Census (`qwen2.5-3b L17 -> smollm2-1.7b L9`) | Held-out Read $R^2$ / Write $R^2$ (filtered vs raw) | 0.6207 / 0.6467 (raw: -3.0343) |
| Massive-Activation Outlier Filter | Dropped tokens / max norm before->after | 8/10559 | 3302.9 -> 109.1 |
| Split-Half Socket Stability | Top-256 PC mean / median cos (`>=0.80`) | 0.8427 / 0.8298 (200/256) |
| Constant Pin $c_B^\top x_B \approx 1$ | $\sigma(c_A^\top x_A)$ / $\sigma(c_B^\top x_B)$ / pin err | 0.01115 / 0.00698 / 2.02e-15 |
| Gate G0 (1000-distractor Read Retrieval) | Known 24 LM Top-1 / Gap-17 B-unknown Top-1 (ratio) | 95.8% / 88.2% (ratio 0.921) |

*Regime 1: 2,000-Neuron Stock MLP Conjugation (`n_eval_tokens = 4,104`):*
| Conjugation Arm | Median $r(g)$ | Frac $r(g)\ge 0.70$ | Median $r(a)$ | Median $|\Delta|/\sigma$ | BOS Max $|y|$ | Channel Model $R^2$ |
| --- | --- | --- | --- | --- | --- | --- |
| `V0_naive_biasfree` | 0.8192 | 99.9% | 0.5072 | 0.8314 | 0.005354 | 0.9756 |
| `V1_affine_bosnull` | 0.8192 | 99.9% | 0.6889 | 0.0459 | 0.000000 | 0.9756 |
| `V1_gamma_cpin_biasfree` | 0.7743 | 98.7% | 0.6116 | 0.0425 | 0.000000 | 0.8690 |
| `V1_ridge_affine_bosnull` | 0.9138 | 100.0% | 0.8484 | 0.0074 | 0.000000 | 0.8170 |

### Table 2B — Write-Once, Implant-Anywhere Across 4 Architectures (`results_organs_multifamily.json` Stage A + Foils)
> `results_organs_multifamily.json` present; inspect custom keys.


#### Pillar 2 Comparison Foils — Output-Layer (L23) MEMIT vs. Non-Destructive Gated Sidecar (`qwen2.5-3b -> smollm2-1.7b`)
| Method | Deployment Regime | Private Table `f` / Deranged `f'` (96 unseen paraphrases) | 196-Fact Locality Drop | Mid-Layer Multi-Hop Ripple |
| --- | --- | --- | --- | --- |
| Live Dual-Adjoint Crossing (`L35->L23`) | Donor alive | f=93.8%/64.6% EM | f'=82.3%/66.7% EM (on old f: 5.2%) | n/a (live) | 0% (post-attention L23) |
| Platonic MEMIT (`W_down` overwrite, `L23`) | Donor unloaded, 0 SGD | f=47.9%/27.1% EM | f'=51.0%/47.9% EM | **+24.49 pp** (collateral damage!) | 0% (post-attention L23) |
| Non-Destructive Gated Sidecar (`L23`, frozen `MLP_orig`) | Donor unloaded, 0 SGD | f=84.4%/62.5% EM | f'=70.8%/60.4% EM (on old f: 5.2%) | **+0.00 pp** (FPR=0.00%, 0/1876 toks) | 0% (output-only foil!) |

## Pillar 3: Multi-Donor Composition, Selective Exact Unlearning & Scaled Knowledge-Gap Transplant
### Table 3A — Native Mid-Layer Multi-Donor Composition (`Qwen2.5-3B` + `Falcon3-3B` -> Solo `SmolLM2-1.7B`) & Selective Unlearning
> **[PENDING]** `results_organs_multifamily.json` Stage B not yet generated.

### Table 3B — Output-Layer 3-Family Split-Donor Composition & Compile-Once Deploy-to-4-Recipients (`results_pilot3*.json`)
| Recipient | Condition (Both Donors Unloaded) | Hop-1 `f` (8w / EM) | Hop-2 `g` (8w / EM) | 2-Step 8-Way (alt) | 2-Step Greedy EM (alt) | 196-Fact Loc Drop | Test FPR (tok / pr) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` | MEMIT `f + g` (`W_down`) | 40.6% / 6.2% | 33.3% / 0.0% | 21.9% (alt 12.5%) | 0.0% | +45.92pp | n/a |
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

*Gate 3 (`lB=9`):* non-BOS $\delta_{99}=0.204$, **with-BOS $\delta_{99}=1.553$** (kills naive zero-bias SwiGLU without explicit BOS null!).

### Table 3D — Pillar 3c: Fable T3 Long-Tail Knowledge-Gap Transplant (`results_graft_t3_gap*.json`)

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
*Screened `265` obscure entities across `9` countries -> `n_gap=21` genuine `A-right, B-wrong` (`strict_blindspot=8`).*
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
*Screened `265` obscure entities across `6` countries -> `n_gap=8` genuine `A-right, B-wrong` (`strict_blindspot=5`).*
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

## Pillar 4: Zero-Magic-Constant Channel-Pushforward Engine (Cap-1 Geography & Strictly Sealed Cap-2 `BIO-RIPPLE`)
### Table 4A — Cap-1 Zero-Magic-Constant Multi-Family Channel-Pushforward (`results_cap1_channel_pushforward.json`)
*Frozen Engine SHA-256:* `22ef57c9d196e12f` | *Frozen Tolerances:* `alpha=0.002, alpha_neg=0.03, delta=0.02` | *Targets:* `smollm2, granite, falcon3, qwen1.5b`.

| Recipient | Arm | Status | $d_{\text{ff}}$ | Direct EM | Unseen 3-Attr Ripple | DS-8 | Prot Ret. | HardNeg FF | 196-Fact Loc (Broken) | KL / ΔPPL | Conf. ff | Bit-UL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` (L9) | `PUSHFWD_DIGITAL_BIASFREE` | COMPILED | 8196 | 18.8% (p=0.2500) | 18.8% (p=0.0039) | 25.0% | 100.0% | 18.8% | 15/175 (91.4%) | 0.00959 / +0.2883 | 0.00136 | True |
| `smollm2-1.7b` (L9) | `PUSHFWD_DIGITAL_AFFINE` | COMPILED | 8196 | 18.8% (p=0.2500) | 14.6% (p=0.0156) | 25.0% | 100.0% | 12.5% | 9/175 (94.9%) | 0.00394 / +0.1248 | 0.00136 | True |
| `smollm2-1.7b` (L9) | `SHUFFLED_READ_AUTO` | **ABSTAIN** | — | — | abstained (France->Japan: r=0.30 a=0.18, Italy->Brazil: r=0.20 a=0.13) | — | — | — | — | — | — | — |
| `smollm2-1.7b` (L9) | `SHUFFLED_READ_FORCED` | FORCED_FROM_ABSTAIN | 8196 | 0.0% | 0.0% | 0.0% | 93.8% | 12.5% | 7/175 (96.0%) | 0.00514 / +0.1082 | 0.00060 | True |
| `smollm2-1.7b` (L9) | `SHUFFLED_WRITE` | COMPILED | 8196 | 0.0% (p=1.0000) | 0.0% (p=1.0000) | 0.0% | 100.0% | 12.5% | 6/175 (96.6%) | 0.00581 / +0.1205 | 0.00136 | True |
| `smollm2-1.7b` (L9) | `NATIVE_B_ORACLE` | COMPILED | 8196 | 62.5% (p=0.0020) | 41.7% (p=1.91e-06) | 87.5% | 81.2% | 100.0% | 133/175 (24.0%) | 0.00915 / +0.1629 | 0.00068 | True |
| `granite-3.1-2b` (L12) | `PUSHFWD_DIGITAL_BIASFREE` | COMPILED | 8196 | 25.0% (p=0.1250) | 22.9% (p=0.0010) | 25.0% | 87.5% | 18.8% | 22/183 (88.0%) | 0.00159 / +0.0267 | 0.00000 | True |
| `granite-3.1-2b` (L12) | `PUSHFWD_DIGITAL_AFFINE` | COMPILED | 8196 | 25.0% (p=0.1250) | 14.6% (p=0.0156) | 25.0% | 81.2% | 18.8% | 19/183 (89.6%) | 0.00117 / +0.0204 | 0.00000 | True |
| `granite-3.1-2b` (L12) | `SHUFFLED_READ_AUTO` | **ABSTAIN** | — | — | abstained (France->Japan: r=0.27 a=0.17, Italy->Brazil: r=0.23 a=0.15) | — | — | — | — | — | — | — |
| `granite-3.1-2b` (L12) | `SHUFFLED_READ_FORCED` | FORCED_FROM_ABSTAIN | 8196 | 0.0% | 0.0% | 0.0% | 56.2% | 12.5% | 13/183 (92.9%) | 0.00051 / +0.0103 | 0.00068 | True |
| `granite-3.1-2b` (L12) | `SHUFFLED_WRITE` | COMPILED | 8196 | 0.0% (p=1.0000) | 0.0% (p=1.0000) | 0.0% | 56.2% | 12.5% | 2/183 (98.9%) | 0.00106 / +0.0161 | 0.00000 | True |
| `falcon3-3b` (L9) | `PUSHFWD_DIGITAL_BIASFREE` | COMPILED | 9220 | 37.5% (p=0.1250) | 6.2% (p=0.2500) | 50.0% | 81.2% | 18.8% | 17/177 (90.4%) | 0.00122 / +0.0532 | 0.00129 | True |
| `falcon3-3b` (L9) | `PUSHFWD_DIGITAL_AFFINE` | COMPILED | 9220 | 25.0% (p=0.3750) | 4.2% (p=0.5000) | 37.5% | 81.2% | 18.8% | 12/177 (93.2%) | 0.00112 / +0.0490 | 0.00120 | True |
| `falcon3-3b` (L9) | `SHUFFLED_READ_AUTO` | **ABSTAIN** | — | — | abstained (France->Japan: r=0.23 a=0.11, Italy->Brazil: r=0.21 a=0.10) | — | — | — | — | — | — | — |
| `falcon3-3b` (L9) | `SHUFFLED_READ_FORCED` | FORCED_FROM_ABSTAIN | 9220 | 6.2% | 0.0% | 0.0% | 87.5% | 18.8% | 16/177 (91.0%) | 0.00061 / +0.0043 | 0.00069 | True |
| `falcon3-3b` (L9) | `SHUFFLED_WRITE` | COMPILED | 9220 | 0.0% (p=1.0000) | 0.0% (p=1.0000) | 0.0% | 87.5% | 12.5% | 2/177 (98.9%) | 0.00048 / +0.0032 | 0.00129 | True |
| `qwen2.5-1.5b` (L13) | `PUSHFWD_DIGITAL_BIASFREE` | COMPILED | 8964 | 50.0% (p=0.0391) | 16.7% (p=0.0391) | 50.0% | 100.0% | 37.5% | 34/155 (78.1%) | 0.00059 / +0.0351 | 0.00042 | True |
| `qwen2.5-1.5b` (L13) | `PUSHFWD_DIGITAL_AFFINE` | COMPILED | 8964 | 50.0% (p=0.0391) | 16.7% (p=0.0391) | 37.5% | 100.0% | 37.5% | 30/155 (80.7%) | 0.00050 / +0.0266 | 0.00034 | True |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_READ_AUTO` | **ABSTAIN** | — | — | abstained (France->Japan: r=0.25 a=0.17, Italy->Brazil: r=0.23 a=0.16) | — | — | — | — | — | — | — |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_READ_FORCED` | FORCED_FROM_ABSTAIN | 8964 | 6.2% | 2.1% | 0.0% | 100.0% | 12.5% | 5/155 (96.8%) | 0.00147 / +0.0029 | 0.00051 | True |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_WRITE` | COMPILED | 8964 | 0.0% (p=1.0000) | 0.0% (p=1.0000) | 0.0% | 100.0% | 12.5% | 3/155 (98.1%) | 0.00034 / -0.0067 | 0.00042 | True |

#### Table 4A-ii — $2 \times 2$ Calibration Ablation (Universal vs. Prose Read $\times$ Concept vs. Prose Write, Solo `smollm2-1.7b` L9)
| 2×2 Cell (`smollm2-1.7b` L9) | Direct EM | Unseen 3-Attr Ripple | DS-8 | Prot Ret. | HardNeg FF | 196-Fact Loc | KL / ΔPPL | Conf. ff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Universal Read × Concept Write (Primary) | 18.8% | 18.8% | 25.0% | 100.0% | 18.8% | 15/175 (91.4%) | 0.00959 / +0.2883 | 0.00136 |
| Universal Read × Prose Write | 0.0% | 8.3% | 12.5% | 100.0% | 12.5% | 17/175 (90.3%) | 0.00879 / +0.1846 | 0.00136 |
| Prose Read × Concept Write | 56.2% | 50.0% | 37.5% | 93.8% | 12.5% | 35/175 (80.0%) | 0.01402 / +0.3578 | 0.00097 |
| Prose Read × Prose Write | 6.2% | 18.8% | 12.5% | 93.8% | 12.5% | 28/175 (84.0%) | 0.01507 / +0.4078 | 0.00097 |

#### Table 4A-iii — Cap-1 Stage B Multi-Donor Composition (`Qwen2.5-3B` Fr→Jp + `Falcon3-3B` It→Br → Solo `smollm2-1.7b` L9)
| State | $d_{\text{ff}}$ | France→Japan (Qwen Donor) | Italy→Brazil (Falcon Donor) | 196-Fact Loc | KL / ΔPPL | Bit-Exact Unlearn |
| --- | --- | --- | --- | --- | --- | --- |
| `JOINT_QWEN_PLUS_FALCON` | 8196 | dir=37.5%, rip3=37.5%, ds=66.7% | dir=0.0%, rip3=0.0%, ds=0.0% | 13/175 (92.6%) | 0.00170 / +0.0122 | — |
| `SELECTIVE_UNLEARN_QWEN` | 8194 | dir=0.0%, rip3=0.0%, ds=0.0% | dir=0.0%, rip3=0.0%, ds=0.0% | 3/175 (98.3%) | 0.00061 / -0.0090 | — |
| `SELECTIVE_UNLEARN_FALCON` | 8194 | dir=37.5%, rip3=37.5%, ds=66.7% | dir=0.0%, rip3=0.0%, ds=0.0% | 11/175 (93.7%) | 0.00112 / +0.0222 | — |
| `UNLEARN_ALL` | 8192 | restored to base | restored to base | 100.0% | 0.00000 | sd=True, logits=True, maxΔ=0.0 |

### Table 4B — Strictly Sealed Capability 2 Benchmark: `BIO-RIPPLE` (`Bird->Fish`, `Mammal->Reptile`, `results_bio_ripple_sealed.json`)
*Frozen Engine SHA-256:* `22ef57c9d196e12f` | *Sealed Cap-2 SHA-256:* `d321fd0255c47cd3` | *Frozen Tolerances:* `alpha=0.002, alpha_neg=0.03, delta=0.02` | *Donor Frame-Only AUROC:* `{'Bird->Fish': 0.5, 'Mammal->Reptile': 0.5}`.

| Recipient | Arm | Status / $d_{\text{ff}}$ | Direct (FC / EM) | 4-Attr Ripple (FC / EM) | 2-Sent Hops (FC / EM) | Invariant Ret (FC/EM) | HardNeg (FF / Strict NT Ret) | Loc / KL / Conf. ff / Bit-UL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` (L9) | `BASE` | elig: dir=48/48, rip=168/192, hop=72/96 | 0.0% / 0.0% | 12.5% / 6.2% | 25.0% / 0.0% | 100.0% / 72.9% | ff=0.0%, nt=47.1% | 181/196 | PPL=26.100 |
| `smollm2-1.7b` (L9) | `IN_CONTEXT_CEILING` | prompt ceiling | 6.2% / 6.2% (elig 6.2%/6.2%) | 55.7% / 38.0% (elig 50.0%/33.9%) | 32.3% / 15.6% (elig 9.7%/16.7%) | 100.0% / 66.7% | ff=0.0%, nt=47.1% | unedited B |
| `smollm2-1.7b` (L9) | `CHANNEL_PUSHFORWARD_BIASFREE` | COMPILED (d_ff=8196) | 41.7% / 8.3% (elig 41.7%/8.3%) | 57.8% / 22.9% (elig 51.8%/19.1%) | 51.0% / 24.0% (elig 34.7%/31.9%) | 100.0% / 89.6% | ff=0.0%, nt=47.1% | loc=—/— KL=0.00107 ff=0.00128 ul=True |
| `smollm2-1.7b` (L9) | `SHUFFLED_R_AUTO` | **ABSTAIN** | — | — | — | — | — | auto-abstained (0 writes) |
| `smollm2-1.7b` (L9) | `SHUFFLED_R_FORCED` | FORCED_FROM_ABSTAIN (d_ff=8196) | 2.1% / 0.0% (elig 2.1%/0.0%) | 15.6% / 7.3% (elig 3.6%/1.8%) | 29.2% / 4.2% (elig 5.6%/5.6%) | 100.0% / 77.1% | ff=0.0%, nt=47.1% | loc=—/— KL=0.00292 ff=0.00119 ul=True |
| `smollm2-1.7b` (L9) | `CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | COMPILED (d_ff=8196) | 0.0% / 0.0% (elig 0.0%/0.0%) | 12.0% / 5.2% (elig 0.6%/0.6%) | 25.0% / 0.0% (elig 0.0%/0.0%) | 100.0% / 72.9% | ff=0.0%, nt=47.1% | loc=—/— KL=0.00959 ff=0.00136 ul=True |
| `smollm2-1.7b` (L9) | `NATIVE_B_ORACLE` | COMPILED (d_ff=8196) | 100.0% / 47.9% (elig 100.0%/47.9%) | 94.3% / 41.1% (elig 93.5%/38.7%) | 71.9% / 46.9% (elig 62.5%/62.5%) | 100.0% / 83.3% | ff=50.0%, nt=35.3% | loc=—/— KL=0.00580 ff=0.00230 ul=True |
| `granite-3.1-2b` (L12) | `BASE` | elig: dir=48/48, rip=162/192, hop=68/96 | 0.0% / 0.0% | 15.6% / 4.2% | 29.2% / 4.2% | 100.0% / 83.3% | ff=0.0%, nt=41.2% | 192/196 | PPL=14.681 |
| `granite-3.1-2b` (L12) | `IN_CONTEXT_CEILING` | prompt ceiling | 0.0% / 0.0% (elig 0.0%/0.0%) | 56.8% / 40.6% (elig 51.2%/42.0%) | 28.1% / 20.8% (elig 4.4%/7.3%) | 100.0% / 54.2% | ff=0.0%, nt=41.2% | unedited B |
| `granite-3.1-2b` (L12) | `CHANNEL_PUSHFORWARD_BIASFREE` | COMPILED (d_ff=8196) | 6.2% / 4.2% (elig 6.2%/4.2%) | 34.9% / 8.8% (elig 24.1%/4.9%) | 43.8% / 18.8% (elig 20.6%/20.6%) | 100.0% / 85.4% | ff=0.0%, nt=41.2% | loc=—/— KL=0.00076 ff=0.00170 ul=True |
| `granite-3.1-2b` (L12) | `SHUFFLED_R_AUTO` | **ABSTAIN** | — | — | — | — | — | auto-abstained (0 writes) |
| `granite-3.1-2b` (L12) | `SHUFFLED_R_FORCED` | FORCED_FROM_ABSTAIN (d_ff=8196) | 0.0% / 0.0% (elig 0.0%/0.0%) | 14.1% / 2.6% (elig 0.0%/0.0%) | 28.1% / 3.1% (elig 0.0%/0.0%) | 100.0% / 89.6% | ff=0.0%, nt=41.2% | loc=—/— KL=0.00033 ff=0.00128 ul=True |
| `granite-3.1-2b` (L12) | `CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | COMPILED (d_ff=8196) | 0.0% / 0.0% (elig 0.0%/0.0%) | 15.6% / 4.2% (elig 0.0%/0.0%) | 29.2% / 4.2% (elig 0.0%/0.0%) | 100.0% / 83.3% | ff=0.0%, nt=41.2% | loc=—/— KL=0.00159 ff=0.00000 ul=True |
| `granite-3.1-2b` (L12) | `NATIVE_B_ORACLE` | COMPILED (d_ff=8196) | 68.8% / 18.8% (elig 68.8%/18.8%) | 84.9% / 24.5% (elig 83.3%/24.7%) | 51.0% / 18.8% (elig 35.3%/26.5%) | 100.0% / 70.8% | ff=0.0%, nt=5.9% | loc=—/— KL=0.01063 ff=0.00213 ul=True |
| `falcon3-3b` (L9) | `BASE` | elig: dir=48/48, rip=169/192, hop=67/96 | 0.0% / 0.0% | 12.0% / 6.2% | 30.2% / 15.6% | 100.0% / 85.4% | ff=0.0%, nt=35.3% | 186/196 | PPL=24.988 |
| `falcon3-3b` (L9) | `IN_CONTEXT_CEILING` | prompt ceiling | 2.1% / 2.1% (elig 2.1%/2.1%) | 64.1% / 49.5% (elig 60.4%/47.3%) | 50.0% / 71.9% (elig 43.3%/59.7%) | 100.0% / 54.2% | ff=0.0%, nt=35.3% | unedited B |
| `falcon3-3b` (L9) | `CHANNEL_PUSHFORWARD_BIASFREE` | COMPILED (d_ff=9220) | 12.5% / 0.0% (elig 12.5%/0.0%) | 51.6% / 13.5% (elig 45.0%/7.1%) | 42.7% / 30.2% (elig 20.9%/20.9%) | 100.0% / 87.5% | ff=4.2%, nt=17.6% | loc=—/— KL=0.00071 ff=0.00137 ul=True |
| `falcon3-3b` (L9) | `SHUFFLED_R_AUTO` | **ABSTAIN** | — | — | — | — | — | auto-abstained (0 writes) |
| `falcon3-3b` (L9) | `SHUFFLED_R_FORCED` | FORCED_FROM_ABSTAIN (d_ff=9220) | 4.2% / 0.0% (elig 4.2%/0.0%) | 16.2% / 6.2% (elig 4.7%/0.6%) | 32.3% / 17.7% (elig 3.0%/3.0%) | 100.0% / 85.4% | ff=0.0%, nt=35.3% | loc=—/— KL=0.00056 ff=0.00172 ul=True |
| `falcon3-3b` (L9) | `CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | COMPILED (d_ff=9220) | 0.0% / 0.0% (elig 0.0%/0.0%) | 12.0% / 6.2% (elig 0.6%/0.6%) | 29.2% / 15.6% (elig 0.0%/0.0%) | 100.0% / 85.4% | ff=0.0%, nt=35.3% | loc=—/— KL=0.00122 ff=0.00129 ul=True |
| `falcon3-3b` (L9) | `NATIVE_B_ORACLE` | COMPILED (d_ff=9220) | 100.0% / 41.7% (elig 100.0%/41.7%) | 87.5% / 42.2% (elig 85.8%/39.1%) | 80.2% / 60.4% (elig 73.1%/59.7%) | 100.0% / 39.6% | ff=41.7%, nt=23.5% | loc=—/— KL=0.00113 ff=0.00137 ul=True |
| `qwen2.5-1.5b` (L13) | `BASE` | elig: dir=48/48, rip=163/192, hop=68/96 | 0.0% / 0.0% | 15.1% / 5.7% | 29.2% / 2.1% | 100.0% / 81.2% | ff=0.0%, nt=47.1% | 163/196 | PPL=28.251 |
| `qwen2.5-1.5b` (L13) | `IN_CONTEXT_CEILING` | prompt ceiling | 2.1% / 2.1% (elig 2.1%/2.1%) | 60.4% / 37.0% (elig 55.2%/35.0%) | 45.8% / 41.7% (elig 23.5%/23.5%) | 100.0% / 58.3% | ff=0.0%, nt=47.1% | unedited B |
| `qwen2.5-1.5b` (L13) | `CHANNEL_PUSHFORWARD_BIASFREE` | COMPILED (d_ff=8964) | 54.2% / 33.3% (elig 54.2%/33.3%) | 52.1% / 14.1% (elig 43.6%/9.8%) | 45.8% / 18.8% (elig 23.5%/13.2%) | 100.0% / 85.4% | ff=4.2%, nt=41.2% | loc=—/— KL=0.00119 ff=0.00161 ul=True |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_R_AUTO` | **ABSTAIN** | — | — | — | — | — | auto-abstained (0 writes) |
| `qwen2.5-1.5b` (L13) | `SHUFFLED_R_FORCED` | FORCED_FROM_ABSTAIN (d_ff=8964) | 0.0% / 0.0% (elig 0.0%/0.0%) | 18.2% / 8.8% (elig 3.7%/3.7%) | 29.2% / 3.1% (elig 0.0%/0.0%) | 100.0% / 83.3% | ff=0.0%, nt=47.1% | loc=—/— KL=0.00046 ff=0.00102 ul=True |
| `qwen2.5-1.5b` (L13) | `CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | COMPILED (d_ff=8964) | 0.0% / 0.0% (elig 0.0%/0.0%) | 15.1% / 6.2% (elig 0.0%/0.6%) | 31.2% / 3.1% (elig 2.9%/1.5%) | 100.0% / 79.2% | ff=0.0%, nt=47.1% | loc=—/— KL=0.00059 ff=0.00042 ul=True |
| `qwen2.5-1.5b` (L13) | `NATIVE_B_ORACLE` | COMPILED (d_ff=8964) | 100.0% / 27.1% (elig 100.0%/27.1%) | 89.1% / 17.7% (elig 89.6%/17.2%) | 50.0% / 15.6% (elig 29.4%/19.1%) | 100.0% / 79.2% | ff=8.3%, nt=11.8% | loc=—/— KL=0.00102 ff=0.00161 ul=True |

#### Table 4B-ii — Sealed Cap-2 Exact Two-Sided McNemar p-Values
| Recipient | Comparison | Direct (n10:n01, p) | 4-Attr Ripple All (n10:n01, p) | 4-Attr Ripple Elig FC | 2-Sent Hops All (n10:n01, p) |
| --- | --- | --- | --- | --- | --- |
| `smollm2-1.7b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_BASE` | FC 20:0 (p=1.91e-06) | EM 4:0 (p=0.1250) | FC 87:0 (p=1.29e-26) | EM 32:0 (p=4.66e-10) | FC 87:0 (p=1.29e-26) | FC 25:0 (p=5.96e-08) | EM 23:0 (p=2.38e-07) |
| `smollm2-1.7b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_SHUFFLED_R_FORCED` | FC 19:0 (p=3.81e-06) | EM 4:0 (p=0.1250) | FC 81:0 (p=8.27e-25) | EM 31:1 (p=1.54e-08) | FC 81:0 (p=8.27e-25) | FC 21:0 (p=9.54e-07) | EM 19:0 (p=3.81e-06) |
| `smollm2-1.7b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | FC 20:0 (p=1.91e-06) | EM 4:0 (p=0.1250) | FC 88:0 (p=6.46e-27) | EM 34:0 (p=1.16e-10) | FC 86:0 (p=2.58e-26) | FC 25:0 (p=5.96e-08) | EM 23:0 (p=2.38e-07) |
| `smollm2-1.7b` | `NATIVE_B_ORACLE_vs_BASE` | FC 48:0 (p=7.11e-15) | EM 23:0 (p=2.38e-07) | FC 157:0 (p=1.09e-47) | EM 67:0 (p=1.36e-20) | FC 157:0 (p=1.09e-47) | FC 45:0 (p=5.68e-14) | EM 45:0 (p=5.68e-14) |
| `granite-3.1-2b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_BASE` | FC 3:0 (p=0.2500) | EM 2:0 (p=0.5000) | FC 39:2 (p=7.84e-10) | EM 12:3 (p=0.0352) | FC 39:0 (p=3.64e-12) | FC 14:0 (p=0.0001) | EM 14:0 (p=0.0001) |
| `granite-3.1-2b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_SHUFFLED_R_FORCED` | FC 3:0 (p=0.2500) | EM 2:0 (p=0.5000) | FC 40:0 (p=1.82e-12) | EM 13:1 (p=0.0018) | FC 39:0 (p=3.64e-12) | FC 15:0 (p=6.10e-05) | EM 15:0 (p=6.10e-05) |
| `granite-3.1-2b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | FC 3:0 (p=0.2500) | EM 2:0 (p=0.5000) | FC 39:2 (p=7.84e-10) | EM 12:3 (p=0.0352) | FC 39:0 (p=3.64e-12) | FC 14:0 (p=0.0001) | EM 14:0 (p=0.0001) |
| `granite-3.1-2b` | `NATIVE_B_ORACLE_vs_BASE` | FC 33:0 (p=2.33e-10) | EM 9:0 (p=0.0039) | FC 135:2 (p=1.09e-37) | EM 42:3 (p=8.65e-10) | FC 135:0 (p=4.59e-41) | FC 24:3 (p=4.92e-05) | EM 18:4 (p=0.0043) |
| `falcon3-3b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_BASE` | FC 6:0 (p=0.0312) | EM 0:0 (p=1.0000) | FC 76:0 (p=2.65e-23) | EM 14:0 (p=0.0001) | FC 76:0 (p=2.65e-23) | FC 14:2 (p=0.0042) | EM 14:0 (p=0.0001) |
| `falcon3-3b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_SHUFFLED_R_FORCED` | FC 4:0 (p=0.1250) | EM 0:0 (p=1.0000) | FC 68:0 (p=6.78e-21) | EM 14:0 (p=0.0001) | FC 68:0 (p=6.78e-21) | FC 12:2 (p=0.0129) | EM 12:0 (p=0.0005) |
| `falcon3-3b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | FC 6:0 (p=0.0312) | EM 0:0 (p=1.0000) | FC 76:0 (p=2.65e-23) | EM 14:0 (p=0.0001) | FC 75:0 (p=5.29e-23) | FC 15:2 (p=0.0023) | EM 14:0 (p=0.0001) |
| `falcon3-3b` | `NATIVE_B_ORACLE_vs_BASE` | FC 48:0 (p=7.11e-15) | EM 20:0 (p=1.91e-06) | FC 145:0 (p=4.48e-44) | EM 69:0 (p=3.39e-21) | FC 145:0 (p=4.48e-44) | FC 49:1 (p=9.06e-14) | EM 44:1 (p=2.61e-12) |
| `qwen2.5-1.5b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_BASE` | FC 26:0 (p=2.98e-08) | EM 16:0 (p=3.05e-05) | FC 71:0 (p=8.47e-22) | EM 17:1 (p=0.0001) | FC 71:0 (p=8.47e-22) | FC 16:0 (p=3.05e-05) | EM 16:0 (p=3.05e-05) |
| `qwen2.5-1.5b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_SHUFFLED_R_FORCED` | FC 26:0 (p=2.98e-08) | EM 16:0 (p=3.05e-05) | FC 66:1 (p=9.22e-19) | EM 12:2 (p=0.0129) | FC 66:1 (p=9.22e-19) | FC 16:0 (p=3.05e-05) | EM 15:0 (p=6.10e-05) |
| `qwen2.5-1.5b` | `CHANNEL_PUSHFORWARD_BIASFREE_vs_CROSS_ORGAN_SWAP_CAP1_ON_CAP2` | FC 26:0 (p=2.98e-08) | EM 16:0 (p=3.05e-05) | FC 71:0 (p=8.47e-22) | EM 16:1 (p=0.0003) | FC 71:0 (p=8.47e-22) | FC 14:0 (p=0.0001) | EM 15:0 (p=6.10e-05) |
| `qwen2.5-1.5b` | `NATIVE_B_ORACLE_vs_BASE` | FC 48:0 (p=7.11e-15) | EM 13:0 (p=0.0002) | FC 146:4 (p=2.92e-38) | EM 28:5 (p=6.62e-05) | FC 146:0 (p=2.24e-44) | FC 20:0 (p=1.91e-06) | EM 14:1 (p=0.0010) |

#### Table 4B-iii — Cross-Capability Co-Implantation on Solo `smollm2-1.7b` L9 (`Cap-1 Geo + Cap-2 Bio`: $d_{\text{ff}}$ `8192 → 8200 → 8196 → 8192`)
| State | $d_{\text{ff}}$ | Cap-1 Geography (EM) | Cap-2 BIO-RIPPLE (FC/EM) | 196-Fact Loc | KL / ΔPPL | Bit-Exact Certificate |
| --- | --- | --- | --- | --- | --- | --- |
| `JOINT_CAP1_PLUS_CAP2` | 8200 | dir=18.8%, rip3=18.8%, ds8=25.0% | dir=41.7%/8.3%, rip=54.7%/20.3%, hop=51.0%/24.0% | 15/175 (91.4%) | 0.01059 / +0.3187 | — |
| `SLICE_OUT_CAP1_KEEP_CAP2` | 8196 | dir=0.0%, rip3=0.0%, ds8=0.0% | dir=41.7%/8.3%, rip=57.8%/22.9%, hop=51.0%/24.0% | 0/175 (100.0%) | 0.00107 / +0.0324 | ==solo_cap2: True (maxΔ=0.0) |
| `SLICE_OUT_CAP2_KEEP_CAP1` | 8196 | dir=18.8%, rip3=18.8%, ds8=25.0% | dir=0.0%/0.0%, rip=12.0%/5.2%, hop=25.0%/0.0% | 15/175 (91.4%) | 0.00959 / +0.2883 | ==solo_cap1: True (maxΔ=0.0) |
| `SLICE_ALL_RESTORE_BASE` | 8192 | restored to base | restored to base | 100.0% | 0.00000 | cap1_eq=True, cap2_eq=True, maxΔ=0.0 |

