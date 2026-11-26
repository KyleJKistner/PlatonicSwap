# 4-Model Outside Council Synthesis: First-Principles Generality & Anti-Overfitting for Platonic Organs

**Panelists (Zero Carbon):**
1. **Opus 5.5 Max:** [council_fp_opus_max.md](council_fp_opus_max.md)
2. **Fable 5.2 Max:** [council_fp_fable_max.md](council_fp_fable_max.md)
3. **Astra 6.2 Ultra:** [council_fp_astra_ultra.md](council_fp_astra_ultra.md)
4. **Neon High (Independent):** [council_fp_neon.md](council_fp_neon.md)

---

## Unanimous Verdict (4/4): HOLD Push -> Adopt Zero-Magic-Constant Channel-Pushforward Law + Pass Sealed Cap-2 (`BIO-RIPPLE`)

All four panelists independently rejected pushing a universal-transfer claim based on 1 recipient (`SmolLM2`), 1 geography task, hand-factored `w_country × w_entity`, hand-tuned constants (`0.45, 0.50, 3.0, 1.5, 1.15`), and the `4.5:1` template-imbalanced negative pool that caused `Qwen1.5B` Italy to score `AU_hn=0.922` vs `AU_unrel=0.307`.

### 1. Unified Task-Agnostic Organ Law (Q1 — Zero Magic Constants)
Replace backward weight pullback + scalar fudge factors with **Channel-Pushforward Twin-SwiGLU**:
- **Generative Channel ($A \to B$, strictly unlabeled multi-register pairs):**
  $$G = \Sigma_{BA}(\Sigma_{AA} + \lambda_{\text{LW}} I)^{-1} \in \mathbb{R}^{D_B \times D_A}, \qquad N = \operatorname{Cov}(X_B^{\text{val}} - (\mu_B + G(X_A^{\text{val}} - \mu_A)))$$
  Naturally rectangular (`2048 <-> 3072 <-> 1536`) — lost modes ($\rho = 0$) receive zero weight automatically; B-private dimensions enter through $N$.
- **Frame-Counterbalanced Donor Contrasts:** Enforce `frame_only_auroc ≈ 0.50` before fitting (eliminates prefix confounds at the source).
- **Channel-Optimal Direction in Recipient B Coordinates (50 ms CPU, convex, Ledoit-Wolf shrinkage):**
  Push Donor-A positives/negatives forward via $\tilde{x}_B = \mu_B + G(x_A - \mu_A) + e$ ($e$ resampled from empirical held-out calibration residuals) and solve shrinkage-LDA against **real unlabeled Recipient-B calibration tokens + simulated in-domain negatives**.
- **Exact Conformal Thresholds & Calibrated Abstention:**
  - $t_1 = \max\big(Q_{1-\alpha}(z_{\text{real }B\text{ cal}}),\; Q_{1-\alpha_{\text{neg}}}(z_{\text{neg sim}})\big)$ — provable exchangeability false-fire bound $\le \alpha$ even for pure-noise directions!
  - $t_2 = Q_{\alpha'}(z_{\text{pos sim}})$; **ABSTAIN iff $\Gamma = t_2 - t_1 \le 0$!**
- **Exact Analytic Sharpness:** $\beta = \frac{0.2785}{\delta \cdot \Gamma}$ (guarantees off-leak and on-ripple $\le \delta$ via $\max_{u<0}|\mathrm{SiLU}(u)| = 0.2785$).
- **Pure Digital `up` (`mlp_bias=False`):** `up ≡ constant pin c_B` (BOS-nulled!), eliminating the analog `up` path that decayed `1.234 -> 0.381` on Granite.
- **Exact Write Isometry:** Variance-matched isometry on RMS-normalized residual states with architectural inversion $W_{d,B} = \frac{1}{\mathrm{rm}_B} M_{\text{out,iso}} W_{d,A}$ (`gain == 1.000`, zero `2.5×` magic filter).

### 2. Universal Task-Agnostic Multi-Register Socket & 2×2 Ablation (Q2)
- Build **ONE** frozen, task-agnostic multi-register unlabeled corpus (`40%` general prose + `30%` open-domain encyclopedic/reference sentences + `30%` science/code/dialogue/QA/lists, strictly n-gram disjoint from BOTH Cap-1 and Cap-2!) and reuse it unchanged across BOTH capabilities.
- Run Fable/Astra's 2×2 diagnostic (`prose-only` vs `multi-register`) to cleanly separate socket coverage from architecture/template effects.

### 3. Model-Intrinsic Mask-Free Echo Neutralization & Sealed Capability 2 (Q3)
- **Mask-Free Earlier-Token Echo Fix (Opus):** Include all entity subtokens (not only final subtoken) in Donor-A positives so the per-token MLP organ natively suppresses earlier-token echoes without attention masks.
- **Task-Free Depth & Band Selection (Fable + Opus + Neon):** Measure unlabeled multi-token resample-patch commit $C_B(\ell)$ and self-healing $H_B(\ell \to \ell')$; pick $\ell^* = \arg\max_\ell \hat P_{\text{hit}}(\ell) C_B(\ell)$ and allocate a 2–3 layer native SwiGLU band ONLY when $H_B$ indicates high self-healing (40L Granite).
- **Sealed Capability 2 (3/4 Unanimous — Opus, Fable, Neon): `BIO-RIPPLE` (`Bird -> Fish`, `Mammal -> Reptile`)**
  - 0% geographic overlap, 48 strictly held-out unseen species (24 per rule), 192 unseen 4-attribute ripple trials + 96 unseen multi-sentence compositional hop trials + invariants + flying non-bird / aquatic mammal hard negatives + Cap-1 cross-talk + 196-fact locality + WikiText KL + `SHUFFLED_R` + cross-organ swap + `Native-B oracle` + co-implantation with Cap-1 & bit-verified `torch.equal` selective unlearning!
