# Reproducing PlatonicSwap

## 1. Environment Setup
All experiments were run on Apple Silicon (M5 Pro, 48 GB unified memory, PyTorch MPS backend, strict `float32` for Granite-3.1-2B and organ evaluations).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `HF_HOME` (or let HuggingFace download the 5 ungated base models: `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-1.5B`, `HuggingFaceTB/SmolLM2-1.7B`, `ibm-granite/granite-3.1-2b-base`, `tiiuae/Falcon3-3B-Base`).

## 2. Fast CPU Verification (No GPU / No Model Downloads Required)
Verify all mathematical invariants (Ledoit-Wolf shrinkage, exact 2-constraint KKT optimality, rectangular Stiefel channels, Cauchy-weighted write bridges, conformal false-fire bounds, automatic `ABSTAIN` on shuffled channels, exact `frame_only_auroc == 0.5000`, `SEALED CAP-2 SHA-256`, and `torch.equal` bit-identical unlearning):

```bash
PYTHONPATH=platonicswap:experiments python experiments/platonic_organ_engine.py --unit-test
PYTHONPATH=platonicswap:experiments python experiments/bio_ripple_sealed.py --unit-test
PYTHONPATH=platonicswap:experiments python experiments/pilot_graft_t3_gap_scaled.py --unit-test
```

## 3. Pillar 1 — Zero-QA Geometry & 85–92% Activation Transfer
```bash
PYTHONPATH=platonicswap:experiments python experiments/pilot0_diagnostics.py
PYTHONPATH=platonicswap:experiments python experiments/pilot1_fisher_scaling.py --pair "qwen2.5-3b->smollm2-1.7b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1_fisher_scaling.py --pair "qwen2.5-3b->granite-3.1-2b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1_fisher_scaling.py --pair "falcon3-3b->smollm2-1.7b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1b_break85.py --pair "qwen2.5-3b->smollm2-1.7b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1b_break85.py --pair "qwen2.5-3b->granite-3.1-2b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1b_break85.py --pair "falcon3-3b->smollm2-1.7b"
```

## 4. Pillar 2 & 3 — Frozen Channel-Pushforward Platonic Organs (Cap-1 & Sealed Cap-2 `BIO-RIPPLE`)
```bash
# Regime 1 pretrained 2,000-neuron conjugation & Gate G0
PYTHONPATH=platonicswap:experiments python experiments/pilot_organs_p0.py --stages 0 --site 17_9

# Cap-1 Geography (builds & caches frozen universal read/write channels across all 4 recipients + Stage B)
PYTHONPATH=platonicswap:experiments python experiments/platonic_organ_engine.py --run-cap1 --targets smollm2,granite,falcon3,qwen1.5b

# One-Shot Sealed Cap-2 BIO-RIPPLE (reuses exact frozen engine & universal channels + Cap-1/Cap-2 Co-Implantation)
PYTHONPATH=platonicswap:experiments python experiments/bio_ripple_sealed.py --run-cap2 --targets smollm2,granite,falcon3,qwen1.5b

# Scaled Knowledge-Gap Transplant (T3)
PYTHONPATH=platonicswap:experiments python experiments/pilot_graft_t3_gap_scaled.py --pair "qwen2.5-3b->smollm2-1.7b"
PYTHONPATH=platonicswap:experiments python experiments/pilot_graft_t3_gap_scaled.py --pair "qwen2.5-3b->granite-3.1-2b"

# Output-Layer Comparison Foil (Pilot 2b Gated Side-Module & Pilot 3 3-Family 2-Hop Split Composition)
PYTHONPATH=platonicswap:experiments python experiments/pilot3_split_composition.py
```

## 5. Rebuild Programmatic Audit Tables
```bash
python experiments/build_repo_tables.py
```
