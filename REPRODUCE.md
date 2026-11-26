# Reproducing PlatonicSwap (`v1.0`)

## 1. Environment Setup
All experiments were run on Apple Silicon (M5 Pro, 48 GB unified memory, PyTorch MPS backend, strict `float32` for `granite-3.1-2b` and organ evaluations).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

## 2. Fast CPU Verification (No GPU / No Model Downloads Required)
Verifies frozen engine SHA-256 provenance, mathematical invariants (Ledoit-Wolf shrinkage, exact 2-constraint KKT optimality, rectangular Stiefel channels, Cauchy-weighted write bridges, conformal false-fire bounds, automatic `ABSTAIN` on shuffled channels, exact `frame_only_auroc == 0.5000`, `SEALED CAP-2 SHA-256`, Cap-1/Cap-2/dataset overlap checks, and `torch.equal` bit-identical modular removal), and regenerates `results/master_audited_results.md`:

```bash
shasum -a 256 experiments/platonic_organ_engine.py platonicswap/platonic_organ_engine.py
# Both must match: 22ef57c9d196e12f2cd3c9ecc79ec0813803df43d5a6009f5422904948802306

PYTHONPATH=platonicswap:experiments python -B experiments/platonic_organ_engine.py --unit-test
PYTHONPATH=platonicswap:experiments python -B experiments/bio_ripple_sealed.py --unit-test
PYTHONPATH=platonicswap:experiments python -B experiments/pilot_graft_t3_gap_scaled.py --unit-test
python experiments/build_repo_tables.py
```

## 3. Model Downloads (Required for MPS Runs)
All experiment scripts load checkpoints strictly offline (`local_files_only=True`) from `experiments/hf_cache/<key>` (override via `PLATONICSWAP_HF_CACHE`). Download all 5 ungated base models once:

```bash
mkdir -p experiments/hf_cache
huggingface-cli download Qwen/Qwen2.5-3B                 --local-dir experiments/hf_cache/qwen2.5-3b
huggingface-cli download Qwen/Qwen2.5-1.5B               --local-dir experiments/hf_cache/qwen2.5-1.5b
huggingface-cli download HuggingFaceTB/SmolLM2-1.7B      --local-dir experiments/hf_cache/smollm2-1.7b
huggingface-cli download ibm-granite/granite-3.1-2b-base --local-dir experiments/hf_cache/granite-3.1-2b
huggingface-cli download tiiuae/Falcon3-3B-Base          --local-dir experiments/hf_cache/falcon3-3b
```

> **Canonical Published JSONs & Socket Manifest:** The gitignored `experiments/cache/*.pt` directory (~5.1 GB of frozen universal read/write sockets and donor rule caches) is indexed by filename, byte size, and SHA-256 in [`results/socket_manifest.json`](results/socket_manifest.json). Running `--run-cap1` generates and caches these sockets first; `--run-cap2` reuses them unchanged. Because PyTorch MPS kernels are not guaranteed bit-identical across OS/driver builds, the committed JSONs in [`results/`](results/) are the canonical audited artifacts referenced by `README.md` and `results/master_audited_results.md`.

## 4. Upstream Baselines & Pillar 1 (Zero-QA Geometry & 85–92% Transfer)
Supporting upstream JSONs (`results_baseline.json`, `results_referee_controls.json`, `results_systems.json`) are already shipped in `results/` so downstream scripts work out-of-the-box. To regenerate from scratch:

```bash
PYTHONPATH=platonicswap:experiments python platonicswap/extract_and_baseline.py --mode all
PYTHONPATH=platonicswap:experiments python experiments/run_referee_controls.py
PYTHONPATH=platonicswap:experiments python experiments/run_systems_dark.py

# Pillar 1 core scripts
PYTHONPATH=platonicswap:experiments python experiments/pilot0_diagnostics.py
PYTHONPATH=platonicswap:experiments python experiments/pilot1_fisher_scaling.py --pair "qwen2.5-3b->smollm2-1.7b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1_fisher_scaling.py --pair "qwen2.5-3b->granite-3.1-2b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1_fisher_scaling.py --pair "falcon3-3b->smollm2-1.7b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1b_break85.py --pair "qwen2.5-3b->smollm2-1.7b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1b_break85.py --pair "qwen2.5-3b->granite-3.1-2b"
PYTHONPATH=platonicswap:experiments python experiments/pilot1b_break85.py --pair "falcon3-3b->smollm2-1.7b"
```

## 5. Pillar 2 & 3 — Native SwiGLU Platonic Organs (Cap-1 & Cap-2 `BIO-RIPPLE`)
Run `--run-cap1` **before** `--run-cap2` so the frozen universal read/write sockets are populated in `experiments/cache/`:

```bash
# Regime 1 pretrained 2,000-neuron conjugation & Gate G0
PYTHONPATH=platonicswap:experiments python experiments/pilot_organs_p0.py --stages 0 --site 17_9

# Earlier single-family prototype (Stage A + Stage B comparison baseline)
PYTHONPATH=platonicswap:experiments python experiments/pilot_organs_multifamily.py --stages A,B

# Frozen Engine Cap-1 Geography (builds & caches universal sockets for all 4 recipients + Stage B)
PYTHONPATH=platonicswap:experiments python experiments/platonic_organ_engine.py --run-cap1 --targets smollm2,granite,falcon3,qwen1.5b

# Single-Execution Cap-2 BIO-RIPPLE (reuses frozen engine & universal sockets + Cap-1/Cap-2 Co-Implantation)
PYTHONPATH=platonicswap:experiments python experiments/bio_ripple_sealed.py --run-cap2 --targets smollm2,granite,falcon3,qwen1.5b

# Grafting Falsification Pilot (T1/T2/Gate 3) & Latent-Knowledge Unlocking (T3 Original + Scaled)
PYTHONPATH=platonicswap:experiments python experiments/pilot_graft_falsify.py
PYTHONPATH=platonicswap:experiments python experiments/pilot_graft_t3_gap.py
PYTHONPATH=platonicswap:experiments python experiments/pilot_graft_t3_gap_scaled.py --pair "qwen2.5-3b->smollm2-1.7b"
PYTHONPATH=platonicswap:experiments python experiments/pilot_graft_t3_gap_scaled.py --pair "qwen2.5-3b->granite-3.1-2b"

# Output-Layer Comparison Foils (L23 non-canonical W_down overwrite, L23 Gated Sidecar, & 3-Family Split Composition)
PYTHONPATH=platonicswap:experiments python experiments/pilot2_fullrank_check.py
PYTHONPATH=platonicswap:experiments python experiments/pilot2_fullrank_check.py --gated-only
PYTHONPATH=platonicswap:experiments python experiments/pilot3_split_composition.py --recipient smollm2-1.7b
PYTHONPATH=platonicswap:experiments python experiments/pilot3_split_composition.py --recipient qwen2.5-1.5b
PYTHONPATH=platonicswap:experiments python experiments/pilot3_split_composition.py --recipient falcon3-3b
PYTHONPATH=platonicswap:experiments python experiments/pilot3_split_composition.py --recipient granite-3.1-2b
```

Note: scripts write outputs to `LAB_DIR` (`experiments/` by default); copy refreshed JSONs to `results/` (or run with `PLATONICSWAP_LAB_DIR=results`) before regenerating the master table.

## 6. Rebuild Programmatic Audit Tables
```bash
python experiments/build_repo_tables.py
```
This reads all JSON artifacts from `results/` and regenerates [`results/master_audited_results.md`](results/master_audited_results.md).
