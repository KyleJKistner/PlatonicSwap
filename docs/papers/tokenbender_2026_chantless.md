---
title: "Chantless: How Do Models Lose Weight Without Crying? (Training-Free Capability Extraction)"
authors: "Abhishek Harshvardhan Mishra (tokenbender / Deus Experiments)"
date: "2026-11-26"
url: "https://deusexperiments.com/research/how-do-models-lose-weight-without-crying/"
cluster: "capability_extraction_complementary"
tags: ["chantless", "capability-extraction", "pruning", "masks", "training-free", "gguf"]
status: "FULL_TEXT"
source: "web_html (39,460 chars) + github_readme"
---

# Chantless: How Do Models Lose Weight Without Crying? (Training-Free Capability Extraction)

- **Authors:** Abhishek Harshvardhan Mishra (tokenbender / Deus Experiments)
- **Date:** 2026-11-26 | **URL:** https://deusexperiments.com/research/how-do-models-lose-weight-without-crying/
- **Cluster:** `capability_extraction_complementary` | **Tags:** `chantless`, `capability-extraction`, `pruning`, `masks`, `training-free`, `gguf`
- **Status:** `FULL_TEXT` (web_html (39,460 chars) + github_readme)
- **Relevance to PlatonicSwap v1:** Complementary subtractive single-model capability extraction (carving a specialist out of 1 model vs our additive cross-architecture splicing).
- **Relevance to PlatonicSwap v2:** Potential upstream 'harvester' for finding organic Donor-A capability subnetworks before PlatonicSwap splices their residual deltas into Recipient B.

---

## Repository README

# CHANTLESS

![Deus Experiments logo](assets/brand/deus-experiments-logo.png)

**Your taste. Multiplied.**

Chantless extracts capabilities from frozen models without training their weights. The extracted capabilities remain independently deployable and adaptable.

Conventional compression work tries to preserve as much of the model as possible, primarily under a size constraint. Chantless utilizes search algorithms to best attribute everything that really contributes to a capability and ensure it is neatly abstracted. Once we identify the capability, we extract it in a novel way that keeps it malleable.

In practice, you hand Chantless a small batch of up to 25 real mistakes. Instead of training new float weight values, it decides which small set of late-layer neurons to switch off. Flipping those switches adds zero bytes to the file.

[**Read the research announcement — How do models lose weight without crying?**](https://deusexperiments.com/research/how-do-models-lose-weight-without-crying/)

---

## Quickstart

```bash
# 1. Build unified llama.cpp@afe77e18 (CUDA / Metal / CPU auto-detect) + C++ scorers & install CLI
./scripts/build_toolchain.sh --backend auto --prefix .local/toolchain
pip install -e ".[train,dev]"
chantless doctor --llama-bin .local/toolchain/llama-afe77e18/build/bin --traj-kl .local/toolchain/bin/traj-kl

# 2. Materialize any shipped winner mask into a standalone SHA256-exact GGUF
chantless build \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --llama-bin .local/toolchain/llama-afe77e18/build/bin \
  --out qwen3-4b-bfcl-apprentice.gguf

# 3. Serve or benchmark directly from stock base + mask (SHA256-cached auto-build + live CUDA/Metal VRAM)
chantless bench \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --compare qwen3-4b-bf16.gguf \
  --llama-bin .local/toolchain/llama-afe77e18/build/bin
```

---

## Verified Winners

All **`16` trophy masks** (`6.4–52.8 KB`, zero trained floats) and SHA256 receipts live in [`masks/README.md`](masks/README.md).

### 1. Stock-Checkpoint Double Winners (`extract` + `<=25`-Mistake `+0 B` `learn`, `4` Architectures)

| Model | Task | Size (`%` BF16) | Mask | Held-Out `exam` | Gain vs Uniform | Apprentice (`<=25` ex → Unseen Fixed) | Controls (`base/empty/flip/shuf`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Gemma 4 E2B`** | `ClsBench` | `928.7 MB` (**`9.97%`**) | `47.7 KB` | **`99.21%`** | **`+49.78` pts** | **`15 / 60`** (`+0 B`) | `3 / 3 / 2 / 8` |
| **`Qwen3.5-4B`** | `ClsBench` | `837.2 MB` (**`9.94%`**) | `48.5 KB` | **`93.40%`** | **`+46.73` pts** | **`33 / 60`** (`55.0%`, `+0 B`) | `4 / 4 / 5 / 20` |
| **`Gemma 4 E2B`** | `BFCL` | `1.95 GB` (`20.92%`) | `47.1 KB` | **`97.13%`** | **`+44.02` pts** | **`26 / 60`** (`43.3%`, `+0 B`) | `0 / 0 / 0 / 7` |
| **`Qwen3-4B`** | `BFCL` | `1.97 GB` (`24.50%`) | `6.4 KB` | **`92.31%`** | **`+5.87` pts** | **`22 / 60`** (`36.7%`, `+0 B`) | `3 / 3 / 1 / 12` |
| **`Qwen3.5-4B`** | `BFCL` | `2.19 GB` (`26.01%`) | `18.6 KB` | **`92.39%`** | `+0.00` pts | **`29 / 60`** (`48.3%`, `+0 B`) | `0 / 0 / 2 / 19` |
| **`Inkling-2-Mini`** (`24B` MoE) | `ClsBench` | `4.17 GiB` (**`8.58%`**) | `33.9 KB` | **`98.67%` @ `5.00%` FPR** | `11.65×` smaller (`BF16` `99.17%`) | **`10 / 60`** (`+0 B`, exam `+0.34 pp`) | `4 / 4 / 3 / 8` |

### 2. Additional Looped-MoE `Inkling-2-Mini` (`24B` / `128` Experts) Specialist Winners

| Mask | Task | Size (`%` BF16) | Mask | Held-Out `exam` (vs `BF16`) | `reasoning_effort="high"` (`0.9`) Stress Test |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `inkling2-mini-clsbench-mask.json` | `ClsBench` | `4.17 GiB` (**`8.58%`**) | `27.9 KB` | **`98.33%` recall** @ `5.33%` FPR, `AUROC=0.9965` (`BF16` `99.17%` @ `5.00%`) | Upgraded by `+0 B` Apprentice above (`98.67%` @ `5.00%` FPR) |
| `inkling2-mini-bfcl-mask.json` | `BFCL` | `14.22 GiB` (**`29.23%`**) | `9.0 KB` | **`115/160` (`71.88%`, `+0.63 pp`)** Gorilla (`BF16` `114/160`) | **BFCL-HIGH (`8k`):** `53/79` Gorilla tie · **DeepMath-HIGH (`16k`):** **`41/48` (`0` regressions)** |
| `inkling2-mini-deepmath-mask.json` | `DeepMath` | `12.48 GiB` (**`25.66%`**) | `8.4 KB` | **`81/96` (`84.38%`)** `"low"` `4k` (`BF16` `84/96`, `1.45×` faster) | **DeepMath-HIGH (`16k`):** **`40/48` strict / `41/48` raw** (`25%` experts pruned) |

### 3. Imported `27B` Winners (`chantless mask import`, Terminal-Bench 2.1)

| Mask | Model | Size (`%` F16) | Mask | Recorded Score (`89` tasks × `4` repeats) |
| :--- | :--- | :---: | :---: | :--- |
| `qwen38-27b-dynquant-k34-mask.json` | `Qwen3.8-27B` | `19.60 GB` (`35.87%`) | `9.7 KB` | **`77.8%` pass@1** (`277/356` audited) · **`91.0%` pass@4** (`81/89`) |
| `qwen38-27b-dynquant-k94-mask.json` | `Qwen3.8-27B` | `14.26 GB` (`26.09%`) | `9.6 KB` | **`74.4%` pass@1** (`265/356` audited) · **`85.4%` pass@4** (`76/89`) |

### 4. Live Hardware Runtime Telemetry (NVIDIA CUDA & Apple Metal)

| Measurement | Hardware | Baseline | Chantless | Delta |
| :--- | :--- | :---: | :---: | :--- |
| **Apprentice Training (Streaming Prefix)** | NVIDIA RTX 4090 (`sm_89`) | `3,072.8 MB` alloc · `18.84s` | **`493.3 MB`** alloc · `15.76s` *(cached `13.10s`)* | **`-83.95%`** CUDA VRAM (bit-exact, `0.0` diff) |
| **Inference (`chantless bench`, `Q4_K` vs `BF16`)** | NVIDIA RTX 4090 (`CUDA0`) | `440.95 tok/s` · `1,844 MiB` VRAM | **`630.0 tok/s`** · **`1,016 MiB`** VRAM (`571 MiB` `CUDA0`) | **`1.429×`** decode · **`0.551×`** `nvidia-smi` VRAM |
| **Inference & Apprentice (`mps` / `MTL0`)** | Apple Silicon | `1,286 MiB` `MTL0` | **`464 MiB`** `MTL0` (`build_s=0.0` cache hit) | **`2.498×`** decode · **`0.361×`** memory · `3.9×–13.5×` learn |

### 4. Production-Scale Classifier Proof & OOD Counter-Baselines

| Capability | Held-out result | Extracted size |
|---|---|---|
| AI writing detection · Gemma 4 E2B | **99.76% retention of source-correct decisions** | **930.97 MB (10% of BF16)** |

**Artifacts:** Source checkpoint, winning extraction mask, evaluation receipt and reproduction instructions.

**Caveat:** Passes pooled non-inferiority; Beemo alone falls outside the two-point margin.

- **`Qwen3.5-0.8B` `200 MB` multi-pass Apprentice vs Small-Model SFT/KD:** Improved **`8/25 → 25/25`** on unseen AI passages (`0/25` human FPR) at **`6.6 × 10¹⁶` FLOPs** replay (`3.2×` cheaper than Hinton KD on `ModernBERT-base`). On the exact same `14,991`-doc splits, labels-only SFT on `ModernBERT-base` (`159 MB`) and `LFM2.5-230M` (`244 MB`) collapsed OOD (**`75.21%`** and **`78.44%`** human false-positive rate on Beemo).

---

## CLI & Capabilities

| Area | Command | Capabilities | Key Options |
| :--- | :--- | :--- | :--- |
| **Toolchain Bootstrap** | [`scripts/build_toolchain.sh`](scripts/build_toolchain.sh) | Builds pinned `llama.cpp @ afe77e18` (single unified patch for `Qwen3` / `Qwen3.5` / `Gemma 4` / `Inkling-2-Mini`) + `traj-kl`, `logodds`, `tokdocs` | `--backend auto\|cuda\|metal\|cpu`, auto single-arch CUDA detection, `--with-fla`, `--dry-run` |
| **Environment Preflight** | `chantless doctor` | JSON health check for `afe77e18` binaries, C++ scorers, CUDA/Metal + `fla`, Python deps, and `/dev/shm` + disk | `--config`, `--model-gb 54` (`27B` scratch/tier-cache layout preview), `--require-cuda`, `--require-classifier` |
| **Declarative Configs** | `chantless validate` / `schema` | Strict Pydantic v2 validation + JSON Schema export | JSON or YAML, file-relative `extends:`, repeatable `--set key.path=val`, `22` opt-in ggml tiers |
| **Dry-Run Planner** | `chantless plan` | Estimates moves, probes, episodes, wall-clock, and spend before compute | `--rates custom.json`, `--source local.gguf`, `--set` overrides |
| **Specialist Extraction** | `chantless extract` | Frozen-base extraction (`prepare → proxy → sanity → search → freeze → exam → report`) with one-shot solver, Wanda widths, Rule-P vocab trim, lossless dead shared-KV drop, and `SplitGuard` | `--upto`, `--reset`, `--out-mask`, `--out`, decoupled `/dev/shm` vs NVMe (`1.5×`/`3.5×`), auto worker clamp, tier cache |
| **Zero-Byte Apprentice** | `chantless learn` | Adapts built specialists from `<=25` mistakes at `+0 B` via `up_row_zero` switches + full-vocab token-KL & A-GEM + `4` strict controls | Prefix activation cache, `stream_prefix="auto"`, CUDA pinned memory, deterministic `judge_slots=1` |
| **Multi-Cell Sweeps** | `chantless sweep` | Declarative matrix runner across models, tasks, and budgets | `--launcher local\|ssh\|modal\|colab`, `--dry-run`, auto-resume, `--stop-on-failure` |
| **Mask Algebra & Import** | `chantless mask <cmd>` | Git-like operations on `chantless.mask/v3` across all `22` native `llama-quantize` ggml types (`F32..IQ1_S`) | `inspect`, `diff`, `fork`, `merge` (`apprentice` / `specialist`), `pull` (`hf://`/`https://` SHA256-pinned), `import` (`--from-gguf`, `--from-tensor-types`, `--from-spec`) |
| **Deterministic Build** | `chantless build` | SHA256-exact standalone GGUF compilation from stock BF16 + mask | Auto sharded `llama-quantize` (`--shards`, `--threads`), `--hf-dir` shared-KV resolution, `--imatrix` isolation |
| **Serve & Telemetry** | `chantless serve` / `bench` | OpenAI-compatible serving & benchmarking with structured JSON receipts | Direct `--model` or `--source + --mask` (SHA256 build cache), `--compare`, live per-PID VRAM, `/metrics`, clean `SIGTERM` |

---

## Documentation

- [Verified Winner Masks & SHA256 Receipts](masks/README.md) · [Engine Reference](mvp/README.md) · [Architecture & Design Notes](explainer/design-notes.md)
- Example configs: [extract-example.yaml](mvp/configs/extract-example.yaml) · [learn-example.yaml](mvp/configs/learn-example.yaml) · [local-smoke.json](mvp/configs/local-smoke.json)


---

## Research Blog Post Text

Imagine there is a really big model, and this big model contains answers to anything you wish to ask questions on. But this model is slow and expensive to work with. It requires lots of compute or cost to even get it to answer your needs once. Now most of your workflow revolves around only one kind of questions. And while you may be asking a variety of questions, but for your business or for a specific workflow, you do not really concern yourselfwith anything outside of one domain. At that point you start pondering - Is it possible for me to just get access to the portion which contains the knowledge for my questions? Now I know what you might be thinking. You may be thinking that I'm talking about creating small language models or using a highly quantized version which can fit into much lesser memory, but you're wrong.

To prepare a small language model that fits your usecase, you first need to find a suitable small base. Something that really fits in your budget, and there is no guarantee that given any flexible budget that you may have, you would actually be able to find something that is able to capture the performance you really want. And your primary need is you wanted to get the outputs that you were getting from the bigger model just for your usecase and much smaller budget. And in the case of a quantized model, you pray that the data that it has been calibrated on matches what you really care for. And your needs are not going to just stop there. When you start using the model, just in some time, you are going to realize that you would want to extend the workflow, do a little bit of tweaking against the behaviour you like vs what you dislike. Your needs aren't going to be satisfied with anything that is the present in the ecosystem right now. And there is a genuine gap in the ecosystem where one simply wishes to use ai capabilities for their needs for cheap, dictated by just their preferences, adaptive and requires no technical knowledge.

Conventional compression work tries to preserve as much of the model as possible, primarily under a size constraint. Chantless, in contrast, utilizes search algorithms to best attribute everything that really contributes to a capability and ensure it is neatly abstracted. Once we identify the capability, we extract it in a novel way that keeps it malleable.

Well then, how do we solve this problem? Well, turns out we have tackled exactly that with Chantless. We walk through it by using a super intuitive example and then we will map that intuition to the decade-long ML research that stands behind us. And then we discuss how do we stand on the shoulders of these giants. So let's say you're a manager and you have a huge team of 100 people. Now this one hundred team group, they are an all rounder group. Together they are superintelligent, can solve all sorts of tasks. The only problem is while they can solve everything, they also have to go everywhere together. So whichever task, however big or small, however simple or difficult we always take everyone. And that is bad because it is costly. We try to solve this problem by running a bunch of experiments. We take a category of tasks and then we ask the team to work on it and then we start observing. We observe which particular team members are the most active, and are the most performant. And we find out which ones are so-so and which ones seem to not be doing great. And then we start trying various configurations or combinations where we try to keep the ones who seem to be highly performant full time. The ones seem to be doing so-so we keep them partial and some other ones we just completely drop. After that we see if we are still able to match exactly the same performance that we were getting when the entire team was frozen. Sometimes we're right, sometimes we're wrong but we identify exactly what particular combination seems to be the most performant. Once we have done that for a task, we realize that now we know exactly how we can say that this particular team can represent this particular capability. We can then follow that up with all sorts of tasks. Now whenever we have to solve any task, we don't have to haul the entire 100 team group. We simply go and pick up the ones who seem to be allocated to different tasks and only pay for their time.

Now, how do we really map this to ML? And here is the part where I initially talked about using what already exists. The existence of sparse, task-specific subnetworks has been explored extensively in prior research [ 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 ] . OpenAI demonstrated interpretable circuits in weight-sparse transformers [ 13 ] , while research in January 2026 showed that language model circuits can be sparse even in the native neuron basis [ 14 ] .

In our own research in May and June 2026, through Honey, I Shrunk the Circuits! and PRISM, we investigated the extraction, conditioning and recovery of specific capabilities [ 15 , 16 ] . More recently, Goodfire’s research has provided additional evidence supporting the interpretability and functional organization of neurons [ 17 , 18 , 19 ] .

So then we combine it with the idea of Lottery Tickets and Supermasks [ 20 , 21 ] , and identify those subnetworks. Then we have activation-aware pruning [ 22 , 23 , 24 ] and mixed-precision quantization [ 25 , 26 , 27 ] that can be quite useful in being able to carve out high-signal pathways. These high-signal pathways can keep performance sharp while crushing or dropping any idle mass around them. And then finally, we can use prior research around continual learning where replay KL and A-GEM [ 28 , 29 , 30 , 31 ] have shown that you can adjust behavior from a few corrections without erasing old skills. The missing piece in this entire puzzle is how do you really identify an entire pathway and capture a specific capability? So that comes from the analogy which I showed previously. The core idea behind Chantless is applying that intuition to a network. You throw your real queries through the frozen model and then you spend search and compute to let the true pathways sharpen. You drown out the noise, then you test for what holds up, and then you keep on feeding it failures. You keep on searching until you reach whatever threshold of performance you're setting. So this allows you to set a budget of exactly how much you want to search, and what total parameter cost you're looking for, and then you can discover that specific pathway suited to that specific budget. And finally, once that extraction is complete, you can flip individual neuron switches on and off based on your preferences, based on your feedback as it keeps on evolving. You can flip the neuron switches and then find out that, okay, if I do it like this, then this particular failure seems to be getting fixed and this particular capability seems to be sharpening out. Using that, you can have a highly elastic, highly adaptive system which captures the entire discovery into a tiny kilobyte file. And your base model stays untouched, untrained.

## Tasks, Environments, and Defining What You Want

Figure 2 Anatomy of a Task and Environment

Figure 2 (Task and environment contract): How Chantless turns what you want into a checkable signal. Each task separates what the model sees ( prompt ) from the hidden grading target inside meta , keeps each declared task family inside one split, grades attempts through either an episode runner or a calibrated classifier scorer, and feeds those verdicts into Stage 1 extraction and Stage 2 adaptation.

Chantless is designed to be a minimal hassle onboarding experience. If you already have your evals or envs you wish to extract and build apprentice models on you can simply point an agent towards the Chantless repo and those evals/envs and ask it to simply run the framework for you. However as long as you have enough data in any directory or exported information which can represent the work that you do sufficiently well, then even that can be used to create envs/tasks in Chantless. There are clear examples and templates inside Chantless framework that one can simply ask any of their favorite agents to setup a task for them provided enough data and preferences are present beforehand. I strongly recommend that in the case you do not have any tasks/evals you iterate a bunch of times on defining precisely what you want with Chantless. This means using some equivalent of "Ask me questions to create a rigorous task/env that covers what i wish to be the task as well as what is desirable/undesirable or what doing it well means." with your agents until you are satisfied. Chantless is quite sample efficient and requires much lesser examples than actual training counterparts.

The quality of your extracted specialist entirely depends on the search compute and data/envs that you provide. Let us see how does a task look like. A task is one rigorously designed set of test functions - an ID, a label (family), split tag, prompt(s) and the metadata needed to graade (can be a label, reference answer, or hidden unit tests). An environment then wraps the collection of those tasks together with your operating envelope which is generation budget, your context window, concurrency, KV type and any optional memory cap. This all comes together to define how a model attempts and gets judged.

Once the contract is in place, then we tell the model what we want in two stages. During Stage 1 extraction , our preferences combine the environment's verifier, a quick task proxy, and guardrails set in the run configuration. In case of all the generative runs we minimize KL divergence to the frozen source in preserve [ 32 ]  mode or we use candidate negative log-likelihood in improve mode if we have reliable, source-successful search trajectories/data. In case of a classifier run we minimize soft partial-AUC damage ( pAUC(source) - pAUC(candidate) , pulling hard negatives dynamically from the candidate's own lowest-scoring negative tail) or preserve source rank order in one-shot mode. Then we have guardrails which check episode detectors, optional executable behavioral items (arithmetic, JSON, counting, reversal, capitals, uppercase; off by default and disallowed in classifier extraction), solo per-group damage-per-byte floors against uniform quantization, and an optional SizeMatchedCap bounding general-data drift. When greedy search enables failure feedback, it logs failed candidate search episodes (plus behavioral failures the source passed), tests reverting recent moves to locate the culprit group, and penalizes or undoes damaging moves without using human-rewritten answers or feeding select failures into search feedback.

And during Stage 2 post-extraction adaptation , giving feedback means supplying concrete mistakes directly. Generative switch adaptation takes (prompt, corrected) lessons (with optional mistaken trajectories) and learns binary neuron switches on frozen weights [ 33 , 34 , 35 ] ; classifier feedback takes labeled tokenized documents and optional contrastive pairs, updating quantization grid levels by default or binary switches when method="gates" is selected. Both paths support configured replay retention holds and A-GEM gradient projection [ 28 , 29 ] to reduce interference with replayed skills, with retention checked against configured margins, while promotion follows path-specific gates: generative switching can gate against up to four matched-byte controls ( base , empty , and conditional flipped and shuffled arms, defaulting to 25 lessons, configurable) plus held-out select retention when held-out tasks are provided (otherwise, corrected-output NLL improvement, additionally gated by select retention when available), and the classifier runner splits select into replay and judge halves, recalibrates each model's cutoff, and gates retention alongside feedback-pair gain.

Table 1 Five environments shipped in mvp/chantless/env/ . Each contract turns a real workflow into concrete inputs and an executable check. Dataset and benchmark sources: [ 36 , 37 , 38 , 39 , 40 ]

| Environment
| What you feed it
| What grading actually checks

| ClsBench
| Binary text examples (Yelp Polarity default or local JSONL)
| Classifier path: log-odds vs. each model's own calibrated cutoff (5% default FPR; 1% in shipped Inkling config).

| AI-text detector ( detect )
| Human and AI documents (WildOutlines and Beemo, including expert-edited AI text)
| Classifier mode ( classifier=True ): model-specific 1% calibration-human cutoff vs. dataset labels (default episode mode uses a supplied probability cutoff).

| BFCL
| Single-invocation tool-calling prompts (pinned BFCL v3 subsets)
| Episode path: exactly one parsed call matching canonicalized reference calls and arguments plus detectors (no live tool execution).

| DeepMath
| Filtered math problems (pinned DeepMath-103K integers or local JSONL)
| Episode path: extracted boxed integer vs. stored reference integer plus detectors (checks the final integer, not the proof).

| CodeTasks
| Procedural Python specs with 2 visible examples
| Episode path: generated code passes hidden unit tests in an isolated subprocess plus detectors (process isolation, not a security sandbox).

## Capability discovery and extraction

Figure 3 Built-in Stage 1 Search

Figure 3 (Built-in Stage 1 search, illustrative): Stage 1 searches for a mix of weight settings that fits your size target without breaking your task. It runs in one of two ways: step by step, testing each precision cut or block drop on the current model toward a stopping target and using failed search-split runs (when enabled) to catch bad moves; or in one shot, optionally selecting a vocabulary first, combining single-block measurements into a heuristic full plan under a required target, and checking the built file (with up to 3 build attempts / 2 size-correction retries). Activation stats help guide quantization ( imatrix ) and structural rankings, while task proxy measurements and select gates decide what stays sharp and what shrinks. Related pruning and quantization methods: [ 22 , 23 , 24 , 41 ]

Figure 4 Stage 1 Masks · Supported Controls and Construction Paths

Figure 4 (Supported controls and build paths): How Chantless compiles a compact, standalone GGUF from a frozen model across dense, hybrid token-mixer/SSM, and MoE architectures (with no Stage 1 weight-repair lineage recorded on any shipped mask). Each built-in run (Path A) turns on whichever of the schema's conditional controls the model adapter and config allow (group precision tiers, adapter-legal group drops, narrower MLP widths, fewer MoE experts, and a trimmed vocabulary; per-tensor overrides in tensors are honored by the builder and populated by direct imports or merged/custom masks), freezes the selected (or fallback) layout into a declarative chantless.mask/v3 JSON file, builds the standalone model, and evaluates it in a post-freeze exam phase. Existing mixed-precision builds can also enter directly through chantless mask import (Path B), keeping caller-supplied benchmark receipts distinct from built-in search and exam claims.

Let's go through Figures 3 and 4; they both represent the same pipeline. Let's start where we have a frozen source checkpoint. This frozen source checkpoint can be a stock open-weights model or a previously trained model. Then we move on to the next stage, where instead of fine-tuning a smaller pretrained network on our task labels, using Chantless we run a huge number of queries through the network and, through various approaches, search and identify the family of neurons that seem to be highly responsive for ensuring the same behavior as the base model. We use this to inform the imatrix quantization and rank neuron and expert proposals [ 41 ] . Once this is done, we probe task score shifts and byte savings directly on the frozen model to decide exactly how the config should be allocated.

Now, because Chantless supports various architectures and tasks, it has to tolerate different kinds of cuts, and thus no single fixed recipe is forced everywhere. The objective of each mask is to enable only the levers that would help it. It goes with mixed precision across the eligible tiers [ 22 , 24 , 25 , 26 , 27 ] , supports adapter-legal group drops, and also has optional support for block-level MLP width slicing, MoE expert selection, and vocabulary trimming. Finally, every single run results in a layout, which we call the Chantless mask.

Table 2 Stage 1 extraction across architectures and tasks (pure extraction only, zero post-extraction switching or lessons). Covers dense, hybrid token-mixer/SSM, and looped-MoE stock models, two fine-tuned AI-text detector extractions, and two 27B imported dynamic-quantization masks. Denominator type (teacher-match, full-split exam, select stress test, conditional retention, or imported provenance) is labeled per row. Base model and benchmark credits: [ 42 , 43 , 1 , 39 , 40 , 44 , 36 , 38 , 37 ]

| Model (Architecture)
| Task / Path
| Built GGUF (% Source)
| Active Stage 1 Levers
| Stage 1 Held-Out Result (vs. Source / Uniform)

| Gemma 4 E2B Dense + shared KV
| ClsBench classifier
| 928.7 MB 9.97% BF16
| Mixed tiers + 9 drops + 14 width cuts + 51,999 vocab + 60 dead KV
| 99.21% teacher-match (n = 2,919 of 3,000 source-correct; +49.78 pts vs. IQ1_S)

| Gemma 4 E2B Dense + shared KV
| BFCL tool calling
| 1,948.0 MB 20.92% BF16
| Mixed tiers + 2 width cuts + 155,000 vocab + 60 dead KV
| 97.13% teacher-match (n = 418 of 869 source-correct; +44.02 pts vs. Q2_K)

| Gemma 4 E2B Dense, fine-tuned source
| AI-text detect classifier
| 930.97 MB 10.0% BF16
| Mixed tiers + 17 drops + 21 width cuts + 80,000 vocab + 60 dead KV
| 99.76% conditional retention (50,944 / 51,066 source-correct; all-doc recall 98.53% WO, 57.98% Beemo)*

| Qwen3-4B Dense
| BFCL tool calling
| 1,972.2 MB 24.50% BF16
| Mixed tiers + 98,000 vocab (6.4 kB mask)
| 92.31% teacher-match (n = 494 of 869 source-correct; +5.87 pts vs. Q3_K)

| Qwen3.5-4B Hybrid GDN / SSM
| ClsBench classifier
| 837.2 MB 9.94% BF16
| Mixed tiers + 9 drops + 27 width cuts + 40,960 vocab
| 93.40% teacher-match (n = 2,923 of 3,000 source-correct; +46.73 pts vs. IQ1_S)

| Qwen3.5-4B Hybrid GDN / SSM
| BFCL tool calling
| 2,191.1 MB 26.01% BF16
| Mixed tiers + 132,000 vocab
| 92.39% teacher-match (n = 460 of 869 source-correct; +0.00 pts tie vs. Q3_K at 1.82 GB)

| Qwen3.5-0.8B Hybrid GDN / SSM, fine-tuned source
| AI-text detect classifier
| 396.45 MB 26.1% BF16
| Mixed tiers + 12 group drops ( detector-cls396.gguf )
| 98.09% WO AI @ 1.83% FPR (48,869 docs vs. 98.39% BF16); 50.66% Beemo AI @ 14.04% FPR

| Inkling-2-Mini (26.4B) Looped MoE, 128 exp.
| ClsBench classifier
| 4.17 GiB 8.58% BF16
| Mixed tiers + 48/128 experts + width 1,024 + 32,768 vocab
| 98.33% full-split exam recall (590/600 @ 5.33% FPR vs. 595/600 BF16; AUROC 0.9965)

| Inkling-2-Mini (26.4B) Looped MoE, 128 exp.
| BFCL tool calling
| 14.22 GiB 29.23% BF16
| Mixed tiers (128/128 experts, full width & vocab)
| 115/160 full-split exam (71.88% Gorilla vs. 114/160 BF16); select stress tests: BFCL-HIGH 53/79 tie, DeepMath-HIGH 41/48

| Inkling-2-Mini (26.4B) Looped MoE, 128 exp.
| DeepMath math
| 12.48 GiB 25.66% BF16
| Mixed tiers + 96/128 experts (25% MoE pruned)
| 81/96 full-split exam (84.38% vs. 84/96 BF16; 1.45x faster ); select stress test: high-16k 40/48 strict

| Qwen3.8-27B K34 / K94 27B · imported
| Terminal-Bench 2.1 Path B import
| 19.60 / 14.26 GB 35.87% / 26.09% F16
| chantless mask import : 130 groups + 42 / 38 tensor overrides
| 77.8% / 74.4% pass@1 (277/356, 265/356); 91.0% / 85.4% pass@4 (81/89, 76/89 · imported receipts)

Reading the denominators across Table 2: On the five stock non-MoE cells (Gemma 4 E2B, Qwen3-4B, Qwen3.5-4B), teacher-match evaluates exam items the full BF16 base answered correctly, and uniform quant baselines are reference points rather than strictly byte-matched files. Full-split sealed exams are reported for Inkling-2-Mini (ClsBench n=1,200, BFCL n=160, DeepMath n=96) and Qwen3.5-0.8B WildOutlines (n=48,869) / Beemo (n=4,549); high-effort Inkling-2-Mini evaluations (BFCL-HIGH n=79 valid from select80 and DeepMath-HIGH n=48 from select48) are select-split stress tests, listed separately from sealed exams. The smaller 200.30 MB Qwen3.5-0.8B mask.

## Teaching a model without training weights from feedback

So now that we have got the compact specialist out of the frozen model, our next job is to apply it to real work and see what areas it is still lacking in. Then we're going to try to fix these mistakes. Now, if this were your typical quantization or small-model distillation, just thinking about that would have been completely impossible from the beginning. But because of Chantless's architecture, it's possible. Now let's dive into how that works.

Chantless tackles this process with a very simple rule. We do not go for training new float weight values. Instead, we decide which small set of late-layer neurons we can switch off. When we look inside a model's feed-forward layers, we find that each hidden neuron multiplies two internal signals together, a gate value and an up value [ 45 ] , before passing its output forward. If we zero out either of those values, then the neuron goes completely quiet, and the remaining weights stay untouched. This is the trick that we will be applying.

Figure 5 Gated-MLP Neuron & Zero-Byte Switch

Figure 5 (Inside one neuron): How a zero-byte off switch silences a gated-MLP neuron (SwiGLU, GeGLU, or SiTU) in place. Zeroing a selected row's bytes in ffn_up and ffn_gate (or the corresponding MoE expert tensors) forces the multiplied activation a = 0.0 and the neuron's output y = 0.0 across supported ggml formats ( IQ1_S..BF16 ) without adding a single byte to the GGUF file. Gated-MLP architecture: [ 45 ]

Even better, flipping those switches adds zero bytes to the file. Inside a quantized GGUF model, weights live in compact blocks that share tiny scaling multipliers. Zeroing a row's block bytes in place (setting its scaling multipliers to zero in mvp/chantless/lattice.py ) silences that neuron without shifting anything on disk or attaching an external adapter, so the model binary stays the exact same size down to the byte.

In practice, you hand Chantless a small batch of up to 25 real mistakes. It optimizes binary on/off switch gates [ 33 , 35 ] on those late-layer rows while replay and A-GEM guardrails [ 28 , 29 , 30 , 31 ] constrain drift on the model's normal answers, then applies the chosen switches in place and checks whether learned switches beat four blind controls.

Here is what that looks like on two real models:

Inkling-2-Mini (26.4B looped MoE, ClsBench): Starting from the 4.17 GiB extracted specialist (masks/inkling2-mini-clsbench-mask.json, built from the architecture in the variable loop model card), switching off 424 of 6,144 late-layer rows across layers 18 to 23 (shared across the 48 retained experts) keeps the file at the exact same size and lifts sealed exam recall from 98.33% to 98.67% (592 of 600 positives at 5.00% FPR). On 60 unseen mistakes, learned switches fix 10 versus 4 for base, 4 for empty, 3 for flipped labels, and 8 for shuffled labels.

Gemma 4 E2B AI-text detector (930.97 MB mask): On a separate Beemo feedback experiment using a related zero-byte edit (clamping selected down-projection columns to the nearest representable zero on the existing quantization grid rather than zeroing entire up-projection row bytes, with each arm's decision threshold calibrated independently; 4,477 counted evaluation documents after excluding 25 feedback mistakes and 47 same-prompt documents from the 4,549-document pool), real-label switching fixed 241 / 1,528 unseen mistakes and broke 151 / 2,949 previously correct answers (net +90), compared with 36 / 1,528 fixed and 40 / 2,949 broken under flipped labels (net -4).

## Results Across Four Model Families

Chantless ships 16 masks across Gemma 4, Qwen3.5, Qwen3, Inkling-2-Mini, and two imported Qwen3.8-27B allocations. Across tasks, targeted mixed-precision and structural cuts beat uniform low-bit quantization cleanly at tight sizes.

Figure 6 Size versus quality across ClsBench, BFCL, and DeepMath / Terminal-Bench 2.1. In Panel A, compare the Inkling specialist square against the two uniform IQ1_S diamonds in the top-left corner.

Table 3 Extraction and zero-byte switching across four open model families. Summarizes all 16 shipped masks (6 passing specialist/apprentice pairs, Inkling-2-Mini BFCL and DeepMath specialists, and 2 imported Qwen3.8-27B allocations) plus the rejected Inkling-2-Mini BFCL switch run.

| Model
| Task
| Built GGUF (% Base)
| Specialist Score vs Uniform / BF16
| Apprentice Unseen Fixes (Controls: B/E/F/S)
| Retention After Patch

| Gemma 4 E2B
| ClsBench
| 928.7 MB (9.97%)
| 99.21% teacher-match (+49.78 pts vs IQ1_S)
| 15 / 60 (3 / 3 / 2 / 8)
| 99.15% to 99.40% (+0.25 pp)

| Qwen3.5-4B
| ClsBench
| 837.2 MB (9.94%)
| 93.40% teacher-match (+46.73 pts vs IQ1_S)
| 33 / 60 (4 / 4 / 5 / 20)
| 88.72% to 92.79% (+4.07 pp)

| Gemma 4 E2B
| BFCL
| 1,948.0 MB (20.92%)
| 97.13% teacher-match (+44.02 pts vs Q2_K)
| 26 / 60 (0 / 0 / 0 / 7)
| 99.21% to 83.33% (-15.88 pp)

| Qwen3-4B
| BFCL
| 1,972.2 MB (24.50%)
| 92.31% teacher-match (+5.87 pts vs Q3_K)
| 22 / 60 (3 / 3 / 1 / 12)
| 95.58% to 87.29% (-8.29 pp)

| Qwen3.5-4B
| BFCL
| 2,191.1 MB (26.01%)
| 92.39% teacher-match (+0.00 pts tie vs Q3_K)
| 29 / 60 (0 / 0 / 2 / 19)
| 93.96% to 83.22% (-10.74 pp)

| Inkling-2-Mini
| ClsBench
| 4.17 GiB (8.58%)
| 98.33% recall (590/600 vs 595/600 BF16)
| 10 / 60 (4 / 4 / 3 / 8)
| Exam 98.33% to 98.67% (val 96.71% to 95.89%)

| Inkling-2-Mini
| BFCL
| 14.22 GiB (29.23%)
| 115/160 vs 114/160 BF16 (high-effort 53/79 tie)
| 6 / 46 (3 / 3 / 6 / 5, rejected)
| Blocked by Flipped control

| Inkling-2-Mini
| DeepMath
| 12.48 GiB (25.66%)
| 81/96 vs 84/96 BF16 (1.45x faster, 25% MoE pruned)
| Not run
| 40/48 strict high-16k

| Qwen3.8-27B K34
| TB 2.1
| 19.60 GB (35.87%)
| 77.8% pass@1 (277/356) · imported
| Not run
| 81/89 pass@4 (91.0%)

| Qwen3.8-27B K94
| TB 2.1
| 14.26 GB (26.09%)
| 74.4% pass@1 (265/356) · imported
| Not run
| 76/89 pass@4 (85.4%)

## Chantless vs. Compact Trained Baselines (KD and SFT)

A natural engineering question is whether extracting a compact specialist from a frozen model beats training a small encoder or language model directly on your search-split labels using supervised fine-tuning (SFT) or Hinton knowledge distillation [ 32 ] (KD). On AI-text detection, we can compare both stages head-to-head against fixed-recipe compact baselines.

Table 4 Pure Stage 1 extraction vs. fixed-recipe compact baselines (AI-text detection, zero post-extraction switching or lessons). Each model uses its own threshold calibrated on the same 20,000 human documents. WildOutlines test holds 48,869 documents (24,295 AI, 24,574 human); held-out Beemo holds 4,549 documents (3,032 AI, 1,517 human). Sizes are decimal MB. Baseline model and dataset credits: [ 46 , 47 , 37 , 38 ]

| Evaluated Model / Stage
| Size (MB)
| WO AI Caught
| WO FPR
| Beemo AI
| Beemo FPR
| Beemo AUROC
| Winner / Caveat

| Qwen3.5-0.8B BF16 detector source
| 1,516.74 file
| 98.39%
| 1.89%
| 48.38%
| 12.59%
| N/A
| Reference frozen source (fine-tuned before extraction).

| Qwen Stage 1 specialist detector-cls396.gguf
| 396.45 file
| 98.09%
| 1.83%
| 50.66%
| 14.04%
| N/A
| Best verified compact WO recall (3.8x smaller); clears source-retention gate; beats KD point estimates on both Beemo rates.

| Qwen Stage 1 200 MB mask detector-200mb-q4cut50.gguf
| 200.30 file
| Not found
| Not found
| 50.4% rep.
| 14.2% rep.
| Not found
| Smallest extraction shown here (7.6x smaller); verified on select gate plus full Beemo log (no full WO exam receipt locally).

| ModernBERT-base Hinton KD Q8-style eval
| 158.99 est.
| 92.95%
| 1.64%
| 49.01%
| 17.01%
| 0.7129
| Best WO recall among compact trained baselines; uses teacher soft targets plus hard labels.

| ModernBERT-base SFT labels-only, Q8-style eval
| 158.99 est.
| 90.21%
| 1.58%
| 89.02%
| 75.21%
| 0.6774*
| Lowest WO FPR (virtual tie near 1.6%), but Beemo recall comes with 75.21% human FPR collapse.

| LFM2.5-230M-Base SFT labels-only, Q8-style eval
| 244.08 est.
| 88.00%
| 1.60%
| 91.66%
| 78.44%
| 0.7191*
| Highest Beemo AUROC among rows with saved scores, yet unusable at its shipped cutoff (78.44% human FPR).

Training exposure: The Qwen source detector was fine-tuned on 842,301 WildOutlines training documents prior to extraction; extraction then held those weights frozen. The three compact baselines trained on 14,991 documents (13,987 WildOutlines validation documents plus 1,004 Beemo documents grouped by prompt) using fixed single-seed recipes ( lr = 5e-5 , 5 passes with validation checkpoint selection; no LFM KD run was executed). This is a practical extraction-versus-compact-baseline comparison, not an equal-total-data comparison.

Size and score accounting: GGUF sizes are measured serialized files on disk. Baseline sizes are calculated packed-weight payload estimates (34 bytes per 32 matrix values plus 2 bytes per non-matrix parameter) from floating-point Q8-style quantize/dequantize evaluation; compact deployed binaries were not serialized. * Tie-aware AUROC recomputed from saved exam scores; N/A indicates AUROC was not recorded in that artifact's exam receipt.

Takeaway on pure extraction vs. compact training: Extracting a task subnetwork directly from a frozen model outperforms the fixed-recipe compact baselines on WildOutlines AI recall and avoids the severe Beemo false-positive collapse of both labels-only SFT runs without updating a single weight.

Table 5 Combining Stage 1 extraction with Stage 2/3 Apprentice overlays (shared OOD benchmarks). Layering neuron switches and targeted value updates onto the 200.30 MB Qwen detector mask ( combo.gguf ) yields a measured 200.30 MB binary compared below against the compact baselines on held-out Beemo (4,549 docs) and TextSight (394 passages: 56 AI, 338 human; literature subset excluded). Dataset sources: [ 38 , 48 ]

| Shared OOD Metric
| Qwen Apprentice Overlay
(200.30 MB file)
| ModernBERT KD
(158.99 MB est.)
| ModernBERT SFT
(158.99 MB est.)
| LFM2.5 SFT
(244.08 MB est.)

| Beemo AI caught / human FPR (4,549 docs)
| 46.01% / 13.05%
| 49.01% / 17.01%
| 89.02% / 75.21%
| 91.66% / 78.44%

| Beemo AUROC
| 0.7190
| 0.7129
| 0.6774
| 0.7191

| TextSight AUROC (394 passages)
| 0.98
| 0.96
| 0.92
| 0.82

| TextSight AI caught (fixed cutoff)
| 85.71% (48/56)
| 67.86% (38/56)
| 83.93% (47/56)
| 58.93% (33/56)

| TextSight human FPR (fixed cutoff)
| 2.66% (9/338)
| 0.30% (1/338)
| 17.75% (60/338)
| 17.16% (58/338)

| TextSight AI caught at matched in-set FPR (0.89%)
| 66.07% (37/56)
| 73.21% (41/56)
| 37.50% (21/56)
| 32.14% (18/56)

| TextSight Opus passages caught (fixed cutoff)
| 12/12
| 1/12
| 4/12
| 3/12

Operating-point and lesson notes: ModernBERT KD wins fixed-cutoff TextSight human FPR (0.30% vs. 2.66%) and aggregate recall when refitting a separate in-set threshold on TextSight humans (flagging 3/338, not a deployment cutoff), while the Qwen overlay leads overall TextSight AUROC (0.98 vs. 0.96), fixed-cutoff AI recall (85.71% vs. 67.86%), and Opus recall (12/12 fixed, and 11/12 even at the matched 0.89% in-set threshold vs. KD 1/12).

## Updating From 25 Mistakes Without Forgetting

Optimizing 6,144 binary switch logits from at most 25 field mistakes on a single 24 GB GPU creates two immediate hazards: small-batch updates can wreck unrelated skills, and backpropagating through a large model overflows VRAM.

To keep updates surgical inside mvp/chantless/learn.py , the trainer combines two guardrails:

Replay KL anchor ( replay_h0 ): snapshots the untouched specialist's output distribution on normal replay prompts and penalizes full-vocabulary float32 KL divergence whenever predictions drift. [ 30 , 31 , 32 ]

Orthogonal gradient projection (A-GEM): at each step, compares the lesson gradient against the replay-protection gradient; whenever their dot product is negative (meaning the new fix pulls against old skills), A-GEM subtracts away the conflicting component and steps with the projected lesson gradient plus the replay gradient. [ 28 , 29 ]

Figure 7 Extraction, Feedback Ablation & A-GEM Guardrail

Figure 7 (How the two stages fit together, illustrative): Stage 1 extracts a compact subnetwork that carries your task, while Stage 2 leaves specialist weights frozen and learns feedback-driven ablations on a small fraction of final switchable FFN rows. When a lesson gradient opposes replay protection ( dot(g_new, g_old) < 0 ), A-GEM removes that negative alignment in the lesson gradient before AdamW steps with g_proj + g_old .

Memory stays tight for the same structural reason. Because switches only live in the final 6 to 10 switchable FFN layers across shipped apprentices, the earlier layers never change during training. Streaming frozen prefix activations one layer at a time cuts peak CUDA allocation by 83.95% (from 3,072.8 MB down to 493.3 MB) with bit-identical outputs.

## What Comes Next

While Chantless ships one shot capabilities extraction and adapting to user feedback, there are some interesting line of sight features for us to work on.

Per-expert MoE switches and retention-safe generation: replacing shared-row zeroing with expert-local coordinates on routed models like Inkling-2-Mini, and closing the -8.3 to -15.9 pp retention drop on strict JSON tool calling.

Multi-round elastic updates: validating sequential update rounds where a mask tightens switches or restores dropped capacity from the frozen base before drift accumulates.

Targeted weight repair when switching saturates: validating low-rank weight patches in repair.py and staged schedules in anneal.py for tasks where pure binary switching reaches its ceiling. [ 49 ]

Multi-seed confidence intervals and semantic circuits: tightening narrow repair margins (like +2 cases on Inkling ClsBench) across repeated seeds and understanding what the 424 switched rows actually compute.

We want capabilities to become things that people can own, compose, exchange and improve, without carrying around the intelligence they have no use for.

Chantless code is available on GitHub . Released masks and model artifacts are hosted on Hugging Face .

## References

Prior research, related work, software, model authors and benchmark sources. Citations to model cards and datasets credit their creators; they do not imply those sources independently verified our experimental results.

Thinking Machines Lab (2026). Inkling 2 Mini — model card . See also TML-d24-varloop-research model card .

Andrew Lee et al. (2026). Interpreting Inkling 2 Mini .

Anna Bair et al. (2026). Compressed Sensing for Capability Localization in Large Language Models .

Maximilian S. Ernst et al. (2026). Distributed Sparse Interventions in Language Models .

Deniz Bayazit et al. (2023). Discovering Knowledge-Critical Subnetworks in Pretrained Language Models .

Nakyeong Yang et al. (2022). Task-specific Compression for Multi-task Language Models using Attribution-based Pruning .

Yirao Zhao et al. (2025). Pruning General Large Language Models into Customized Expert Models .

Waleed Reda, Abhinav Jangda, Krishna Chintalapudi (2025). How Many Parameters Does Your Task Really Need? Task Specific Pruning with LLM-Sieve .

Yuanhe Tian et al. (2025). Frustratingly Easy Task-aware Pruning for Large Language Models .

Steven Cao, Victor Sanh, Alexander M. Rush (2021). Low-Complexity Probing via Finding Subnetworks .

Omar Naim, Krish Sharma, Niyar R. Barman, Nicholas Asher (2026). TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination .

Jorge García-Carrasco, Alejandro Maté, Juan Trujillo (2025). Extracting Interpretable Task-Specific Circuits from Large Language Models for Faster Inference .

Leo Gao et al. (2025; ICML 2026). Weight-sparse transformers have interpretable circuits .

Aryaman Arora, Zhengxuan Wu, Jacob Steinhardt, Sarah Schwettmann (January 2026). Language Model Circuits Are Sparse in the Neuron Basis .

Abhishek Mishra (May 2026). Honey, I Shrunk the Circuits! .

Abhishek Mishra, Krishna Pagare (June 2026). PRISM: Unlocking Language Model Capability Extraction .

Nick Cammarata (2026). Neural Networks Are Already Clean .

Curt Tigges et al. (2026). Neurons as an Interpretable Basis .

Igor Shilov et al. (2026). Discovering Neuron Communities in Language Models .

Jonathan Frankle, Michael Carbin (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks .

Hattie Zhou, Janice Lan, Rosanne Liu, Jason Yosinski (2019). Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask .

Mingjie Sun et al. (2023). A Simple and Effective Pruning Approach for Large Language Models (Wanda) .

Elias Frantar, Dan Alistarh (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot .

Saleh Ashkboos et al. (2024). SliceGPT: Compress Large Language Models by Deleting Rows and Columns .

Ji Lin et al. (2024). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration .

Elias Frantar et al. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers .

Unsloth (2025). Dynamic 2.0 GGUFs .

Arslan Chaudhry et al. (2019). Efficient Lifelong Learning with A-GEM .

David Lopez-Paz, Marc’Aurelio Ranzato (2017). Gradient Episodic Memory for Continual Learning .

Pietro Buzzega et al. (2020). Dark Experience for General Continual Learning: a Strong, Simple Baseline .

Zhizhong Li, Derek Hoiem (2016). Learning without Forgetting .

Geoffrey Hinton, Oriol Vinyals, Jeff Dean (2015). Distilling the Knowledge in a Neural Network .

Arun Mallya, Dillon Davis, Svetlana Lazebnik (2018). Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights .

Arun Mallya, Svetlana Lazebnik (2018). PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning .

Yoshua Bengio, Nicholas Léonard, Aaron Courville (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation .

Xiang Zhang, Junbo Zhao, Yann LeCun (2015). Character-level Convolutional Networks for Text Classification (Yelp Polarity) .

Rishanth Rajendhran (2026). WildOutlines dataset .

Ekaterina Artemova et al. (2025). Beemo: Benchmark of Expert-edited Machine-generated Outputs .

Shishir G. Patil et al. (2025). The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models .

Zhiwei He et al. (2025). DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning .

ggml contributors. llama.cpp: importance matrix and quantization tooling . See also importance matrix ; quantization .

Google DeepMind (2026). Gemma 4 E2B — model card .

Qwen Team (2025–2026). Qwen3–Qwen3.8 — model cards . Qwen3-4B ; Qwen3.5-4B ; Qwen3.5-0.8B ; Qwen3.8-27B .

Harbor Framework. Terminal-Bench 2.1 .

Noam Shazeer (2020). GLU Variants Improve Transformer .

Benjamin Warner et al. (2024). ModernBERT: Smarter, Better, Faster, Longer .

Liquid AI (2026). LFM2.5-230M-Base — model card .

TextSight (2026). AI Text Detection Benchmark .

Edward J. Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models .
