---
title: "Rotate2Think: Geometric Priming via Orthogonal Rotation to Improve Language Model Reasoning"
authors: "Aditya Sharma, Christopher J. Pal, Amal Zouaq"
date: "2026-06-02"
arxiv_id: "2606.09873"
url: "https://arxiv.org/abs/2606.09873"
cluster: "activation_steering_transfer"
tags: ["procrustes", "orthogonal-rotation", "reasoning", "training-free", "priming"]
status: "FULL_TEXT"
source: "arxiv_html (48,029 chars)"
---

# Rotate2Think: Geometric Priming via Orthogonal Rotation to Improve Language Model Reasoning

- **Authors:** Aditya Sharma, Christopher J. Pal, Amal Zouaq
- **Published:** 2026-06-02 | **arXiv:** [2606.09873](https://arxiv.org/abs/2606.09873)
- **Cluster:** `activation_steering_transfer` | **Tags:** `procrustes`, `orthogonal-rotation`, `reasoning`, `training-free`, `priming`
- **Status:** `FULL_TEXT` (arxiv_html (48,029 chars))
- **Relevance to PlatonicSwap v1:** Training-free orthogonal Procrustes geometric priming for reasoning.
- **Relevance to PlatonicSwap v2:** Orthogonal rotation operators for reasoning traces.

## Abstract

Reasoning models achieve strong performance on challenging tasks by generating explicit intermediate reasoning traces before producing a final answer. Yet the internal structure of representation space when reasoning remains poorly understood: how do a model's hidden representations differ during thinking versus the embeddings of the input prompt, and can this structure be exploited to elicit stronger reasoning at inference time? We show that both input embeddings and thinking embeddings (mean-pooled last-layer hidden states over the prompt and reasoning trace, respectively) exhibit extremely high conicity, with all vectors clustering tightly around a single mean direction. Crucially, these mean input and thinking directions are non-collinear, with thinking embeddings occupying a geometrically distinct region of embedding space across many different models and benchmark tasks. This observation motivates casting the input-to-thinking transition as a rotation problem admitting a closed-form solution via orthogonal Procrustes analysis. We propose Rotate2Think, a training-free method that estimates this rotation from a small set of correctly solved examples and injects the resulting synthetic thinking vector between thinking delimiters at inference time, providing a geometric primer at the onset of the reasoning trace. Evaluated across multiple benchmarks and model families, Rotate2Think improves accuracy in 30 of 32 model-benchmark configurations across mathematics, science, and code tasks, and generalizes zero-shot to multimodal reasoning on MATH-Vision.

---

## Full Extracted Text (Searchable)

# Rotate2Think: Geometric Priming via Orthogonal Rotation
to Improve Language Model Reasoning

Aditya Sharma

Affiliation:  Polytechnique Montréal

Affiliation:  Mila - Quebec AI Institute

Affiliation:  LAMA-WeST Lab

Christopher J. Pal,
Amal Zouaq

Affiliation:  Polytechnique Montréal

Affiliation:  Mila - Quebec AI Institute

Affiliation:  LAMA-WeST Lab

Affiliation:  Canada CIFAR AI Chair
Correspondence: aditya.sharma@mila.quebec

Abstract

Reasoning models achieve strong performance on challenging tasks by generating explicit intermediate reasoning traces before producing a final answer. Yet the internal structure of representation space when reasoning remains poorly understood: how do a model’s hidden representations differ during thinking versus the embeddings of the input prompt, and can this structure be exploited to elicit stronger reasoning at inference time?
We show that both input embeddings and thinking embeddings (mean-pooled last-layer hidden states over the prompt and reasoning trace, respectively) exhibit extremely high conicity, with all vectors clustering tightly around a single mean direction. Crucially, these mean input and thinking directions are non-collinear, with thinking embeddings occupying a geometrically distinct region of embedding space across many different models and benchmark tasks. This observation motivates casting the input-to-thinking transition as a rotation problem admitting a closed-form solution via orthogonal Procrustes analysis. We propose Rotate2Think, a training-free method that estimates this rotation from a small set of correctly solved examples and injects the resulting synthetic thinking vector between thinking delimiters at inference time, providing a geometric primer at the onset of the reasoning trace.Evaluated across multiple benchmarks and model families, Rotate2Think improves accuracy in 30 of 32 model–benchmark configurations across mathematics, science, and code tasks, and generalizes zero-shot to multimodal reasoning on MATH-Vision.

## 1 Introduction

The advent of chain-of-thought prompting  ( Wei et al., 2022 ) and large-scale reinforcement learning-based post-training ( Shao et al., 2024 ; Ouyang et al., 2022 ) has given rise to a new class of reasoning models : language models that produce explicit intermediate reasoning traces, delimited by special thinking tokens, before generating a final answer  ( Jaech et al., 2024 ; Guo et al., 2025 ; Yang et al., 2025 ) . These models achieve strong performance on challenging benchmarks spanning mathematical olympiad problems, graduate-level science questions, and programming tasks precisely because they can allocate additional test-time compute to the problem at hand ( Snell et al., 2024 ) . Yet despite this empirical success, remarkably little is understood about what geometrically distinguishes a model’s hidden representations during thinking from those during direct generation.
If the onset of a reasoning trace shapes the trajectory that follows, then steering the model into a better initial representational state could improve reasoning quality without spending additional test-time compute.

This motivates three main research questions about the geometry of reasoning:
RQ1. How does the geometry of a model’s hidden representations evolve from the embeddings of the input prompt to those of the reasoning trace?
RQ2. If thinking corresponds to a specific geometric structure in representation space, can we prime the start of a reasoning trace, providing both reasoning and base models a stronger initial position from which they then reason fully?
RQ3. If the input-to-thinking geometry is a property of the model rather than of the input distribution, does a rotation fit on text reasoning transfer zero-shot to other modalities, such as visual mathematical reasoning?

We approach these questions empirically by extracting and comparing two types of mean-pooled last-layer embeddings for each problem in a reasoning benchmark: the input embedding e in ​ ( q ) e_{\mathrm{in}}(q) , obtained from a single pass over the user query without any generation, and the thinking embedding e th ​ ( q ) e_{\mathrm{th}}(q) , the mean-pooled hidden state over the <think> … </think> span of a model-generated reasoning trace. This allows us to directly characterize how representational geometry evolves from the prompt to the reasoning trace.
We discover a striking regularity in both embedding sets: they exhibit extremely high conicity   ( Chandrahas et al., 2018 ) , with all vectors clustering tightly around a single mean direction and average cosine similarity to that mean approaching 1.0 1.0 . This holds consistently across diverse model families and benchmark domains spanning mathematics, science, and code.
Crucially, with average cosine similarity of 0.66 0.66 across models and benchmarks tested, the mean direction of the input space ( μ in \mu_{\mathrm{in}} ) and that of the thinking space ( μ th \mu_{\mathrm{th}} ) are consistently misaligned relative to within-space alignment confirming that thinking embeddings occupy a geometrically distinct subspace, even as each space individually is nearly one-dimensional.
Together, these two observations, high within-space conicity and cross-space misalignment, reduce the input-to-thinking mapping to a rotation problem admitting a closed-form solution via orthogonal Procrustes analysis  ( Schönemann, 1966 ) .
We show this rotation can be estimated from a small set of correctly solved examples drawn from held-out datasets, requiring no examples from the target distribution. Further, the resulting transformation reconstructs thinking embeddings with high fidelity across all models and benchmarks tested (cosine similarity > 0.96 >0.96 , mean square error < 0.05 <0.05 ), validating both the geometric hypothesis and the cross-domain generality of the learned rotation.

We call this approach Rotate2Think .
At inference time, we apply the fitted rotation to the mean-pooled input embedding of a new query to produce a synthetic thinking vector e ^ th ​ ( q ) \hat{e}_{\mathrm{th}}(q) , which is injected into the model’s context as a single token position between the <think> and </think> delimiters via the model’s inputs_embeds interface. This geometrically primes the onset of the reasoning trace, providing the model a stronger initial representational position from which it can reason. Rotate2Think is entirely training-free: it requires only a small collection of correctly solved examples from held-out datasets to fit the rotation, which amounts to a single D × D D\times D matrix computed in seconds and generalizes to unseen benchmarks without any target-domain examples.
We make the following contributions:

1.

A geometric characterization of reasoning representations.
We show that input and thinking embeddings both form nearly one-dimensional cones with non-aligned axes across diverse model families and benchmarks, and that a single rotation maps between them with high fidelity (cosine similarity > 0.96 >0.96 , mean square error < 0.05 <0.05 across models and benchmarks tested).

2.

Rotate2Think, a training-free inference-time method.
We fit an orthogonal Procrustes rotation from correctly solved examples on held-out datasets and inject the resulting synthetic thinking vector between thinking delimiters, geometrically priming the onset of the reasoning trace without any target-domain data or model modification.

3.

Cross-domain generalization. A single rotation, fit on held-out datasets, improves accuracy in 30 of 32 model–benchmark configurations across mathematics, science, and code, spanning four model families from 4B to 31B parameters, with no target-domain examples or per-benchmark refitting.

4.

Zero-shot cross-modal transfer. Fit purely on text reasoning, the same rotation transfers zero-shot to visual mathematical reasoning on MATH-Vision (testmini). On Gemma-4 E4B, Base+R2T (38.1%) exceeds full reasoning-mode accuracy (34.9%) on a modality the rotation never observed, suggesting the input-to-thinking geometry is a property of the model, not the input distribution.

## 2 Related Work

Test-Time Reasoning and Scaling.
Chain-of-thought prompting  ( Wei et al., 2022 ) established that intermediate reasoning steps improve LLM performance, inspiring RL-trained reasoning models, o1  ( Jaech et al., 2024 ) , DeepSeek-R1  ( Guo et al., 2025 ) , Qwen3  ( Yang et al., 2025 ) , that generate extended thinking traces. Subsequent work has largely sought to reshape these traces: Snell et al. (2024) scale compute up via multi-response search and process verifiers; Muennighoff et al. (2025) manipulate trace length through budget forcing; Aghajohari et al. (2025) restructure the reasoning environment itself to achieve linear compute scaling.
All require either additional compute at inference or gradient updates to the target model.
Rotate2Think is orthogonal to these directions: operating on frozen models with no fine-tuning or additional inference cost, it geometrically primes the start of the reasoning trace with a single synthetic embedding. As these approaches manipulate the trace itself, Rotate2Think could naturally complement them by providing a better-positioned initial state; we leave such combinations for future work.

Geometry of Transformer Representations.
Rotate2Think rests on the empirical finding that both input and thinking embeddings exhibit extreme conicity; we use the Alignment-to-Mean and conicity metrics of Chandrahas et al. (2018) .
Ethayarajh (2019) first demonstrated that all layers of BERT, ELMo, and GPT-2 produce anisotropic cone-like representations; Rotate2Think’s observed conicity of ≈ 1.0 \approx 1.0 (see Figure  1 ) in last-layer means of reasoning models is the extreme end of this spectrum.
Gao et al. (2019) identify an underlying mechanism: auto-regressive LMs with weight tying collapse output embeddings into a narrow cone during training.
Razzhigaev et al. (2024) show that decoder anisotropy peaks in upper layers and varies systematically with architecture and training phase, consistent with the high conicity Rotate2Think measures at the final layer.
Prior work has proposed differentiable isotropy measures, finding that anisotropy can improve task performance  ( Rudman and Eickhoff, 2024 ) , while Tsukagoshi and Sasano (2025) demonstrate that the intrinsic dimensionality of text embeddings is low enough that 25% of dimensions suffice for most tasks, corroborating the effective rank-one geometry that justifies fitting Procrustes from only a handful of solved examples.
Karkada et al. (2026) show that translation symmetries in language co-occurrence statistics analytically predict manifold structure in model representations, providing a theoretical mechanism for why a systematic input-to-thinking rotation arises from training data structure.
The Platonic Representation Hypothesis  ( Huh et al., 2024 ) further suggests that representational geometry converges universally with model scale, supporting the generalizability of the learned rotation across problem domains and model families.
Finally, Sun et al. (2019) model knowledge-graph relations as rotations in complex vector space, establishing rotation as a natural and expressive transformation between semantic spaces; Rotate2Think instantiates this intuition in the real-valued mapping from input to thinking geometry in language models.

Representational Analysis and Steering.
Park et al. (2024) formalize the linear representation hypothesis, showing that high-level semantic properties are encoded as linear directions in the representation space, providing some theoretical basis for treating the thinking embedding as a meaningful direction toward which the input embedding can be rotated.
Representation Engineering ( Zou et al., 2023 ) extracts population-level concept directions and adds them to intermediate activations to monitor and control model behavior.
Turner et al. (2023) similarly construct contrastive steering vectors injected at chosen hidden layers.
Rotate2Think shares this spirit but differs structurally: rather than adding to activations at a manually chosen layer, we inject a synthesized vector as a full token position between the thinking delimiters, letting the complete transformer stack process it naturally.
Additionally, our focus is on the geometric structure specific to reasoning traces, and we intervene through orthogonal rotation rather than linear probing.
Supporting this design, Esakkiraja et al. (2026) show that LLMs encode action decisions in pre-generation activations with reasoning text serving as post-hoc rationalization, directly motivating the use of the input embedding as the seed for rotation into thinking space.

## 3 Preliminaries

Figure 1: Within-space conicity is high (avg. ≈ 0.93 \approx 0.93 ), indicating tight cones in both thinking and input spaces. Cross-space ISA is markedly lower (avg. ≈ 0.66 \approx 0.66 ), confirming that input and thinking cones are narrow but don’t share an axis, the structure Rotate2Think exploits.

### 3.1 Notation and Embedding Extraction

Let ℳ \mathcal{M} be a decoder-only transformer language model with L L layers and hidden dimension D D .
For an input token sequence 𝐱 = ( x 1 , … , x T ) \mathbf{x}=(x_{1},\ldots,x_{T}) , we write 𝐡 ( L ) ​ ( 𝐱 ) ∈ ℝ T × D \mathbf{h}^{(L)}(\mathbf{x})\in\mathbb{R}^{T\times D} for the matrix of final-layer hidden states, where 𝐡 t ( L ) ​ ( 𝐱 ) \mathbf{h}^{(L)}_{t}(\mathbf{x}) is the hidden state at position t t .

#### Input embeddings.

Given a query q q formatted as a chat-template prompt of T T tokens, the input embedding is the mean of the final-layer hidden states:

|
| e in ​ ( q ) = 1 T ​ ∑ t = 1 T 𝐡 t ( L ) ​ ( q ) ∈ ℝ D . e_{\mathrm{in}}(q)=\frac{1}{T}\sum_{t=1}^{T}\mathbf{h}^{(L)}_{t}(q)\;\in\;\mathbb{R}^{D}.
|
| (1)

Computing e in ​ ( q ) e_{\mathrm{in}}(q) requires only a single forward pass through ℳ \mathcal{M} ; no tokens are generated.

#### Thinking embeddings.

Reasoning models generate an explicit reasoning trace delimited by special tokens like <think> and </think> ( Yang et al., 2025 ; Guo et al., 2025 ) .
Given query q q , let 𝐫 = ( r 1 , … , r S ) \mathbf{r}=(r_{1},\ldots,r_{S}) denote the S S tokens generated within the thinking span.
The thinking embedding is the mean of the final-layer hidden states at the thinking positions:

|
| e th ​ ( q ) = 1 S ​ ∑ s = 1 S 𝐡 | q | + s ( L ) ​ ( [ 𝐪 ; r 1 , … , r S ] ) ∈ ℝ D , e_{\mathrm{th}}(q)=\frac{1}{S}\sum_{s=1}^{S}\mathbf{h}^{(L)}_{|q|+s}\!\left([\mathbf{q};\,r_{1},\ldots,r_{S}]\right)\;\in\;\mathbb{R}^{D},
|
| (2)

where [ 𝐪 ; r 1 , … , r S ] [\mathbf{q};\,r_{1},\ldots,r_{S}] is the concatenated prompt-plus-thinking sequence and | q | |q| is the number of prompt tokens.

Computing e th ​ ( q ) e_{\mathrm{th}}(q) uses a two-phase procedure:
(1) Generation phase. Run thinking-mode generation to produce the full response; record the token indices of the <think>/</think> boundaries in the generated sequence.
(2) Extraction phase. Run a single forward pass on the concatenated prompt-plus-thinking sequence to obtain 𝐡 ( L ) \mathbf{h}^{(L)} , then mean-pool over the thinking-span positions.

### 3.2 Conicity of Embedding Spaces

Chandrahas et al. (2018) introduced the notions of alignment to mean (ATM) and conicity to analyse the geometry of knowledge graph embedding spaces.
The alignment to mean of a single vector v v with respect to a set 𝒱 \mathcal{V} is its cosine similarity to the set mean, ATM ⁡ ( v , 𝒱 ) = cos ⁡ ( v , 𝝁 𝒱 ) \mathrm{ATM}(v,\mathcal{V})=\cos(v,\boldsymbol{\mu}_{\mathcal{V}}) .
Conicity is the average ATM over the set: given 𝒱 = { v i } i = 1 N ⊂ ℝ D \mathcal{V}=\{v_{i}\}_{i=1}^{N}\subset\mathbb{R}^{D} with mean 𝝁 𝒱 = 1 N ​ ∑ i v i \boldsymbol{\mu}_{\mathcal{V}}=\tfrac{1}{N}\sum_{i}v_{i} , the conicity of 𝒱 \mathcal{V} is

|
| Conicity ⁡ ( 𝒱 ) = 1 N ​ ∑ i = 1 N cos ⁡ ( v i , 𝝁 𝒱 ) , \mathrm{Conicity}(\mathcal{V})=\frac{1}{N}\sum_{i=1}^{N}\cos\!\left(v_{i},\;\boldsymbol{\mu}_{\mathcal{V}}\right),
|
| (3)

where cos ⁡ ( u , v ) = u ⊤ ​ v / ( ‖ u ‖ ​ ‖ v ‖ ) \cos(u,v)=u^{\top}v/(\|u\|\|v\|) .

Conicity ⁡ ( 𝒱 ) ∈ [ − 1 , 1 ] \mathrm{Conicity}(\mathcal{V})\in[-1,1] ; a value near 1 1 indicates that all vectors point approximately in the direction of the mean, the set occupies a narrow cone around 𝝁 𝒱 \boldsymbol{\mu}_{\mathcal{V}} rather than being spread across the sphere.

We also introduce a complementary metric, Inter-Space Alignment (ISA), defined as the cosine similarity between the centroids of two vector sets.
Formally, for sets 𝒫 \mathcal{P} and 𝒬 \mathcal{Q} we define ISA ⁡ ( 𝒫 , 𝒬 ) = cos ⁡ ( 𝝁 𝒫 , 𝝁 𝒬 ) \mathrm{ISA}(\mathcal{P},\mathcal{Q})=\cos(\boldsymbol{\mu}_{\mathcal{P}},\boldsymbol{\mu}_{\mathcal{Q}}) , which quantifies the directional agreement between their mean representations.
Low ISA values signify directional divergence between 𝒫 \mathcal{P} and 𝒬 \mathcal{Q} , suggesting that the aggregate representations of the two sets occupy distinct or unrelated regions of the latent space.

### 3.3 The Orthogonal Procrustes Problem

Given paired matrices 𝐀 = [ a 1 , … , a N ] ⊤ ∈ ℝ N × D \mathbf{A}=[a_{1},\ldots,a_{N}]^{\top}\in\mathbb{R}^{N\times D} and 𝐁 = [ b 1 , … , b N ] ⊤ ∈ ℝ N × D \mathbf{B}=[b_{1},\ldots,b_{N}]^{\top}\in\mathbb{R}^{N\times D} , the orthogonal Procrustes problem   ( Schönemann, 1966 ) seeks the orthogonal matrix 𝐖 ∈ ℝ D × D \mathbf{W}\in\mathbb{R}^{D\times D} (satisfying 𝐖 ⊤ ​ 𝐖 = 𝐈 D \mathbf{W}^{\top}\mathbf{W}=\mathbf{I}_{D} ) that minimises the Frobenius-norm reconstruction error:

|
| 𝐖 ∗ = argmin 𝐖 : 𝐖 ⊤ ​ 𝐖 = 𝐈 D ‖ 𝐁 − 𝐀𝐖 ⊤ ‖ F 2 . \mathbf{W}^{*}=\argmin_{\mathbf{W}:\;\mathbf{W}^{\top}\mathbf{W}=\mathbf{I}_{D}}\left\|\mathbf{B}-\mathbf{A}\mathbf{W}^{\top}\right\|_{F}^{2}.
|
| (4)

Schönemann (1966) showed that the closed-form solution is obtained via the SVD of the cross-covariance matrix 𝐌 = 𝐁 ⊤ ​ 𝐀 \mathbf{M}=\mathbf{B}^{\top}\mathbf{A} :

|
| 𝐌 = 𝐔 ​ 𝚺 ​ 𝐕 ⊤ ⟹ 𝐖 ∗ = 𝐔𝐕 ⊤ . \mathbf{M}=\mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^{\top}\;\implies\;\mathbf{W}^{*}=\mathbf{U}\mathbf{V}^{\top}.
|
| (5)

The solution 𝐖 ∗ \mathbf{W}^{*} is an orthogonal matrix.

Figure 2: Schematic of the Rotate2Think method. An orthogonal Procrustes rotation 𝐖 ∗ \mathbf{W}^{*} maps the mean-pooled input embedding e in ​ ( q ) e_{\mathrm{in}}(q) into thinking space. The resulting synthetic thinking vector e ^ th ​ ( q ) \hat{e}_{\mathrm{th}}(q) is injected between thinking delimiters.

#### Centering.

In practice, we mean-center both matrices before fitting.
Let 𝝁 A \boldsymbol{\mu}_{A} and 𝝁 B \boldsymbol{\mu}_{B} be the column means of 𝐀 \mathbf{A} and 𝐁 \mathbf{B} .
We apply Equation  5 to the centered matrices ( 𝐀 − 𝟏 ​ 𝝁 A ⊤ ) (\mathbf{A}-\mathbf{1}\boldsymbol{\mu}_{A}^{\top}) and ( 𝐁 − 𝟏 ​ 𝝁 B ⊤ ) (\mathbf{B}-\mathbf{1}\boldsymbol{\mu}_{B}^{\top}) , and account for the centroid shift at inference time:

|
| e ^ th ​ ( q ) = 𝐖 ∗ ​ ( e in ​ ( q ) − 𝝁 A ) + 𝝁 B . \hat{e}_{\mathrm{th}}(q)=\mathbf{W}^{*}\!\left(e_{\mathrm{in}}(q)-\boldsymbol{\mu}_{A}\right)+\boldsymbol{\mu}_{B}.
|
| (6)

## 4 Rotate2Think

Rotate2Think is a training-free, inference-time method that steers a language model toward its reasoning mode, with base models exhibiting ”thinking” patterns without RL-based post-training and post-trained reasoning models showing improved reasoning.
At the heart of the approach is the geometric observation from Section  3.2 and Figure  1 that both the input embedding space and the thinking embedding space are effectively one-dimensional cones, with distinct but consistently oriented mean directions.
This structure makes the mapping from one space to the other computable with very few examples and expressible as a single orthogonal rotation.

The method operates in two stages:

1.

Fitting stage (offline). Extract paired input and thinking embeddings for a set of training problems; fit an orthogonal Procrustes rotation 𝐖 ∗ \mathbf{W}^{*} on the correctly-answered subset.

2.

Inference stage (online). For each new query, compute a synthetic thinking vector via the fitted rotation, inject it between thinking delimiters using inputs_embeds , and generate the answer.

Figure  2 shows the schematic of Rotate2Think, while Algorithm  1 gives the full procedure, and the following subsections describe each component.

### 4.1 Rotation Fitting

#### Training example selection.

We fit the Procrustes rotation using only problems the model answered correctly in thinking mode, on the hypothesis that correct solutions yield thinking vectors representative of productive reasoning, while incorrect ones may introduce noise that corrupts the rotation estimate.
When fitting from K K datasets other than the one being tested, 𝒟 1 , … , 𝒟 K \mathcal{D}_{1},\ldots,\mathcal{D}_{K} , we sample up to 30 problems from each dataset and retain only those answered correctly in thinking mode.
Let 𝒞 k ⊆ 𝒟 k \mathcal{C}_{k}\subseteq\mathcal{D}_{k} denote this correct subset for dataset k k .
We directly stack the individual input and thinking embeddings across all datasets into the paired matrices:

|
| 𝐀 = [ e in ​ ( q i ) ] k = 1 , … , K i ∈ 𝒞 k ⊤ , 𝐁 = [ e th ​ ( q i ) ] k = 1 , … , K i ∈ 𝒞 k ⊤ , \mathbf{A}=\bigl[e_{\mathrm{in}}(q_{i})\bigr]_{\begin{subarray}{c}k=1,\ldots,K\\
i\in\mathcal{C}_{k}\end{subarray}}^{\top},\mathbf{B}=\bigl[e_{\mathrm{th}}(q_{i})\bigr]_{\begin{subarray}{c}k=1,\ldots,K\\
i\in\mathcal{C}_{k}\end{subarray}}^{\top},
|
| (7)

where 𝐀 , 𝐁 ∈ ℝ N × D \mathbf{A},\mathbf{B}\in\mathbb{R}^{N\times D} and N = ∑ k = 1 K | 𝒞 k | N=\sum_{k=1}^{K}|\mathcal{C}_{k}| is the total number of correctly answered examples across all source datasets.
Each row of 𝐀 \mathbf{A} and the corresponding row of 𝐁 \mathbf{B} form a paired ( e in , e th ) (e_{\mathrm{in}},e_{\mathrm{th}}) example used to fit the Procrustes rotation.

#### SVD fit.

We apply the centered Procrustes solution (Equations  4 – 5 ) to 𝐀 \mathbf{A} and 𝐁 \mathbf{B} , obtaining 𝐖 ∗ \mathbf{W}^{*} , 𝝁 A \boldsymbol{\mu}_{A} , and 𝝁 B \boldsymbol{\mu}_{B} .

### 4.2 Rotated Inference

#### Synthetic thinking vector.

Given a new test query q q , we first compute e in ​ ( q ) e_{\mathrm{in}}(q) via a forward pass, then apply the rotation fitted in Equation  6 on K K given datasets other than the one being evaluated to obtain the synthetic thinking vector e ^ th ​ ( q ) \hat{e}_{\mathrm{th}}(q) .
This is a single matrix-vector multiply and incurs negligible overhead compared to a model forward pass.

#### Context construction and answer generation.

We construct the inputs_embeds tensor ( 𝐄 ctx \mathbf{E}_{\mathrm{ctx}} ) for the generation call as a concatenation of four parts - 𝐄 ⁡ ( q p ​ r ​ o ​ m ​ p ​ t ) \mathbf{E}(q_{prompt}) , 𝐄 ⁡ ( <think>\n ) \mathbf{E}(\texttt{<think>\textbackslash n}) , e ^ th ​ ( q ) \hat{e}_{\mathrm{th}}(q) , and 𝐄 ⁡ ( \n</think> ) \mathbf{E}(\texttt{\textbackslash n</think>}) ,
where 𝐄 ⁡ ( ⋅ ) \mathbf{E}(\cdot) is the model’s token embedding lookup and e ^ th ​ ( q ) ∈ ℝ 1 × D \hat{e}_{\mathrm{th}}(q)\in\mathbb{R}^{1\times D} occupies a single token position.
The thinking delimiters surrounding the injected vector ensure that the positional context matches the format the model encounters during thinking-mode generation.
We generate the answer by calling model.generate(inputs_embeds= 𝐄 ctx \mathbf{E}_{\mathrm{ctx}} ) .

Algorithm 1 Rotate2Think: fitting and inference

1:

Model ℳ \mathcal{M} , query q q , pre-extracted paired embeddings { ( e in ​ ( q i ) , e th ​ ( q i ) ) } i = 1 N \{(e_{\mathrm{in}}(q_{i}),e_{\mathrm{th}}(q_{i}))\}_{i=1}^{N} with correctness labels { y i ∈ { 0 , 1 } } \{y_{i}\in\{0,1\}\} (up to 30 examples per dataset) for rotation dataset list 𝒟 1 , … , 𝒟 K \mathcal{D}_{1},\ldots,\mathcal{D}_{K} with 𝒞 k \mathcal{C}_{k} being the list of correct examples from dataset 𝒟 k \mathcal{D}_{k} for k = 1 , … , K k=1,\ldots,K

2:

Rotation parameters ( 𝐖 ∗ , 𝝁 A , 𝝁 B ) (\mathbf{W}^{*},\boldsymbol{\mu}_{A},\boldsymbol{\mu}_{B}) ; answer for test query q q

3:

– Offline: Rotation Fitting –

4:

𝐀 ← [ e in ​ ( q i ) ] k = 1 , … , K i ∈ 𝒞 k ⊤ \mathbf{A}\leftarrow\bigl[e_{\mathrm{in}}(q_{i})\bigr]_{\begin{subarray}{c}k=1,\ldots,K\\
i\in\mathcal{C}_{k}\end{subarray}}^{\top} ⊳ \triangleright Eq.  7

5:

𝐁 ← [ e th ​ ( q i ) ] k = 1 , … , K i ∈ 𝒞 k ⊤ \mathbf{B}\leftarrow\bigl[e_{\mathrm{th}}(q_{i})\bigr]_{\begin{subarray}{c}k=1,\ldots,K\\
i\in\mathcal{C}_{k}\end{subarray}}^{\top}
⊳ \triangleright Eq.  7

6:

𝝁 A ← mean ⁡ ( 𝐀 ) \boldsymbol{\mu}_{A}\leftarrow\mathrm{mean}(\mathbf{A}) ;  𝝁 B ← mean ⁡ ( 𝐁 ) \boldsymbol{\mu}_{B}\leftarrow\mathrm{mean}(\mathbf{B})

7:

𝐌 ← ( 𝐁 − 𝟏 ​ 𝝁 B ⊤ ) ⊤ ​ ( 𝐀 − 𝟏 ​ 𝝁 A ⊤ ) \mathbf{M}\leftarrow(\mathbf{B}-\mathbf{1}\boldsymbol{\mu}_{B}^{\top})^{\top}(\mathbf{A}-\mathbf{1}\boldsymbol{\mu}_{A}^{\top})

8:

𝐔 ​ 𝚺 ​ 𝐕 ⊤ ← SVD ⁡ ( 𝐌 ) \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}\leftarrow\mathrm{SVD}(\mathbf{M})

9:

𝐖 ∗ ← 𝐔𝐕 ⊤ \mathbf{W}^{*}\leftarrow\mathbf{U}\mathbf{V}^{\top} ⊳ \triangleright Eq.  5

10:

– Online: Inference for test query q q –

11:

e in ​ ( q ) ← e_{\mathrm{in}}(q)\leftarrow forward pass on prompt tokens ⊳ \triangleright no generation needed

12:

e ^ th ​ ( q ) ← 𝐖 ∗ ​ ( e in ​ ( q ) − 𝝁 A ) + 𝝁 B \hat{e}_{\mathrm{th}}(q)\leftarrow\mathbf{W}^{*}(e_{\mathrm{in}}(q)-\boldsymbol{\mu}_{A})+\boldsymbol{\mu}_{B} ⊳ \triangleright Eq.  6

13:

𝐄 ctx ← [ 𝐄 ⁡ ( q prompt ) ​ ∣ 𝐄 ⁡ ( <think>\n ) ∣ ​ e ^ th ​ ( q ) ∣ 𝐄 ⁡ ( \n</think> ) ] \mathbf{E}_{\mathrm{ctx}}\leftarrow[\mathbf{E}(q_{\mathrm{prompt}})\mid\mathbf{E}(\texttt{<think>\textbackslash n})\mid\hat{e}_{\mathrm{th}}(q)\mid\mathbf{E}(\texttt{\textbackslash n</think>})]

14:

return ℳ . generate ​ ( 𝐄 ctx ) \mathcal{M}.\texttt{generate}(\mathbf{E}_{\mathrm{ctx}})

## 5 Experimental Setup

### 5.1 Datasets

We evaluate on five benchmarks spanning mathematics, science, code, and visual-mathematics chosen to represent diverse reasoning domains.

#### AIME 2025 and AIME 2026.

The American Invitational Mathematics Examination (AIME) is a competition-level mathematics benchmark consisting of 30 integer-answer problems per year, drawn from number theory, algebra, combinatorics, and geometry.
Problems require multi-step reasoning and yield an integer answer in [ 0,999 ] [0,999] .
We use the official 2025 and 2026 sets, treating each year as a separate benchmark ( Balunovic et al., 2025 ) .

#### GPQA Diamond.

GPQA ( Rein et al., 2024 ) is a multiple-choice benchmark of graduate-level science questions authored by domain experts.
We evaluate on the Diamond split (198 questions spanning Biology, Chemistry, and Physics), the hardest subset, which requires deep subject-matter knowledge and eliminates most surface-level shortcuts.

#### BigCodeBench.

BigCodeBench ( Zhuo et al., 2025 ) is a function-level Python code generation benchmark with 1,140 diverse tasks covering library usage, data manipulation, and algorithm implementation.
Each task is evaluated by executing the generated code against a suite of unit tests.

#### MATH-Vision (testmini).

MATH-Vision ( Wang et al., 2025 ) is a multimodal mathematical problem-solving benchmark with 3040 visual reasoning problems in the full test set. We use the testmini split of the dataset containing 304 problems.

### 5.2 Baselines and Evaluation Protocol

We report two metrics for each condition: Acc (accuracy, %), and Tok (average number of tokens generated per problem).
For each model-benchmark pair, we evaluate four modes:

•

Base. Standard (non-thinking) generation: the model generates its answer directly without producing a reasoning trace.

•

Reasoning. Full thinking-mode generation: the model produces a complete reasoning trace before its answer.

•

Base+R2T. The rotation is fitted from the base model variant’s input and to the reasoning model’s thinking embeddings. During inference, the base model receives the synthetic thinking vector injected between thinking delimiters.

•

Reasoning+R2T. The rotation is fitted from the reasoning model’s embeddings. The same reasoning model is used for inference, with the synthetic thinking vector injected between thinking delimiters.

|
|
| Base
| Base+R2T
| Reasoning
| Reasoning+R2T

| Benchmark
| Model
| Acc %
| Tok
| Acc %
| Tok
| Acc %
| Tok
| Acc %
| Tok

| AIME 2025
| Qwen3 4B
| 43.33
| 10444
| 46.67
| 8299
| 70.00
| 20944
| 80.00
| 20894

| Gemma-4 E4B
| 36.67
| 3438
| 40.00
| 5408
| 40.00
| 6968
| 50.00
| 5891

| Phi-4 14B
| 16.67
| 1672
| 20.00
| 1639
| 73.33
| 15053
| 76.67
| 17108

| Gemma-4 31B
| 70.00
| 2927
| 73.33
| 4110
| 83.33
| 7170
| 86.67
| 4240

| AIME 2026
| Qwen3 4B
| 40.00
| 9359
| 53.33
| 8855
| 70.00
| 20609
| 73.33
| 21076

| Gemma-4 E4B
| 33.33
| 3691
| 46.67
| 5422
| 43.33
| 6829
| 53.33
| 5586

| Phi-4 14B
| 10.00
| 1380
| 16.67
| 1515
| 76.67
| 17077
| 80.00
| 17540

| Gemma-4 31B
| 80.00
| 2119
| 83.33
| 2036
| 83.33
| 7894
| 86.67
| 3098

| GPQA Diamond
| Qwen3 4B
| 46.67
| 737
| 48.48
| 549
| 67.17
| 8018
| 69.19
| 9126

| Gemma-4 E4B
| 55.07
| 1979
| 64.00
| 3254
| 58.67
| 3646
| 60.33
| 3342

| Phi-4 14B
| 54.55
| 658
| 56.57
| 746
| 69.19
| 11353
| 69.70
| 11569

| Gemma-4 31B
| 74.24
| 1284
| 78.28
| 1390
| 84.34
| 5728
| 86.87
| 5614

| BigCodeBench
| Qwen3 4B
| 38.24
| 409
| 38.51
| 404
| 41.58
| 5787
| 41.84
| 5670

| Gemma-4 E4B
| 36.67
| 608
| 38.38
| 1238
| 36.84
| 1227
| 37.98
| 1252

| Phi-4 14B
| 39.91
| 242
| 39.74
| 582
| 43.68
| 4172
| 44.03
| 3962

| Gemma-4 31B
| 46.49
| 364
| 47.54
| 1577
| 48.51
| 2198
| 48.25
| 2291

Table 1: Accuracy (%) and average tokens generated (Tok). Rotate2Think improves accuracy in 30 of 32 model–benchmark configurations, with comparable token budgets. R2T = Rotate2Think.

### 5.3 Models Evaluated

We evaluate four language models spanning three families and a range of parameter counts.

#### Qwen3 4B.

Qwen/Qwen3-4B-Instruct-2507 (which we use as base model) and Qwen/Qwen3-4B-Thinking-2507 (which we use as reasoning model) ( Yang et al., 2025 ) are 4B-parameter dense language models.

#### Phi-4 14B.

We use microsoft/phi-4 as the base model and microsoft/Phi-4-reasoning-plus as the reasoning model. These are 14B, dense models ( Abdin et al., 2024 ) .

#### Gemma-4 E4B and 31B.

We use google/gemma-4-E4B-it (8B MoE model) and google/gemma-4-31B-it (31B dense model) ( Google DeepMind, 2026 ) in our evaluation.

The reasoning versions of Qwen3 4B, and Phi-4 14B use <think> / </think> delimiters to bracket the reasoning trace.
Gemma-4 models, on the other hand, use the <|think|> flag to toggle between non-thinking (which we use as base model for Gemma-4) and thinking (which we use as reasoning model for Gemma-4) modes with model-specific thinking delimiters ( <|channel> / <channel|> ) and enforce an exclusive-or constraint between input_ids and inputs_embeds . We handle this via a forward hook that replaces a designated placeholder token’s embedding with the synthetic thinking vector during prefill.
We use the 4-bit quantized version of Gemma-4 31B model.

## 6 Results

Geometric structure of reasoning hidden states. Figure  1 reports conicity and ISA scores across all models and benchmarks.
Conicity is uniformly high across all conditions (0.87–0.99), confirming that reasoning model hidden states occupy a narrow cone in the representation space, which is a prerequisite for the low-rank geometric characterization that Rotate2Think exploits.
ISA values between input and thinking embeddings are substantially lower on average (0.659 base, 0.670 reasoning),
indicating that input and thinking cones are individually narrow but do not share an axis, motivating our Procrustes-based rotation.
Both properties hold consistently across benchmarks and model families, suggesting that high conicity and low ISA are stable geometric properties of current reasoning models.

Base → \to Base+Rotate2Think.
Injecting a synthetic thinking vector into base models (Base+R2T) yields consistent accuracy gains across every model and benchmark tested, as seen in Table  1 . AIME improvements reach up to + 13 +13 percentage points. On GPQA Diamond and AIME 2026, Gemma-4 E4B under Base+R2T surpasses the same model in full reasoning mode. These results suggest the geometric vector carries a genuine reasoning signal independent of the chain-of-thought length.
Beyond accuracy, we qualitatively observe that Qwen3-4B under Rotate2Think produces self-correcting phrases such as ”Wait…” and ”Let me reconsider…” — patterns characteristic of RL-trained reasoning models but absent in standard base-model generation. This behavior does not appear consistently across Phi-4 and Gemma-4, so we report it as an incidental observation rather than a general finding.

Reasoning → \to Reasoning+Rotate2Think.
As seen in Table  1 , augmenting reasoning models with the Rotate2Think primer yields consistent accuracy improvements across all models and benchmarks, with gains present regardless of model scale or task domain.
On AIME, improvements range from modest to substantial with Gemma-4 E4B gaining + 10 +10 percentage points on both AIME 2025 and AIME 2026, while Qwen3 4B improves by + 10 +10 and + 3.3 +3.3 points, respectively. GPQA Diamond and BigCodeBench follow the same trend, with almost all models registering gains under Reasoning+R2T. Notably, these improvements occur without a commensurate increase in token generation, suggesting that R2T does not prompt the model to reason more, but rather more effectively. The synthetic thinking vector appears to prime the model’s internal reasoning process, improving the quality of the chain-of-thought that follows.

Figure 3: Rotate2Think results on MATH-Vision (testmini) for Gemma-4 E4B. The rotation is fit on text benchmarks only and applied zero-shot to a multimodal benchmark. Base+R2T ( 38.1 38.1 ) exceeds Reasoning ( 34.9 34.9 ) and approaches Reasoning+R2T ( 37.8 37.8 ), described further in Section  6 .

Generalizing to Multimodal Reasoning.
To examine whether Rotate2Think transfers beyond language-only inputs, we evaluate on MATH-Vision (testmini) using Gemma-4 E4B, with the rotation matrix fit on 30 examples each from the four text benchmarks (AIME 2025, AIME 2026, GPQA Diamond, BigCodeBench) — without any multimodal data or modality-specific adaptation.
As shown in Figure  3 , Rotate2Think improves accuracy for both the base model and the reasoning model, consistent with the pattern observed on text benchmarks. That these gains carry over to a multimodal LLM (MLLM) without any modification suggests Rotate2Think may extend naturally to multimodal settings, broadening its practical applicability beyond language-only reasoning.

## 7 Discussion

Taken together, the results in Section  6 support a coherent picture of what Rotate2Think is doing.

The thinking embedding is a meaningful priming signal.
Across diverse model families and parameter scales, rotating the input embedding into thinking space and injecting a single token consistently steers the model toward stronger reasoning behavior on hard problems.
This is consistent with the geometric hypothesis in Section  3.2 : if both input and thinking embeddings form tight cones with non-aligned axes, then a single rotation can reliably shift the model’s context from one regime to the other.

Reasoning capacity may be latent and geometrically accessible.
Rotate2Think intervenes at a single point (one rotated embedding, one injected token) with no gradient updates, no additional training, and no modification to model weights.
That such a minimal perturbation produces consistent accuracy gains across both base and reasoning models hints that the injected thinking vector may provide a useful initialization of the reasoning process, steering the model toward a representational regime where reasoning is more reliably expressed.

Thinking-space geometry transcends input modality.
The transfer of Rotate2Think to MATH-Vision (testmini), with the rotation fit purely on text reasoning, suggests that the thinking-space geometry is a property of the model rather than of the input modality. That such a transfer occurs without any multimodal data or modality-specific refitting is consistent with the Platonic Representation Hypothesis ( Huh et al., 2024 ) , and hints that the learned rotation may capture a modality-agnostic signature of the model’s reasoning regime.

## 8 Conclusion

We introduced Rotate2Think, a training-free method that improves language model reasoning by rotating the mean-pooled input embedding into “thinking space” via orthogonal Procrustes analysis and injecting the result as a single token between thinking delimiters.
Our key geometric insight is that input and thinking embeddings both exhibit high conicity, i.e., they occupy narrow cones in representation space, but with systematically misaligned mean directions, reducing the cross-space mapping to a pure rotation.
Rotate2Think consistently improves accuracy for both base and reasoning models across benchmarks and LLMs evaluated.
Notably, Rotate2Think also extends to multimodal settings as evaluated on MATH-Vision (testmini), it improves reasoning accuracy in MLLMs without any modality-specific adaptation, suggesting that the thinking-space geometry exploited by Rotate2Think may not be exclusive to language-only pretraining, and could reflect a more general structural property of the representation space.

## Limitations

There are limitations of the current work that warrant acknowledgment.
First, Rotate2Think in its current form requires access to a reasoning-capable variant of the target model to extract thinking embeddings; it cannot be applied to model families for which no such variant exists.
Second, while gains are consistent across mathematical reasoning and scientific knowledge benchmarks, improvements on code generation tasks are modest, suggesting the method’s effectiveness is tied to task domains where extended deliberation is most consequential.
Third, all evaluations are conducted under a single-token injection scheme; whether more expressive interventions, such as injecting a sequence of rotated embeddings, for instance, could yield further gains is an open question.
Finally, Rotate2Think operates directly on hidden state representations and therefore requires white-box access to the model’s internals, specifically, the last hidden state of the input sequence. This precludes application to closed-source models, where only token-level outputs are accessible, and limits the method to open-weight models for which activation extraction is feasible.
We hope this perspective encourages further work at the intersection of representational geometry and inference-time reasoning, and that the lightweight nature of Rotate2Think makes it a practical building block for future reasoning systems.

## Acknowledgements

We thank NSERC, Samsung, IVADO, and the Canada First Research Excellence Fund for supporting this work and CIFAR for their support under the Canada CIFAR AI Chair program.

## References

Abdin et al. (2024)
M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan, S. Gunasekar, M. Harrison, R. J. Hewett, M. Javaheripi, P. Kauffmann, et al.

Phi-4 technical report .

arXiv preprint arXiv:2412.08905 .

External Links: Link

Cited by: §5.3 .

Aghajohari et al. (2025)
M. Aghajohari, K. Chitsaz, A. Kazemnejad, S. Chandar, A. Sordoni, A. Courville, and S. Reddy

The markovian thinker: architecture-agnostic linear scaling of reasoning .

arXiv preprint arXiv:2510.06557 .

Cited by: §2 .

Balunovic et al. (2025)
M. Balunovic, J. Dekoninck, I. Petrov, N. Jovanović, and M. Vechev

MathArena: evaluating llms on uncontaminated math competitions .

In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track ,

Cited by: §5.1 .

Chandrahas et al. (2018)
Chandrahas, A. Sharma, and P. Talukdar

Towards understanding the geometry of knowledge graph embeddings .

In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) ,

Melbourne, Australia , pp. 122–131 .

External Links: Link ,
Document

Cited by: §1 ,
§2 ,
§3.2 .

Esakkiraja et al. (2026)
E. Esakkiraja, S. Rajeswar, D. Akhiyarov, and R. Venkatesaramani

Therefore i am. i think .

arXiv preprint arXiv:2604.01202 .

Cited by: §2 .

Ethayarajh (2019)
K. Ethayarajh

How contextual are contextualized word representations? comparing the geometry of bert, elmo, and gpt-2 embeddings .

In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP) ,

pp. 55–65 .

Cited by: §2 .

Gao et al. (2019)
J. Gao, D. He, X. Tan, T. Qin, L. Wang, and T. Liu

Representation degeneration problem in training natural language generation models .

In International Conference on Learning Representations ,

Cited by: §2 .

Google DeepMind (2026)
Google DeepMind

Gemma 4 model card .

Note: https://ai.google.dev/gemma/docs/core/model_card_4

Cited by: §5.3 .

Guo et al. (2025)
D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al.

Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning .

arXiv preprint arXiv:2501.12948 .

Cited by: §1 ,
§2 ,
§3.1 .

Huh et al. (2024)
M. Huh, B. Cheung, T. Wang, and P. Isola

The platonic representation hypothesis .

arXiv preprint arXiv:2405.07987 .

Cited by: §2 ,
§7 .

Jaech et al. (2024)
A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al.

Openai o1 system card .

arXiv preprint arXiv:2412.16720 .

Cited by: §1 ,
§2 .

Karkada et al. (2026)
D. Karkada, D. J. Korchinski, A. Nava, M. Wyart, and Y. Bahri

Symmetry in language statistics shapes the geometry of model representations .

arXiv preprint arXiv:2602.15029 .

Cited by: §2 .

Muennighoff et al. (2025)
N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Candès, and T. B. Hashimoto

S1: simple test-time scaling .

In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing ,

pp. 20286–20332 .

Cited by: §2 .

Ouyang et al. (2022)
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al.

Training language models to follow instructions with human feedback .

Advances in neural information processing systems 35 , pp. 27730–27744 .

Cited by: §1 .

Park et al. (2024)
K. Park, Y. J. Choe, and V. Veitch

The linear representation hypothesis and the geometry of large language models .

In Proceedings of the 41st International Conference on Machine Learning ,

pp. 39643–39666 .

Cited by: §2 .

Razzhigaev et al. (2024)
A. Razzhigaev, M. Mikhalchuk, E. Goncharova, I. Oseledets, D. Dimitrov, and A. Kuznetsov

The shape of learning: anisotropy and intrinsic dimensions in transformer-based models .

In Findings of the Association for Computational Linguistics: EACL 2024 ,

pp. 868–874 .

Cited by: §2 .

Rein et al. (2024)
D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman

Gpqa: a graduate-level google-proof q&a benchmark .

In First conference on language modeling ,

Cited by: §5.1 .

Rudman and Eickhoff (2024)
W. Rudman and C. Eickhoff

Stable anisotropic regularization .

In International Conference on Learning Representations ,

Vol. 2024 , pp. 23594–23609 .

Cited by: §2 .

Schönemann (1966)
P. H. Schönemann

A generalized solution of the orthogonal procrustes problem .

Psychometrika 31 ( 1 ), pp. 1–10 .

Cited by: §1 ,
§3.3 ,
§3.3 .

Shao et al. (2024)
Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, et al.

Deepseekmath: pushing the limits of mathematical reasoning in open language models .

arXiv preprint arXiv:2402.03300 .

Cited by: §1 .

Snell et al. (2024)
C. Snell, J. Lee, K. Xu, and A. Kumar

Scaling llm test-time compute optimally can be more effective than scaling model parameters .

arXiv preprint arXiv:2408.03314 .

Cited by: §1 ,
§2 .

Sun et al. (2019)
Z. Sun, Z. Deng, J. Nie, and J. Tang

RotatE: knowledge graph embedding by relational rotation in complex space .

In International Conference on Learning Representations ,

Cited by: §2 .

Tsukagoshi and Sasano (2025)
H. Tsukagoshi and R. Sasano

Redundancy, isotropy, and intrinsic dimensionality of prompt-based text embeddings .

In Findings of the Association for Computational Linguistics: ACL 2025 ,

pp. 25915–25930 .

Cited by: §2 .

Turner et al. (2023)
A. M. Turner, L. Thiergart, G. Leech, D. Udell, J. J. Vazquez, U. Mini, and M. MacDiarmid

Steering language models with activation engineering .

arXiv preprint arXiv:2308.10248 .

Cited by: §2 .

Wang et al. (2025)
K. Wang, J. Pan, L. Wei, A. Zhou, W. Shi, Z. Lu, H. Xiao, Y. Yang, H. Ren, M. Zhan, et al.

Mathcoder-vl: bridging vision and code for enhanced multimodal mathematical reasoning .

In Findings of the Association for Computational Linguistics: ACL 2025 ,

pp. 2505–2534 .

Cited by: §5.1 .

Wei et al. (2022)
J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al.

Chain-of-thought prompting elicits reasoning in large language models .

Advances in neural information processing systems 35 , pp. 24824–24837 .

Cited by: §1 ,
§2 .

Yang et al. (2025)
A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al.

Qwen3 technical report .

arXiv preprint arXiv:2505.09388 .

Cited by: §1 ,
§2 ,
§3.1 ,
§5.3 .

Zhuo et al. (2025)
T. Y. Zhuo, M. C. Vu, J. Chim, H. Hu, W. Yu, R. Widyasari, I. N. B. Yusuf, H. Zhan, J. He, I. Paul, et al.

BigCodeBench: benchmarking code generation with diverse function calls and complex instructions .

In International Conference on Learning Representations 2025 ,

pp. 99488–99542 .

Cited by: §5.1 .

Zou et al. (2023)
A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A. Dombrowski, et al.

Representation engineering: a top-down approach to ai transparency .

arXiv preprint arXiv:2310.01405 .

Cited by: §2 .

## Appendix A Appendix: Hyperparameters

### A.1 Generation

All single-run experiments, except Phi-4 (base), use sampling. Sampling parameters follow each model family’s recommended settings from their respective huggingface model cards and are listed in Table  2 .

| Model
| temperature
| top_k
| top_p

| Gemma-4 E4B / 31B
| 1.0
| 64
| 0.95

| Qwen3 4B (base)
| 0.7
| 20
| 0.80

| Qwen3 4B (reasoning)
| 1.0
| 20
| 0.95

| Phi-4 14B (base)
| –
| –
| –

| Phi-4 14B (reasoning)
| 0.6
| 20
| 0.95

Table 2: Sampling hyperparameters used per model family.

### A.2 Rotation Fitting

The orthogonal Procrustes rotation 𝐖 ∗ \mathbf{W}^{*} is fitted offline using a deterministic SVD (no gradient descent, no learning rate). Fitting proceeds as follows:

•

At most 30 examples per dataset are considered ( MAX_ROTATION_SAMPLES = 30 ).

•

Only correctly-answered examples are retained.

•

The mean input embedding and mean thinking embedding are computed per dataset over the retained examples.

•

These per-dataset means are stacked and used to fit a single Procrustes rotation with centroid alignment.

For the multimodal transfer experiment (MATH-Vision), the rotation is fitted on 120 examples in total (4 text datasets × \times 30 samples each: AIME 2025, AIME 2026, GPQA Diamond, and BigCodeBench), and applied zero-shot to visual problems.

### A.3 Quantization

Gemma-4 31B is loaded in 4-bit (NF4 + double quantization, bfloat16 compute dtype) via bitsandbytes . All other models are loaded in bfloat16 without quantization.

### A.4 Embedding Extraction

Embeddings are obtained by mean-pooling the last-layer hidden states over all non-padding token positions. Procrustes fitting is performed in float64 ; injected synthetic thinking vectors are cast to bfloat16 for inference.
