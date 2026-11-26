---
title: "Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion"
authors: "Zhen Cheng, Hao-Bo Yang, Wan-Yi Huang, Jin-Long Li"
date: "2026-04-07"
arxiv_id: "2604.05688"
url: "https://arxiv.org/abs/2604.05688"
cluster: "weight_adapter_transfer"
tags: ["attention-editing", "architecture-conversion", "intermediate-supervision"]
status: "FULL_TEXT"
source: "arxiv_html (60,564 chars)"
---

# Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion

- **Authors:** Zhen Cheng, Hao-Bo Yang, Wan-Yi Huang, Jin-Long Li
- **Published:** 2026-04-07 | **arXiv:** [2604.05688](https://arxiv.org/abs/2604.05688)
- **Cluster:** `weight_adapter_transfer` | **Tags:** `attention-editing`, `architecture-conversion`, `intermediate-supervision`
- **Status:** `FULL_TEXT` (arxiv_html (60,564 chars))
- **Relevance to PlatonicSwap v1:** Background on editing/converting attention mechanisms in pretrained LLMs.
- **Relevance to PlatonicSwap v2:** Progressive intermediate activation supervision for cross-attention translation.

## Abstract

Key-Value (KV) cache memory and bandwidth increasingly dominate large language model inference cost in long-context and long-generation regimes. Architectures such as multi-head latent attention (MLA) and hybrid sliding-window attention (SWA) can alleviate this bound, but integrating them into existing models remains difficult. Prior methods impose fine-grained structural requirements on both source and target attention modules, which cannot meet the feasible requirement in practical deployment. We present Attention Editing, a practical framework for converting already-trained large language models (LLMs) with new attention architectures without re-pretraining from scratch. Attention editing replaces the original attention with a learnable target module and trains it using progressive distillation, consisting of (1) layer-wise teacher-forced optimization with intermediate activation supervision to prevent cold-start error accumulation, and (2) model-level distillation on next-token distributions, optionally regularized by weak feature matching. We instantiate the framework on two different target--MLA and GateSWA, a gated hybrid SWA design, and apply it to Qwen3-8B and Qwen3-30B-A3B. The resulting models maintain competitive performance while delivering substantial efficiency improvements, demonstrating that large-scale attention conversion is both feasible and robust. Notably, experiments are conducted on an Ascend 910B clusters, offering a practical training case study on domestic hardware.

---

## Full Extracted Text (Searchable)

# Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion

Zhen Cheng

Hao-Bo Yang

Wan-Yi Huang

Jin-Long Li

Affiliation:  China Merchants Bank Artificial Intelligence Laboratory

Affiliation:  {chengzhen1005, yanghaobo, huang_wanyi, lucida}@cmbchina.com

Abstract

Key–Value (KV) cache memory and bandwidth increasingly dominate large language model inference cost in long-context and long-generation regimes. Architectures such as multi-head latent attention (MLA) and hybrid sliding-window attention (SWA) can alleviate this bound, but integrating them into existing models remains difficult. Prior methods impose fine-grained structural requirements on both source and target attention modules, which cannot meet the feasible requirement in practical deployment.
We present Attention Editing , a practical framework for converting already-trained large language models (LLMs) with new attention architectures without re-pretraining from scratch. Attention editing replaces the original attention with a learnable target module and trains it using progressive distillation , consisting of (1) layer-wise teacher-forced optimization with intermediate activation supervision to prevent cold-start error accumulation, and (2) model-level distillation on next-token distributions, optionally regularized by weak feature matching. We instantiate the framework on two different targets—MLA and GateSWA , a gated hybrid SWA design, and apply it to Qwen3-8B and Qwen3-30B-A3B. The resulting models maintain competitive performance while delivering substantial efficiency improvements, demonstrating that large-scale attention conversion is both feasible and robust. Notably, experiments are conducted on an Ascend 910B clusters, offering a practical training case study on domestic hardware.

Technical Report

|

## 1 Introduction

Agentic large language model (LLM) applications, from tool-using assistants to end-to-end AGI agents, increasingly execute long interactive trajectories that interleave user instructions, intermediate reasoning, tool calls, and tool observations ( Yao et al., 2022 ; Schick et al., 2023 ; OpenAI, 2025 ) . Such workflows naturally push both the input context (conversation history, retrieved evidence, and tool outputs) and the output length (multi-step plans, long-form explanations, and iterative self-correction) upward. Meanwhile, recent open-weight reasoning and agent-capable models explicitly target these use cases and emphasize controllable “reasoning effort” or “thinking” behavior ( OpenAI, 2025 ; Yang and others, 2025 ) . In this regime, the inference bottleneck is increasingly memory- and bandwidth-bound: autoregressive decoding caches per-layer keys and values for all previous tokens, and the KV cache grows with sequence length and batch size ( Kwon et al., 2023 ) .

To alleviate KV-cache overhead, the community has developed a wide spectrum of efficient attention mechanisms. At the architecture level, Multi-head Latent Attention (MLA) compresses key/value states into a low-rank latent representation, substantially reducing cached memory while retaining expressiveness ( DeepSeek-AI, 2024a ; DeepSeek-AI, 2024b ; DeepSeek-AI, 2025 ; Team, 2025a ; Team, 2026a ) . Linear attention and its variants redesign attention to avoid quadratic scaling and can change the structure of state carried across long contexts ( Katharopoulos et al., 2020 ; Choromanski et al., 2021 ; Peng and others, 2023 ; Gu and Dao, 2023 ; Yang et al., 2025 ) . In parallel, sliding-window attention (SWA) restrict attention to a local window for most layers (often with periodic global layers), achieving favorable compute and cache scaling for long sequences ( Beltagy et al., 2020 ; OpenAI, 2025 ; LLM-Core Xiaomi, 2026 ; Team, 2026b ) . However, adopting a new architecture typically requires training from scratch. This has motivated post-hoc conversion methods such as TransMLA and MHA2MLA that aim to obtain MLA’s KV-cache benefits while reusing existing model weights ( Meng et al., 2025 ; Ji et al., 2025 ) .

However, existing approaches face three practical limitations for real-world applications. First , many conversions impose fine-grained structural requirements on both source and target attention modules, because they rely on matrix factorization / low-rank linear approximations (e.g., SVD-based projections) and attention-specific handling of positional encoding (e.g., RoPE variants) to initialize or constrain transformed weights ( Ji et al., 2025 ; Bercovich et al., 2025b ; Koike-Akino et al., 2026 ) . In practice, deployed model implementations are often flexible (e.g., mixing kernel variants, hybrid attention patterns, MoE blocks, and serving-optimized layouts), which can deviate from the assumptions required by a given conversion recipe. Second , conversions are frequently demonstrated on base model checkpoints and require re-running substantial post-training (SFT/RL) to recover chat/reasoning behavior, potentially discarding expensive gains from post-training stage ( Ouyang et al., 2022 ; Yang and others, 2025 ; Meng et al., 2025 ) . Third , successful recovery can heavily depend on training data quality and distribution: Both TransMLA and MHA2MLA adopt the SmolLM series as their primary backbone models, owing to the availability of its open-source pretraining corpus. Since modern LLMs are pretrained on proprietary mixtures, practitioners often cannot obtain original pre-training corpus, making data-efficient and distribution-robust editing essential.

To address these limitations, we introduce Attention Editing , shown in Figure 1 , a general framework for substantially converting the attention architecture of an already-trained LLM without training from scratch. Unlike prior recipes that depend on delicate weight surgery, we treat the target attention modules as learnable replacements : we allow most parameters inside the new attention modules to be randomly initialized and then trained efficiently via progressive distillation . Concretely, we propose a two-stage recipe. (i) Layer-wise teacher forcing trains each decoder layer using teacher-provided intermediate activations to avoid deep error accumulation at cold start, bringing the newly initialized attention weights to a strong working point (intermediate “hints” style supervision) ( Romero et al., 2014 ) . (ii) We then apply model-level distillation by matching teacher and student next-token distributions with KL divergence (knowledge distillation) ( Hinton et al., 2015 ) . We optionally add weak intermediate-feature matching as regularization.

Figure 1: Illustration of attention editing. Attention editing is a general framework for substantially modifying the attention architecture of an already-trained LLM without re-pretraining from scratch. It does not depend on delicate weight surgery, and treats the target attention modules as learnable replacements .

We validate Attention Editing by converting GQA → \rightarrow MLA on Qwen3-8B and Qwen3-30B-A3B ( Yang and others, 2025 ) . To demonstrate generality beyond MLA, we also edit GQA into Gate Sliding-Window Attention (GateSWA) : a tiny-window sliding window attention (SWA) hybrid inspired by combining GPT-OSS and Qwen3-Next ( OpenAI, 2025 ; Team, 2025b ) . Motivated by recent evidence that a simple gate applied to the attention output can mitigate attention sinks and improve stability, we remove explicit learnable sink biases/tokens and instead adopt an element-wise gate on the SWA output ( Xiao and others, 2023 ; Qiu et al., 2025 ) . We apply Attention Editing to convert GQA-based models into GateSWA-GQA using a 5:1 sliding-to-full schedule. Results indicate competitive quality with strong hardware-efficiency gains. All training runs are conducted on an Ascend 910B cluster, and our setup follows the growing evidence that large-scale model training on Ascend clusters is feasible in practice.

These two targets (MLA and GateSWA) jointly support our claims that attention can be substantially refactored post-training and that progressive distillation enables data-efficient learning from random-init modules. Our contributions can be summarized as follows:

•

We formalize Attention Editing : fundamental attention-architecture changes are feasible for already-trained LLMs, without the requirement of delicate weight surgery.

•

We propose progressive distillation as a robust, structure-insensitive, and data-efficient method for attention editing.

•

We introduce GateSWA , an efficient hybrid attention variant that replaces learnable sink mechanisms in hybrid SWA models with an element-wise gate.

•

We present a practical case study of large-model attention editing trained entirely on Ascend 910B, providing an actionable recipe for large-scale training on domestic hardware.

## 2 Related Works

#### Efficient attention architectures.

To reduce the KV-cache memory occupied by MHA, MQA ( Shazeer, 2019 ) and GQA ( Ainslie et al., 2023 ) are designed to share KV heads to reduce decoding bandwidth. Furthermore, a large body of work studies the trade-off between model quality and inference efficiency by redesigning the attention mechanism, especially to reduce KV-cache memory and bandwidth during autoregressive decoding. Representative directions include linear attention , which replaces softmax attention with kernelized or recurrent formulations to improve scaling with sequence length ( Katharopoulos et al., 2020 ; Choromanski et al., 2021 ; Peng and others, 2023 ; Gu and Dao, 2023 ; Yang et al., 2025 ; Zhang et al., 2025 ) ; Multi-head Latent Attention , which compresses key/value states into a low-rank latent representation and has been adopted in recent reasoning-oriented model families ( DeepSeek-AI, 2024a ; DeepSeek-AI, 2024b ; DeepSeek-AI, 2025 ) ; and sliding-window or hybrid attention , which restricts most layers to local attention while retaining periodic global layers for long-context modeling ( Beltagy et al., 2020 ; OpenAI, 2025 ; LLM-Core Xiaomi, 2026 ) . These approaches are complementary to other efficiency techniques like KV-cache quantization or system-level serving optimizations, which are important but orthogonal to the attention-architecture focus of this work.

#### Attention architecture conversion.

Since replacing the attention module usually requires expensive re-training, recent work has explored post-hoc conversion from one attention architecture form to another. In particular, TransMLA and MHA2MLA study conversion from GQA/MHA to MLA in order to inherit MLA’s KV-cache advantages while reusing existing model weights ( Meng et al., 2025 ; Ji et al., 2025 ) . Besides that, the other work from NVIDIA Nemotron treat attention configuration itself, reducing KV heads or modifying GQA layouts, as an important efficiency knob in deployment-oriented model design ( Bercovich et al., 2025b ; Bercovich et al., 2025a ) . Compared with these methods, our goal is different: rather than relying on attention-specific weight surgery or fine-grained structural assumptions, we view the target attention as a learnable replacement, enabling much greater architectural changes and making the approach applicable not only to base models but also to already post-trained chat or reasoning LLMs.

#### Knowledge distillation.

Knowledge distillation transfers behavior by matching teacher and student output distributions ( Hinton et al., 2015 ) , while intermediate-feature or hint-based distillation supervises hidden representations to stabilize training when the student architecture differs substantially from the teacher ( Romero et al., 2014 ) . For autoregressive language models, recent work further emphasizes on-policy or generation-aware distillation to reduce the mismatch between teacher-forced training and student-generated inference trajectories ( Agarwal et al., 2024 ; Gu and others, 2023 ) . Our method combines these lines into a progressive recipe: hidden-state distillation first brings randomly initialized replacement modules to a workable regime, and output-level distillation then refines next-token behavior.

## 3 Preliminary

### 3.1 Multi Head Softmax Attention

Let h t ∈ ℝ d h_{t}\in\mathbb{R}^{d} denote the input representation at position t t , n h n_{h} the number of attention heads, and d h d_{h} the per-head dimension. In standard multi-head attention (MHA) ( Vaswani et al., 2017 ) , we first compute

|
| q t \displaystyle q_{t}
| = W Q ​ h t , \displaystyle=W^{Q}h_{t},
|
| (1)

|
| k t \displaystyle k_{t}
| = W K ​ h t , \displaystyle=W^{K}h_{t},
|
| (2)

|
| v t \displaystyle v_{t}
| = W V ​ h t , \displaystyle=W^{V}h_{t},
|
| (3)

where q t , k t , v t ∈ ℝ n h ​ d h q_{t},k_{t},v_{t}\in\mathbb{R}^{n_{h}d_{h}} . We then split them into n h n_{h} heads, i.e.,

|
| [ q t , 1 ; q t , 2 ; ⋯ ; q t , n h ] \displaystyle[q_{t,1};q_{t,2};\cdots;q_{t,n_{h}}]
| = q t , \displaystyle=q_{t},
|
| (4)

|
| [ k t , 1 ; k t , 2 ; ⋯ ; k t , n h ] \displaystyle[k_{t,1};k_{t,2};\cdots;k_{t,n_{h}}]
| = k t , \displaystyle=k_{t},
|
| (5)

|
| [ v t , 1 ; v t , 2 ; ⋯ ; v t , n h ] \displaystyle[v_{t,1};v_{t,2};\cdots;v_{t,n_{h}}]
| = v t , \displaystyle=v_{t},
|
| (6)

with q t , i , k t , i , v t , i ∈ ℝ d h q_{t,i},k_{t,i},v_{t,i}\in\mathbb{R}^{d_{h}} . For causal self-attention, the output of the i i -th head at step t t is

|
| α t , i , j \displaystyle\alpha_{t,i,j}
| = exp ⁡ ( q t , i ⊤ ​ k j , i / d h ) ∑ s = 1 t exp ⁡ ( q t , i ⊤ ​ k s , i / d h ) , 1 ≤ j ≤ t , \displaystyle=\frac{\exp\!\left(q_{t,i}^{\top}k_{j,i}/\sqrt{d_{h}}\right)}{\sum_{s=1}^{t}\exp\!\left(q_{t,i}^{\top}k_{s,i}/\sqrt{d_{h}}\right)},\qquad 1\leq j\leq t,
|
| (7)

|
| o t , i \displaystyle o_{t,i}
| = ∑ j = 1 t α t , i , j ​ v j , i , \displaystyle=\sum_{j=1}^{t}\alpha_{t,i,j}v_{j,i},
|
| (8)

and the final attention output is obtained by concatenating all heads and applying the output projection,

|
| u t = W O ​ [ o t , 1 ; o t , 2 ; ⋯ ; o t , n h ] . \displaystyle u_{t}=W^{O}[o_{t,1};o_{t,2};\cdots;o_{t,n_{h}}].
|
| (9)

Equivalently, if we stack all positions into matrices Q i Q_{i} , K i K_{i} , and V i V_{i} for the i i -th head, then O i = softmax ⁡ ( Q i ​ K i ⊤ / d h ) ​ V i O_{i}=\mathrm{softmax}(Q_{i}K_{i}^{\top}/\sqrt{d_{h}})V_{i} , which matches the standard scaled dot-product attention form.

### 3.2 Efficient Attention Architectures

To reduce the decoding-time memory cost of KV-cache, several efficient attention variants modify the above computation while retaining the same head-level notation. We next review three efficient attention variants.

#### Multi-Head Latent Attention (MLA).

Instead of caching all per-token keys and values, MLA compresses them into a shared latent representation:

|
| c t K ​ V \displaystyle c_{t}^{KV}
| = W D ​ K ​ V ​ h t , \displaystyle=W^{DKV}h_{t},
|
| (10)

|
| k t C \displaystyle k_{t}^{C}
| = W U ​ K ​ c t K ​ V , \displaystyle=W^{UK}c_{t}^{KV},
|
| (11)

|
| v t C \displaystyle v_{t}^{C}
| = W U ​ V ​ c t K ​ V , \displaystyle=W^{UV}c_{t}^{KV},
|
| (12)

where c t K ​ V ∈ ℝ d c c_{t}^{KV}\in\mathbb{R}^{d_{c}} is a low-dimensional latent code with d c ≪ n h ​ d h d_{c}\ll n_{h}d_{h} . To remain compatible with rotary position embeddings, MLA further introduces decoupled positional components q t R q_{t}^{R} and k t R k_{t}^{R} , and computes attention with concatenated queries and keys, e.g., q t , i = [ q t , i C ; q t , i R ] q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}] and k j , i = [ k j , i C ; k j R ] k_{j,i}=[k_{j,i}^{C};k_{j}^{R}] . The key idea is to replace KV-cache of each token in the original head space by a shared low-rank latent cache, so that the per-token cache over ℓ \ell layers becomes ( d c + d h R ) ​ ℓ (d_{c}+d_{h}^{R})\ell rather than 2 ​ n h ​ d h ​ ℓ 2n_{h}d_{h}\ell .

#### Linear hybrid attention.

Linear attention replaces the growing prefix-wise KV cache with a fixed-size recurrent memory. For clarity, we write the update rule for a single head. A broad class of linear attention methods maintains a finite-state memory S t S_{t} instead of the full prefix KV cache:

|
| S t \displaystyle S_{t}
| = A t ​ S t − 1 + k t ​ v t ⊤ , \displaystyle=A_{t}S_{t-1}+k_{t}v_{t}^{\top},
|
| (13)

|
| o t \displaystyle o_{t}
| = S t ⊤ ​ q t , \displaystyle=S_{t}^{\top}q_{t},
|
| (14)

where A t A_{t} is a data-dependent or data-independent transition matrix that controls forgetting and gating. Different linear attention variants mainly differ in how A t A_{t} is parameterized.
In a hybrid architecture, linear-attention layers are interleaved with a subset of full-attention layers. The linear layers only maintain fixed-size recurrent states, whose memory cost does not grow with the context length, while only the full-attention layers keep token-wise KV caches. For example, Kimi Linear ( Zhang et al., 2025 ) instantiates this design by interleaving Kimi Delta Attention (KDA) and MLA, and delivers substantial reductions in KV-cache memory relative to full attention.

#### Sliding-window attention (SWA).

SWA restricts each query to a local window of size w w , namely

|
| o t , i SWA = ∑ j = max ⁡ ( 1 , t − w + 1 ) t α t , i , j ​ v j , i , \displaystyle o_{t,i}^{\mathrm{SWA}}=\sum_{j=\max(1,\,t-w+1)}^{t}\alpha_{t,i,j}v_{j,i},
|
| (15)

where the softmax normalization is performed only over the visible window. After the cache is filled, SWA requires only a rolling buffer of size O ⁡ ( 2 ​ w ​ n h ​ d h ) O(2wn_{h}d_{h}) per layer, instead of a cache that grows linearly with the full prefix length. A practical challenge in streaming SWA is the attention sink , namely that certain early positions absorb disproportionate attention mass even when they carry little semantic information ( Xiao and others, 2023 ; Gu et al., 2025 ) . To tackle the influence of attention sink, one remedy is to introduce a dedicated learnable sink token during training, so that excess attention mass is redirected to a parameterized placeholder rather than to ordinary context tokens. Another remedy is gated attention, which applies a query-dependent gate to the SDPA output, e.g.,

|
| o ~ t , i = g t , i ⊙ o t , i , g t , i = σ ⁡ ( W g ​ h t ) , \displaystyle\tilde{o}_{t,i}=g_{t,i}\odot o_{t,i},\qquad g_{t,i}=\sigma(W_{g}h_{t}),
|
| (16)

thereby suppressing query-irrelevant attention outputs. Recent evidence shows that such post-SDPA gating can mitigate attention sink and improve long-context extrapolation ( Qiu et al., 2025 ) .

In the experimental setup of this work, we ultimately select GateSWA and MLA as the target architectures. On the one hand, these attentions have relatively stable inference implementations in the community. On the other hand, they are better aligned with the Ascend clusters we employ, thereby facilitating more efficient experimentation.

## 4 Method: Progressive Distillation for Attention Editing

We consider attention editing , where the chat/reasoning LLM is converted into a new architecture by replacing the attention architecture, while keeping the remainder of the network as intact as possible. Let 𝒯 \mathcal{T} denote the original pretrained model (the teacher ) and 𝒮 \mathcal{S} the edited model (the student ). We write the student parameters as

|
| θ 𝒮 = θ keep ∪ θ edit , \displaystyle\theta^{\mathcal{S}}=\theta_{\mathrm{keep}}\cup\theta_{\mathrm{edit}},
|
| (17)

where θ keep \theta_{\mathrm{keep}} contains parameters that remain structurally compatible with the teacher and are copied from 𝒯 \mathcal{T} , and θ edit \theta_{\mathrm{edit}} contains parameters introduced by the edited attention modules. We intentionally initialize θ edit \theta_{\mathrm{edit}} at random, rather than deriving them from a carefully structured matrix decomposition of pretrained attention weights.

Figure 2: Illustration of the forward propogation in two stages of progressive distillation. (a) Block-wise teacher forcing distillation: the inputs to each layer comes from the original model; (b) Model-level distillation: each architecture maintain its own inputs. The gray shading indicates that these two modules can be merged into a single model and trained efficiently using sharding strategies such as FSDP or DeepSpeed.

This design choice distinguishes our setting from prior conversion methods such as TransMLA and MHA2MLA, which exploit strong structural constraints and explicitly construct initial MLA parameters from the original attention weights via low-rank factorization or related decompositions  ( Meng et al., 2025 ; Ji et al., 2025 ) . While such initialization is elegant and effective when the source and target attention forms are tightly aligned, it becomes increasingly restrictive when the architectural edit is large. Our goal is therefore different: we seek an optimization procedure that remains effective even when the edited attention differs substantially from the original one. Randomly initializing the edited attention parameters avoids imposing a brittle source-to-target correspondence and allows the new attention modules to be optimized more freely. At the same time, we still preserve structurally compatible parameters whenever possible. The exact retained weights is described in Section  5.1 .

### 4.1 Problem Setup

Let 𝒟 = { x ( i ) } i = 1 N \mathcal{D}=\{x^{(i)}\}_{i=1}^{N} denote the training corpus, where each sequence

|
| x ( i ) = ( x 1 ( i ) , … , x n i ( i ) ) \displaystyle x^{(i)}=(x^{(i)}_{1},\dots,x^{(i)}_{n_{i}})
|
| (18)

is presented in standard pre-training format. Unlike instruction-tuning distillation, our data do not contain an explicit prompt/answer split, and we therefore apply supervision to all valid next-token positions instead of restricting the loss to response spans only.

Let L L be the number of Transformer blocks, and let ℬ ⊆ { 1 , … , L } \mathcal{B}\subseteq\{1,\dots,L\} denote the set of layers whose attention modules are edited. For a token position t t and layer ℓ \ell , h t 𝒯 , ℓ h_{t}^{\mathcal{T},\ell} and h t 𝒮 , ℓ h_{t}^{\mathcal{S},\ell} are denoted as the hidden states of the teacher and student, respectively. We further denote by

|
| u t 𝒯 , ℓ , u t 𝒮 , ℓ ∈ ℝ d \displaystyle u_{t}^{\mathcal{T},\ell},\quad u_{t}^{\mathcal{S},\ell}\in\mathbb{R}^{d}
|
| (19)

the output of the attention branch after the output projection ( o_proj ) and before residual addition at layer ℓ \ell . This quantity is the basic intermediate representation used in our first training stage.

A natural way to recover the chat and reasoning ability of the teacher is to distill the student from the teacher by minimizing a token-level divergence between their output distributions, following standard knowledge distillation  ( Hinton et al., 2015 ) . However, in our setting, directly applying model-level distillation from the beginning is highly unstable. Since the edited attention parameters in shallow layers are randomly initialized, the student receives poor fine-grained signals, and the resulting representation error accumulates rapidly with depth. In practice, this causes the optimization of output-level distillation to stall. To address this issue, we propose progressive distillation : we first align intermediate representations under a block-wise teacher-forcing regime to obtain a reliable initialization, and only then transition to end-to-end distillation on the model outputs.

### 4.2 Progressive Distillation

Our training procedure consists of two stages:

1.

Block-wise teacher-forcing distillation. Each edited attention block is trained independently using the teacher’s hidden state as input, so that early errors from an under-trained student block do not propagate to deeper layers.

2.

Model-level distillation. After the edited blocks acquire a meaningful initialization, the entire student is trained end-to-end using token-level knowledge distillation, optionally augmented with a small intermediate-state similarity loss.

The key idea is to move from local matching to global matching. Stage I turns the difficult problem of optimizing a randomly initialized edited network into a collection of well-conditioned local regression problems. Stage II then restores full autoregressive behavior by matching the teacher’s output distribution under the student’s own forward dynamics, as shown in Figure 2 .

#### Stage I: Block-Wise Teacher-Forcing Distillation.

For each edited layer ℓ ∈ ℬ \ell\in\mathcal{B} , we construct a block-wise training problem in which the input to the edited student attention block is clamped to the teacher hidden state. Concretely, let

|
| H 𝒯 , ℓ − 1 = [ h 1 𝒯 , ℓ − 1 ; … ; h n 𝒯 , ℓ − 1 ] ∈ ℝ n × d \displaystyle H^{\mathcal{T},\ell-1}=[h_{1}^{\mathcal{T},\ell-1};\dots;h_{n}^{\mathcal{T},\ell-1}]\in\mathbb{R}^{n\times d}
|
| (20)

be the teacher representation entering layer ℓ \ell . We then feed H 𝒯 , ℓ − 1 H^{\mathcal{T},\ell-1} into the edited attention block of the student and obtain

|
| U ^ 𝒮 , ℓ = 𝒜 ℓ 𝒮 ​ ( H 𝒯 , ℓ − 1 ) , \displaystyle\widehat{U}^{\mathcal{S},\ell}=\mathcal{A}^{\mathcal{S}}_{\ell}(H^{\mathcal{T},\ell-1}),
|
| (21)

where 𝒜 ℓ 𝒮 \mathcal{A}^{\mathcal{S}}_{\ell} denotes the edited attention branch up to and including o_proj . The corresponding target is the teacher attention-branch output

|
| U 𝒯 , ℓ = 𝒜 ℓ 𝒯 ​ ( H 𝒯 , ℓ − 1 ) . \displaystyle U^{\mathcal{T},\ell}=\mathcal{A}^{\mathcal{T}}_{\ell}(H^{\mathcal{T},\ell-1}).
|
| (22)

We supervise the edited block by matching these post- o_proj activations, using Mean Square Error (MSE) loss. To improve numerical stability, we normalize the MSE loss before computing the regression loss as ( Bercovich et al., 2025b ) . The block-wise loss for layer ℓ \ell is

|
| ℒ blk ( ℓ ) = ‖ U ^ 𝒮 , ℓ − U 𝒯 , ℓ ‖ F 2 ‖ U 𝒯 , ℓ ‖ F 2 + ϵ . \displaystyle\mathcal{L}_{\mathrm{blk}}^{(\ell)}=\frac{\left\|\widehat{U}^{\mathcal{S},\ell}-U^{\mathcal{T},\ell}\right\|_{F}^{2}}{\left\|U^{\mathcal{T},\ell}\right\|_{F}^{2}+\epsilon}.
|
| (23)

In practice, we optimize each edited layer independently. That is, when training layer ℓ \ell , the supervision signal is always computed from the teacher input H 𝒯 , ℓ − 1 H^{\mathcal{T},\ell-1} rather than from representations produced by previously edited student layers. This block-wise teacher-forcing strategy removes the main source of optimization failure in direct distillation: poor signals from shallow edited blocks can no longer cascade through the network and corrupt deeper targets.

The choice of the post- o_proj output as the regression target is deliberate. First, it is the quantity that is directly injected into the residual stream, and therefore most immediately controls the downstream hidden-state trajectory. Second, unlike attention logits or head-wise Q ​ K ⊤ QK^{\top} statistics, it remains well defined even when the source and target attention parameterizations differ substantially. This makes it a robust architecture-agnostic intermediate target for attention editing.

#### Stage II: Model-Level Distillation.

At this stage, the student is sufficiently well initialized for output-level distillation to become effective. Let z t 𝒯 z_{t}^{\mathcal{T}} and z t 𝒮 z_{t}^{\mathcal{S}} denote the teacher and student logits at position t t , respectively. We use the standard token-level distillation objective  ( Hinton et al., 2015 )

|
| p t 𝒯 \displaystyle p_{t}^{\mathcal{T}}
| = softmax ⁡ ( z t 𝒯 / τ ) , \displaystyle=\mathrm{softmax}\!\left(z_{t}^{\mathcal{T}}/\tau\right),
|
| (24)

|
| p t 𝒮 \displaystyle p_{t}^{\mathcal{S}}
| = softmax ⁡ ( z t 𝒮 / τ ) , \displaystyle=\mathrm{softmax}\!\left(z_{t}^{\mathcal{S}}/\tau\right),
|
| (25)

and define

|
| ℒ KD = τ 2 ∑ i = 1 N n i ∑ i = 1 N ∑ t = 1 n i KL ( p i , t 𝒯 ∥ p i , t 𝒮 ) . \displaystyle\mathcal{L}_{\mathrm{KD}}=\frac{\tau^{2}}{\sum_{i=1}^{N}n_{i}}\sum_{i=1}^{N}\sum_{t=1}^{n_{i}}\mathrm{KL}\!\left(p_{i,t}^{\mathcal{T}}\,\|\,p_{i,t}^{\mathcal{S}}\right).
|
| (26)

Here τ > 0 \tau>0 is the distillation temperature. Since our corpus is in pre-training format, the loss in Eq. ( 26 ) is evaluated on all non-padding positions, rather than only on an answer segment.

Empirically, we find that optimization becomes more reliable when a small intermediate-state regularizer is added on top of token-level distillation. Following distillation practices used in the Nemotron/Minitron line  ( Bercovich et al., 2025a ) , we introduce a low-weight cosine similarity loss on a selected set of intermediate layers ℳ ⊆ { 1 , … , L } \mathcal{M}\subseteq\{1,\dots,L\} :

|
| ℒ cos = 1 ∑ i = 1 N n i ​ | ℳ | ​ ∑ i = 1 N ∑ t = 1 n i ∑ ℓ ∈ ℳ ( 1 − ⟨ h i , t 𝒮 , ℓ , h i , t 𝒯 , ℓ ⟩ ‖ h i , t 𝒮 , ℓ ‖ 2 ​ ‖ h i , t 𝒯 , ℓ ‖ 2 ) . \displaystyle\mathcal{L}_{\mathrm{cos}}=\frac{1}{\sum_{i=1}^{N}n_{i}|\mathcal{M}|}\sum_{i=1}^{N}\sum_{t=1}^{n_{i}}\sum_{\ell\in\mathcal{M}}\left(1-\frac{\left\langle h_{i,t}^{\mathcal{S},\ell},\,h_{i,t}^{\mathcal{T},\ell}\right\rangle}{\|h_{i,t}^{\mathcal{S},\ell}\|_{2}\,\|h_{i,t}^{\mathcal{T},\ell}\|_{2}}\right).
|
| (27)

The final model-level objective is

|
| ℒ model = ℒ KD + λ cos ​ ℒ cos , \displaystyle\mathcal{L}_{\mathrm{model}}=\mathcal{L}_{\mathrm{KD}}+\lambda_{\mathrm{cos}}\mathcal{L}_{\mathrm{cos}},
|
| (28)

where λ cos \lambda_{\mathrm{cos}} is a small coefficient. Intuitively, ℒ KD \mathcal{L}_{\mathrm{KD}} restores the teacher’s predictive behavior at the token level, while ℒ cos \mathcal{L}_{\mathrm{cos}} provides a weak geometric regularization on the internal representation trajectory, which is especially helpful during the early phase of end-to-end training.

#### Discussions.

The overall procedure can be viewed as a progressive path from intermediate-state distillation to output-level distillation . Stage I stabilizes optimization by solving local, feature matching problem for the edited attention blocks. Stage II then restores the student as a coherent autoregressive model under its own hidden-state dynamics. This design is particularly suitable for attention editing because the architectural perturbation is localized yet potentially severe: randomly initialized edited attention blocks are difficult to optimize from logits alone, but become trainable once they are first anchored to teacher-provided intermediate signals. Our training pipeline is inspired by that of Llama-Nemotron Bercovich et al. (2025a) . Although both approaches adopt distillation from the block level to the model level, several notable differences remain. Our objective is to introduce fundamental modifications to the attention architecture, and we retrain nearly all attention parameters from scratch, resulting in greater generality. In contrast, Llama-Nemotron prunes the existing number of heads in GQA, without making substantive structural changes.

## 5 Implementation Details

### 5.1 Instantiation with MLA and GateSWA

We instantiate our attention editing pipeline on Qwen3-8B and Qwen3-30B-A3B ( Yang and others, 2025 ) , which together provide a representative pair of open-weight hybrid-thinking backbones. This choice allows us to evaluate the proposed training procedure under both dense and mixture-of-experts settings while keeping the post-training objective unchanged. One principle in all edited variants is that we always keep the original output projection ( o_proj ) unchanged, which we find could perform better. To enable exact reuse of o_proj (i.e., W O W^{O} ), we require the concatenated attention output is shape-compatible with the inherited o_proj . All remaining attention parameters, unless otherwise stated, are randomly initialized and subsequently optimized by the progressive distillation procedure described in Section  4 .

Table 1: Comparison (%) of KV cache per token among GQA, MLA, and GateSWA. We follow the notations in DeepSeek-V2 ( DeepSeek-AI, 2024a ) . Here the approximation ignores the bounded local-window ( w = 128 w=128 ) cache in SWA layers and only accounts for the unbounded cache carried by full-attention layers. Under a 5:1 sliding-to-full schedule, one out of every six layers is a full-attention layer.

|     Attention Mechanisms
|     KV cache per token
|    KV memory

|    Qwen3-8B
|    Qwen3-30B-A3B

|    GQA
|     2 ​ n g ​ d h ​ l 2n_{g}d_{h}l
|     100 % 100\%
|     100 % 100\%

|    MLA
|     ( d c + d h R ) ​ l (d_{c}+d_{h}^{R})l
|    28%
|    56%

|    GateSWA
|     ≈ 2 ​ n g ​ d h ⋅ l 6 \approx 2n_{g}d_{h}\cdot\frac{l}{6}
|    17%
|    17%

#### MLA instantiation.

We adopt a hardware-aware configuration tailored to FlashMLA kernels ( Li and Liu, 2025 ) . Let d c d_{c} denote the KV compression dimension and d r d_{r} the RoPE dimension used in the key branch. We set d c = 512 d_{c}=512 , d r = 64 d_{r}=64 .
This choice is motivated by practical compatibility with FlashMLA kernels, whose decode path can be viewed as an MQA-style computation with a single KV head, key width 576 576 , and value width 512 512 . We therefore obtain an efficient MQA decode path with n q n_{q} query heads and one shared KV head, while retaining the expressive advantages of multi-head queries.

Unlike the original MLA parameterization, we discard the query down-projection bottleneck, and preserve the original number of query heads from Qwen3. The rationale is to avoid unnecessarily constraining query expressiveness after a substantial architectural edit. In other words, we compress K/V aggressively for decoding efficiency, but keep Q relatively unconstrained so that the edited attention can better recover the representational capacity of the teacher. In addition, to further improve inference speed, we set the non-positional key dimension to d k NoPE = 64 d_{k}^{\mathrm{NoPE}}=64 .
Together, these choices yield an MLA configuration that is both kernel-friendly and sufficiently expressive: KV states are aggressively compressed for decoding, the query pathway remains high-capacity, and the inherited o_proj can still be reused exactly through the dimensionality constraint.

#### GateSWA instantiation.

Most full-attention layers are replaced by sliding-window attention, and we introduce an element-wise output gate ( Qiu et al., 2025 ) for each attention layer. Specifically, let o t ℓ ∈ ℝ d o_{t}^{\ell}\in\mathbb{R}^{d} denote the SDPA output of the attention branch at position t t and layer ℓ \ell , similar to notations in Section 3 . We apply a learned gate

|
| g t ℓ = σ ⁡ ( W g ℓ ​ h t ℓ − 1 ) , g t ℓ ∈ ℝ d , \displaystyle g_{t}^{\ell}=\sigma(W_{g}^{\ell}h_{t}^{\ell-1}),\qquad g_{t}^{\ell}\in\mathbb{R}^{d},
|
| (29)

and define the gated output as

|
| o ~ t ℓ = g t ℓ ⊙ o t ℓ , \displaystyle\widetilde{o}_{t}^{\ell}=g_{t}^{\ell}\odot o_{t}^{\ell},
|
| (30)

where σ ⁡ ( ⋅ ) \sigma(\cdot) is the element-wise sigmoid function and ⊙ \odot denotes element-wise multiplication. This gate is intentionally lightweight, yet it introduces additional nonlinearity at the attention output and eliminates attention sink. Aside from this gating operation, all attention-related architectural settings are kept identical to the original Qwen3 backbone.

For the sliding-window component, we use a window size of w = 128 w=128 , following the same local-attention regime adopted in GPT-OSS and MiMo-V2-Flash ( OpenAI, 2025 ; LLM-Core Xiaomi, 2026 ) . We further interleave sliding-window and full-attention layers with a ratio of SWA : Full = 5 : 1 \text{SWA}:\text{Full}=5:1 . Besides that, we keep setting the first layer to full attention. Under the window size of 128, the overall KV cache is approximately reduced to one-sixth of that of the original model. The comparison of KV cache is listed in Table 1 .

### 5.2 Data Construction

We emphasize that our distillation setup differs from the common <prompt, answer> pair used in instruction distillation. All training data are in pre-training format, and our objective does not isolate an answer span. Consequently, both the block-wise objective in Eq. ( 23 ) and the model-level objective in Eq. ( 28 ) are defined over the full sequence, making the procedure naturally compatible with continual pre-training or architecture conversion scenarios.

Table 2: Data mix in stage I.

| Data source
| Percentage

| General domain
| 40%

| Math & code
| 35%

| Chinese corpora
| 25%

In Stages I and II, we employed entirely different dataset configurations. In Stage I, since the attention weights were randomly initialized, lower-difficulty data were more suitable; accordingly, we used a large amount of general-domain data. In Stage II, we progressively incorporated a greater proportion of complex data related to mathematics, coding, and reasoning. These data were primarily drawn from YiZhao ( HITsz-TMG, 2026 ) and community open-source datasets such as ( Li and others, 2024 ) , FineWeb ( Penedo and others, 2024 ) , Nemotron-CC ( Su and others, 2024 ) , MegaMath ( Zhou and others, 2025 ) , and StarCoder2 ( Lozhkov and others, 2024 ) . It is worth noting that, because the full pretraining corpus of Qwen3 is not publicly available ( Yang and others, 2025 ) , the data we use may differ substantially from the true distribution of its original pretraining data. This further demonstrates the robustness of our method to variations in data quality. Our training data do not overlap with the following test set.

Figure 3: Data mix in stage II. At this stage, curriculum learning was employed to control the data composition, gradually increasing the proportion of reasoning-related data while reducing that of general domains. During training, the optimizer state keeps preserved across changes in the data mixture.

In Stage I, the vast majority of the data consisted of general-domain content, and the overall proportions are shown in Table 2 . Our data categories mainly include general-domain, mathematics and code, and Chinese corpora. These relatively simple data were used to enable the model to optimize smoothly from random initialization. At this stage, we used 2B tokens and trained with a sequence length of 8K.

In Stage II, the dataset still comprised the same four categories as in Stage I. In our experiments, we found that if we directly used a larger proportion of more difficult corpora, such as complex mathematics, code, and reasoning data, the model was often difficult to optimize. Conversely, using a larger amount of easier data leads to inferior model performance. We therefore adopt a curriculum learning strategy, in which three groups of data with progressively increasing difficulty are concatenated, as illustrated in Figure 3 , with the proportion of more challenging data gradually increasing. During training, when switching between datasets, the optimizer states are preserved. In this stage, we use a total of 6B tokens, arranged as three consecutive segments of 2B tokens each, as illustrated in Figure 3 . It is noteworthy that the amount of training data used in our study is less than one-thousandth of that used for Qwen3.

## 6 Experimental Results

In this section, we present the experimental results, focusing on factors such as model performance, inference speed, and improvements in hardware utilization efficiency. Since the weights of the modified attention modules are randomly initialized, we first verify that the proposed method can recover the base-model performance, mainly through a set of few-shot evaluations of fundamental capabilities. In addition, because our target model is a hybrid reasoning model, we also report several chat- and reasoning-related benchmarks to assess its effectiveness. Finally, we provide a detailed analysis of the inference acceleration achieved by the modified architecture.

### 6.1 Model Performance

Table 3: Performance (%) of different attention variants under few-shot evaluation. The model is required to directly output the answer, and accuracy is computed by matching the predicted answer against the ground truth.

|     Models
|    ARC-E
|    ARC-C
|    C-Eval
|    MMLU

|    Qwen3-8B
|    96.46
|    91.46
|    77.04
|    74.47

|    Qwen3-8B-GateSWA
|    96.54
|    90.35
|    76.82
|    73.26

|    Qwen3-8B-MLA
|    96.71
|    90.01
|    73.70
|    71.77

|    Qwen3-30B-A3B
|    96.89
|    92.58
|    83.06
|    78.68

|    Qwen3-30B-A3B-GateSWA
|    98.19
|    91.98
|    81.20
|    77.24

|    Qwen3-30B-A3B-MLA
|    98.15
|    93.77
|    81.05
|    76.99

#### Pre-training evaluation.

To validate that the attention editing has recovered the basic capacity, we first test the few-shot performance. We conduct evaluation in a few-shot setting, requiring the model to directly output one token as the answer (e.g., one of “A”, “B”, “C”, or “D”), which is then extracted and matched against the ground truth ( Zheng and others, 2024 ) . Since nearly all weights in the attention modules are randomly initialized, the model is initially unable to answer any questions. This evaluation therefore serves to verify whether the model parameters have been effectively optimized, particularly with respect to its memory-related capabilities. Concretely, the few-shot performance benchmarks include ARC-asy, ARC-challenge ( Clark and others, 2018 ) , MMLU ( Hendrycks and others, 2021 ) , and C-Eval ( Huang and others, 2023 ) . The results are listed in Table 3 .

As shown in Table 3 , the fundamental capabilities are well preserved, particularly on relatively simple memory-related benchmarks. This demonstrates the effectiveness of the progressive distillation method proposed in this paper, indicating that it can indeed transform one attention architecture into another. For example, on ARC-Easy, Qwen3-8B-MLA outperforms the original Qwen3-8B by 0.25%. This is a rather interesting phenomenon, and we speculate that the following explanation may account for it: prior works ( Geva and others, 2021 ; Dai and others, 2022 ; Meng and others, 2022 ; Niu and others, 2024 ) suggest that the attention module primarily serves as a mechanism for knowledge retrieval, whereas the FFN module is regarded as the locus of knowledge storage. Since we leave all FFN parameters frozen, the model’s knowledge storage remains intact, and only the retrieval functionality needs to be relearned. Gated attention is often considered to enhance nonlinearity and mitigate the attention sink phenomenon ( Qiu et al., 2025 ) , which may in some cases improve retrieval capability. MLA, in contrast, is generally regarded as having stronger expressive capacity than GQA, and may therefore also yield performance gains in certain settings ( DeepSeek-AI, 2024a ; DeepSeek-AI, 2024b ) .

#### Post-training evaluation.

In this part, we evaluate the model’s chat and reasoning capabilities, noting that these are abilities not possessed by the previously converted models. We aim to examine whether the method proposed in this paper can recover, to some extent, the model’s capacity for chat and even reasoning. Specifically, we primarily adopt GSM8K ( Cobbe and others, 2021 ) , a mathematics benchmark of moderate difficulty, and C-Eval ( Huang and others, 2023 ) , a more diverse general-domain benchmark covering a broad range of subjects. In addition, we report results under both thinking-enabled modes.

Table 4: Accuracy (%) of different attention-variant models on GSM8K and C-Eval in thinking-mode.

| Models
|

GSM8K

(thinking)

|

C-Eval

(thinking)

| Qwen3-8B
| 95.98
| 83.80

| Qwen3-8B-GateSWA
| 94.31
| 78.90

| Qwen3-8B-MLA
| 95.00
| 79.20

| Qwen3-30B
| 96.44
| 86.40

| Qwen3-30B-A3B-GateSWA
| 95.15
| 81.28

| Qwen3-30B-A3B-MLA
| 94.39
| 81.13

The results are listed in Table 4 , and we adopt AISBench 1 1
1

https://github.com/AISBench/benchmark tool to conduct evaluation. To our surprise, on the GSM8K task, the model with the modified attention architecture is able to achieve an accuracy remarkably close to that of the original 8B model. Qwen3 series underwent 36 trillion tokens of pre-training, along with highly sophisticated post-training , whereas our method attains comparable performance using fewer than 10B tokens . Moreover, the training data we use are expected to differ substantially in distribution from those used for Qwen3. Although GSM8K is not considered as a challenging benchmark by current standards, these results nevertheless provide clear evidence for the success of our attention editing approach. On a general knowledge benchmark such as C-Eval, the converted model exhibits a certain performance gap. We attribute this to the fact that knowledge-oriented tasks depend more directly on the model’s internal knowledge, whereas mathematical tasks are more amenable to compensation through chain-of-thought reasoning.

### 6.2 Inference Speedup

In the previous part, we showed that progressive distillation enables the converted models to recover strong general-domain and reasoning capabilities. We now turn to evaluate whether attention conversion also delivers the intended hardware-level inference benefits . Our goal in this part is to verify that, beyond maintaining model performance, the edited attention architectures substantially improve deployment efficiency.

Figure 4: Performance under different concurrency levels for three input lengths. The upper row shows the variation in throughput , while the lower row shows the variation in TTFT. The experimental setup is as follows: inference is conducted on a single H800 GPU using vLLM version 0.16.0, with gpu-memory-utilization set to 0.8. Missing data in the figure indicate that the model is unable to handle that level of concurrency simultaneously.

Figure 5: Throughput under different concurrency levels. The experimental setup is as follows: inference is conducted on a single Ascend 910B using vLLM/vLLM-Ascend version 0.16.0. Missing data in the figure indicate that the model is unable to handle that level of concurrency simultaneously. The results about Qwen3-8B-MLA are missing, because the MLA kernels in higher-version vLLM-Ascend impose customized requirements on specific parameter configuration.

#### Evaluation protocol.

We evaluate inference performance under a controlled decoding setup and report two standard serving metrics: time to first token (TTFT) and output throughput . TTFT measures the elapsed time between the arrival of a request and the emission of its first output token, and therefore captures the latency perceived by the user before generation starts. Output throughput measures the total number of generated output tokens per unit time across all served requests, and reflects the steady-state decoding efficiency of the serving system.

These two metrics are complementary: TTFT emphasizes responsiveness, whereas output throughput emphasizes sustained generation capacity under load. We benchmark the original and converted models under three input lengths, namely 3K/8K/16K tokens, while fixing the output length to 1K tokens. For each input length, we vary the request concurrency and record both TTFT and output throughput.

#### KV-cache reduction.

The memory benefit of attention conversion is summarized in Table  1 . For example, the converted models Qwen3-8B-GateSWA reduce KV-cache usage by approximately 80 % 80\% relative to the original full-attention backbone, which is a substantial reduction. From a systems perspective, such a drop in KV-cache footprint directly increases the effective memory budget available for active requests and long contexts. In practice, this means that the same hardware can either accommodate more cached tokens per request or serve more concurrent requests before hitting memory limits, both of which translate into lower serving cost and improved deployment flexibility.

#### Decoding throughput and TTFT.

The throughput and TTFT results are shown in Figure  4 - 5 . The advantage of inference speed becomes increasingly pronounced as concurrency grows. This trend is consistent with the intended systems effect of attention conversion. When concurrency is low, the GPU is less constrained by KV-cache residency and memory traffic, so the benefit of reducing the attention-state footprint is present but limited. As concurrency increases, however, the decoding system must maintain more active sequences simultaneously, and KV-cache pressure becomes increasingly dominant. In this regime, the converted models enjoy a large advantage because they require substantially less memory per request and incur lower cache-management overhead.

For TTFT, performance depends more heavily on input tokens processed during the prefill stage and is typically compute-bound. In the TTFT plot, the trends of MLA and the original model largely overlap. For MLA, the prefill stage follows an MHA-style computation, and although its FLOPs are lower than those of an MHA model of comparable size, they are not necessarily lower than those of GQA. Consequently, MLA does not exhibit a clear advantage in TTFT. By contrast, GateSWA substantially reduces computational cost through the sliding-window mechanism, while the additional computation introduced by the gating operation is relatively minor, leading to a more noticeable improvement in TTFT, which is consistent with the tendency shown in Figure 4 .

These results show that the proposed attention conversion could preserve model capability after distillation, and it also yields a meaningful systems-level payoff. The converted models substantially reduce KV-cache consumption and, as a consequence, deliver higher decoding throughput and better TTFT. The gain is especially evident when concurrency is high, which is precisely the operating regime where deployment cost and serving efficiency matter most.

## 7 Conclusion and Future Work

In this paper, we present Attention Editing, a general framework for converting the attention architecture of a trained LLM without re-pretraining from scratch. By treating the target attention as a learnable replacement, and training it with progressive distillation, our method avoids delicate architecture-specific weight surgery and enables substantial post hoc attention refactoring. Experiments on converting GQA-based Qwen3 models to both MLA and GateSWA show that competitive quality can be retained while improving efficiency. In addition, all experiments were conducted entirely on an Ascend 910B cluster, providing a practical case study of large-model post-training on domestic hardware.

Although some current results are promising, several important questions remain open. First, the amount of training data used in our study is less than one-thousandth of that used for Qwen3 , so it remains unclear how Attention Editing will scale with substantially larger and more diverse data. Understanding whether further scaling can yield stronger recovery and better downstream efficiency-quality trade-offs is an especially interesting direction. Second, our current training mixture contains relatively limited a gent-oriented data , and our evaluation focuses more on chat and reasoning abilities than on fully interactive agent performance. Extending attention editing toward stronger tool-use and long-horizon agent capabilities is therefore a natural next step.

## References

Agarwal et al. (2024)
R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos, M. Geist, and O. Bachem

On-policy distillation of language models: learning from self-generated mistakes .

In International Conference on Learning Representations ,

Cited by: §2 .

Ainslie et al. (2023)
J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai

GQA: training generalized multi-query transformer models from multi-head checkpoints .

arXiv preprint arXiv:2305.13245 .

Cited by: §2 .

Beltagy et al. (2020)
I. Beltagy, M. E. Peters, and A. Cohan

Longformer: the long-document transformer .

arXiv preprint arXiv:2004.05150 .

Cited by: §1 ,
§2 .

Bercovich et al. (2025a)
A. Bercovich, I. Levy, I. Golan, et al.

Llama-nemotron: efficient reasoning models .

arXiv preprint arXiv:2505.00949 .

Cited by: §2 ,
§4.2 ,
§4.2 .

Bercovich et al. (2025b)
A. Bercovich, T. Ronen, T. Abramovich, et al.

Puzzle: distillation-based nas for inference-optimized llms .

In International Conference on Machine Learning ,

Cited by: §1 ,
§2 ,
§4.2 .

Choromanski et al. (2021)
K. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane, T. Sarlos, P. Hawkins, J. Davis, A. Mohiuddin, L. Kaiser, D. Belanger, L. Colwell, and A. Weller

Rethinking attention with performers .

In International Conference on Learning Representations ,

Cited by: §1 ,
§2 .

Clark et al. (2018)
P. Clark et al.

Think you have solved question answering? try arc, the ai2 reasoning challenge .

arXiv preprint arXiv:1803.05457 .

Cited by: §6.1 .

Cobbe et al. (2021)
K. Cobbe et al.

Training verifiers to solve math word problems .

arXiv preprint arXiv:2110.14168 .

Cited by: §6.1 .

Dai et al. (2022)
D. Dai et al.

Knowledge neurons in pretrained transformers .

In Proceedings of the Annual Meeting of the Association for Computational Linguistics ,

Cited by: §6.1 .

DeepSeek-AI (2024a)
DeepSeek-AI

DeepSeek-v2: a strong, economical, and efficient mixture-of-experts language model .

External Links: 2405.04434

Cited by: §1 ,
§2 ,
Table 1 ,
§6.1 .

DeepSeek-AI (2024b)
DeepSeek-AI

DeepSeek-v3 technical report .

arXiv preprint arXiv:2412.19437 .

Cited by: §1 ,
§2 ,
§6.1 .

DeepSeek-AI (2025)
DeepSeek-AI

DeepSeek-r1: incentivizing reasoning capability in llms via reinforcement learning .

arXiv preprint arXiv:2501.12948 .

Cited by: §1 ,
§2 .

Geva et al. (2021)
M. Geva et al.

Transformer feed-forward layers are key-value memories .

In Proceedings of the Conference on Empirical Methods in Natural Language Processing ,

Cited by: §6.1 .

Gu and Dao (2023)
A. Gu and T. Dao

Mamba: linear-time sequence modeling with selective state spaces .

arXiv preprint arXiv:2312.00752 .

Cited by: §1 ,
§2 .

Gu et al. (2025)
X. Gu, T. Pang, C. Du, Q. Liu, F. Zhang, C. Du, Y. Wang, and M. Lin

When attention sink emerges in language models: an empirical view .

arXiv preprint arXiv:2410.10781 .

Cited by: §3.2 .

Gu et al. (2023)
Y. Gu et al.

MiniLLM: knowledge distillation of large language models .

arXiv preprint arXiv:2306.08543 .

Cited by: §2 .

Hendrycks et al. (2021)
D. Hendrycks et al.

Measuring massive multitask language understanding .

In International Conference on Learning Representations ,

Cited by: §6.1 .

Hinton et al. (2015)
G. Hinton, O. Vinyals, and J. Dean

Distilling the knowledge in a neural network .

arXiv preprint arXiv:1503.02531 .

Cited by: §1 ,
§2 ,
§4.1 ,
§4.2 .

HITsz-TMG (2026)
HITsz-TMG

YiZhao: a 2tb open financial corpus .

GitHub .

Cited by: §5.2 .

Huang et al. (2023)
Y. Huang et al.

C-eval: a multi-level multi-discipline chinese evaluation suite for foundation models .

arXiv preprint arXiv:2305.08322 .

Cited by: §6.1 ,
§6.1 .

Ji et al. (2025)
T. Ji, B. Guo, Y. Wu, Q. Guo, L. Shen, Z. Chen, X. Qiu, Q. Zhang, and T. Gui

Towards economical inference: enabling deepseek’s multi-head latent attention in any transformer-based llms .

arXiv preprint arXiv:2502.14837 .

Cited by: §1 ,
§1 ,
§2 ,
§4 .

Katharopoulos et al. (2020)
A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret

Transformers are rnns: fast autoregressive transformers with linear attention .

In International Conference on Machine Learning ,

Cited by: §1 ,
§2 .

Koike-Akino et al. (2026)
T. Koike-Akino, X. Chen, J. Liu, Y. Wang, P. Wang, and M. Brand

LatentLLM: activation-aware transform to multi-head latent attention .

Technical report

Mitsubishi Electric Research Laboratories .

Cited by: §1 .

Kwon et al. (2023)
W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica

Efficient memory management for large language model serving with pagedattention .

arXiv preprint arXiv:2309.06180 .

Cited by: §1 .

Li et al. (2024)
J. Li et al.

DataComp-lm: in search of the next generation of training sets for language models .

arXiv preprint arXiv:2406.11794 .

Cited by: §5.2 .

Li and Liu (2025)
J. Li and S. Liu

FlashMLA: efficient multi-head latent attention kernels .

GitHub .

Cited by: §5.1 .

LLM-Core Xiaomi (2026)
LLM-Core Xiaomi

MiMo-v2-flash technical report .

arXiv preprint arXiv:2601.02780 .

Cited by: §1 ,
§2 ,
§5.1 .

Lozhkov et al. (2024)
A. Lozhkov et al.

StarCoder2 and the stack v2: the next generation .

arXiv preprint arXiv:2402.19173 .

Cited by: §5.2 .

Meng et al. (2025)
F. Meng, P. Tang, X. Tang, Z. Yao, X. Sun, and M. Zhang

TransMLA: multi-head latent attention is all you need .

arXiv preprint arXiv:2502.07864 .

Cited by: §1 ,
§1 ,
§2 ,
§4 .

Meng et al. (2022)
K. Meng et al.

Locating and editing factual associations in gpt .

In Advances in Neural Information Processing Systems ,

Cited by: §6.1 .

Niu et al. (2024)
J. Niu et al.

What does the knowledge neuron thesis have to do with knowledge? .

In International Conference on Learning Representations ,

Cited by: §6.1 .

OpenAI (2025)
OpenAI

Gpt-oss-120b & gpt-oss-20b model card .

Note: Model card

Cited by: §1 ,
§1 ,
§1 ,
§2 ,
§5.1 .

Ouyang et al. (2022)
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, J. Leike, and R. Lowe

Training language models to follow instructions with human feedback .

In Advances in Neural Information Processing Systems ,

Cited by: §1 .

Penedo et al. (2024)
G. Penedo et al.

Decanting the web for the finest text data at scale .

arXiv preprint arXiv:2406.17557 .

Cited by: §5.2 .

Peng et al. (2023)
B. Peng et al.

RWKV: reinventing rnns for the transformer era .

arXiv preprint arXiv:2305.13048 .

Cited by: §1 ,
§2 .

Qiu et al. (2025)
Z. Qiu, Z. Wang, B. Zheng, Z. Huang, K. Wen, S. Yang, R. Men, L. Yu, F. Huang, S. Huang, D. Liu, J. Zhou, and J. Lin

Gated attention for large language models: non-linearity, sparsity, and attention-sink-free .

arXiv preprint arXiv:2505.06708 .

Cited by: §1 ,
§3.2 ,
§5.1 ,
§6.1 .

Romero et al. (2014)
A. Romero, N. Ballas, S. Ebrahimi Kahou, A. Chassang, C. Gatta, and Y. Bengio

FitNets: hints for thin deep nets .

arXiv preprint arXiv:1412.6550 .

Cited by: §1 ,
§2 .

Schick et al. (2023)
T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom

Toolformer: language models can teach themselves to use tools .

arXiv preprint arXiv:2302.04761 .

Cited by: §1 .

Shazeer (2019)
N. Shazeer

Fast transformer decoding: one write-head is all you need .

arXiv preprint arXiv:1911.02150 .

Cited by: §2 .

Su et al. (2024)
D. Su et al.

Nemotron-cc: transforming common crawl into a refined long-horizon pretraining dataset .

arXiv preprint arXiv:2412.02595 .

Cited by: §5.2 .

Team (2026a)
G. Team

GLM-5: from vibe coding to agentic engineering .

arXiv preprint arXiv:2602.15763 .

Cited by: §1 .

Team (2025a)
K. Team

Kimi k2: open agentic intelligence .

arXiv preprint arXiv:2507.20534 .

Cited by: §1 .

Team (2025b)
Q. Team

Qwen3-next-80b-a3b-instruct .

Hugging Face .

Cited by: §1 .

Team (2026b)
S. Team

Step 3.5 flash: open frontier-level intelligence with 11b active parameters .

arXiv preprint arXiv:2602.10604 .

Cited by: §1 .

Vaswani et al. (2017)
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin

Attention is all you need .

In Advances in Neural Information Processing Systems ,

Cited by: §3.1 .

Xiao et al. (2023)
G. Xiao et al.

Efficient streaming language models with attention sinks .

arXiv preprint arXiv:2309.17453 .

Cited by: §1 ,
§3.2 .

Yang et al. (2025)
A. Yang et al.

Qwen3 technical report .

arXiv preprint arXiv:2505.09388 .

Cited by: §1 ,
§1 ,
§1 ,
§5.1 ,
§5.2 .

Yang et al. (2025)
S. Yang, J. Kautz, and A. Hatamizadeh

Gated delta networks: improving mamba2 with delta rule .

In International Conference on Learning Representations ,

Cited by: §1 ,
§2 .

Yao et al. (2022)
S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao

ReAct: synergizing reasoning and acting in language models .

arXiv preprint arXiv:2210.03629 .

Cited by: §1 .

Zhang et al. (2025)
Y. Zhang, Z. Lin, X. Yao, et al.

Kimi linear: an expressive, efficient attention architecture .

arXiv preprint arXiv:2510.26692 .

Cited by: §2 ,
§3.2 .

Zheng et al. (2024)
L. Zheng et al.

SGLang: efficient execution of structured language model programs .

In Advances in Neural Information Processing Systems ,

Cited by: §6.1 .

Zhou et al. (2025)
F. Zhou et al.

MegaMath: pushing the limits of open math corpora .

arXiv preprint arXiv:2504.02807 .

Cited by: §5.2 .
