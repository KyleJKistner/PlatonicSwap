---
title: "Cross-LoRA: A Data-Free LoRA Transfer Framework across Heterogeneous LLMs"
authors: "Feifan Xia, Mingyang Liao, Yuyang Fang, Defang Li, Yantong Xie, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang"
date: "2025-08-07"
arxiv_id: "2508.05232"
url: "https://arxiv.org/abs/2508.05232"
cluster: "weight_adapter_transfer"
tags: ["cross-lora", "lora-transfer", "data-free", "training-free", "svd", "heterogeneous-llms"]
status: "FULL_TEXT"
source: "arxiv_html (50,643 chars)"
---

# Cross-LoRA: A Data-Free LoRA Transfer Framework across Heterogeneous LLMs

- **Authors:** Feifan Xia, Mingyang Liao, Yuyang Fang, Defang Li, Yantong Xie, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang
- **Published:** 2025-08-07 | **arXiv:** [2508.05232](https://arxiv.org/abs/2508.05232)
- **Cluster:** `weight_adapter_transfer` | **Tags:** `cross-lora`, `lora-transfer`, `data-free`, `training-free`, `svd`, `heterogeneous-llms`
- **Status:** `FULL_TEXT` (arxiv_html (50,643 chars))
- **Relevance to PlatonicSwap v1:** Closest weight-space transfer baseline (data-free SVD subspace alignment for LoRA adapters vs our surgical mid-layer SwiGLU organs with unseen ripple).
- **Relevance to PlatonicSwap v2:** LoRA-Align / LoRA-Shift subspace math for translating multi-layer attention+MLP LoRA skills.

## Abstract

Traditional parameter-efficient fine-tuning (PEFT) methods such as LoRA are tightly coupled with the base model architecture, which constrains their applicability across heterogeneous pretrained large language models (LLMs). To address this limitation, we introduce Cross-LoRA, a data-free framework for transferring LoRA modules between diverse base models without requiring additional training data. Cross-LoRA consists of two key components: (a) LoRA-Align, which performs subspace alignment between source and target base models through rank-truncated singular value decomposition (SVD) and Frobenius-optimal linear transformation, ensuring compatibility under dimension mismatch; and (b) LoRA-Shift, which applies the aligned subspaces to project source LoRA weight updates into the target model parameter space. Both components are data-free, training-free, and enable lightweight adaptation on a commodity GPU in 20 minutes. Experiments on ARCs, OBOA and HellaSwag show that Cross-LoRA achieves relative gains of up to 5.26% over base models. Across other commonsense reasoning benchmarks, Cross-LoRA maintains performance comparable to that of directly trained LoRA adapters.

---

## Full Extracted Text (Searchable)

# Cross-LoRA: A Data-Free LoRA Transfer Framework across Heterogeneous LLMs

Feifan Xia

Mingyang Liao

Yuyang Fang

Defang Li

Yantong Xie

Weikang Li

Yang Li
† † thanks: Corresponding author

Deguo Xia

Jizhou Huang

Abstract

Traditional parameter-efficient fine-tuning (PEFT) methods such as LoRA are tightly coupled with the base model architecture, which constrains their applicability across heterogeneous pretrained large language models (LLMs). To address this limitation, we introduce Cross-LoRA , a data-free framework for transferring LoRA modules between diverse base models without requiring additional training data. Cross-LoRA consists of two key components: (a) LoRA-Align , which performs subspace alignment between source and target base models through rank-truncated singular value decomposition (SVD) and Frobenius-optimal linear transformation, ensuring compatibility under dimension mismatch; and (b) LoRA-Shift , which applies the aligned subspaces to project source LoRA weight updates into the target model parameter space. Both components are data-free , training-free , and enable lightweight adaptation on a commodity GPU in 20 minutes. Experiments on ARCs, OBQA and HellaSwag show that Cross-LoRA achieves relative gains of up to 5.26% over base models . Across other commonsense reasoning benchmarks, Cross-LoRA maintains performance comparable to that of directly trained LoRA adapters.

1 Baidu Inc

2 Imperial College London   3 Peking University

4 Zhejiang University   5 Carnegie Mellon University

{xiafeifan, liaomingyang, lidefang, xieyantong, liyang164, xiadeguo, huangjizhou01}@baidu.com

feifan.xia23@imperial.ac.uk, 2301210317@stu.pku.edu.cn, wavejkd@pku.edu.cn,

fangyuyang@zju.edu.cn, yantongx@andrew.cmu.edu

## Introduction

Large Foundation Models (LFMs) such as GPT - 4, LLaMA - 3, Qwen2.5, and Gemma - 2 have become the cornerstone of modern artificial intelligence, delivering state - of - the - art performance across diverse domains, particularly natural language processing   ( OpenAI et al. 2024 ; Qwen et al. 2025 ; Team et al. 2024 ) . Owing to their strong capabilities, fine-tuning LFMs for downstream tasks has become the prevalent strategy. However, conventional fine-tuning generally requires saving a new checkpoint of comparable size to the original model for each task, resulting in substantial storage demands and memory overhead as model scale and task diversity increase.

Figure 1: Overview of the Cross-LoRA framework.
The figure illustrates the process of transferring a LoRA adapter from a source model to a target model.
It highlights the rank-truncated SVD decomposition of source and target bases, subspace alignment via Frobenius-optimal projection,
and the generation of target-compatible LoRA updates.

To address this challenge, Parameter-Efficient Fine-Tuning (PEFT)   ( Xu et al. 2023 ) methods have been widely adopted. Among them, Low-Rank Adaptation (LoRA)   ( Hu et al. 2021 ) , DoRA  ( Liu et al. 2024 ) , and LoRA+   ( Hayou et al. 2024 ) are particularly effective, representing weight updates as low-rank matrices to reduce the number of trainable parameters. These designs achieve strong task performance with relatively modest computational cost.

Nevertheless, LoRA suffers from a key limitation: its dependency on the base model. A LoRA adapter fine-tuned on one model cannot be directly applied to another without retraining. This issue becomes especially problematic when the base model is replaced or deprecated, as the associated LoRA adapters must either be retrained or discarded. Retraining is often infeasible due to the unavailability of the original fine-tuning data and the high computational cost of additional tuning.

This raises a crucial question:

|

Question: Can we transfer LoRA adapters trained on one base model to another, without requiring any training data or additional fine-tuning?

We propose Cross-LoRA , a data-free and training-free framework for transferring LoRA adapters across heterogeneous large language models. Cross-LoRA introduces two key components: LoRA-Align, which performs rank-truncated SVD–based subspace alignment and Frobenius-optimal linear transformation to ensure compatibility across dimension mismatches; and LoRA-Shift, which projects the aligned source LoRA weight updates into the target model parameter space.

Our contributions can be summarized as follows:

1.

Data-free and training-free LoRA transfer at LLM scale. To our knowledge, Cross-LoRA is the first framework enabling LoRA migration across heterogeneous large language models without requiring access to the original training data or additional fine-tuning.

2.

Efficient alignment mechanism tailored for LLMs. We introduce a Frobenius-optimal subspace alignment approach with rank-truncated singular value decomposition (SVD), which ensures numerical stability under dimension mismatch and enables practical, lightweight transfer on autoregressive LLMs without the need for costly computation or retraining.

3.

Robust generalization across architectures and model families. Cross-LoRA demonstrates strong transferability across heterogeneous base models, such as Qwen, LLaMA, and Gemma. It is architecture agnostic and plug and play, requiring no modification to the tokenizer or the internal structure of the base model.

## Related Work

Transferring knowledge between models has long been a central goal in machine learning.
A common approach is Knowledge Distillation ( Hinton et al. 2015 ) ,
where a smaller “student” learns to mimic a more powerful “teacher.”
Later studies examined its convergence and generalization ( Phuong and Lampert 2021 ; Kaplun et al. 2022 ) .
The student acquires the teacher’s behavior through training on this dataset,
thus obtaining W 0 W_{0} or a W 0 W_{0} -equivalent when base models differ.

Using original or synthetic data as a proxy for transferring delta weights has also shown viability. Trans-LoRA  ( Wang et al. 2025 ) generates synthetic data from the source model combined with a trained LoRA, then uses this data to perform LoRA training on the target model. This method has proven effective across model families, yet proxy data may fail to represent the original distribution, and the transfer process requires training time comparable to fine-tuning the original LoRA.

To reduce reliance on data and training, LoRASuite  ( Li et al. 2025 ) compares subspaces between the source and target models and constructs a transfer matrix, followed by lightweight fine-tuning on the target. While efficient, it is limited to same-architecture transfers (e.g., small-to-large within a model family) and still requires partial access to target model gradients.

In contrast, LoRA-X  ( Farhadzadeh et al. 2025a ) avoids both data dependency and training. It proposes a subspace-constrained alternative to LoRA that can align with the subspaces across different models. However, it requires specially designed adapters and has thus far been primarily applied to diffusion models.

ProLoRA  ( Farhadzadeh et al. 2025b ) further removes this constraint by enabling training-free transfer of standard LoRA adapters. It decomposes LoRA weights into subspace and null-space components, projecting them into the target model’s structure without any data or training. Though more general, its application remains focused on text-to-image diffusion models.

In summary, existing methods either rely on proxy data and training(Trans-LoRA, LoRASuite), or are adapter- or domain-specific (LoRA-X, ProLoRA). We aim to bridge this gap with a general, training-free transfer method for standard LoRA adapters, applicable to Large Language Models.

## Method

Figure 2: Overall framework of Cross-LoRA. The source base model (left) provides
LoRA updates Δ ​ W s \Delta W_{s} trained on a specific task.
In Step 1, rank-truncated SVD is applied to both source and target base weights
to extract compact subspaces. Lora-Align computes Frobenius-optimal linear transformations
to align the source subspaces with the target.
In Step 2, Lora-Shift projects the source LoRA updates into the aligned target subspace,
yielding target-compatible LoRA weights Δ ​ W t \Delta W_{t} .
The process requires no access to training data and runs efficiently on commodity GPUs.

### Problem Formulation

Traditional parameter-efficient fine-tuning (PEFT) methods such as LoRA are highly effective but intrinsically tied to the base model, preventing direct reuse across heterogeneous architectures. Inspired by LoRA-X  ( Farhadzadeh et al. 2025a ) , which demonstrated data-free transfer for diffusion models with structurally similar backbones, we extend this idea to heterogeneous large language models (LLMs) and propose a general framework for transferring LoRA modules without access to training data or further optimization.

Formally, let the source base model weight be
W s ∈ ℝ m × n W_{s}\in\mathbb{R}^{m\times n}
and the corresponding LoRA update be
Δ ​ W s ∈ ℝ m × n \Delta W_{s}\in\mathbb{R}^{m\times n} .
Given a target base model weight
W t ∈ ℝ m ′ × n ′ W_{t}\in\mathbb{R}^{m^{\prime}\times n^{\prime}} ,
our objective is to construct a target-compatible LoRA update
Δ ​ W t ∈ ℝ m ′ × n ′ \Delta W_{t}\in\mathbb{R}^{m^{\prime}\times n^{\prime}}
such that the adapted target model retains the knowledge encoded by Δ ​ W s \Delta W_{s} , without requiring original data or further training.

### Framework Overview: Cross-LoRA

Cross-LoRA consists of two complementary components:

LoRA-Align.
We perform subspace alignment between source and target base models using rank-truncated singular value decomposition (SVD). This resolves dimensional mismatches and identifies a shared subspace for LoRA transfer.

LoRA-Shift.
We project the source LoRA update Δ ​ W s \Delta W_{s} into the aligned subspace to obtain a target-compatible update Δ ​ W t \Delta W_{t} . This transformation is both data-free and training-free.

### Mathematical Formulation

#### SVD Decomposition.

We first compute rank- r r truncated SVDs for the source and target base weights:

|
| W s ≈ U s ​ Σ s ​ V s ⊤ W_{s}\approx U_{s}\Sigma_{s}V_{s}^{\top}
|
| (1)

|
| W t ≈ U t ​ Σ t ​ V t ⊤ W_{t}\approx U_{t}\Sigma_{t}V_{t}^{\top}
|
| (2)

where U s ∈ ℝ m × r U_{s}\in\mathbb{R}^{m\times r} , V s ∈ ℝ n × r V_{s}\in\mathbb{R}^{n\times r} , and similarly for U t , V t U_{t},V_{t} . The truncated singular vectors capture the dominant directions of the parameter spaces.

#### Subspace Alignment.

To establish subspace compatibility, we construct Frobenius-optimal linear transforms that align the source and target bases:

|
| P ^ U = arg ⁡ min P ⁡ ‖ P ​ U s − U t ‖ F 2 \hat{P}_{U}=\arg\min_{P}\|PU_{s}-U_{t}\|_{F}^{2}
|
| (3)

|
| P ^ V = arg ⁡ min P ⁡ ‖ P ​ V s − V t ‖ F 2 \hat{P}_{V}=\arg\min_{P}\|PV_{s}-V_{t}\|_{F}^{2}
|
| (4)

These least-squares problems admit closed-form solutions and remain stable even when U s U_{s} or V s V_{s} is rank-deficient. In practice, we solve them using efficient batched routines (e.g., torch.linalg.lstsq ). The aligned subspaces are then defined as:

|
| U ~ s = P ^ U ​ U s \tilde{U}_{s}=\hat{P}_{U}U_{s}
|
| (5)

|
| V ~ s = P ^ V ​ V s \tilde{V}_{s}=\hat{P}_{V}V_{s}
|
| (6)

#### Frobenius Projection.

We then project the source LoRA update Δ ​ W s \Delta W_{s} into the aligned target subspace:

|
| Δ ​ W t = U ~ s ​ ( U ~ s ⊤ ​ Δ ​ W s ​ V ~ s ) ​ V ~ s ⊤ \Delta W_{t}=\tilde{U}_{s}\left(\tilde{U}_{s}^{\top}\Delta W_{s}\tilde{V}_{s}\right)\tilde{V}_{s}^{\top}
|
| (7)

This operation minimizes the Frobenius norm ‖ Δ ​ W s − Δ ​ W t ‖ F \|\Delta W_{s}-\Delta W_{t}\|_{F} under the aligned latent basis, enabling adaptation to the target model geometry without requiring training data or explicit pseudo-inverses.

### Complexity and Efficiency

Algorithm 1 Cross-LoRA Transfer via Subspace Projection

Input : LoRA update Δ ​ W s \Delta W_{s} , source weights W s W_{s} , target weights W t W_{t}

Parameter : Truncated rank r r

Output : Transferred update Δ ​ W t \Delta W_{t}

1:

Initialize empty Δ ​ W t \Delta W_{t} and counters.

2:

for each LoRA parameter k k in Δ ​ W s \Delta W_{s} do

3:

Determine base key b b from k k .

4:

if b ∉ W s b\notin W_{s} or b ∉ W t b\notin W_{t} then

5:

continue

6:

end if

7:

Compute rank- r r SVDs for W s ​ [ b ] W_{s}[b] , W t ​ [ b ] W_{t}[b] .

8:

Derive aligned basis U ~ s , V ~ s \tilde{U}_{s},\tilde{V}_{s} via least-squares.

9:

if k k is a left LoRA weight then

10:

Project: Δ ​ W t ​ [ k ] ← U ~ s ​ ( U ~ s ⊤ ​ Δ ​ W s ​ [ k ] ) \Delta W_{t}[k]\leftarrow\tilde{U}_{s}(\tilde{U}_{s}^{\top}\Delta W_{s}[k])

11:

else

12:

Project: Δ ​ W t ​ [ k ] ← ( Δ ​ W s ​ [ k ] ​ V ~ s ) ​ V ~ s ⊤ \Delta W_{t}[k]\leftarrow(\Delta W_{s}[k]\tilde{V}_{s})\tilde{V}_{s}^{\top}

13:

end if

14:

Cast to FP16 and update statistics.

15:

end for

16:

return Δ ​ W t \Delta W_{t}

We analyze the computational complexity and accuracy implications
of rank-truncated SVD in Cross-LoRA.

Let the base weight

|
| W 0 ∈ ℝ m × n W_{0}\in\mathbb{R}^{m\times n}
|

have singular value decomposition (SVD):

|
| W 0 = U ​ Σ ​ V ⊤ , W_{0}=U\Sigma V^{\top},
|

with truncated form

|
| W 0 ( r ) = U r ​ Σ r ​ V r ⊤ . W_{0}^{(r)}=U_{r}\Sigma_{r}V_{r}^{\top}.
|

By the Eckart–Young–Mirsky theorem  ( MIRSKY 1960 ) ,
W 0 ( r ) W_{0}^{(r)} is the best rank- r r approximation of W 0 W_{0} under the Frobenius norm.

Since a LoRA update can be written as

|
| Δ ​ W s = B s ​ A s , \Delta W_{s}=B_{s}A_{s},
|

with rank ⁡ ( Δ ​ W s ) ≤ r \mathrm{rank}(\Delta W_{s})\leq r ,
it can be expressed in the truncated subspace as

|
| Δ ​ W s ≈ U r ​ C ​ V r ⊤ , C ∈ ℝ r × r . \Delta W_{s}\approx U_{r}CV_{r}^{\top},\quad C\in\mathbb{R}^{r\times r}.
|

Projecting into the truncated subspace gives

|
| Δ ​ W s ^ = U r ​ ( U r ⊤ ​ Δ ​ W s ​ V r ) ​ V r ⊤ . \widehat{\Delta W_{s}}=U_{r}\big(U_{r}^{\top}\Delta W_{s}V_{r}\big)V_{r}^{\top}.
|

The resulting error is

|
| E = ‖ Δ ​ W s − Δ ​ W s ^ ‖ F ≈ 0 , E=\|\Delta W_{s}-\widehat{\Delta W_{s}}\|_{F}\;\;\approx 0,
|

since discarded singular directions contribute negligibly
when r r is chosen at least as large as the LoRA rank
(typically 16 16 – 64 64 ).

Truncation reduces the time complexity from

|
| O ⁡ ( m ​ n ⋅ min ⁡ ( m , n ) ) (full SVD) O(mn\cdot\min(m,n))\quad\text{(full SVD)}
|

to

|
| O ⁡ ( r ​ m ​ n ) (truncated SVD) , O(rmn)\quad\text{(truncated SVD)},
|

and memory cost from

|
| O ⁡ ( m ​ n ) to O ⁡ ( r ⁡ ( m + n ) ) . O(mn)\quad\text{to}\quad O(r(m+n)).
|

This efficiency enables transfer on a single 8 8 GB GPU
within 20 20 minutes.
Previous spectral studies show that the first r = 320 r=320 singular values capture over 99 % 99\% of the Frobenius norm energy.

| Method
| Time
| Memory
| F.N. Energy

| Full-rank
| O ( m n ⋅ min ) O(mn\!\cdot\!\min)
| O ⁡ ( m ​ n ) O(mn)
| 100%

| Truncated ( r ≪ min r\!\ll\!\min )
| O ⁡ ( r ​ m ​ n ) O(rmn)
| O ⁡ ( r ⁡ ( m + n ) ) O(r(m+n))
| ≥ \geq 99%

Table 1: Computational and representational trade-off between full-rank and rank-truncated SVD in Cross-LoRA.
Truncation substantially reduces time and memory complexity while retaining at least 99% of the Frobenius norm energy,
as detailed in Appendix C .

## Experiments

| Model
| Adapter
| Training
| Arc-c (Acc.)
| Arc-e (Acc.)
| OpenBookQA (Acc.)
| HellaSwag (Acc.)

| LLaMA-3.2-3B
| Base Model
| No
| 0.6838
| 0.8260
| 0.7260
| 0.5657

| Trained LoRA
| Yes
| 0.7082 (+3.57%)
| 0.8229 (-0.38%)
| 0.7200 (-0.83%)
| 0.6092 (+7.69%)

| Transferred LoRA
| No
| 0.7065 (+3.32%)
| 0.8331 (+0.86%)
| 0.7260 (+0.00%)
| 0.5667 (+0.18%)

| Gemma-2-2B
| Base Model
| No
| 0.7144
| 0.8269
| 0.7120
| 0.5254

| Trained LoRA
| Yes
| 0.7355 (+2.95%)
| 0.8345 (+0.92%)
| 0.7220 (+1.40%)
| 0.5100 (-2.93%)

| Transferred LoRA
| No
| 0.7520 (+5.26%)
| 0.8289 (+0.24%)
| 0.7120 (+0.00%)
| 0.5254 (+0.00%)

| Qwen2.5-3B
| Base Model
| No
| 0.8129
| 0.9061
| 0.8000
| 0.6040

| Trained LoRA
| Yes
| 0.8172 (+0.53%) †
| 0.9061 (+0.00%)
| 0.8020 (+0.25%) †
| 0.6125 (+1.41%) †

| Transferred LoRA
| No
| 0.8130 (+0.01%)
| 0.9063 (+0.02%) †
| 0.8000 (+0.00%)
| 0.6028 (-0.20%)

| Qwen2.5-1.5B
| Base Model
| No
| 0.7338
| 0.8483
| 0.7440
| 0.4980

| Trained LoRA
| Yes
| 0.7457 (+1.62%)
| 0.8485 (+0.02%)
| 0.7500 (+0.81%)
| 0.5180 (+4.02%)

| Transferred LoRA
| No
| 0.7372 (+0.46%)
| 0.8485 (+0.02%)
| 0.7460 (+0.27%)
| 0.4995 (+0.30%)

Table 2: Performance comparison of Cross-LoRA with base models and directly trained LoRA adapters across four benchmarks.
Transferred adapters are data-free and training-free. The best results per dataset are marked with † . Percentage change is relative to the corresponding base model.

This section evaluates the effectiveness of Cross-LoRA for training-free LoRA transfer across heterogeneous large language models (LLMs). We first describe the experimental setup and then present results on multiple NLP benchmarks.

### Experiment Setup

We evaluate the effectiveness of Cross-LoRA across multiple large language models (LLMs) and benchmark datasets.
Our goal is to assess whether Cross-LoRA can transfer LoRA adapters trained on a source model to a heterogeneous target model
without requiring any additional training or access to the original fine-tuning data.

#### Models.

We select four representative instruction-tuned LLMs as the base models for our experiments:
LLaMA-3.2-3B, Qwen2.5-1.5B, Qwen2.5-3B, and Gemma-2-2B.
These models cover both decoder-only architectures and different model families,
enabling evaluation of transferability across heterogeneous base models.

#### Datasets.

We benchmark performance on four widely used NLP tasks:
ARC-c, ARC-e, OpenBookQA, and HellaSwag  ( Clark et al. 2018 ; Mihaylov et al. 2018 ; Zellers et al. 2019 ) .
These tasks are standard in evaluating reasoning, knowledge retrieval, and commonsense understanding,
providing a comprehensive view of model performance under transfer.

#### Baselines.

We compare three settings:
(1) the Base Model , evaluated without any LoRA adapters;
(2) Trained LoRA , where adapters are directly fine-tuned on the target model for each dataset. This training process follows common practices in prior PEFT transfer studies  ( Liang et al. 2025 ) ;
and (3) Transferred LoRA (Cross-LoRA), where adapters fine-tuned on a source model are transferred to the target model using our framework,
without any training data or fine-tuning.
This comparison allows us to quantify the relative utility of Cross-LoRA, highlighting both its advantages and limitations in zero-shot adaptation settings.

#### Implementation.

Cross-LoRA employs rank-truncated SVD and Frobenius-optimal subspace alignment to map LoRA weight updates into the target model space.
All experiments are executed on a single NVIDIA V100 GPU.
No additional training or optimization is performed on the transferred adapters. A detailed description of the setup is provided in Appendix A.

### Main Results

Table  2 presents the results of Cross-LoRA across four QA benchmarks. We compare transferred adapters against base models and directly trained LoRA adapters, reporting accuracy and relative gains. Overall, transferred adapters yield consistent improvements over base models, achieving an average relative gain of +0.848% , closely approaching the +0.976% gain from trained LoRA. Notably, the best observed gains are +5.26% on ARC-c and +3.32% on ARC-e , demonstrating the effectiveness of training-free subspace alignment in recovering task-specific knowledge.

Compared to trained LoRA, transferred adapters deliver competitive results. On ARC-e and OpenBookQA, Cross-LoRA matches or slightly exceeds the trained counterpart on multiple architectures. On HellaSwag, performance varies more across models, with some configurations showing minor drops. This variation suggests architecture-dependent transfer fidelity, especially in tasks requiring fine-grained reasoning.

These findings indicate that Cross-LoRA enables efficient adapter transfer in both homogeneous and heterogeneous settings, offering a practical alternative when fine-tuning data or computational resources are limited. While minor performance gaps remain in specific tasks, the results suggest room for further refinement, such as lightweight post-transfer adaptation.

### Effect of Subspace Alignment

To further understand the contribution of subspace alignment, we compare Cross-LoRA against a linear interpolation baseline, as well as base and directly trained LoRA adapters.
The results, reported in Table  3 , are obtained when transferring a LoRA trained on Qwen2.5-1.5B into LLaMA-3.2-3B.
The interpolation baseline adjusts LoRA weights through direct dimension matching without subspace projection,
while Cross-LoRA applies rank-truncated SVD and Frobenius-optimal alignment.

The results demonstrate that simple interpolation provides little benefit and in some cases degrades performance relative to the base model.
By contrast, Cross-LoRA consistently improves accuracy on ARC-c and ARC-e, reaching 0.7065 and 0.8331 respectively, which is close to or above the directly trained LoRA.
The gain is particularly evident compared to interpolation, highlighting the necessity of subspace alignment in capturing transferable task-specific knowledge.
On HellaSwag, both methods achieve only marginal improvements over the base model, suggesting that the task requires finer-grained reasoning that may not be fully preserved by projection alone.

| Method
| ARC-c (Acc.)
| ARC-e (Acc.)

| Base
| 0.6838
| 0.8260

| Base + Trained LoRA
| 0.7082
| 0.8229

| Interpolation Baseline
| 0.6820
| 0.8320

| Cross-LoRA (Projection)
| 0.7065
| 0.8331

Table 3: Impact of subspace alignment when transferring from Qwen2.5-1.5B to LLaMA-3.2-3B.
Cross-LoRA employs Frobenius projection, whereas the interpolation baseline directly resizes weights.

Taken together, these findings confirm that the Frobenius-optimal projection in Cross-LoRA is essential for effective transfer.
While direct interpolation fails to preserve critical structure, subspace alignment enables the transferred LoRA to approach the performance of direct fine-tuning,
validating the design choice of aligning singular vector subspaces during transfer.

## Ablation Studies

### Rank of Cross-Lora

(a)

(b)

Figure 3: Effect of adapter rank on Cross-LoRA performance.
Results on ARC-C and ARC-E demonstrate that higher adapter ranks consistently improve accuracy for both trained and transferred LoRA adapters.
Notably, transferred LoRA exhibits smaller performance degradation at low ranks compared to directly trained LoRA,
highlighting the robustness of Cross-LoRA under resource-constrained settings.

Figure  3 illustrates the effect of adapter rank on Cross-LoRA performance, evaluated on ARC-C and ARC-E benchmarks. We observe a consistent trend that accuracy improves as the adapter rank increases for both trained and transferred LoRA . This aligns with the intuition that low-rank approximations restrict representational capacity , whereas higher ranks provide richer subspaces to capture task-relevant features. However, this accuracy gain comes with increased storage cost: the LoRA checkpoint size grows from 3.7MB at r = 80 r=80 to 32.4MB at r = 320 r=320 , highlighting a trade-off between accuracy and efficiency .

Notably, transferred LoRA even surpasses directly trained LoRA on the target model in some cases , as shown in Figure  3b . This phenomenon can be explained by differences in training dynamics across models. Certain target models, even under the same training recipe, may fail to converge or overfit, limiting their ability to extract effective features from the dataset. In contrast, the source model may better capture task-relevant information during LoRA fine-tuning, which Cross-LoRA can then efficiently transfer to the target. This suggests that Cross-LoRA can leverage source models better suited to a given dataset or domain to improve weaker target models.

At lower ranks, such as r = 80 r=80 , both trained and transferred LoRA experience performance degradation. Notably, the decline is less severe for transferred LoRA , indicating that the Frobenius-optimal projection mechanism of Cross-LoRA provides additional robustness under low-rank constraints. This highlights a key advantage of Cross-LoRA: it maintains competitive performance even with significantly reduced adapter size , making it particularly suitable for resource-constrained scenarios .

### Cross Model Transferability

Figure 4: Cross-model transferability heatmap of Cross-LoRA.
Each cell reports the average percentage change in accuracy across ARC-C, ARC-E, and OpenBookQA,
relative to the corresponding base models.
Transfers between models with similar architectures (e.g., LLaMA-3.2-3B and Qwen2.5-1.5B, both using GQA and SwiGLU) yield stronger gains,
while transfers involving models with architectural mismatches (e.g., Gemma-2-2B with MHA and GeLU) show weaker improvements.

To further evaluate the generalization capability of Cross-LoRA, we examine the effectiveness of transferring adapters across different large language models. Figure  4 reports the average percentage change relative to the corresponding base models, aggregated across ARC-C, ARC-E, and OpenBookQA.

The results reveal clear trends regarding the role of architectural similarity. Transfers between models with highly aligned configurations show consistent improvements. For instance, both LLaMA-3.2-3B and Qwen2.5 models employ grouped-query attention (GQA, 1KV), SwiGLU activation, and RMSNorm normalization. This structural alignment explains the stable positive transfer between these models, such as LLaMA-3.2-3B → \rightarrow Qwen2.5-1.5B (+0.38%) and Qwen2.5-1.5B → \rightarrow LLaMA-3.2-3B (+1.66%). These findings indicate that selecting GQA + SwiGLU + RMSNorm models as source models is particularly effective for Cross-LoRA.

In contrast, when transferring from Gemma-2-2B, which employs multi-head attention (MHA) with multi-KV and a mixed GeLU/SwiGLU activation scheme, the benefits are less consistent. While Gemma-2-2B → \rightarrow LLaMA-3.2-3B yields a notable +2.63% improvement, reverse transfer or transfers to other targets show limited gains (e.g., Qwen2.5-3B → \rightarrow Gemma-2-2B at +0.01%). This highlights that attention mechanism differences can hinder subspace alignment, limiting the robustness of the projection.

We also observe non-symmetric behaviors in transferability. For example, Qwen2.5-3B → \rightarrow LLaMA-3.2-3B exhibits a slight negative effect (-0.20%), despite similar parameter counts. This suggests that differences in hidden layer width or attention head distributions can challenge the Frobenius projection, even when overall model scales are close.

Overall, these results indicate that Cross-LoRA transfer is more stable and beneficial when the source model shares the GQA, SwiGLU, and RMSNorm design with the target. This architecture-agnostic yet practically guided choice enables robust transfer performance across heterogeneous LLMs.

## Discussion and Future Work

Cross-LoRA demonstrates that data-free and training-free transfer of LoRA adapters is feasible at the scale of large language models. By leveraging rank-truncated SVD and Frobenius-optimal subspace alignment, it effectively projects task-specific updates across heterogeneous architectures while maintaining efficiency on commodity GPUs. The framework consistently improves over base models and delivers performance comparable to directly trained LoRA adapters, while reducing computational and memory requirements to enable transfer on a single 8GB GPU within 20 minutes.

Despite these advantages, several limitations remain. First, Cross-LoRA does not fully close the gap with directly trained adapters in certain reasoning-heavy tasks, such as HellaSwag, indicating that some task-specific signals are lost during projection. Second, the effectiveness of transfer is sensitive to architectural mismatches. In particular, differences in attention mechanisms (e.g., grouped-query vs. multi-head attention) or activation functions (e.g., SwiGLU vs. ReLU) can reduce alignment quality. Our experiments indicate that source models equipped with grouped-query attention (GQA), SwiGLU activations, and RMSNorm normalization provide stronger transferability, suggesting that architectural choices substantially influence alignment effectiveness. Third, the current framework performs one-shot projection without any post-transfer adaptation, which may limit performance when the source and target distributions diverge significantly.

An additional insight from our analysis is the quantification of LoRA transferability across heterogeneous LLM architectures. Together with the plug-and-play nature of Cross-LoRA, this suggests a new paradigm: training LoRA adapters on a “learner” model well-suited for a given domain, and then transferring them to a “receiver” model for deployment, without requiring access to the original training data. This could substantially reduce the cost of LoRA-based adaptation and accelerate knowledge sharing across model families.

Future work could explore several directions. Hybrid approaches that combine data-free projection with lightweight task-agnostic adaptation may help recover residual task-specific performance. Extending the method to larger-scale models (e.g., 13B or 70B parameters) and multimodal architectures is another promising avenue. Automated strategies for layer-wise or subspace selection could further reduce computational overhead. Finally, a more rigorous theoretical analysis of projection error and its correlation with downstream performance would provide deeper insights into the robustness and limitations of Cross-LoRA.

## Conclusion

We present Cross-LoRA, a simple yet effective framework for data-free and training-free transfer of LoRA adapters across heterogeneous language models. By aligning subspaces with truncated SVD and applying Frobenius-optimal projection, Cross-LoRA enables fast and reliable adapter migration without requiring original training data.

Experiments on multiple reasoning benchmarks show that Cross-LoRA achieves consistent improvements over base models and matches the performance of directly trained LoRA adapters, while completing transfer in minutes on a single 8GB GPU. These results highlight the practicality of Cross-LoRA for low-resource and real-world deployment scenarios.

We believe Cross-LoRA offers a scalable and lightweight alternative to conventional fine-tuning, and we hope it serves as a foundation for future research on modular, transferable, and architecture-agnostic adaptation in large language models.

## References

Aghajanyan et al. [2020]

Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta.

Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning, December 2020.

URL http://arxiv.org/abs/2012.13255 .

arXiv:2012.13255 [cs].

Clark et al. [2018]

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.

Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge, March 2018.

URL http://arxiv.org/abs/1803.05457 .

arXiv:1803.05457 [cs].

Eckart and Young [1936]

Carl Eckart and Gale Young.

The approximation of one matrix by another of lower rank.

Psychometrika , 1(3):211–218, September 1936.

ISSN 1860-0980.

doi: 10.1007/BF02288367 .

URL https://doi.org/10.1007/BF02288367 .

Farhadzadeh et al. [2025a]

Farzad Farhadzadeh, Debasmit Das, Shubhankar Borse, and Fatih Porikli.

LoRA-X: Bridging Foundation Models with Training-Free Cross-Model Adaptation, February 2025a.

URL http://arxiv.org/abs/2501.16559 .

arXiv:2501.16559 [cs].

Farhadzadeh et al. [2025b]

Farzad Farhadzadeh, Debasmit Das, Shubhankar Borse, and Fatih Porikli.

Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models, May 2025b.

URL http://arxiv.org/abs/2506.04244 .

arXiv:2506.04244 [cs].

Hayou et al. [2024]

Soufiane Hayou, Nikhil Ghosh, and Bin Yu.

LoRA+: Efficient Low Rank Adaptation of Large Models, July 2024.

URL http://arxiv.org/abs/2402.12354 .

arXiv:2402.12354 [cs].

Hinton et al. [2015]

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.

Distilling the Knowledge in a Neural Network, March 2015.

URL http://arxiv.org/abs/1503.02531 .

arXiv:1503.02531 [stat].

Hu et al. [2021]

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.

LoRA: Low-Rank Adaptation of Large Language Models, October 2021.

URL http://arxiv.org/abs/2106.09685 .

arXiv:2106.09685 [cs].

Kaplun et al. [2022]

Gal Kaplun, Eran Malach, Preetum Nakkiran, and Shai Shalev-Shwartz.

Knowledge Distillation: Bad Models Can Be Good Role Models, March 2022.

URL http://arxiv.org/abs/2203.14649 .

arXiv:2203.14649 [cs].

Kovaleva et al. [2019]

Olga Kovaleva, Alexey Romanov, Anna Rogers, and Anna Rumshisky.

Revealing the Dark Secrets of BERT, September 2019.

URL http://arxiv.org/abs/1908.08593 .

arXiv:1908.08593 [cs].

Li et al. [2025]

Yanan Li, Fanxu Meng, Muhan Zhang, Shiai Zhu, Shangguang Wang, and Mengwei Xu.

LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades, May 2025.

URL http://arxiv.org/abs/2505.13515 .

arXiv:2505.13515 [cs].

Liang et al. [2025]

Zhiyuan Liang, Dongwen Tang, Yuhao Zhou, Xuanlei Zhao, Mingjia Shi, Wangbo Zhao, Zekai Li, Peihao Wang, Konstantin Schürholt, Damian Borth, Michael M. Bronstein, Yang You, Zhangyang Wang, and Kai Wang.

Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights, June 2025.

URL http://arxiv.org/abs/2506.16406 .

arXiv:2506.16406 [cs].

Liu et al. [2024]

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen.

DoRA: Weight-Decomposed Low-Rank Adaptation, July 2024.

URL http://arxiv.org/abs/2402.09353 .

arXiv:2402.09353 [cs].

Loshchilov and Hutter [2019]

Ilya Loshchilov and Frank Hutter.

Decoupled Weight Decay Regularization, January 2019.

URL http://arxiv.org/abs/1711.05101 .

arXiv:1711.05101 [cs].

Mihaylov et al. [2018]

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal.

Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering, September 2018.

URL http://arxiv.org/abs/1809.02789 .

arXiv:1809.02789 [cs].

MIRSKY [1960]

L. MIRSKY.

SYMMETRIC GAUGE FUNCTIONS AND UNITARILY INVARIANT NORMS.

The Quarterly Journal of Mathematics , 11(1):50–59, January 1960.

ISSN 0033-5606.

doi: 10.1093/qmath/11.1.50 .

URL https://doi.org/10.1093/qmath/11.1.50 .

OpenAI et al. [2024]

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo
Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan
Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez,
Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, C. J. Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia
Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph.

GPT-4 Technical Report, March 2024.

URL http://arxiv.org/abs/2303.08774 .

arXiv:2303.08774 [cs].

Phuong and Lampert [2021]

Mary Phuong and Christoph H. Lampert.

Towards Understanding Knowledge Distillation, May 2021.

URL http://arxiv.org/abs/2105.13093 .

arXiv:2105.13093 [cs].

Qwen et al. [2025]

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu.

Qwen2.5 Technical Report, January 2025.

URL http://arxiv.org/abs/2412.15115 .

arXiv:2412.15115 [cs].

Team et al. [2024]

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar, Dominika Rogozińska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins,
Hadi Hashemi, Hanna Klimczak-Plucińska, Harleen Batra, Harsh Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju-yeong Ji, Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, Machel Reid, Manvinder Singh, Mark Iverson, Martin Görner, Mat Velloso, Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin,
Petko Georgiev, Phil Culliton, Pradeep Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, Sébastien M. R. Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev.

Gemma 2: Improving Open Language Models at a Practical Size, October 2024.

URL http://arxiv.org/abs/2408.00118 .

arXiv:2408.00118 [cs].

Wang et al. [2025]

Runqian Wang, Soumya Ghosh, David Cox, Diego Antognini, Aude Oliva, Rogerio Feris, and Leonid Karlinsky.

Trans-lora: Towards data-free transferable parameter-efficient finetuning, May 2025.

URL https://arxiv.org/abs/2405.17258 .

arXiv preprint arXiv:2405.17258.

Xu et al. [2023]

Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, and Fu Lee Wang.

Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment, December 2023.

URL http://arxiv.org/abs/2312.12148 .

arXiv:2312.12148 [cs].

Zellers et al. [2019]

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.

HellaSwag: Can a Machine Really Finish Your Sentence?, May 2019.

URL http://arxiv.org/abs/1905.07830 .

arXiv:1905.07830 [cs].

## Appendix A. Additional Experimental Details

### Datasets

We evaluate Cross-LoRA on four widely used natural language understanding benchmarks, each targeting different aspects of reasoning and commonsense knowledge.

ARC-Challenge (ARC-C) : A benchmark designed to test grade-school science reasoning ability. The dataset is composed of approximately 1.2k training examples and 1k evaluation questions, requiring multi-step reasoning beyond simple fact retrieval  [ Clark et al., 2018 ] .

ARC-Easy (ARC-E) : A companion benchmark to ARC-C with around 2.2k training examples. Unlike ARC-C, its questions are more straightforward, relying on direct factual recall. Together with ARC-C, it provides a balanced assessment of reasoning difficulty  [ Clark et al., 2018 ] .

OpenBookQA : A benchmark focused on open-domain science knowledge with roughly 5k training questions. It requires combining information from a small science textbook (“open book”) with general commonsense reasoning  [ Mihaylov et al., 2018 ] .

HellaSwag : A large-scale commonsense inference benchmark with about 39k validation examples. It challenges models with adversarially filtered multiple-choice questions designed to be trivial for humans yet difficult for language models  [ Zellers et al., 2019 ] .

These datasets cover diverse reasoning challenges and scales, making them suitable to rigorously assess the generalization ability of Cross-LoRA.

### LoRA Training Recipe

We trained LoRA adapters  [ Hu et al., 2021 ] across all four benchmarks, maintaining consistency in optimizer  [ Loshchilov and Hutter, 2019 ] , learning rate, and dropout, while adjusting training steps for dataset size.
Table  4 summarizes the settings used.

| Setting
| ARC-C
| ARC-E
| OBQA
| HellaSwag

| Batch Size
| 16
| 16
| 16
| 16

| Optimizer
| AdamW
| AdamW
| AdamW
| AdamW

| Learning Rate
| 1e-5
| 1e-5
| 1e-5
| 1e-5

| Training Steps
| 600
| 600
| 600
| 300

| Weight Decay
| 0.1
| 0.1
| 0.1
| 0.1

| Max Grad Norm
| 1.0
| 1.0
| 1.0
| 1.0

| LoRA Alpha
| 32
| 32
| 32
| 32

| LoRA Dropout
| 0.1
| 0.1
| 0.1
| 0.1

| LoRA Bias
| None
| None
| None
| None

| LoRA Rank
| 16
| 16
| 16
| 16

Table 4: LoRA training recipe across ARC-C, ARC-E, OBQA, and HellaSwag.

### Cross-LoRA Transferring

For Cross-LoRA transfer, we applied adapters on a wide set of target modules:

|

Target Modules

| Rank
| LoRA Alpha

|

q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj

| 320
| 64

Table 5: Cross-LoRA transferring configuration.

## Appendix B. Experiment Hardware

We provide details of the hardware used for Cross-LoRA experiments. Batch experiments were carried out on NVIDIA Tesla V100 GPUs, while additional small-scale experiments were conducted on RTX 4090 for comparison.

Based on these results, we conservatively infer that Cross-LoRA should complete transfer in under 20 minutes on edge devices with 8GB GPU memory.

| Setting
| V100
| RTX 4090

| OS
| Ubuntu 22.04.1
| Ubuntu 22.04.1

| CPU
| Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz

| GPU
| NVIDIA Tesla V100 32GB
| NVIDIA RTX 4090 24GB

| CUDA Version
| 11.8
| 11.8

|

Cross-LoRA
Transfer Time

| 349.287s
| 564.043s

|

Cross-LoRA
Memory Cost

| 5508MB
| 2291MB

Table 6: Hardware platforms and Cross-LoRA transfer cost.

## Appendix C. Detailed Derivation of Rank-Truncated SVD

We analyze the approximation quality of rank-truncated singular value decomposition (SVD) as employed in Cross-LoRA.

### Preliminaries

Consider a weight matrix W 0 ∈ ℝ 3072 × 3072 W_{0}\in\mathbb{R}^{3072\times 3072} with singular value decomposition:

|
| W 0 = U ​ Σ ​ V ⊤ , W_{0}=U\Sigma V^{\top},
|

where Σ = diag ⁡ ( σ 1 , σ 2 , … , σ 3072 ) \Sigma=\mathrm{diag}(\sigma_{1},\sigma_{2},\dots,\sigma_{3072}) and σ 1 ≥ σ 2 ≥ ⋯ ≥ 0 \sigma_{1}\geq\sigma_{2}\geq\cdots\geq 0 .
The rank- r r truncated approximation of W 0 W_{0} is:

|
| W 0 ( r ) = U r ​ Σ r ​ V r ⊤ , W_{0}^{(r)}=U_{r}\Sigma_{r}V_{r}^{\top},
|

where U r , V r U_{r},V_{r} contain the top- r r singular vectors, and Σ r = diag ⁡ ( σ 1 , … , σ r ) \Sigma_{r}=\mathrm{diag}(\sigma_{1},\dots,\sigma_{r}) .
By the Eckart–Young–Mirsky theorem  [ Eckart and Young, 1936 , MIRSKY, 1960 ] , W 0 ( r ) W_{0}^{(r)} minimizes the Frobenius norm error among all rank- r r matrices:

|
| ‖ W 0 − W 0 ( r ) ‖ F 2 = ∑ i = r + 1 3072 σ i 2 . \|W_{0}-W_{0}^{(r)}\|_{F}^{2}=\sum_{i=r+1}^{3072}\sigma_{i}^{2}.
|

The fraction of Frobenius energy retained is defined as:

|
| η ⁡ ( r ) = ∑ i = 1 r σ i 2 ∑ i = 1 3072 σ i 2 . \eta(r)=\frac{\sum_{i=1}^{r}\sigma_{i}^{2}}{\sum_{i=1}^{3072}\sigma_{i}^{2}}.
|

### LoRA Updates in the Truncated Subspace

Given a LoRA update Δ ​ W s = B s ​ A s \Delta W_{s}=B_{s}A_{s} with rank ⁡ ( Δ ​ W s ) ≤ r \mathrm{rank}(\Delta W_{s})\leq r , it can be approximated in the truncated subspace as:

|
| Δ ​ W s ≈ U r ​ C ​ V r ⊤ , C ∈ ℝ r × r . \Delta W_{s}\approx U_{r}CV_{r}^{\top},\quad C\in\mathbb{R}^{r\times r}.
|

Projecting into the subspace yields:

|
| Δ ​ W s ^ = U r ​ ( U r ⊤ ​ Δ ​ W s ​ V r ) ​ V r ⊤ , \widehat{\Delta W_{s}}=U_{r}(U_{r}^{\top}\Delta W_{s}V_{r})V_{r}^{\top},
|

with corresponding Frobenius reconstruction error:

|
| E 2 = ‖ Δ ​ W s − Δ ​ W s ^ ‖ F 2 = ∑ i = r + 1 3072 σ i 2 . E^{2}=\|\Delta W_{s}-\widehat{\Delta W_{s}}\|_{F}^{2}=\sum_{i=r+1}^{3072}\sigma_{i}^{2}.
|

### Numerical Illustration

Empirical analyses suggest that low-rank approximations are sufficient for strong performance.
Aghajanyan et al.  [ Aghajanyan et al., 2020 ] showed that tuning only 200 intrinsic dimensions retains roughly 90% downstream accuracy, implying η ⁡ ( 200 ) ≈ 0.9 \eta(200)\approx 0.9 .
Moreover, spectral studies  [ Kovaleva et al., 2019 ] reveal rapid singular value decay in transformers, which can be approximated by a geometric decay σ i 2 ≈ σ 1 2 ​ ρ i \sigma_{i}^{2}\approx\sigma_{1}^{2}\rho^{\,i} , with ρ ∈ [ 0.92 , 0.97 ] \rho\in[0.92,0.97] .

Under this assumption:

|
| η ⁡ ( r ) ≈ 1 − ρ r . \eta(r)\approx 1-\rho^{r}.
|

For r = 320 r=320 and ρ = 0.94 \rho=0.94 :

|
| η ⁡ ( 320 ) ≈ 1 − ( 0.94 ) 320 ≈ 1 − e − 19.8 ≈ 0.999999998 . \eta(320)\approx 1-(0.94)^{320}\approx 1-e^{-19.8}\approx 0.999999998.
|

Even for ρ = 0.97 \rho=0.97 :

|
| η ⁡ ( 320 ) ≈ 1 − ( 0.97 ) 320 ≈ 0.99994 . \eta(320)\approx 1-(0.97)^{320}\approx 0.99994.
|

Thus, in typical transformer layers, rank- 320 320 truncation retains at least 99% of the Frobenius norm energy. The reconstruction error is bounded by:

|
| E 2 = ( 1 − η ⁡ ( 320 ) ) ⋅ ‖ W 0 ‖ F 2 , so  ​ E ≤ 0.001 ≈ 0.031 . E^{2}=(1-\eta(320))\cdot\|W_{0}\|_{F}^{2},\quad\text{so }E\leq\sqrt{0.001}\approx 0.031.
|

### Conclusion

These results justify that rank- 320 320 truncation preserves nearly all semantically meaningful structure in W 0 W_{0} .
Given that LoRA updates themselves are rank- 16 16 or lower, the approximation error is negligible in practice.
This explains why Cross-LoRA achieves performance comparable to directly trained LoRA adapters.
