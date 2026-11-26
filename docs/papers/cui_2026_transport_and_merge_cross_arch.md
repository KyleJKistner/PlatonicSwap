---
title: "Transport and Merge: Cross-Architecture Merging for Large Language Models"
authors: "Chenhang Cui, Binyun Yang, Fei Shen, Yuxin Chen, Jingnan Zheng, Xiang Wang, An Zhang, Tat-Seng Chua"
date: "2026-02-05"
arxiv_id: "2602.05495"
url: "https://arxiv.org/abs/2602.05495"
cluster: "weight_adapter_transfer"
tags: ["optimal-transport", "cross-architecture-merging", "neuron-matching", "weight-fusion", "icml-2026"]
status: "FULL_TEXT"
source: "arxiv_html (82,606 chars)"
---

# Transport and Merge: Cross-Architecture Merging for Large Language Models

- **Authors:** Chenhang Cui, Binyun Yang, Fei Shen, Yuxin Chen, Jingnan Zheng, Xiang Wang, An Zhang, Tat-Seng Chua
- **Published:** 2026-02-05 | **arXiv:** [2602.05495](https://arxiv.org/abs/2602.05495)
- **Cluster:** `weight_adapter_transfer` | **Tags:** `optimal-transport`, `cross-architecture-merging`, `neuron-matching`, `weight-fusion`, `icml-2026`
- **Status:** `FULL_TEXT` (arxiv_html (82,606 chars))
- **Relevance to PlatonicSwap v1:** OT neuron alignment for cross-architecture weight merging (destructive global merge vs our non-destructive additive organs + exact unlearning).
- **Relevance to PlatonicSwap v2:** Optimal Transport cross-neuron correspondence for multi-layer/attention mechanism translation.

## Abstract

Large language models (LLMs) achieve strong capabilities by scaling model capacity and training data, yet many real-world deployments rely on smaller models trained or adapted from low-resource data. This gap motivates the need for mechanisms to transfer knowledge from large, high-resource models to smaller, low-resource targets. While model merging provides an effective transfer mechanism, most existing approaches assume architecture-compatible models and therefore cannot directly transfer knowledge from large high-resource LLMs to heterogeneous low-resource targets. In this work, we propose a cross-architecture merging framework based on optimal transport (OT) that aligns activations to infer cross-neuron correspondences between heterogeneous models. The resulting transport plans are then used to guide direct weight-space fusion, enabling effective high-resource to low-resource transfer using only a small set of inputs. Extensive experiments across low-resource languages and specialized domains demonstrate consistent improvements over target models.

---

## Full Extracted Text (Searchable)

# Transport and Merge:
Cross-Architecture Merging for Large Language Models

Abstract

Large language models (LLMs) achieve strong capabilities by scaling model capacity and training data, yet many real-world deployments rely on smaller models trained or adapted from low-resource data. This gap motivates the need for mechanisms to transfer knowledge from large, high-resource models to smaller, resource-constrained targets.
While model merging provides an effective transfer mechanism, most existing approaches assume architecture-compatible models and therefore cannot directly transfer knowledge from large high-resource LLMs to heterogeneous low-resource targets.
In this work, we propose a cross-architecture merging framework based on optimal transport (OT) that aligns activations to infer cross-neuron correspondences between heterogeneous models.
The resulting transport matrices are then used to guide direct weight-space fusion, enabling effective high-resource to low-resource transfer using only a small set of inputs.
Extensive experiments across low-resource languages and specialized domains demonstrate consistent improvements over target base models. The code is available at https://github.com/chenhangcuisg-code/Cross-Architecture-Merging-for-\Large-Language-Models/ .

Keywords:  Machine Learning, ICML

† † affiliationnotice: ∗ Equal contribution. † Corresponding authors.

Chenhang Cui 1,∗ ,
Binyun Yang 2,∗ ,
Fei Shen 1,† ,
Yuxin Chen 1 ,
Jingnan Zheng 1 ,

Xiang Wang 3 ,
An Zhang 3 ,
Tat-Seng Chua 1

1 National University of Singapore (NUS), Singapore

2 University of Electronic Science and Technology of China (UESTC), China

3 University of Science and Technology of China (USTC), China

## 1 Introduction

Large language models (LLMs) have achieved remarkable success in general language understanding and generation  ( Achiam et al., 2023 ; Grattafiori et al., 2024 ; Bai et al., 2023 ) , enabling a wide range of downstream applications, e.g., in medicine  ( Xie et al., 2024 ) , finance  ( Konstantinidis et al., 2024 ) , and education  ( Wen et al., 2024 ) . This success is largely driven by scaling model capacity and pretraining on massive, high-quality, and diverse corpora.

In practice, many real-world deployments rely on models with limited parameter budgets trained or adapted from low-resource data. Typical examples include low-resource languages such as Malaysian languages ( Hew et al., 2025 ) and Cantonese dialects ( Liu, 2022 ) . Due to constraints on both model capacity and data availability, models trained from low-resource languages are unable to acquire specific knowledge comparable to large and data-rich models.
A natural direction to address this challenge is high-resource to low-resource transfer ( Adimulam et al., 2022 ; Myakala & Naayini, 2023 ; Cao et al., 2025 ) , which leverages representations learned by large, well-trained models to improve performance in low-resource domains.
Among existing approaches, model merging ( Wortsman et al., 2022a ; Jin et al., 2022 ; Imfeld et al., 2023a ) has emerged as an alternative to gradient-based adaptation. It directly aggregates parameters from multiple expert models, enabling capability fusion without access to massive training data or repeated optimization.
With the rise of large language models (LLMs)  ( Yang et al., 2024 ; Tao et al., 2024 ; Perin et al., 2024 ) , this idea naturally extends to LLM merging.

Despite its appeal, many current techniques ( Tao et al., 2024 ; Perin et al., 2024 ; Tian et al., 2025 ) assume that LLMs being merged share the same architecture, enabling direct weight-space operations. This assumption restricts their applicability in low-resource transfer scenarios, where the source model is often a large high-resource LLM, while the target model is a smaller architecture designed for deployment efficiency or domain-specific constraints ( Liu et al., 2024 ; Liu et al., 2023a ; Kolomeitsev, 2025 ) . Direct parameter merging between heterogeneous architectures becomes challenging.
Although some recent methods attempt cross-architecture merging or knowledge fusion for LLMs ( Wan et al., 2024 ; Wan et al., 2025 ) , they typically rely on distillation-based training ( Hinton et al., 2015 ; Hsieh et al., 2023 ; Cai et al., 2025 ) to first distill knowledge into the parameter space of a target model, followed by merging among architecture-compatible models.

These limitations naturally raise the following question:
Can we enable high-resource to low-resource knowledge transfer across heterogeneous model architectures through direct model merging?
Motivated by this question, we propose a cross-model merging framework based on optimal transport ( Peyré & Cuturi, 2019 ; Rout et al., 2021 ) to enable parameter transfer between heterogeneous architectures.
Our approach is motivated by the observation that, despite differences in architecture and model capacity, language models often exhibit correlated internal activations when processing the same inputs ( Huh et al., 2024 ; Shah & Khosla, 2025 ) .
Building on this insight, we establish a structured correspondence between heterogeneous models by aligning their internal activations using an optimal transport formulation.
Given a small set of inputs, we extract intermediate activations from both the source and target models and compute cross-model similarity signals at the feature level.
These signals are used to infer cross-neuron relationships between the two models, yielding a global transfer plan that captures transferable structure across layers and neurons.
The resulting transport relationships are then converted into weight-space fusion operators, enabling direct parameter fusion across architectures.

In summary, our contribution is a cross-architecture model merging framework that enables high-resource to low-resource knowledge transfer.
By aligning heterogeneous language models in activation space through an optimal transport formulation, our method infers structured neuron-level correspondences across models with different architectures and capacities using only a small set of inputs.
Crucially, we show that these activation-space correspondences admit a principled weight-space interpretation, allowing them to be converted into direct fusion for parameter transfer across heterogeneous models.
Extensive experiments on low-resource languages and domains demonstrate consistent improvements, establishing our approach as a practical alternative to distillation-based transfer.
As shown in Figure  1 , our method consistently improves performance across different settings under multiple strategies.

Figure 1 :
Cross-domain comparison using normalized scores (higher is better).
Top: Domain-specific performance trajectories for Cantonese, Malaysian, and Thai under different transfer strategies.
Bottom-left: Relative improvements within each domain.
Bottom-right: Average normalized performance across all domains.
See Section  5.3 for detailed analysis.

## 2 Related Work

Model Merging.
Model merging seeks to combine multiple expert models into a single model, typically without access to the original training data and without costly training ( Ilharco et al., 2022 ; Wortsman et al., 2022b ; Ainsworth et al., 2022 ) .
Existing approaches primarily involve two stages: pre-merging and during-merging, covering both foundation models (LLMs and MLLMs) ( Yang et al., 2024 ) and other deep learning settings ( Li et al., 2023 ) .
Pre-merging methods focus on preparing models for subsequent fusion by modifying their parameters or representations in advance.
Typical approaches include adaptation schemes designed to reduce parameter interference ( Jacot et al., 2018 ; Ortiz-Jimenez et al., 2023 ) and alignment techniques that map different checkpoints into compatible parameter spaces prior to merging ( Singh & Jaggi, 2020 ; Imfeld et al., 2023b ) .
During-merging methods perform the actual fusion, ranging from simple parameter averaging ( Wortsman et al., 2022a ) to more structured strategies such as subspace-based merging ( Yadav et al., 2023 ) , as well as optimization-based merging ( Wei et al., 2025 ) .
Although some methods attempt cross-architecture merging or knowledge fusion in LLMs ( Wan et al., 2024 ; Wan et al., 2025 ) , they generally rely on distillation-based training to first transfer knowledge into the parameter space of a target model.

Optimal Transport and Applications.
Optimal transport (OT) provides a principled framework for comparing probability measures by
finding a minimum-cost plan that transports mass from a source distribution to a target
distribution ( Peyré & Cuturi, 2019 ) .
Building on solid theory, OT has become a versatile tool across a broad range of applications.
In machine learning, OT is commonly used for distribution matching in generative modeling ( Rout et al., 2021 ) , domain adaptation under distribution shift ( Liu et al., 2023b ; Redko et al., 2019 ) , imitation learning ( Luo et al., 2023 ) , and clustering ( Del Barrio et al., 2019 ; Lin et al., 2023 ) .
More recently, OT has been explored as a mechanism for model merging, where a transport plan provides a soft correspondence between neurons and enables weight merging across models ( Imfeld et al., 2023a ; Singh & Jaggi, 2020 ) .
However, while recent studies have used optimal transport (OT) to reveal cross-model feature similarity even between heterogeneous LLMs ( Shah & Khosla, 2025 ) , how to effectively leverage OT for model merging with different architectures remains underexplored.

## 3 Preliminary

### 3.1 Optimal Transport

Optimal transport (OT) provides a principled way to compute a soft correspondence between
two sets of objects under a pairwise cost.
Given a cost matrix C ∈ ℝ n × m C\in\mathbb{R}^{n\times m} and two discrete distributions
a ∈ Δ n a\in\Delta_{n} , b ∈ Δ m b\in\Delta_{m} , OT solves

|
| min Q ∈ ℝ + n × m ⁡ ⟨ C , Q ⟩ s.t. Q ​ 𝟏 = a , Q ⊤ ​ 𝟏 = b . \min_{Q\in\mathbb{R}_{+}^{n\times m}}\ \langle C,Q\rangle\quad\text{s.t.}\quad Q\mathbf{1}=a,\ \ Q^{\top}\mathbf{1}=b.
|
| (1)

We adopt the entropically regularized variant,

|
| min Q ∈ ℝ + n × m ⁡ ⟨ C , Q ⟩ − ε ​ H ​ ( Q ) s.t. Q ​ 𝟏 = a , Q ⊤ ​ 𝟏 = b , \min_{Q\in\mathbb{R}_{+}^{n\times m}}\ \langle C,Q\rangle-\varepsilon H(Q)\quad\text{s.t.}\quad Q\mathbf{1}=a,\ \ Q^{\top}\mathbf{1}=b,
|
| (2)

where H ( Q ) = − ∑ i , j Q i ​ j ( log Q i ​ j − 1 ) H(Q)=-\sum_{i,j}Q_{ij}(\log Q_{ij}-1) and ε > 0 \varepsilon>0 .
The solution admits the Sinkhorn scaling form ( Cuturi, 2013 )
Q = diag ⁡ ( u ) ​ K ​ diag ​ ( v ) Q=\mathrm{diag}(u)\,K\,\mathrm{diag}(v) with K = exp ( − C / ε ) K=\exp(-C/\varepsilon) , and can be computed
by alternating updates on u u and v v .

### 3.2 Neurons and Features in Transformers

In this paper, we adopt a structural view of neurons in Transformer models.
Specifically, we use the term “neuron” to refer to a unit in the weight space of a linear
sublayer, corresponding to a single row or column of the associated weight matrix.
The values propagated through the network are
referred to as activated features.

Specifically, consider a linear transformation

|
| y = W ​ x , y=Wx,
|
| (3)

where W ∈ ℝ d out × d in W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}} .
We distinguish two neuron spaces induced by W W :
(i) input-side neurons correspond to the columns of W W (indexed by d in d_{\mathrm{in}} ), and
(ii) output-side neurons correspond to the rows of W W (indexed by d out d_{\mathrm{out}} ).
Accordingly, x ∈ ℝ d in x\in\mathbb{R}^{d_{\mathrm{in}}} and y ∈ ℝ d out y\in\mathbb{R}^{d_{\mathrm{out}}} represent
features in the input and output spaces, respectively.

In a Transformer block, the primary linear sublayers include the attention projections

|
| Q = W Q ​ h , K = W K ​ h , V = W V ​ h , Q=W_{Q}h,\quad K=W_{K}h,\quad V=W_{V}h,
|
| (4)

and the MLP projections

|
| z = W 1 ​ h , h ′ = σ ⁡ ( z ) , o = W 2 ​ h ′ , z=W_{1}h,\quad h^{\prime}=\sigma(z),\quad o=W_{2}h^{\prime},
|
| (5)

where σ ⁡ ( ⋅ ) \sigma(\cdot) denotes an elementwise nonlinearity.
In all these cases, rows and columns of the projection matrices define neurons in weight space,
while each dimension of the activations corresponds to a feature induced by these
neurons.

Figure 2 :
Illustration of cross-architecture merging pipeline.
Given a small dataset 𝒟 \mathcal{D} , we extract intermediate activations from a high-resource source model and a low-resource target model with heterogeneous architectures.
We then use optimal transport to infer layer- and feature-level correspondences, and leverage the resulting transport plans for direct parameter fusion.
Finally, the fused model can be optionally refined via residual-frozen adaptation, where the transferred residuals are kept fixed and only base weights are updated.

## 4 Method

Problem Definition.
Our goal is to merge models across heterogeneous architectures.
We consider a target model M A M_{A} with L L layers and a source model M B M_{B} with M M layers.

Given a set of T T samples 𝒟 = { x 1 , … , x T } \mathcal{D}=\{x_{1},\dots,x_{T}\} , we record
activations from both models across transformer modules:

|
| X ℓ \displaystyle X_{\ell}
| ∈ ℝ T × n ℓ , ℓ = 1 , … , L , \displaystyle\in\mathbb{R}^{T\times n_{\ell}},\quad\ell=1,\dots,L,
|
| (6)

|
| Y m \displaystyle Y_{m}
| ∈ ℝ T × n m ′ , m = 1 , … , M , \displaystyle\in\mathbb{R}^{T\times n^{\prime}_{m}},\quad m=1,\dots,M,
|

where each row corresponds to one input sample and each column corresponds to one feature channel.
Details of the data used for activation extraction and transport estimation are provided in
Appendix  B.1 .

### 4.1 Finding Feature and Layer Relationships using Optimal Transport

We use optimal transport (OT)  ( Peyré & Cuturi, 2019 ) to infer transport relationship matrices from
activations, which reveal how activated feature channels in a source model can be recombined to
match those in a target model.
Although the ultimate goal is to merge parameters (i.e., neuron weights), directly establishing
neuron-to-neuron correspondences is challenging across heterogeneous architectures.
Instead, we first infer correspondences in the activation space and then lift these
activation-level relationships to the neuron level for parameter transport.

Concretely, for each target layer ℓ \ell and source layer m m , we quantify cross-model similarity at
the level of activation channels by constructing a correlation-based cost matrix

|
| C ℓ ​ m [ i , j ] = d corr ( X ℓ [ : , i ] , Y m [ : , j ] ) ≜ 1 − ρ ( X ℓ [ : , i ] , Y m [ : , j ] ) , C^{\ell m}[i,j]=d_{\mathrm{corr}}\!\left(X_{\ell}[:,i],\,Y_{m}[:,j]\right)\triangleq 1-\rho\!\left(X_{\ell}[:,i],\,Y_{m}[:,j]\right),
|
| (7)

where ρ ⁡ ( ⋅ , ⋅ ) \rho(\cdot,\cdot) denotes the Pearson correlation coefficient.

Based on this cost matrix, we compute a feature relationship matrix
Q ℓ ​ m ∈ ℝ n ℓ × n m ′ Q^{\ell m}\in\mathbb{R}^{n_{\ell}\times n^{\prime}_{m}} by solving an entropically regularized optimal transport
problem:

|
| Q ℓ ​ m = arg ⁡ min Q ∈ 𝒯 ⁡ ( n ℓ , n m ′ ) ​ ⟨ C ℓ ​ m , Q ⟩ − ε ​ H ​ ( Q ) , Q^{\ell m}=\arg\min_{Q\in\mathcal{T}(n_{\ell},n^{\prime}_{m})}\ \langle C^{\ell m},Q\rangle-\varepsilon H(Q),
|
| (8)

where H ( Q ) = − ∑ i , j Q i ​ j ( log Q i ​ j − 1 ) H(Q)=-\sum_{i,j}Q_{ij}(\log Q_{ij}-1) and ε > 0 \varepsilon>0 .
The transportation polytope enforces balanced marginals,

|
| 𝒯 ( n ℓ , n m ′ ) = { Q ∈ ℝ + n ℓ × n m ′ : Q 𝟏 = 1 n ℓ 𝟏 , Q ⊤ 𝟏 = 1 n m ′ 𝟏 } . \mathcal{T}(n_{\ell},n^{\prime}_{m})=\Big\{Q\in\mathbb{R}_{+}^{n_{\ell}\times n^{\prime}_{m}}:Q\mathbf{1}=\tfrac{1}{n_{\ell}}\mathbf{1},\ Q^{\top}\mathbf{1}=\tfrac{1}{n^{\prime}_{m}}\mathbf{1}\Big\}.
|
| (9)

The resulting transport plan Q ℓ ​ m Q^{\ell m} specifies how each target activation channel can be
represented as a weighted combination of source channels.
Although inferred purely from activation statistics, Q ℓ ​ m Q^{\ell m} will later be used as a
neuron-level mixing operator for transporting source parameters into the target neuron coordinates.

To summarize the compatibility between target layer ℓ \ell and source layer m m , we aggregate the
feature-level correspondence into a scalar transport cost inspired by ( Shah & Khosla, 2025 )

|
| C layer ​ [ ℓ , m ] = ⟨ C ℓ ​ m , Q ℓ ​ m ⟩ , C_{\mathrm{layer}}[\ell,m]=\langle C^{\ell m},Q^{\ell m}\rangle,
|
| (10)

which yields a layer-to-layer cost matrix
C layer ∈ ℝ L × M C_{\mathrm{layer}}\in\mathbb{R}^{L\times M} .

Using these layer-wise costs, we further infer a global correspondence matrix
P ∈ ℝ L × M P\in\mathbb{R}^{L\times M} by solving another entropically regularized optimal transport problem:

|
| P = arg ⁡ min P ∈ 𝒯 ⁡ ( L , M ) ​ ⟨ C layer , P ⟩ − η ​ H ​ ( P ) , P=\arg\min_{P\in\mathcal{T}(L,M)}\ \langle C_{\mathrm{layer}},P\rangle-\eta H(P),
|
| (11)

where η > 0 \eta>0 and

|
| 𝒯 ( L , M ) = { P ∈ ℝ + L × M : P 𝟏 = 1 L 𝟏 , P ⊤ 𝟏 = 1 M 𝟏 } . \mathcal{T}(L,M)=\Big\{P\in\mathbb{R}_{+}^{L\times M}:P\mathbf{1}=\tfrac{1}{L}\mathbf{1},\ P^{\top}\mathbf{1}=\tfrac{1}{M}\mathbf{1}\Big\}.
|
| (12)

Each entry P ℓ ​ m P_{\ell m} quantifies the degree of correspondence between target layer ℓ \ell and source
layer m m .
Together, the feature-level transport plans { Q ℓ ​ m } \{Q^{\ell m}\} and the global matrix P P
specify how source neurons are used to construct target neurons.
All optimal transport problems are solved using Sinkhorn iterations ( Cuturi, 2013 ) ; implementation details about the algorithm are
provided in Appendix  A .

### 4.2 From Activated Features to Neurons: Weight Fusion via Feature and Layer Transport

Our transport matrices are estimated from activated features, while the ultimate objective
is to merge neurons and their associated parameters.
The key observation is that each feature dimension corresponds to a well-defined neuron axis in
the underlying weight matrix, specifically, a column in the input space or a row in the output space.
This correspondence allows us to lift feature-level relations into neuron-level mixing operators,
which are then used to transport source weights into the target neuron basis.

Feature- and Layer-wise Transport.
In attention modules, input-side (pre-projection) and output-side (post-projection) features lie in
different representation spaces and may induce different cross-layer alignment signals.
Accordingly, we estimate two feature-level transport plans,
Q in ℓ ​ m Q^{\ell m}_{\mathrm{in}} from pre-activation features and
Q out ℓ ​ m Q^{\ell m}_{\mathrm{out}} from post-activation features.
At the layer level, we compute two compatibility costs by aggregating the corresponding feature-level
transport objectives, and solve the layer-level OT separately for the pre- and post-sides,
yielding P pre P_{\mathrm{pre}} and P post P_{\mathrm{post}} , respectively.
We then combine them into an effective layer correspondence

|
| P eff ​ [ ℓ , m ] = P pre ​ [ ℓ , m ] ⋅ P post ​ [ ℓ , m ] , P_{\mathrm{eff}}[\ell,m]=\sqrt{P_{\mathrm{pre}}[\ell,m]\cdot P_{\mathrm{post}}[\ell,m]},
|
| (13)

Selective transport via top- k k neuron replacement.
To improve robustness across heterogeneous architectures and avoid unnecessary interference,
we restrict fusion to a small subset of neurons that are highly activated on the same dataset
𝒟 \mathcal{D} used for alignment.
Concretely, for each layer ℓ \ell and each projection module,
we run the target model on 𝒟 \mathcal{D} and record neuron activations via forward hooks.
For each neuron, we compute an activation strength score as the mean absolute activation over
samples
s j = 1 T ​ ∑ t = 1 T | h t , j | , s_{j}=\frac{1}{T}\sum_{t=1}^{T}\bigl|h_{t,j}\bigr|,
where h t , j h_{t,j} denotes the activation of neuron j j at sample t t .
We then select the top- k k neurons with the largest scores in each layer and projection module,
and record their indices. Details about selection can be found in Appendix A.5

Fusion is applied only to these selected neuron indices, while all remaining parameters
in the target model are left unchanged.
For attention projections, this corresponds to replacing the rows or columns of the weight
matrices associated with the selected neurons.
This design yields a partial weight fusion scheme that injects source knowledge only into
neurons that are both highly active on the target data and well-aligned across models.
Under this design, the fused weights at target layer ℓ \ell can be written as:

|
|
| W A ℓ , fused = W A ℓ \displaystyle W_{A}^{\ell,\mathrm{fused}}=W_{A}^{\ell}
|
| (14)

|
|
| + α ⋅ 𝐌 ℓ ⊙ ( ∑ m = 1 M P eff [ ℓ , m ] Q ℓ ​ m out W B m ( Q ℓ ​ m in ) ⊤ − W A ℓ ) , \displaystyle+\alpha\cdot\mathbf{M}^{\ell}\odot\Bigg(\sum_{m=1}^{M}P_{\mathrm{eff}}[\ell,m]\,Q^{\ell m}_{\mathrm{out}}\,W_{B}^{m}\,\bigl(Q^{\ell m}_{\mathrm{in}}\bigr)^{\!\top}-W_{A}^{\ell}\Bigg),
|

where 𝐌 ℓ \mathbf{M}^{\ell} is a binary neuron-level mask that is nonzero only on the selected
top- k k neuron indices, ⊙ \odot denotes masking at the neuron level, and
α ∈ [ 0 , 1 ] \alpha\in[0,1] controls the strength of replacement.
Bias terms are handled analogously.

### 4.3 Residual-Frozen Adaptation after Fusion

After fusion, only a small subset of neurons in the target model are modified by
transported source parameters, while the remaining parameters remain identical to the original
target model.
During this stage we freeze the transported residual parameters and train only
the original base parameters of the target model.
This design ensures that the transferred knowledge is not overwritten by subsequent optimization,
while still allowing the target model to adapt to the downstream task distribution.
Empirically, this residual-frozen adaptation consistently outperforms training the target model
alone under the same conditions, indicating that fusion provides a strong and stable initialization.

Residual Parameterization.
We parameterize fusion through a residual form that separates transferred neuron-level
knowledge from the original model parameters.
Specifically, weight of target layer ℓ \ell is expressed as

|
| W A ℓ , fused = W A ℓ , base + α ⋅ 𝐌 ℓ ⊙ Δ ​ W A ℓ , W_{A}^{\ell,\mathrm{fused}}=W_{A}^{\ell,\mathrm{base}}+\alpha\cdot\mathbf{M}^{\ell}\odot\Delta W_{A}^{\ell},
|
| (15)

where W A ℓ , base W_{A}^{\ell,\mathrm{base}} denotes the original weights of the target model M A M_{A} ,
𝐌 ℓ \mathbf{M}^{\ell} is a neuron-level mask that is nonzero only on the selected top- k k neurons,
⊙ \odot denotes neuron-level masking, and α ∈ [ 0 , 1 ] \alpha\in[0,1] controls the strength of replacement.
The transported residual Δ ​ W A ℓ \Delta W_{A}^{\ell} aggregates source operators across layers via
activation-aligned transport,

|
| Δ ​ W A ℓ = ∑ m = 1 M P eff ​ [ ℓ , m ] ​ Q out ℓ ​ m ​ W B m ​ ( Q in ℓ ​ m ) ⊤ , \Delta W_{A}^{\ell}=\sum_{m=1}^{M}P_{\mathrm{eff}}[\ell,m]\,Q^{\ell m}_{\mathrm{out}}\,W_{B}^{m}\,(Q^{\ell m}_{\mathrm{in}})^{\!\top},
|
| (16)

where P eff ​ [ ℓ , m ] P_{\mathrm{eff}}[\ell,m] is the effective layer correspondence and
Q in ℓ ​ m , Q out ℓ ​ m Q^{\ell m}_{\mathrm{in}},Q^{\ell m}_{\mathrm{out}} are the feature-level transport maps inferred
from activation statistics.
Only the neurons selected by 𝐌 ℓ \mathbf{M}^{\ell} participate in fusion; all others
remain unchanged.

Residual-Freezing and Adaptation.
During post-fusion adaptation, we freeze the transported residual Δ ​ W A ℓ \Delta W_{A}^{\ell} entirely and
optimize only the base parameters W A ℓ , base W_{A}^{\ell,\mathrm{base}} .
Gradients are therefore applied uniformly to the base weights, without any neuron-level freezing:

|
| ∂ ℒ ∂ Δ ​ W A ℓ = 0 , ∂ ℒ ∂ W A ℓ , base ≠ 0 . \frac{\partial\mathcal{L}}{\partial\Delta W_{A}^{\ell}}=0,\qquad\frac{\partial\mathcal{L}}{\partial W_{A}^{\ell,\mathrm{base}}}\neq 0.
|
| (17)

This separation preserves the transferred neuron-level representations while allowing the target
model to adapt globally.

Weight Folding.
After adaptation, we fold the residual back into the base weights to obtain a
parameter representation for inference:

|
| W A ℓ , final = W A ℓ , base + α ⋅ 𝐌 ℓ ⊙ Δ ​ W A ℓ . W_{A}^{\ell,\mathrm{final}}=W_{A}^{\ell,\mathrm{base}}+\alpha\cdot\mathbf{M}^{\ell}\odot\Delta W_{A}^{\ell}.
|
| (18)

This folding operation yields a model
that is architecturally identical to the target model, with transferred neuron-level
knowledge permanently absorbed into its parameters.

We summarize the overall transport-and-merge procedure with masked neuron replacement in
Alg.  1 .

Algorithm 1 Transport and Merge for Cross-Architecture Model Merging

0:

Target model M A M_{A} with L L layers, source model M B M_{B} with M M layers, samples 𝒟 = { x t } t = 1 T \mathcal{D}=\{x_{t}\}_{t=1}^{T} , fusion coefficient α \alpha

0:

Fused target model M A fused M_{A}^{\text{fused}}

1:

Activation extraction:

2:

Run M A M_{A} and M B M_{B} on 𝒟 \mathcal{D} to obtain activations { X ℓ } ℓ = 1 L \{X_{\ell}\}_{\ell=1}^{L} and { Y m } m = 1 M \{Y_{m}\}_{m=1}^{M}

3:

Feature-level transport:

4:

for each target layer ℓ \ell and source layer m m do

5:

Construct cost matrix C ℓ ​ m C^{\ell m} (Eq.  7 )

6:

Solve entropic OT to obtain Q in ℓ ​ m , Q out ℓ ​ m Q^{\ell m}_{\mathrm{in}},Q^{\ell m}_{\mathrm{out}} (Eq.  8 )

7:

end for

8:

Layer-level transport:

9:

Compute pre/post layer costs from the corresponding Q in ℓ ​ m Q^{\ell m}_{\mathrm{in}} and Q out ℓ ​ m Q^{\ell m}_{\mathrm{out}} (Eq.  10 )

10:

Solve OT to obtain P pre P_{\mathrm{pre}} and P post P_{\mathrm{post}} (Eq.  11 )

11:

Compute P eff P_{\mathrm{eff}} (Eq.  13 )

12:

Weight fusion:

13:

for each target layer ℓ \ell do

14:

Fuse weights using Eq.  14

15:

end for

16:

Optional post-fusion adaptation:

17:

Freeze transferred residuals, fine-tune base parameters, and fold residuals for inference (Eqs.  15 – 18 )

18:

Output: ℳ A fused \mathcal{M}_{\!A}^{\text{fused}}

### 4.4 Weight Transport as Representation-Space Transfer

Transporting source weights into the target model admits an exact interpretation in representation
space: it is equivalent to mapping target features into the source feature coordinates, applying the
source linear map there, and mapping the result back to the target space.

Setup.
Fix a target sublayer ℓ \ell in M A M_{A} and a source sublayer m m in M B M_{B} with linear operators (bias omitted)

|
| W A ℓ ∈ ℝ d A , out ℓ × d A , in ℓ , W B m ∈ ℝ d B , out m × d B , in m . W_{A}^{\ell}\in\mathbb{R}^{d_{A,\mathrm{out}}^{\ell}\times d_{A,\mathrm{in}}^{\ell}},\qquad W_{B}^{m}\in\mathbb{R}^{d_{B,\mathrm{out}}^{m}\times d_{B,\mathrm{in}}^{m}}.
|

Let Q in ℓ ​ m Q^{\ell m}_{\mathrm{in}} and Q out ℓ ​ m Q^{\ell m}_{\mathrm{out}} be transport relationship matrices estimated from
pre- and post-activation features. Define the induced coordinate maps

|
| Φ in ℓ ​ m ≜ ( Q in ℓ ​ m ) ⊤ , Φ out ℓ ​ m ≜ Q out ℓ ​ m . \Phi^{\ell m}_{\mathrm{in}}\triangleq(Q^{\ell m}_{\mathrm{in}})^{\top},\qquad\Phi^{\ell m}_{\mathrm{out}}\triangleq Q^{\ell m}_{\mathrm{out}}.
|
| (19)

We then define the transported source operator acting on target features as

|
| W ~ B → A ℓ ​ m ≜ Φ out ℓ ​ m ​ W B m ​ Φ in ℓ ​ m . \widetilde{W}_{B\to A}^{\ell m}\triangleq\Phi^{\ell m}_{\mathrm{out}}\,W_{B}^{m}\,\Phi^{\ell m}_{\mathrm{in}}.
|
| (20)

Theorem 4.1 (Representation-Space Interpretation of Weight Transport) .

For any target representation h A ∈ ℝ d A , in ℓ h_{A}\in\mathbb{R}^{d_{A,\mathrm{in}}^{\ell}} and any ( ℓ , m ) (\ell,m) ,

|
| W ~ B → A ℓ ​ m ​ h A = Φ out ℓ ​ m ​ ( W B m ​ ( Φ in ℓ ​ m ​ h A ) ) , \widetilde{W}_{B\to A}^{\ell m}h_{A}=\Phi^{\ell m}_{\mathrm{out}}\Big(W_{B}^{m}\big(\Phi^{\ell m}_{\mathrm{in}}h_{A}\big)\Big),
|
| (21)

Our fused target operator aggregates transported source operators across layers:

|
| W A ℓ , fused = ( 1 − α ) ​ W A ℓ + α ​ ∑ m = 1 M P eff ​ [ ℓ , m ] ​ W ~ B → A ℓ ​ m . W_{A}^{\ell,\mathrm{fused}}=(1-\alpha)\,W_{A}^{\ell}+\alpha\sum_{m=1}^{M}P_{\mathrm{eff}}[\ell,m]\,\widetilde{W}_{B\to A}^{\ell m}.
|
| (22)

Combining Eq.  22 with Theorem  4.1 shows that fusion realizes a
mixture of source-space computations driven by target features transferred into the corresponding
source coordinate systems.

Corollary 4.2 (Uniqueness Under Invertible Coordinate Maps) .

If Φ in ℓ ​ m \Phi^{\ell m}_{\mathrm{in}} and Φ out ℓ ​ m \Phi^{\ell m}_{\mathrm{out}} are invertible, then
W ~ B → A ℓ ​ m \widetilde{W}_{B\to A}^{\ell m} is the unique operator on the target space that reproduces the
source-layer computation under the target-to-source coordinate transfer in
Eq. 21 .

Proofs are provided in Appendix  A.4 .

Table 1 : Low-resource language transfer results on MalayMMLU.
Accuracy (%; higher is better).

| Category
| base model
| Fused w/o adaptation
| Fused w/ adaptation

| Humanities
| 42.30
| 46.19
| 48.81

| Language
| 38.37
| 40.52
| 41.68

| Others
| 44.50
| 46.63
| 48.69

| STEM
| 42.61
| 45.19
| 46.58

| Social Science
| 40.92
| 43.77
| 46.91

## 5 Experiments

### 5.1 High- to Low-Resource Language Transfer

Setup.
We refer to the original unfused target model as the Base Model . We consider two variants of our method: Fused w/o Adaptation, which performs transport-guided parameter fusion without any post-training, and Fused w/ Adaptation, which applies an adaptation stage. To ensure experimental consistency, we fix the source architecture to the LLaMA-3 8B family ( Grattafiori et al., 2024 ) in
this section.
Comparisons with other model families are deferred to later experiments in Section 5.4 . We adopt four low-resource languages.
Malaysian: Target is Malaysian-LLaMA-3-1B-Instruct  ( Mesolitica, 2025 ) .
Indonesian: Target is LLaMA-3-1B-Indonesian-QLoRA  ( Wahyurejeki, 2025 ) .
Thai: Target is Typhoon2-1B-Instruct  ( Pipatanakul et al., 2024 ) .
Cantonese: To better match linguistic proximity, we use a Chinese instruction-tuned source LLaMA3-Chinese-8B-Instruct  ( FlagAlpha, 2025 ) , and fuse into the target model LLaMA-3-1B-Instruct.
Details of the data used for adaptation and evaluation benchmarks are provided in Appendix  B.1 and Appendix  B.2 , respectively.

Malaysian.
Table  1 shows results of our method on MalayMMLU ( Poh et al., 2024 ) .
We can observe that our method consistently achieves the best performance.
Relative to the Base Model, Fused w/ Adaptation gains
+ 6.51 +6.51 (Humanities), + 3.31 +3.31 (Language), + 4.19 +4.19 (Others), + 3.97 +3.97 (STEM), and + 5.99 +5.99 (Social Science),
indicating successful transfer from the high-resource language.

Table 2 : Low-resource language transfer results on Indonesian benchmarks.
Accuracy (%; higher is better).

| Benchmark
| Base Model
| Fused w/o Adaptation
| Fused w/ Adaptation

| ARC (Indonesian)
| 23.4
| 24.0
| 23.7

| Belebele (Indonesian)
| 34.8
| 36.4
| 36.9

| TruthfulQA-MC (Indonesian)
| 35.9
| 36.5
| 36.6

| XCOPA (Indonesian)
| 59.6
| 59.8
| 60.0

| Average
| 38.43
| 39.18
| 39.30

Indonesian.
Table  2 reports results on Indonesian benchmarks ( Clark et al., 2018 ; Bandarkar et al., 2024 ; Lin et al., 2022 ; Ponti et al., 2020 ) .
Fusion alone already improves most tasks, suggesting effective language capability transfer without extra training.
With post-fusion adaptation, the model attains the best scores on three of four benchmarks,
while Fused w/o Adaptation remains best on ARC.

Table 3 : Low-resource language transfer results on CMMLU (Cantonese) ( Jiang et al., 2025 ) .
Accuracy (%; higher is better).

| Category
| Base Model
| Fused w/o Adaptation
| Fused w/ Adaptation

| Humanities
| 25.44
| 27.72
| 27.34

| Social Science
| 25.07
| 27.10
| 27.36

| STEM
| 25.22
| 26.63
| 26.63

| Others
| 25.84
| 29.21
| 28.99

| Average
| 25.26
| 27.41
| 27.44

Cantonese.
Table  3 shows accuracy of CMMLU benchmark ( Jiang et al., 2025 ) .
Fused w/ Adaptation improves all categories and raises the overall average from 25.26 25.26 to 27.44 27.44 ( + 2.18 +2.18 ).
The consistent gains suggest our fusion can transfer relevant knowledge despite different architectures.

Table 4 : Low-resource language transfer results on Thai benchmarks.
Score (higher is better).

| Benchmark
| Base Model
| Fused w/o Adaptation
| Fused w/ Adaptation

| MMLU (Thai)
| 0.15
| 0.13
| 0.17

| MGSM (Thai)
| 0.50
| 0.64
| 0.72

| XCOPA (Thai)
| 0.58
| 0.58
| 0.60

| Average
| 0.41
| 0.45
| 0.50

Thai.
Table  4 summarizes Thai benchmark ( Hendrycks et al., 2020 ; Ponti et al., 2020 ; Shi et al., 2022 ) results.
Fusion substantially boosts the MGSM task even without adaptation ( 0.50 → 0.64 0.50\rightarrow 0.64 ),
and adaptation further improves all reported benchmarks, achieving the best overall scores.
This indicates fusion provides effective transfer, while adaptation stabilizes and calibrates task performance.

### 5.2 General-to-Expert Domain Transfer

Setup.
We also evaluate general-domain to expert-domain transfer.
For Finance , we fuse LLaMA-3-1B-TEL-A-finance  ( TEL-LLM, 2025 ) with
Qwen2.5-7B  ( Bai et al., 2023 ) .
For Medical , we fuse medical version LLaMA ( Grattafiori et al., 2024 ) with LLaMA-3.2-8B.

Table 5 : General-domain to finance-domain transfer results on financial benchmarks.
Score (higher is better).

| Benchmark
| Base Model
| Fused w/o Adaptation
| Fused w/ Adaptation

| MMLU (Business Ethics)
| 0.36
| 0.37
| 0.38

| MMLU (Microeconomics)
| 0.34
| 0.39
| 0.37

| MMLU (Professional Accounting)
| 0.26
| 0.27
| 0.29

| Average
| 0.32
| 0.34
| 0.35

Finance.
Table  5 reports results on three finance-related MMLU subsets ( Hendrycks et al., 2020 ) .
Direct fusion without adaptation largely preserves the Base Model’s performance, yielding either marginal gains or near-parity across tasks, which indicates that cross-domain fusion does not introduce negative interference.
With post-fusion adaptation, the fused model consistently achieves the best performance on all three benchmarks.
These results suggest that fusion serves as an effective initialization that transfers general knowledge into the target domain.

Table 6 : General-domain to medical-domain transfer results on medical benchmarks.
Accuracy (%; higher is better).

| Benchmark
| Base Model
| Fused w/o Adaptation
| Fused w/ Adaptation

| MMLU (Anatomy)
| 49.6
| 50.0
| 51.1

| MMLU (Medical Genetics)
| 50.0
| 49.0
| 52.0

| MMLU (Professional Medicine)
| 39.3
| 39.7
| 40.5

| Average
| 46.30
| 46.23
| 47.87

Medical.
Table  6 shows consistent improvements from fusion on medical benchmarks ( Hendrycks et al., 2020 ) .
Even without adaptation, fusion yields positive gains on anatomy and professional medicine.
Moreover, Fused w/ Adaptation achieves the best results across all three tasks,
improving over the Base Model by + 1.5 +1.5 , + 2.0 +2.0 , and + 1.2 +1.2 , indicating stable consolidation of transferred medical knowledge.

### 5.3 Effectiveness of Cross-Architecture Merging

Figure 3 :
Average transport mass explained by the top- k k neuron correspondences, computed from the optimal transport plans and averaged over layers and modules.

Evidence of cross-architecture Representation Similarity.
We analyze representation similarity across heterogeneous models using the feature-level optimal
transport plans Q ℓ ​ m Q^{\ell m} learned during fusion (Eq. ( 8 )).
Each entry Q i ​ j ℓ ​ m Q^{\ell m}_{ij} measures the amount of activation mass transported from a source
feature channel to a target feature channel, under balanced marginal constraints.
To quantify how concentrated these correspondences are, we compute the percentage of transport mass captured by the top- k k entries of Q ℓ ​ m Q^{\ell m} .
Specifically, for each Q ℓ ​ m Q^{\ell m} , we sort all entries in descending order and accumulate the transport mass of the top- k k entries, reporting this quantity as a percentage of the total transport mass.
We then average this quantity across all layer pairs and modules.
Figure  3 plots the averaged transport mass.
The horizontal axis corresponds to the top- k k neuron correspondences, while the vertical axis shows
the average percentage of source-to-target transport mass.
Across all domains, a small number of neuron pairs explains a large fraction of the transport mass,
indicating sparse yet aligned internal representations across heterogeneous LLMs.

Performance Comparison with Merging Source Models of Different Sizes.
We further analyze how the capacity of the source model affects cross-architecture fusion
performance.
Using MalayMMLU as a representative benchmark, we merge Malaysian-LLaMA-3-1B-Instruct with
general-purpose LLaMA models of different scales (1B, 8B, and 32B).
Table  7 shows that increasing the source model size leads to consistent
performance improvements.
While merging with an 8B source already yields substantial gains across all categories,
scaling the source to 32B provides additional improvements.
This indicates that larger source models offer richer transferable representations that can be
selectively injected through our neuron-level fusion mechanism.

Table 7 : Effect of source model size on MalayMMLU.
Results report accuracy (%) of Fused w/ Adaptation with a fixed 1B target model.
Higher is better.

| Source Size
| Humanities
| Language
| Others
| STEM
| Social Science

| 1B
| 47.65
| 41.76
| 47.97
| 46.05
| 46.00

| 8B
| 48.81
| 41.68
| 48.69
| 46.58
| 46.91

| 32B
| 49.17
| 42.27
| 49.17
| 46.75
| 47.28

Compared with SFT Training and Distillation.
As illustrated in Figure  1 , our method consistently outperforms both SFT and distillation  ( Hinton et al., 2015 ) under normalized evaluation.
Notably, the adapted fusion achieves the highest relative performance in every domain, whereas SFT and distillation show domain-dependent variability.
These results suggest that effective cross-domain transfer requires not only shared representations, but also post-fusion adaptation to properly align domain-specific features. Details can be found in Appendix C.2 .

### 5.4 Robustness of Cross-Architecture Merging

Sensitivity to Source Model Backbone.
Figure  4 studies the sensitivity of our approach to the choice of
high-resource source backbone. We compare our method (Fused w/ Adaptation) using two different
source models, LLaMA3-8B and Qwen2.5-7B, while keeping the same low-resource target model.
Across both Malay and Indonesian benchmarks, we observe consistent improvements over the
Base Model.
This suggests that our method transfers task-relevant features rather than
overfitting to a specific source architecture, and remains robust across heterogeneous
source models.

Figure 4 :
Sensitivity to the choice of source backbone.
On the Malay and Indonesian benchmarks, our method consistently outperforms the Base Model when using either LLaMA3-8B or Qwen2.5-7B as the source model.

Sensitivity to Fusion Coefficient α \alpha .

Figure  5 analyzes the sensitivity of performance to the fusion coefficient α \alpha , which controls the contribution of the source model during parameter merging.
Across all categories, performance exhibits a consistent trend as α \alpha varies, indicating that our method is not overly sensitive to precise hyperparameter tuning.
Moderate values of α \alpha (around 0.05 0.05 – 0.15 0.15 ) yield the strongest overall performance.
However, when α \alpha is too small, the influence of the source model is limited, resulting in conservative improvements.
Conversely, overly large α \alpha values lead to performance degradation in several categories, suggesting that excessive source injection may introduce mismatched or noisy information under architectural differences.

Figure 5 :
Sensitivity analysis of the fusion coefficient α \alpha for our method (Fused w/ Adaptation) on MalayMMLU.
We report category-wise accuracy (%; higher is better); dashed lines denote the corresponding base-model performance.

Figure 6 :
Impact of cross-architecture fusion on general abilities across domains.
Bars show the average general score (higher is better) for the Base Model and fused variants, with and without post-fusion adaptation.

Impact on General Abilities.
Figure  6 evaluates how cross-architecture fusion affects general capabilities. Details of the general benchmarks used in this evaluation are provided in Appendix  B.2 .
Overall, fusion does not degrade general abilities, and preserves performance close to the Base Model.
For Malay and Cantonese, fusion with adaptation slightly improves the average general score, indicating that transferred knowledge can be integrated without harming generalization.
For Indonesian, Thai, and Medical settings, fused models achieve performance comparable to the Base Model, with differences within a narrow margin.
These results suggest that our transport-based merging selectively injects transferable knowledge while largely maintaining the target model’s original general capabilities.

## 6 Conclusion

We proposed an OT-based framework for cross-architecture model merging that aligns heterogeneous models in activation space and converts the resulting correspondences into direct weight-space fusion.
Across low-resource languages and expert domains, our approach consistently improves target models using only a small calibration set, and residual-frozen adaptation further strengthens performance while preserving general capabilities. Overall, transport-guided cross-architecture merging provides a practical and principled alternative to distillation when architectures differ.

## Impact Statement

This work improves knowledge transfer from high-resource models to smaller low-resource or domain-specific models with heterogeneous architectures.
We use activation-based optimal transport to estimate cross-architecture correspondences from a small input set for direct parameter fusion, optionally followed by lightweight adaptation.
Positive impacts include reducing data/compute requirements for low-resource languages and specialized domains.
Negative impacts include propagating biases, factual errors, or unsafe behaviors from the source model, and overstating robustness if evaluation is insufficient.
We therefore emphasize rigorous evaluation, transparent reporting, and appropriate safeguards when applying the method in practice.

## References

Achiam et al. (2023)

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.

Gpt-4 technical report.

arXiv preprint arXiv:2303.08774 , 2023.

Adimulam et al. (2022)

Adimulam, T., Chinta, S., and Pattanayak, S. K.

Transfer learning in natural language processing: Overcoming low-resource challenges.

International Journal of Enhanced Research In Science Technology & Engineering , 11:65–79, 2022.

Ainsworth et al. (2022)

Ainsworth, S. K., Hayase, J., and Srinivasa, S.

Git re-basin: Merging models modulo permutation symmetries.

arXiv preprint arXiv:2209.04836 , 2022.

Bai et al. (2023)

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al.

Qwen technical report.

arXiv preprint arXiv:2309.16609 , 2023.

Bandarkar et al. (2024)

Bandarkar, L., Liang, D., Muller, B., Artetxe, M., Shukla, S. N., Husa, D., Goyal, N., Krishnan, A., Zettlemoyer, L., and Khabsa, M.

The belebele benchmark: a parallel reading comprehension dataset in 122 language variants.

In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pp. 749–775, 2024.

Bisk et al. (2020)

Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al.

Piqa: Reasoning about physical commonsense in natural language.

In Proceedings of the AAAI conference on artificial intelligence , volume 34, pp. 7432–7439, 2020.

Cai et al. (2025)

Cai, Y., Zhang, J., He, H., He, X., Tong, A., Gan, Z., Wang, C., Xue, Z., Liu, Y., and Bai, X.

Llava-kd: A framework of distilling multimodal large language models.

In Proceedings of the IEEE/CVF International Conference on Computer Vision , pp. 239–249, 2025.

Cao et al. (2025)

Cao, X., Xu, M., Yu, X., Yao, J., Ye, W., Huang, S., Zhang, M., Tsang, I., Ong, Y.-S., Kwok, J. T., et al.

Analytical survey of learning with low-resource data: From analysis to investigation.

ACM Computing Surveys , 58(6):1–47, 2025.

Clark et al. (2018)

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O.

Think you have solved question answering? try arc, the ai2 reasoning challenge.

arXiv preprint arXiv:1803.05457 , 2018.

Cuturi (2013)

Cuturi, M.

Sinkhorn distances: Lightspeed computation of optimal transport.

Advances in neural information processing systems , 26, 2013.

Del Barrio et al. (2019)

Del Barrio, E., Cuesta-Albertos, J. A., Matrán, C., and Mayo-Íscar, A.

Robust clustering tools based on optimal transportation.

Statistics and Computing , 29(1):139–160, 2019.

FlagAlpha (2025)

FlagAlpha.

Llama3‐chinese‐8b‐instruct.

Hugging Face Model Card, 2025.

URL https://huggingface.co/FlagAlpha/LLaMA3-Chinese-8B-Instruct .

Flowers (2025)

Flowers, J. G.

Finance-Instruct-500k.

https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k , 2025.

Accessed: 2026-01-23.

Grattafiori et al. (2024)

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al.

The llama 3 herd of models.

arXiv preprint arXiv:2407.21783 , 2024.

Hendrycks et al. (2020)

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J.

Measuring massive multitask language understanding.

arXiv preprint arXiv:2009.03300 , 2020.

Hew et al. (2025)

Hew, Z. K., Low, J. X., Yang, S. J., and Chan, C. S.

Myculture: Exploring malaysia’s diverse culture under low-resource language constraints.

arXiv preprint arXiv:2508.05429 , 2025.

Hinton et al. (2015)

Hinton, G., Vinyals, O., and Dean, J.

Distilling the knowledge in a neural network.

arXiv preprint arXiv:1503.02531 , 2015.

Hsieh et al. (2023)

Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., and Pfister, T.

Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes.

In Findings of the Association for Computational Linguistics: ACL 2023 , pp. 8003–8017, 2023.

Huh et al. (2024)

Huh, M., Cheung, B., Wang, T., and Isola, P.

Position: the platonic representation hypothesis.

In Proceedings of the 41st International Conference on Machine Learning , pp. 20617–20642, 2024.

Ilharco et al. (2022)

Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., and Farhadi, A.

Editing models with task arithmetic.

arXiv preprint arXiv:2212.04089 , 2022.

Imfeld et al. (2023a)

Imfeld, M., Graldi, J., Giordano, M., Hofmann, T., Anagnostidis, S., and Singh, S. P.

Transformer fusion with optimal transport.

arXiv preprint arXiv:2310.05719 , 2023a.

Imfeld et al. (2023b)

Imfeld, M., Graldi, J., Giordano, M., Hofmann, T., Anagnostidis, S., and Singh, S. P.

Transformer fusion with optimal transport.

arXiv preprint arXiv:2310.05719 , 2023b.

Jacot et al. (2018)

Jacot, A., Gabriel, F., and Hongler, C.

Neural tangent kernel: Convergence and generalization in neural networks.

Advances in neural information processing systems , 31, 2018.

Jiang et al. (2025)

Jiang, J., Chen, P., Chen, L., Wang, S., Bao, Q., Kong, L., Li, Y., and Wu, C.

How well do llms handle cantonese? benchmarking cantonese capabilities of large language models.

In Findings of the Association for Computational Linguistics: NAACL 2025 , pp. 4464–4505, 2025.

Jin et al. (2022)

Jin, X., Ren, X., Preotiuc-Pietro, D., and Cheng, P.

Dataless knowledge fusion by merging weights of language models.

arXiv preprint arXiv:2212.09849 , 2022.

Kolomeitsev (2025)

Kolomeitsev, K.

Llm modules: Knowledge transfer from a large to a small model using enhanced cross-attention.

arXiv preprint arXiv:2502.08213 , 2025.

Konstantinidis et al. (2024)

Konstantinidis, T., Iacovides, G., Xu, M., Constantinides, T. G., and Mandic, D.

Finllama: Financial sentiment classification for algorithmic trading applications.

arXiv preprint arXiv:2403.12285 , 2024.

Li et al. (2023)

Li, W., Peng, Y., Zhang, M., Ding, L., Hu, H., and Shen, L.

Deep model fusion: A survey.

arXiv preprint arXiv:2309.15698 , 2023.

Lin et al. (2023)

Lin, R., Du, S., Wang, S., and Guo, W.

Multi-view clustering via optimal transport algorithm.

Knowledge-Based Systems , 279:110954, 2023.

Lin et al. (2022)

Lin, S., Hilton, J., and Evans, O.

Truthfulqa: Measuring how models mimic human falsehoods.

In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers) , pp. 3214–3252, 2022.

Liu (2022)

Liu, E. K.-Y.

Low-resource neural machine translation: A case study of cantonese.

In Proceedings of the Ninth Workshop on NLP for Similar Languages, Varieties and Dialects , pp. 28–40, 2022.

Liu et al. (2023a)

Liu, J., Liu, J., Wang, Q., Wang, J., Cai, X., Zhao, D., Wang, R., and Yan, R.

Retrieval-based knowledge transfer: An effective approach for extreme large language model compression.

In Findings of the association for computational linguistics: EMNLP 2023 , pp. 8643–8657, 2023a.

Liu et al. (2023b)

Liu, Y., Zhou, Z., and Sun, B.

Cot: Unsupervised domain adaptation with clustering and optimal transport.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pp. 19998–20007, 2023b.

Liu et al. (2024)

Liu, Z., Zhao, C., Iandola, F., Lai, C., Tian, Y., Fedorov, I., Xiong, Y., Chang, E., Shi, Y., Krishnamoorthi, R., et al.

Mobilellm: Optimizing sub-billion parameter language models for on-device use cases.

arXiv preprint arXiv:2402.14905 , 2024.

Luo et al. (2023)

Luo, Y., Jiang, Z., Cohen, S., Grefenstette, E., and Deisenroth, M. P.

Optimal transport for offline imitation learning.

arXiv preprint arXiv:2303.13971 , 2023.

Lynn (2024)

Lynn, S.

Cantonese Dialogue.

https://huggingface.co/datasets/stvlynn/Cantonese-Dialogue , 2024.

Accessed: 2026-01-23.

Mesolitica (2024)

Mesolitica.

Malaysian SFT.

https://huggingface.co/datasets/mesolitica/Malaysian-SFT , 2024.

Accessed: 2026-01-23.

Mesolitica (2025)

Mesolitica.

Malaysian‐llama‐3.2‐1b‐instruct.

Hugging Face Model Card, 2025.

URL https://huggingface.co/mesolitica/Malaysian-LLaMA-3.2-1B-Instruct .

Myakala & Naayini (2023)

Myakala, P. K. and Naayini, P.

Bridging the gap: Leveraging transfer learning for low-resource nlp tasks.

International Journal of Computer Techniques , 10(5), 2023.

Ortiz-Jimenez et al. (2023)

Ortiz-Jimenez, G., Favero, A., and Frossard, P.

Task arithmetic in the tangent space: Improved editing of pre-trained models.

Advances in Neural Information Processing Systems , 36:66727–66754, 2023.

Penedo et al. (2024)

Penedo, G., Kydlíček, H., Lozhkov, A., Mitchell, M., Raffel, C. A., Von Werra, L., Wolf, T., et al.

The fineweb datasets: Decanting the web for the finest text data at scale.

Advances in Neural Information Processing Systems , 37:30811–30849, 2024.

Perin et al. (2024)

Perin, G., Chen, X., Liu, S., Kailkhura, B., Wang, Z., and Gallagher, B.

Rankmean: Module-level importance score for merging fine-tuned llm models.

In Findings of the Association for Computational Linguistics: ACL 2024 , pp. 1776–1782, 2024.

Peyré & Cuturi (2019)

Peyré, G. and Cuturi, M.

Computational optimal transport: With applications to data science .

Now Foundations and Trends, 2019.

Pipatanakul et al. (2024)

Pipatanakul, K., Manakul, P., Nitarach, N., Sirichotedumrong, W., Nonesung, S., Jaknamon, T., Pengpun, P., Taveekitworachai, P., Na-Thalang, A., Sripaisarnmongkol, S., Jirayoot, K., and Tharnpipitchai, K.

Typhoon 2: A family of open text and multimodal thai large language models, 2024.

URL https://arxiv.org/abs/2412.13702 .

Poh et al. (2024)

Poh, S. C., Yang, S. J., Tan, J. M. L., Chieng, L. L. T. Y., Tan, J. X., Yu, Z., Mun, F. C., and Chan, C. S.

Malaymmlu: A multitask benchmark for the low-resource malay language.

In Findings of the Association for Computational Linguistics: EMNLP 2024 , pp. 650–669, 2024.

Ponti et al. (2020)

Ponti, E. M., Glavaš, G., Majewska, O., Liu, Q., Vulić, I., and Korhonen, A.

Xcopa: A multilingual dataset for causal commonsense reasoning.

arXiv preprint arXiv:2005.00333 , 2020.

Redko et al. (2019)

Redko, I., Courty, N., Flamary, R., and Tuia, D.

Optimal transport for multi-source domain adaptation under target shift.

In The 22nd International Conference on artificial intelligence and statistics , pp. 849–858. PMLR, 2019.

Rout et al. (2021)

Rout, L., Korotin, A., and Burnaev, E.

Generative modeling with optimal transport maps.

arXiv preprint arXiv:2110.02999 , 2021.

Sakaguchi et al. (2021)

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y.

Winogrande: An adversarial winograd schema challenge at scale.

Communications of the ACM , 64(9):99–106, 2021.

Sap et al. (2019)

Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y.

Socialiqa: Commonsense reasoning about social interactions.

arXiv preprint arXiv:1904.09728 , 2019.

Shah & Khosla (2025)

Shah, S. and Khosla, M.

Representational alignment across model layers and brain regions with hierarchical optimal transport.

arXiv preprint arXiv:2510.01706 , 2025.

Shekswess (2024)

Shekswess.

Medical LLaMA3 Instruct Dataset.

https://huggingface.co/datasets/Shekswess/medical_LLaMA3_instruct_dataset , 2024.

Accessed: 2026-01-23.

Shi et al. (2022)

Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S., Vosoughi, S., Chung, H. W., Tay, Y., Ruder, S., Zhou, D., et al.

Language models are multilingual chain-of-thought reasoners.

arXiv preprint arXiv:2210.03057 , 2022.

Singh & Jaggi (2020)

Singh, S. P. and Jaggi, M.

Model fusion via optimal transport.

Advances in Neural Information Processing Systems , 33:22045–22055, 2020.

Talmor et al. (2019)

Talmor, A., Herzig, J., Lourie, N., and Berant, J.

Commonsenseqa: A question answering challenge targeting commonsense knowledge.

In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , pp. 4149–4158, 2019.

Tao et al. (2024)

Tao, M., Zhang, C., Huang, Q., Ma, T., Huang, S., Zhao, D., and Feng, Y.

Unlocking the potential of model merging for low-resource languages.

arXiv preprint arXiv:2407.03994 , 2024.

TEL-LLM (2025)

TEL-LLM.

Llama‐3.2‐1b‐tel‐a‐finance.

Hugging Face Model Card, 2025.

URL https://huggingface.co/TEL-LLM/LLaMA-3.2-1B-TEL-A-finance .

Tian et al. (2025)

Tian, Y., Han, Y., Chen, X., Wang, W., and Chawla, N. V.

Beyond answers: Transferring reasoning capabilities to smaller llms using multi-teacher knowledge distillation.

In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining , pp. 251–260, 2025.

Wahyurejeki (2025)

Wahyurejeki.

Llama-3.2-1b-indonesian-qlora.

Hugging Face Model Card, 2025.

URL https://huggingface.co/wahyurejeki/LLaMA-3.2-1B-Indonesian-QLora .

Wan et al. (2024)

Wan, F., Huang, X., Cai, D., Quan, X., Bi, W., and Shi, S.

Knowledge fusion of large language models.

arXiv preprint arXiv:2401.10491 , 2024.

Wan et al. (2025)

Wan, F., Zhong, L., Yang, Z., Chen, R., and Quan, X.

Fusechat: Knowledge fusion of chat models.

In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing , pp. 21629–21653, 2025.

Wei et al. (2025)

Wei, Y., Tang, A., Shen, L., Hu, Z., Yuan, C., and Cao, X.

Modeling multi-task model merging as adaptive projective gradient descent.

arXiv preprint arXiv:2501.01230 , 2025.

Wen et al. (2024)

Wen, Q., Liang, J., Sierra, C., Luckin, R., Tong, R., Liu, Z., Cui, P., and Tang, J.

Ai for education (ai4edu): Advancing personalized education with llm and adaptive learning.

In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , pp. 6743–6744, 2024.

Wortsman et al. (2022a)

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., et al.

Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.

In International conference on machine learning , pp. 23965–23998. PMLR, 2022a.

Wortsman et al. (2022b)

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., et al.

Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.

In International conference on machine learning , pp. 23965–23998. PMLR, 2022b.

Xie et al. (2024)

Xie, Q., Chen, Q., Chen, A., Peng, C., Hu, Y., Lin, F., Peng, X., Huang, J., Zhang, J., Keloth, V., et al.

Me-llama: Foundation large language models for medical applications.

Research square , pp. rs–3, 2024.

Yadav et al. (2023)

Yadav, P., Tam, D., Choshen, L., Raffel, C., and Bansal, M.

Resolving interference when merging models.

arXiv preprint arXiv:2306.01708 , 1, 2023.

Yang et al. (2024)

Yang, E., Shen, L., Guo, G., Wang, X., Cao, X., Zhang, J., and Tao, D.

Model merging in llms, mllms, and beyond: Methods, theories, applications, and opportunities.

ACM Computing Surveys , 2024.

## Appendix A Optimal Transport Formulation Details

This appendix provides the full formulations underlying the transport plans used for
model merging. We present both the neuron-level transport within each layer pair and the global
layer-level mixing transport across the full depth.

### A.1 Neuron-Level Transport Polytope and Objective

To manage computation and filter out noise, we apply top- k k selection strategies at both the neuron and transport matrix levels.

Transport Matrix Sparsification.

For a target layer ℓ \ell with n ℓ n_{\ell} feature channels and a source layer m m with n m ′ n^{\prime}_{m} feature channels,
we define a cost matrix C inner ℓ ​ m ∈ ℝ n ℓ × n m ′ C^{\ell m}_{\mathrm{inner}}\in\mathbb{R}^{n_{\ell}\times n^{\prime}_{m}} , where each entry measures
the dissimilarity between activation channels (Eq.  7 ).

We compute a soft relationship matrix Q ℓ ​ m ∈ ℝ n ℓ × n m ′ Q^{\ell m}\in\mathbb{R}^{n_{\ell}\times n^{\prime}_{m}} by solving an entropically
regularized optimal transport problem:

|
| Q ℓ ​ m = arg ⁡ min Q ∈ 𝒯 ⁡ ( n ℓ , n m ′ ) ​ ⟨ C inner ℓ ​ m , Q ⟩ − ε ​ H ​ ( Q ) , Q^{\ell m}=\arg\min_{Q\in\mathcal{T}(n_{\ell},n^{\prime}_{m})}\ \langle C^{\ell m}_{\mathrm{inner}},Q\rangle-\varepsilon H(Q),
|
| (23)

where ε > 0 \varepsilon>0 , ⟨ A , B ⟩ = ∑ i , j A i ​ j ​ B i ​ j \langle A,B\rangle=\sum_{i,j}A_{ij}B_{ij} , and the entropy term is

|
| H ( Q ) = − ∑ i = 1 n ℓ ∑ j = 1 n m ′ Q i ​ j ( log Q i ​ j − 1 ) . H(Q)=-\sum_{i=1}^{n_{\ell}}\sum_{j=1}^{n^{\prime}_{m}}Q_{ij}\big(\log Q_{ij}-1\big).
|
| (24)

The transportation polytope enforces balanced marginals:

|
| 𝒯 ( n ℓ , n m ′ ) = { Q ∈ ℝ + n ℓ × n m ′ : Q 𝟏 = a , Q ⊤ 𝟏 = b } , \mathcal{T}(n_{\ell},n^{\prime}_{m})=\Big\{Q\in\mathbb{R}_{+}^{n_{\ell}\times n^{\prime}_{m}}:Q\mathbf{1}=a,\ Q^{\top}\mathbf{1}=b\Big\},
|
| (25)

where we use uniform marginals

|
| a = 1 n ℓ ​ 𝟏 ∈ ℝ n ℓ , b = 1 n m ′ ​ 𝟏 ∈ ℝ n m ′ . a=\tfrac{1}{n_{\ell}}\mathbf{1}\in\mathbb{R}^{n_{\ell}},\qquad b=\tfrac{1}{n^{\prime}_{m}}\mathbf{1}\in\mathbb{R}^{n^{\prime}_{m}}.
|
| (26)

Uniform marginals ensure that each target channel distributes equal total mass and each source channel
receives equal total mass, preventing degenerate, one-sided matchings.

### A.2 Layer-Level Transport for Global Mixing

Given neuron-level solutions { Q ℓ ​ m } \{Q^{\ell m}\} , we form the layer-level cost matrix
C layer ∈ ℝ L × M C_{\mathrm{layer}}\in\mathbb{R}^{L\times M} by

|
| C layer ​ [ ℓ , m ] = ⟨ C inner ℓ ​ m , Q ℓ ​ m ⟩ . C_{\mathrm{layer}}[\ell,m]=\langle C^{\ell m}_{\mathrm{inner}},Q^{\ell m}\rangle.
|
| (27)

We then compute the global layer mixing matrix P ∈ ℝ L × M P\in\mathbb{R}^{L\times M} via

|
| P = arg ⁡ min P ∈ 𝒯 ⁡ ( L , M ) ​ ⟨ C layer , P ⟩ − η ​ H ​ ( P ) , P=\arg\min_{P\in\mathcal{T}(L,M)}\ \langle C_{\mathrm{layer}},P\rangle-\eta H(P),
|
| (28)

where η > 0 \eta>0 and

|
| 𝒯 ( L , M ) = { P ∈ ℝ + L × M : P 𝟏 = a ¯ , P ⊤ 𝟏 = b ¯ } , \mathcal{T}(L,M)=\Big\{P\in\mathbb{R}_{+}^{L\times M}:P\mathbf{1}=\bar{a},\ P^{\top}\mathbf{1}=\bar{b}\Big\},
|
| (29)

with uniform marginals

|
| a ¯ = 1 L ​ 𝟏 ∈ ℝ L , b ¯ = 1 M ​ 𝟏 ∈ ℝ M . \bar{a}=\tfrac{1}{L}\mathbf{1}\in\mathbb{R}^{L},\qquad\bar{b}=\tfrac{1}{M}\mathbf{1}\in\mathbb{R}^{M}.
|
| (30)

These constraints enforce global balance: each target layer allocates its mass across source layers,
and each source layer receives comparable total mass, improving utilization during merging.

### A.3 Sinkhorn Solver and Stabilization

We solve the entropically regularized OT problems in
Eqs. ( 23 ) and ( 28 ) using Sinkhorn iterations.
Below we summarize the scaling-form derivation and the iterative updates.

#### A.3.1 Scaling Form

Consider the generic entropic OT problem

|
| min Q ∈ ℝ + n × m ⁡ ⟨ C , Q ⟩ − ε ​ H ​ ( Q ) s.t. Q ​ 𝟏 = a , Q ⊤ ​ 𝟏 = b . \min_{Q\in\mathbb{R}_{+}^{n\times m}}\ \langle C,Q\rangle-\varepsilon H(Q)\quad\text{s.t.}\quad Q\mathbf{1}=a,\ \ Q^{\top}\mathbf{1}=b.
|
| (31)

Define the Gibbs kernel

|
| K = exp ( − C / ε ) . K=\exp(-C/\varepsilon).
|
| (32)

The optimal solution has the form

|
| Q ⋆ = diag ⁡ ( u ) ​ K ​ diag ​ ( v ) , Q^{\star}=\mathrm{diag}(u)\,K\,\mathrm{diag}(v),
|
| (33)

where u ∈ ℝ + n u\in\mathbb{R}_{+}^{n} and v ∈ ℝ + m v\in\mathbb{R}_{+}^{m} are scaling vectors chosen to match the
marginals.

#### A.3.2 Sinkhorn Iterations

Given K K , Sinkhorn iterations alternately normalize rows and columns to satisfy the marginal
constraints:

|
| u ← a . / ( K v ) , v ← b . / ( K ⊤ u ) , u\leftarrow a./(Kv),\qquad v\leftarrow b./(K^{\top}u),
|
| (34)

where . / ./ denotes element-wise division.
After convergence, Q = diag ⁡ ( u ) ​ K ​ diag ​ ( v ) Q=\mathrm{diag}(u)K\mathrm{diag}(v) satisfies Q ​ 𝟏 ≈ a Q\mathbf{1}\approx a and
Q ⊤ ​ 𝟏 ≈ b Q^{\top}\mathbf{1}\approx b .

For the neuron-level problem, we use C = C inner ℓ ​ m C=C^{\ell m}_{\mathrm{inner}} , a = 1 n ℓ ​ 𝟏 a=\tfrac{1}{n_{\ell}}\mathbf{1} ,
b = 1 n m ′ ​ 𝟏 b=\tfrac{1}{n^{\prime}_{m}}\mathbf{1} .
For the layer-level problem, we use C = C layer C=C_{\mathrm{layer}} , a ¯ = 1 L ​ 𝟏 \bar{a}=\tfrac{1}{L}\mathbf{1} ,
b ¯ = 1 M ​ 𝟏 \bar{b}=\tfrac{1}{M}\mathbf{1} .

#### A.3.3 Practical Stabilization

In practice, the kernel K = exp ( − C / ε ) K=\exp(-C/\varepsilon) may suffer from numerical underflow when costs are
large or ε \varepsilon is small.
We therefore optionally employ standard stabilization strategies:
(i) log-domain Sinkhorn updates, and/or (ii) periodic rescaling of ( u , v ) (u,v) .
We stop iterations when the maximum marginal violation
max ⁡ { ‖ Q ​ 𝟏 − a ‖ ∞ , ‖ Q ⊤ ​ 𝟏 − b ‖ ∞ } \max\{\|Q\mathbf{1}-a\|_{\infty},\ \|Q^{\top}\mathbf{1}-b\|_{\infty}\} falls below a tolerance.

#### A.3.4 Hyperparameter Settings

We solve the entropically regularized OT problems using the Sinkhorn algorithm with the following specific hyperparameters:

Inner OT (Feature Alignment).

For computing Q ℓ ​ m Q^{\ell m} (Eq. ( 23 )):

•

Regularization ( ε \varepsilon ): We use ε = 0.1 \varepsilon=0.1 for standard text datasets (e.g., Malay, Cantonese) and a tighter ε = 0.03 \varepsilon=0.03 for the GSM8K math reasoning task.

•

Iterations: We use a memory-efficient streaming Sinkhorn solver with fixed 200 200 iterations.

•

Tolerance: Convergence tolerance is set to 10 − 6 10^{-6} .

Outer OT (Layer Mixing).

For computing P P (Eq. ( 28 )):

•

Regularization ( η \eta ): We set the default η = 0.1 \eta=0.1 . We employ an adaptive scaling mechanism where η \eta is increased if the maximum value of the cost matrix C layer C_{\mathrm{layer}} exceeds 1000, ensuring numerical stability.

•

Iterations: We allow up to 1000 1000 iterations.

•

Tolerance: We use a strict convergence tolerance of 10 − 9 10^{-9} with a numerical stability epsilon of 10 − 12 10^{-12} .

### A.4 Proofs for Representation-Space Interpretation

Proof of Theorem  4.1 .

By definition,

|
| W ~ B → A ℓ ​ m ​ h A = ( Φ out ℓ ​ m ​ W B m ​ Φ in ℓ ​ m ) ​ h A = Φ out ℓ ​ m ​ ( W B m ​ ( Φ in ℓ ​ m ​ h A ) ) , \widetilde{W}_{B\to A}^{\ell m}h_{A}=(\Phi^{\ell m}_{\mathrm{out}}W_{B}^{m}\Phi^{\ell m}_{\mathrm{in}})h_{A}=\Phi^{\ell m}_{\mathrm{out}}\big(W_{B}^{m}(\Phi^{\ell m}_{\mathrm{in}}h_{A})\big),
|

which is exactly Eq. ( 21 ).
∎

Proof of Corollary  4.2 .

Assume Φ in ℓ ​ m \Phi^{\ell m}_{\mathrm{in}} and Φ out ℓ ​ m \Phi^{\ell m}_{\mathrm{out}} are invertible.
Let U : ℝ d A , in ℓ → ℝ d A , out ℓ U:\mathbb{R}^{d_{A,\mathrm{in}}^{\ell}}\rightarrow\mathbb{R}^{d_{A,\mathrm{out}}^{\ell}} be any linear map such that

|
| U ​ h A = Φ out ℓ ​ m ​ ( W B m ​ ( Φ in ℓ ​ m ​ h A ) ) for all  ​ h A . Uh_{A}=\Phi^{\ell m}_{\mathrm{out}}\Big(W_{B}^{m}\big(\Phi^{\ell m}_{\mathrm{in}}h_{A}\big)\Big)\quad\text{for all }h_{A}.
|

Then

|
| U = Φ out ℓ ​ m ​ W B m ​ Φ in ℓ ​ m = W ~ B → A ℓ ​ m , U=\Phi^{\ell m}_{\mathrm{out}}W_{B}^{m}\Phi^{\ell m}_{\mathrm{in}}=\widetilde{W}_{B\to A}^{\ell m},
|

establishing uniqueness.
∎

### A.5 Neuron Selection for Replacement.

For the top- k k neuron replacement strategy, we set the default number of neurons to k = 128 k=128 .
The selection rule matches the main text: we run the target model on the calibration set 𝒟 \mathcal{D}
(used for transport estimation) and record neuron activations via forward hooks.
For each neuron j j in layer ℓ \ell , we compute its activation strength as the mean absolute
activation over the T T samples,

|
| s j = 1 T ​ ∑ t = 1 T | h t , j | , s_{j}=\frac{1}{T}\sum_{t=1}^{T}\bigl|h_{t,j}\bigr|,
|

where h t , j h_{t,j} denotes the activation of neuron j j on sample t t .
We then select the indices of the k k neurons with the highest s j s_{j} .

### A.6 Computational Complexity

We analyze the computational cost of the proposed optimal transport (OT)–based alignment in terms of
feature- and layer-level transport.

Feature-level OT.
For each target layer ℓ \ell and source layer m m , we solve an entropically regularized OT problem
between n ℓ n_{\ell} target features and n m ′ n^{\prime}_{m} source features using Sinkhorn iterations.
Each Sinkhorn update involves matrix–vector multiplications with the kernel
K ℓ ​ m ∈ ℝ n ℓ × n m ′ K^{\ell m}\in\mathbb{R}^{n_{\ell}\times n^{\prime}_{m}} , incurring
𝒪 ⁡ ( n ℓ ​ n m ′ ) \mathcal{O}(n_{\ell}n^{\prime}_{m}) time per iteration.
With I in I_{\mathrm{in}} Sinkhorn iterations, the total cost of one inner OT problem is
𝒪 ⁡ ( I in ​ n ℓ ​ n m ′ ) \mathcal{O}(I_{\mathrm{in}}\,n_{\ell}n^{\prime}_{m}) .

Across all layer pairs, the total feature-level OT cost is

|
| 𝒪 ⁡ ( I in ​ ∑ ℓ = 1 L ∑ m = 1 M n ℓ ​ n m ′ ) . \mathcal{O}\!\left(I_{\mathrm{in}}\sum_{\ell=1}^{L}\sum_{m=1}^{M}n_{\ell}n^{\prime}_{m}\right).
|

Layer-level OT.
The layer-level transport problem operates on a cost matrix
C layer ∈ ℝ L × M C_{\mathrm{layer}}\in\mathbb{R}^{L\times M} .
Each Sinkhorn iteration costs 𝒪 ⁡ ( L ​ M ) \mathcal{O}(LM) , and with I out I_{\mathrm{out}} iterations,
the total cost is 𝒪 ⁡ ( I out ​ L ​ M ) \mathcal{O}(I_{\mathrm{out}}\,LM) , which is negligible compared to
feature-level OT when n ℓ , n m ′ ≫ L , M n_{\ell},n^{\prime}_{m}\gg L,M .

Overall complexity and practicality.
In practice, I in I_{\mathrm{in}} and I out I_{\mathrm{out}} denote the numbers of Sinkhorn iterations for the
inner and outer OT problems (not to be confused with T T , the number of samples used for activation
extraction in Eq. ( 6 )).
We fix I in I_{\mathrm{in}} and I out I_{\mathrm{out}} to small constants (e.g., 200 and 1000), and feature
dimensions are moderate for selected layers and projection modules.
Moreover, OT estimation is performed once using a small dataset and does not
introduce overhead during inference.
As a result, the OT-based alignment adds a one-time preprocessing cost and remains practical for
cross-architecture fusion.

## Appendix B Experimental Setup

In this section, we detail the datasets used for training our models and the comprehensive benchmarks employed to evaluate their performance across diverse domains and languages.

### B.1 Stimulus and Training Datasets

We use stimulus datasets to extract representative activations for correspondence estimation,
and training datasets for task-specific adaptation when required.
Both are tailored to each domain and language.
For each dataset, we randomly sample 2000 examples.

#### B.1.1 Domain-Specific

•

Medical LLaMA3: We employ a specialized medical dataset (medical_LLaMA3) ( Shekswess, 2024 ) containing high-quality biomedical literature, clinical case reports, and medical guidelines to instill domain-specific medical knowledge.

•

Finance: To cover the financial domain, we utilize a dedicated finance dataset (finance) ( Flowers, 2025 ) comprising financial news, reports, and economic analysis texts, enabling the model to grasp complex economic concepts and terminology.

#### B.1.2 Multilingual Languages

•

Thai: We utilize fineweb_thai ( Penedo et al., 2024 ) , a subset of the FineWeb corpus focused on high-quality Thai web text, to improve language modeling performance in Thai.

•

Indonesian: We incorporate indonesian_conversation, a dataset designed to capture natural dialogue and conversational nuances in the Indonesian language.

•

Malay: For the Malay language, we use malaysian_sft ( Mesolitica, 2024 ) , a supervised adaptation dataset tailored to improve instruction following in Malaysian contexts.

•

Cantonese: We include a dedicated cantonese dataset ( Lynn, 2024 ) to address the linguistic specificities and colloquialisms of this regional language.

### B.2 Evaluation Benchmarks

We evaluate our model on a diverse set of benchmarks, categorized by language and domain to provide a holistic view of performance.

#### B.2.1 Malay Benchmarks

•

MMLU (Malay): To assess general world knowledge in Malay, we utilize the Malay version of the Massive Multitask Language Understanding (MMLU) benchmark  ( Poh et al., 2024 ) . This task evaluates the model’s accuracy in a zero-shot setting across a wide range of subjects—including STEM, humanities, and social sciences—adapted to the Malay language to test cross-lingual knowledge transfer.

#### B.2.2 Indonesian Benchmarks

•

ARC (Indonesian): The AI2 Reasoning Challenge (ARC)  ( Clark et al., 2018 ) consists of grade-school science questions designed to test complex reasoning and scientific knowledge. We utilize the Indonesian version to evaluate the model’s reasoning capabilities in a low-resource language context.

•

Belebele (Indonesian): Belebele  ( Bandarkar et al., 2024 ) is a multilingual machine reading comprehension (MRC) dataset. Based on the FLORES-200 passages, it evaluates the model’s ability to answer multiple-choice questions given a specific context. We report results on the Indonesian ( ind_Latn ) subset.

•

TruthfulQA-MC (Indonesian): TruthfulQA  ( Lin et al., 2022 ) evaluates the model’s truthfulness and its tendency to generate imitative falsehoods. We use the Indonesian translated version and report the average accuracy across the single-true (MC1) and multi-true (MC2) multiple-choice tasks.

•

XCOPA (Indonesian): The Cross-lingual Choice of Plausible Alternatives (XCOPA)  ( Ponti et al., 2020 ) assesses causal commonsense reasoning. The task requires the model to identify the correct cause or effect given a premise. We evaluate performance on the Indonesian subset ( xcopa_id ).

#### B.2.3 Cantonese Benchmarks

•

CMMLU (Cantonese): We evaluate the model’s proficiency in the Cantonese (Yue) dialect using the CMMLU dataset from the Yue-Benchmark suite  ( Jiang et al., 2025 ) . This benchmark consists of multiple-choice questions covering diverse disciplines such as history, physics, and culture. It is specifically designed to test the model’s ability to understand and reason with Cantonese-specific vocabulary, grammar, and colloquialisms.

#### B.2.4 Thai Benchmarks

•

MMLU (Thai): We assess general world knowledge in Thai from the MMLU benchmark  ( Hendrycks et al., 2020 ) (adapted as mmlu_prox_lite_th_other ). This subset comprises diverse multiple-choice questions across various uncategorized domains, evaluating the model’s breadth of knowledge beyond specific academic disciplines in the Thai language.

•

XCOPA (Thai): To evaluate causal reasoning in Thai, we utilize the Thai subset ( xcopa_th ) of the XCOPA benchmark  ( Ponti et al., 2020 ) . Similar to the Indonesian task, the model must select the most plausible alternative between cause and effect based on a premise.

•

MGSM (Thai): The Multilingual Grade School Math (MGSM) benchmark  ( Shi et al., 2022 ) evaluates arithmetic reasoning capabilities. We employ the Chain-of-Thought (CoT) prompting setting on the Thai subset ( mgsm_cot ) to assess the model’s ability to perform multi-step mathematical reasoning in Thai.

#### B.2.5 Finance Benchmarks

To assess the model’s capabilities in the finance and business domains, we utilize three relevant subsets from the MMLU benchmark  ( Hendrycks et al., 2020 ) :

•

MMLU (Business Ethics): This task ( global_mmlu_full_en_business_ethics ) evaluates the model’s ability to apply ethical principles in business scenarios, covering topics such as corporate governance, moral reasoning, and professional standards.

•

MMLU (Microeconomics): This subset ( global_mmlu_full_en_high_school_microeconomics ) covers fundamental economic concepts including supply and demand, market structures, and consumer behavior, reflecting a high school level understanding of microeconomic theory.

•

MMLU (Professional Accounting): Designed to test advanced accounting knowledge, this task ( global_mmlu_full_en_professional_ accounting ) includes questions on financial accounting, reporting standards, and auditing, corresponding to professional certification levels.

#### B.2.6 Medical Benchmarks

We evaluate domain-specific medical knowledge using three subsets from the Massive Multitask Language Understanding (MMLU) benchmark  ( Hendrycks et al., 2020 ) :

•

MMLU (Anatomy): This task tests the model’s knowledge of human anatomy through multiple-choice questions covering various body systems and structures, derived from academic and professional sources.

•

MMLU (Medical Genetics): This subset evaluates the model’s understanding of genetic principles, hereditary diseases, and clinical genetics, requiring specialized medical knowledge to answer correctly.

•

MMLU (Professional Medicine): This task assesses the model’s ability to apply medical knowledge in professional contexts. It includes questions typical of medical board examinations, covering diagnosis, treatment, and clinical decision-making.

### B.3 General Capabilities

To assess the model’s fundamental reasoning and commonsense abilities independent of specific domains or languages, we employ the following benchmarks:

•

ARC-Easy ( Clark et al., 2018 ) : The AI2 Reasoning Challenge (Easy Set) consists of grade-school science questions. It is designed to test the model’s ability to answer questions that require basic commonsense reasoning and world knowledge.

•

CommonsenseQA ( Talmor et al., 2019 ) : This dataset evaluates commonsense reasoning through multiple-choice questions that require prior knowledge to distinguish between plausible answers, challenging the model’s understanding of semantic relationships.

•

PIQA ( Bisk et al., 2020 ) : The Physical Interaction QA (PIQA) benchmark focuses on physical commonsense reasoning. It presents the model with everyday situations and asks it to predict the most plausible physical outcome or interaction.

•

Social IQA ( Sap et al., 2019 ) : To evaluate social intelligence, we use Social IQA, which tests the model’s ability to reason about social interactions, including understanding motivations, emotional reactions, and likely next steps in social contexts.

•

WinoGrande ( Sakaguchi et al., 2021 ) : This benchmark is a large-scale dataset for commonsense reasoning, formulated as fill-in-the-blank problems. It is designed to be more robust against annotation artifacts than the original Winograd Schema Challenge, requiring deep understanding to resolve ambiguous pronouns.

## Appendix C Details about Experiments

### C.1 Visualization of Feature-Level Transport Maps

We provide qualitative visualizations of feature-level optimal transport maps to further
illustrate cross-architecture representation similarity across heterogeneous architectures.
Figure  7 and Figure  8 visualize representative
transport plans Q ℓ ​ m Q^{\ell m} computed between target and source layers.

Each entry of Q ℓ ​ m Q^{\ell m} encodes the amount of transport mass between a target neuron (horizontal
axis) and a source neuron (vertical axis), where brighter values indicate stronger cross-architecture
feature similarity.
Across both language settings, the transport mass is highly non-uniform and concentrates on sparse,
localized regions rather than being evenly distributed.
Only a small subset of neuron pairs exhibits strong transport values, while the majority of entries
remain close to zero.

These sparse yet structured patterns indicate that heterogeneous language models share
well-aligned internal representations at the neuron level, rather than exhibiting random or
diffuse correspondence.
Such visual evidence complements our quantitative analysis and supports the design choice of
top- k k neuron replacement, as the most informative cross-architecture alignments are concentrated in a
small number of highly similar neuron pairs.

Figure 7 :
Feature-level optimal transport map between a LLaMA-3-1B model and a Chinese
LLaMA-8B model at layer 5 ( K projection).
Brighter values indicate stronger neuron-level alignment.

Figure 8 :
Feature-level optimal transport map between a Malaysian-version LLaMA-3-1B model and a LLaMA-8B model
at layer 0 ( K projection).
Similar to Cantonese, transport mass concentrates on sparse, high-similarity neuron pairs.

### C.2 Details about Comparison with SFT training and Distillation.

Evaluation Protocol and Normalization.

To enable a fair comparison across domains with heterogeneous score ranges, we normalize results on a per-domain basis.
Specifically, let x d , m x_{d,m} denote the raw performance of method m m on domain d d .
We apply two complementary normalization schemes.

Z-score normalization.
For each domain d d , we compute

|
| z d , m = x d , m − μ d σ d , z_{d,m}=\frac{x_{d,m}-\mu_{d}}{\sigma_{d}},
|

where μ d \mu_{d} and σ d \sigma_{d} are the mean and standard deviation of scores across all methods for domain d d .
This normalization highlights relative performance within each domain and is used to visualize method-wise trajectories and average trends.
Z-scores are shown in the top row (domain-specific trajectories) and the bottom-right panel (average across domains).

Min–max scaling.
We additionally apply min–max normalization per domain,

|
| x ^ d , m = x d , m − min d max d − min d , \hat{x}_{d,m}=\frac{x_{d,m}-\min_{d}}{\max_{d}-\min_{d}},
|

where min d \min_{d} and max d \max_{d} denote the minimum and maximum scores across methods for domain d d .
This scaling preserves the relative shape of improvements within each domain and is used in the bottom-left panel to compare improvement patterns across domains.

Figure Interpretation.

The top row of Figure  1 shows z-score–normalized performance trajectories for representative domains (Cantonese, Malay, and Thai), illustrating how different training strategies compare within each domain.
The bottom-left panel visualizes min–max–scaled scores across all domains, highlighting the relative improvement patterns of each method.
The bottom-right panel reports the average z-score across domains, summarizing overall cross-domain performance.

Together, these normalized views demonstrate that our adapted merging method consistently achieves the strongest relative performance across domains, while SFT and distillation exhibit more domain-dependent variability.

## Appendix D Additional Discussion

Q1: How sensitive is the method to the choice of the calibration set 𝒟 \mathcal{D} ?
Our framework estimates transport plans { Q ℓ ​ m } \{Q^{\ell m}\} and P P from activation statistics, and therefore relies on 𝒟 \mathcal{D} to elicit representative behaviors of the target task or domain.
If 𝒟 \mathcal{D} is severely mismatched, the resulting correspondences may become noisy and less informative.
To reduce this sensitivity, our design explicitly incorporates several stabilizing choices:
(i) using a moderately sized but task-relevant calibration set,
(ii) employing entropic regularization in OT to smooth noisy similarity estimates, and
(iii) restricting fusion to a small set of top- k k highly activated neurons to limit interference.
In practice, we find that lightweight data from the target language or domain is usually sufficient.
When no such data is available, approaches that rely on explicit supervision (e.g., distillation or adapters) may be more appropriate for that setting.

Q2: Under what conditions can cross-architecture fusion fail despite sharp OT correspondences?
Even when OT yields seemingly confident correspondences, effective fusion is not guaranteed if the target model lacks sufficient capacity to absorb the injected knowledge, or if the transferred features are misaligned with the target task.
These cases reflect fundamental capacity or task-mismatch constraints rather than failures of the transport mechanism itself.
Our method adopts a conservative strategy—top- k k neuron replacement combined with optional residual-frozen adaptation—to mitigate such risks by limiting the scope of transfer and allowing controlled recalibration of the remaining parameters.

Q3: Can fusion propagate undesirable behaviors or biases from the source model?
As with any direct parameter transfer method, our approach may propagate both beneficial capabilities and undesirable behaviors present in the source model.
This is an inherent consideration of weight-level reuse rather than a property unique to our framework.
In practical deployments, we recommend applying standard post-hoc safety and alignment pipelines (e.g., safety evaluation, filtering, or targeted alignment) to the fused model.
Incorporating safety-aware objectives directly into the transport formulation is an interesting direction for future work.
