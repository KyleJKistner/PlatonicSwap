---
title: "The Indra Representation Hypothesis for Multimodal Alignment"
authors: "Jianglin Lu, Hailing Wang, Kuo Yang, Yitian Zhang, Simon Jenni, Yun Fu"
date: "2026-04-06"
arxiv_id: "2604.04496"
url: "https://arxiv.org/abs/2604.04496"
cluster: "foundations_platonic"
tags: ["indra", "relational-representations", "angular-distance", "yoneda", "neurips-2025"]
status: "FULL_TEXT"
source: "arxiv_html (65,946 chars)"
---

# The Indra Representation Hypothesis for Multimodal Alignment

- **Authors:** Jianglin Lu, Hailing Wang, Kuo Yang, Yitian Zhang, Simon Jenni, Yun Fu
- **Published:** 2026-04-06 | **arXiv:** [2604.04496](https://arxiv.org/abs/2604.04496)
- **Cluster:** `foundations_platonic` | **Tags:** `indra`, `relational-representations`, `angular-distance`, `yoneda`, `neurips-2025`
- **Status:** `FULL_TEXT` (arxiv_html (65,946 chars))
- **Relevance to PlatonicSwap v1:** Foundational relational representation theory (angular distance used by Bakker & Louizos).
- **Relevance to PlatonicSwap v2:** Category-theoretic relational alignment across architectures/modalities.

## Abstract

Recent studies have uncovered an interesting phenomenon: unimodal foundation models tend to learn convergent representations, regardless of differences in architecture, training objectives, or data modalities. However, these representations are essentially internal abstractions of samples that characterize samples independently, leading to limited expressiveness. In this paper, we propose The Indra Representation Hypothesis, inspired by the philosophical metaphor of Indra's Net. We argue that representations from unimodal foundation models are converging to implicitly reflect a shared relational structure underlying reality, akin to the relational ontology of Indra's Net. We formalize this hypothesis using the V-enriched Yoneda embedding from category theory, defining the Indra representation as a relational profile of each sample with respect to others. This formulation is shown to be unique, complete, and structure-preserving under a given cost function. We instantiate the Indra representation using angular distance and evaluate it in cross-model and cross-modal scenarios involving vision, language, and audio. Extensive experiments demonstrate that Indra representations consistently enhance robustness and alignment across architectures and modalities, providing a theoretically grounded and practical framework for training-free alignment of unimodal foundation models. Our code is available at https://github.com/Jianglin954/Indra.

---

## Full Extracted Text (Searchable)

# The Indra Representation Hypothesis for
Multimodal Alignment

Jianglin Lu Hailing Wang Kuo Yang Yitian Zhang Simon Jenni Yun Fu
† † thanks: Equal Contribution. Corresponding author: JianglinLu@outlook.com .
Affiliation:  Department of Electrical and Computer Engineering, Northeastern University

Affiliation:  Khoury College of Computer Science, Northeastern University

Affiliation:  Adobe Research

Abstract

Recent studies have uncovered an interesting phenomenon: unimodal foundation models tend to learn convergent representations, regardless of differences in architecture, training objectives, or data modalities.
However, these representations are essentially internal abstractions of samples that characterize samples independently, leading to limited expressiveness.
In this paper,
we propose The Indra Representation Hypothesis , inspired by the philosophical metaphor of Indra’s Net.
We argue that representations from unimodal foundation models are converging to implicitly reflect a shared relational structure underlying reality, akin to the relational ontology of Indra’s Net.
We formalize this hypothesis using the 𝒱 \mathcal{V} -enriched Yoneda embedding from category theory, defining the Indra representation as a relational profile of each sample with respect to others. This formulation is shown to be unique, complete, and structure-preserving under a given cost function.
We instantiate the Indra representation using angular distance and evaluate it in cross-model and cross-modal scenarios involving vision, language, and audio. Extensive experiments demonstrate that Indra representations consistently enhance robustness and alignment across architectures and modalities, providing a theoretically grounded and practical framework for training-free alignment of unimodal foundation models.
Our code is available at https://github.com/Jianglin954/Indra .

## 1 Introduction

Through large-scale pretraining, foundation models have emerged as a transformative paradigm in artificial general intelligence, demonstrating impressive progress across diverse domains, such as natural language processing Devlin et al. (2019) ; He et al. (2020b) ; Liu et al. (2019) , computer vision Liu et al. (2022) ; Woo et al. (2023) ; Oquab et al. (2023) ; Dosovitskiy et al. (2021) , and speech processing Schneider et al. (2019) ; Baevski et al. (2020) .
These unimodal models are typically trained on web-scale datasets and acquire generalized representations that can be adapted to a broad range of downstream tasks.
The representative models, such as BERT Devlin et al. (2019) for language, ViT Dosovitskiy et al. (2021) for vision, and Wav2Vec Schneider et al. (2019) for audio, have demonstrated promising performance within their respective domains.

Since real-world information is inherently multimodal (text, images, and audio frequently co-occur and complement each other), relying on a single modality for understanding is typically insufficient.
To extend unimodal foundation models for cross-modal tasks, a growing body of research has exploited strong unimodal encoders as the core components to build multimodal systems Radford et al. (2021) ; Li et al. (2022) ; Radford et al. (2023) .
They assume that unimodal models are able to provide specialized, high-quality, and high-level representations for each individual modality, which can then be aligned, fused, or bridged to enable multimodal interactions Li et al. (2023) ; Liu et al. (2023) .
A common strategy to achieve multimodal understanding is to align unimodal outputs in a shared representation space through cross-modal objectives Radford et al. (2021) ; Li et al. (2023) ; Liu et al. (2023) ; Merullo et al. (2023) .
This usually relies on external mechanisms, such as alignment losses, fusion modules, and prompt tuning, thus requiring large-scale datasets and extensive retraining for modality alignment.

Interestingly, recent studies OpenAI (2024) ; Ng et al. (2023) ; Liu et al. (2023) ; Sharma et al. (2024) suggest that powerful unimodal models may already exhibit latent cross-modal capabilities, as the representations they produce (when grounded on the same physical entity) tend to describe the same underlying semantics from different sensory perspectives.
Previous evidence has revealed that adding only a single linear transformation is capable of bridging an auditory model with a large language model (LLM) Ngo and Kim (2024) , integrating a vision model into an LLM Merullo et al. (2023) , or conversely, stitching an LLM towards a vision model Koh et al. (2023) .
Even without retraining, well-pretrained vision encoders exhibit high semantic similarity with language encoders Maniparambil et al. (2024) .
Further studies Huh et al. (2024) ; Khosla et al. (2024) ; Hosseini et al. (2024) ; Lu et al. (2025b) have revealed that models trained on different data modalities converge, as different models are all trying to arrive at a representation of reality.
Thus, unimodal models may encode modality-agnostic representations in abstract representational space, even without explicit alignment.
While conceptually appealing, the specific form of convergent representations and their eventual convergence targets remain elusive and largely unexplored.

In this paper, we posit that the underlying convergent representation is inherently the Indra Representation—a conceptual abstraction inspired by the philosophical metaphor of Indra’s Net.
Originating in ancient Buddhist philosophy, Indra’s Net describes a vast, infinite web of jewels, each reflecting all others. Every jewel is both a part and a reflection of the whole, suggesting that all phenomena are interdependent, mutually defining, and inherently connected.
We draw an analogy between this worldview and the notion of representation convergence, and introduce The Indra Representation Hypothesis : well-trained unimodal models tend to produce convergent representations that implicitly reflect a shared relational structure underlying reality, echoing the relational ontology of Indra’s Net .
In this view, the representation of each entity is not defined in isolation, but rather emerges from its relational context, i.e., its reflections of all other entities.

To explore this hypothesis, we introduce a theoretical definition of the Indra representation grounded in category theory Kelly (1982) . Specifically, we define it as the 𝒱 \mathcal{V} -enriched Yoneda embedding of a sample within a category enriched over a Cost -category.
This formulation effectively maps each sample to its covariant Hom-functors in the sample category, thereby encoding its relational profile within the structure of the dataset.
We provide theoretical guarantees that the Indra representation is unique, complete, and structure-preserving, offering a principled foundation for its effectiveness.
In particular, we prove that it uniquely and completely characterizes each sample within the relational structure induced by a given cost function, while preserving essential properties of that structure.

To instantiate this theory in practice, we adopt angular distance as the cost function, yielding a simple yet powerful realization of the Indra representation. This concrete formulation enables empirical evaluation and allows us to investigate how Indra representations can uncover and support the latent cross-modal capabilities of unimodal models.
We validate our approach across a range of scenarios involving cross-model and cross-modal understanding, including single-modality settings, vision-language pairs, and speech-language pairs.
Extensive experiments demonstrate the effectiveness and generality of the proposed Indra representation across different architectures and modalities.

## 2 Preliminaries

### 2.1 Indra’s Net

Indra’s Net is a philosophical metaphor originating in ancient Indian and Mahāyāna Buddhist thought, particularly from the Avataṃsaka Sūtra Cleary (1993) ; Harvard FAS CAMLab (2022) .
It is used to symbolize the universe as a web of interdependent connections among all of its members, expressing the concept of interconnectedness, non-duality, and the interpenetration of all phenomena. Francis H. Cook describes it as Cook (1977) :

Imagine a vast, infinite cosmic net belonging to the god Indra. At each node of the net is a jewel or crystal. Each jewel reflects every other jewel in the net, and in each of those reflections are further reflections of all other jewels, recursively and infinitum.

The metaphor of Indra’s Net resonates deeply with the foundational principles across diverse disciplines.
For instance, Gergen et al. Gergen (2009) posit that identities, thoughts, and actions are not solely products of isolated minds but are co-constructed through interactions and relationships with others.
Markus et al. Markus and Kitayama (2014) propose the theory of interdependent self-construal, emphasizing that the self is defined relationally through one’s social roles, group memberships, and interpersonal obligations.
In physics, field theory Faraday (1859) ; Maxwell (1873) illustrates a relational structure where the field at any point depends on all sources throughout space, a concept reminiscent of Indra’s Net, in which each jewel reflects and is reflected by all others. Similarly, modern particle physics Griffiths (2020) ; Beringer et al. (2012) reveals the properties of an elementary particle through its interactions with other particles.
In linguistics, the linguistic principle articulated by J.R. Firth Firth (1957) , “You shall know a word by the company it keeps", asserts that a word’s meaning is derived from its co-occurrence with other words.
This principle forms the basis for modern language models like Word2Vec Mikolov et al. (2013) and its successors.
Furthermore, Kasulis Kasulis (2018) suggests DNA as a better image of Indra’s Net, where every cell contains the blueprint for the whole organism.

### 2.2 Representation Convergence

Recent studies Huh et al. (2024) ; Wentworth (2021) ; Gurnee et al. (2024) ; Wang et al. (2025) ; Lee et al. (2025) have revealed a striking phenomenon: unimodal foundation models tend to learn convergent representations, regardless of their architectures, training objectives, or data modalities.
For example, Bürger et al. Bürger et al. (2024) show that a two-dimensional representation of truth emerges universally across LLMs of varying sizes and from different model families.
Roeder et al. Roeder et al. (2021) prove that a wide class of discriminative and autoregressive models are identifiable in function space up to a linear transformation.
Tan et al. Tan et al. (2024) find strong correlations in both in-distribution and out-of-distribution steerability between LLaMA Touvron et al. (2023) and Qwen Bai et al. (2023) .
Huh et al. Huh et al. (2024) attribute this convergence to a shared goal: approximating an underlying representation of reality.
Khosla et al. Khosla et al. (2024) argue that both artificial and biological systems converge toward representations that capture the causal structure of the world.
Hosseini et al. Hosseini et al. (2024) further observe that high-performing artificial neural networks and biological brains tend to develop similar internal representations under naturalistic training conditions.
Additional evidence and analyses on representation convergence can be found in the comprehensive survey Lu et al. (2025b) .

Despite this emerging consensus, it remains unclear how these representations converge and what they ultimately converge to.
Notably, prior studies rely primarily on model outputs (embeddings) as proxies for representations, but these representations suffer from
① structural myopia: representations are typically treated as isolated carriers of information, ignoring structural interrelations within the broader data manifold;
② limited expressiveness: unimodal representations often exhibit inferior quality in matching and alignment compared to those from multimodal foundation models; and
③ dimensional incompatibility: representations across models and modalities often differ in dimensionality, thus requiring additional post-processing for cross-modal matching.
In light of these challenges, we argue that representations from model outputs do not reflect the final converged form, but instead serve as the foundation upon which such a form can be built .
In the next section, we introduce a novel representation hypothesis inspired by the metaphor of Indra’s Net to hypothesize the concrete structure of convergent representations and illuminate what they ultimately converge to.

## 3 Methodology

### 3.1 The Indra Representation Hypothesis

In this paper, we advocate a shift in perspective on representation convergence, inspired by the metaphor of Indra’s Net: a sample should be represented not in isolation, but through its pattern of relationships to other samples.
In this view, representations emerge from a structure of mutual interdependence.
We formalize this perspective through the Indra Representations Hypothesis:

This hypothesis posits that unimodal foundation models, after extensive pretraining, tend to produce representations that converge to capture the inherent relational structure of reality—a structure characterized by interdependence, contextuality, and mutual influence.
However, current methods that treat model outputs as final representations fail to reflect this structure.
These methods typically emphasize individual embedding information while neglecting the crucial relational patterns between samples.
In the next section, we introduce the Indra Representation , a novel representation framework inspired by the metaphor of Indra’s Net, to reveal the underlying relational structure.
In this framework, representations are not independent embeddings but mutually reflective entities woven into a web of interdependent relationships, revealing the deeper relational structure that underpins the data.

### 3.2 From Metaphor to Theory

Indra’s Net is a philosophical metaphor that symbolizes the interconnectedness of the universe. It aligns with the foundational principles in modern science, as mentioned in Section 2.1 .
To translate this philosophical insight into representation learning, we first introduce the Yoneda Lemma and its corollary, which provide the theoretical foundation for defining our proposed Indra representation.

Lemma 1 (Yoneda Lemma Kelly (1982) ; Riehl (2017) ) .

Let 𝒞 \mathcal{C} be a locally small category, A A be an object in 𝒞 \mathcal{C} , and F : 𝒞 → Set F:\mathcal{C}\to\textbf{Set} be a functor from 𝒞 \mathcal{C} to the category of sets. Then, there exists a bijection, natural in both A A and F F , between the set of natural transformations from the hom-functor h A = Hom 𝒞 ​ ( A , − ) h_{A}=\text{Hom}_{\mathcal{C}}(A,-) to F F , and the set F ⁡ ( A ) F(A) . This bijection is given by:

|
| Nat ​ ( h A , F ) ≅ F ⁡ ( A ) . \text{Nat}(h_{A},F)\cong F(A).
|
| (1)

Corollary 1 (Yoneda Embedding Kelly (1982) ; Riehl (2017) ) .

For any two objects A , B A,B in a locally small category 𝒞 \mathcal{C} , there is a bijection:

|
| Nat ​ ( Hom 𝒞 ​ ( A , − ) , Hom 𝒞 ​ ( B , − ) ) ≅ Hom 𝒞 ​ ( B , A ) . \text{Nat}(\text{Hom}_{\mathcal{C}}(A,-),\text{Hom}_{\mathcal{C}}(B,-))\cong\text{Hom}_{\mathcal{C}}(B,A).
|
| (2)

This demonstrates that the functor Y : 𝒞 o ​ p → [ 𝒞 , Set ] Y:\mathcal{C}^{op}\to[\mathcal{C},\textbf{Set}] , defined by Y ⁡ ( A ) = h A = Hom 𝒞 ​ ( A , − ) Y(A)=h_{A}=\text{Hom}_{\mathcal{C}}(A,-) , is fully faithful. This functor Y Y is known as the Yoneda embedding.

The Yoneda Lemma provides a profound understanding of how an object in a category is characterized by its relationships (morphisms) with all other objects, rather than by its internal properties. Its corollary further shows that any locally small category 𝒞 \mathcal{C} can be embedded into a category of presheaves on 𝒞 \mathcal{C} .
To introduce our Indra representations, we give the following definitions:

Definition 1 (Sample Category) .

Let 𝒳 \mathcal{X} be a set of samples, possibly infinite.
The sample category 𝒞 \mathcal{C} enriched over the Cost -category 𝒱 = ( [ 0 , ∞ ] , ≥ , 0 , + ) \mathcal{V}=([0,\infty],\geq,0,+) consists of:
① Objects: Ob ​ ( 𝒞 ) = 𝒳 \text{Ob}(\mathcal{C})=\mathcal{X} .
② Hom-objects: for every X i , X j ∈ Ob ​ ( 𝒞 ) X_{i},X_{j}\in\text{Ob}(\mathcal{C}) , the hom-object 𝒞 ⁡ ( X i , X j ) \mathcal{C}(X_{i},X_{j}) is given by a cost function d ⁡ ( X i , X j ) ∈ [ 0 , ∞ ] d(X_{i},X_{j})\in[0,\infty] , which is an object in 𝒱 \mathcal{V} .
③ Identity: for all X i ∈ Ob ​ ( 𝒞 ) X_{i}\in\text{Ob}(\mathcal{C}) , the identity morphism i ​ d X i : I → 𝒞 ⁡ ( X i , X i ) id_{X_{i}}:I\to\mathcal{C}(X_{i},X_{i}) in 𝒱 \mathcal{V} is 0 → d ⁡ ( X i , X i ) 0\to d(X_{i},X_{i}) , where d ⁡ ( X i , X i ) = 0 d(X_{i},X_{i})=0 .
④ Composition: for all X i , X j , X k ∈ Ob ​ ( 𝒞 ) X_{i},X_{j},X_{k}\in\text{Ob}(\mathcal{C}) , the composition morphism M X i , X j , X k : 𝒞 ⁡ ( X j , X k ) ⊗ 𝒞 ⁡ ( X i , X j ) → 𝒞 ⁡ ( X i , X k ) M_{X_{i},X_{j},X_{k}}:\mathcal{C}(X_{j},X_{k})\otimes\mathcal{C}(X_{i},X_{j})\to\mathcal{C}(X_{i},X_{k}) in 𝒱 \mathcal{V} is: d ⁡ ( X j , X k ) + d ⁡ ( X i , X j ) → d ⁡ ( X i , X k ) d(X_{j},X_{k})+d(X_{i},X_{j})\to d(X_{i},X_{k}) . This morphism exists in 𝒱 \mathcal{V} if and only if d ⁡ ( X j , X k ) + d ⁡ ( X i , X j ) ≥ d ⁡ ( X i , X k ) d(X_{j},X_{k})+d(X_{i},X_{j})\geq d(X_{i},X_{k}) , which is precisely the triangle inequality.

Definition 2 ( 𝒱 \mathcal{V} -enriched Yoneda embedding) .

Let [ 𝒞 o ​ p , 𝒱 ] [\mathcal{C}^{op},\mathcal{V}] be the category of 𝒱 \mathcal{V} -presheaves on 𝒞 \mathcal{C} . The 𝒱 \mathcal{V} -enriched Yoneda embedding is a 𝒱 \mathcal{V} -functor Y : 𝒞 → [ 𝒞 o ​ p , 𝒱 ] Y:\mathcal{C}\to[\mathcal{C}^{op},\mathcal{V}] .
For each object X i ∈ Ob ​ ( 𝒞 ) X_{i}\in\text{Ob}(\mathcal{C}) , Y ⁡ ( X i ) Y(X_{i}) is the 𝒱 \mathcal{V} -presheaf h X i : 𝒞 o ​ p → 𝒱 h_{X_{i}}:\mathcal{C}^{op}\to\mathcal{V} defined by: h X i ​ ( X j ) = 𝒞 ⁡ ( X j , X i ) = d ⁡ ( X j , X i ) h_{X_{i}}(X_{j})=\mathcal{C}(X_{j},X_{i})=d(X_{j},X_{i}) for any X j ∈ 𝒞 o ​ p X_{j}\in\mathcal{C}^{op} .
For every X i , X j ∈ Ob ​ ( 𝒞 ) X_{i},X_{j}\in\text{Ob}(\mathcal{C}) , Y Y defines a map Y X i , X j : 𝒞 ⁡ ( X i , X j ) → [ 𝒞 o ​ p , 𝒱 ] ​ ( Y ⁡ ( X i ) , Y ⁡ ( X j ) ) Y_{X_{i},X_{j}}:\mathcal{C}(X_{i},X_{j})\to[\mathcal{C}^{op},\mathcal{V}](Y(X_{i}),Y(X_{j})) .

Theorem 1 .

The 𝒱 \mathcal{V} -enriched Yoneda embedding Y : 𝒞 → [ 𝒞 o ​ p , 𝒱 ] Y:\mathcal{C}\to[\mathcal{C}^{op},\mathcal{V}] for the sample category 𝒞 \mathcal{C} enriched over 𝒱 = ( [ 0 , ∞ ] , ≥ , 0 , + ) \mathcal{V}=([0,\infty],\geq,0,+) with the cost function d d is 𝒱 \mathcal{V} -fully faithful.

The sample category 𝒞 \mathcal{C} actually forms a Lawvere metric space Lawvere (1973) , and the corresponding 𝒱 \mathcal{V} -enriched Yoneda embedding maps each sample X i X_{i} to a functor Y ⁡ ( X i ) Y(X_{i}) , which captures the cost profile from all other samples to X i X_{i} .
Theorem 1 shows that the sample category can be fully and faithfully represented within a category of 𝒱 \mathcal{V} -presheaf, preserving its entire structure including its metric information.
In other words, each sample X i X_{i} can be uniquely represented by its cost vector d ⁡ ( ⋅ , X i ) d(\cdot,X_{i}) with all samples.
Based on these, we introduce the Indra representation and state the following theorems:

Definition 3 ( Indra Representation ) .

For each sample X i ∈ Ob ​ ( 𝒞 ) X_{i}\in\text{Ob}(\mathcal{C}) , we define its Indra representation as the 𝒱 \mathcal{V} -functor 𝒞 ⁡ ( X i , − ) \mathcal{C}(X_{i},-) , i.e., the collection of values obtained by evaluating it under the (covariant) 𝒱 \mathcal{V} -enriched Yoneda embedding on all objects of the category 𝒞 \mathcal{C} .

Proposition 1 .

If two samples X i , X j ∈ Ob ​ ( 𝒞 ) X_{i},X_{j}\in\text{Ob}(\mathcal{C}) have 𝒱 \mathcal{V} -naturally isomorphic Indra representations and the cost function d d satisfies the T 0 T_{0} separation axiom, then X i = X j X_{i}=X_{j} .

Theorem 2 .

For any 𝒱 \mathcal{V} -functor P : 𝒞 → 𝒱 P:\mathcal{C}\to\mathcal{V} , the 𝒱 \mathcal{V} -hom-object of 𝒱 \mathcal{V} -natural transformations from the Indra representation of sample X i X_{i} to P P , denoted by [ 𝒞 , 𝒱 ] ​ ( 𝒞 ⁡ ( X i , − ) , P ) [\mathcal{C},\mathcal{V}](\mathcal{C}(X_{i},-),P) , is 𝒱 \mathcal{V} -isomorphic to P ⁡ ( X i ) P(X_{i}) .

Corollary 2 .

The relational structure among objects in the sample category 𝒞 \mathcal{C} is preserved and reflected in the relationships between their Indra representations.

The Indra representation of a sample X i X_{i} , defined as the 𝒱 \mathcal{V} -functor 𝒞 ⁡ ( X i , − ) \mathcal{C}(X_{i},-) , can be interpreted as a relational profile of X i X_{i} , that is, the cost from X i X_{i} to every other sample in the category. Proposition  1 establishes that the Indra representation is a faithful encoding: no two distinct samples share the same representation.
Theorem  2 further shows that the Indra representation is complete, in the sense that it encapsulates all the information needed to determine how distances from X i X_{i} behave under any admissible distance assignment.
Furthermore, Corollary  2 demonstrates that the relationships between samples are in one-to-one correspondence with the relationships between their Indra representations.

### 3.3 Instantiation of Indra Representation

We now demonstrate how to instantiate the Indra representation for a real dataset 𝒳 = { X 1 , … , X n } \mathcal{X}=\{X_{1},\ldots,X_{n}\} consisting of n n samples.
We define the object set of the enriched category as Ob ​ ( 𝒞 ) = 𝒳 \text{Ob}(\mathcal{C})=\mathcal{X} and specify the hom-object 𝒞 ⁡ ( X i , X j ) \mathcal{C}(X_{i},X_{j}) as the cost d ⁡ ( X i , X j ) d(X_{i},X_{j}) between samples X i X_{i} and X j X_{j} , ∀ X i , X j ∈ Ob ​ ( 𝒞 ) \forall X_{i},X_{j}\in\text{Ob}(\mathcal{C}) .
To define a valid Indra representation, the cost function d d must satisfy two properties:
① d ⁡ ( X i , X i ) = 0 d(X_{i},X_{i})=0 for ∀ X i ∈ 𝒳 \forall X_{i}\in\mathcal{X} ; and
② d ⁡ ( X i , X k ) ≤ d ⁡ ( X i , X j ) + d ⁡ ( X j , X k ) d(X_{i},X_{k})\leq d(X_{i},X_{j})+d(X_{j},X_{k}) for ∀ X i , X j , X k ∈ 𝒳 \forall X_{i},X_{j},X_{k}\in\mathcal{X} .
Several distance metrics satisfy these conditions. In this work, we adopt a simple yet effective choice by defining d ⁡ ( X i , X j ) d(X_{i},X_{j}) as the angular distance between the embeddings of X i X_{i} and X j X_{j} :

|
| d ⁡ ( X i , X j ) := arccos ⁡ ( f ​ ( X i ) ⊤ ​ f ​ ( X j ) ‖ f ⁡ ( X i ) ‖ ​ ‖ f ⁡ ( X j ) ‖ ) , ∀ X i , X j ∈ 𝒳 , d(X_{i},X_{j}):=\arccos\left(\frac{f(X_{i})^{\top}f(X_{j})}{\|f(X_{i})\|\|f(X_{j})\|}\right),\quad\forall X_{i},X_{j}\in\mathcal{X},
|
| (3)

where f : 𝒳 → ℝ d 𝒳 f:\mathcal{X}\rightarrow\mathbb{R}^{d^{\mathcal{X}}} is a modality-specific foundation model, f ⁡ ( X i ) f(X_{i}) denotes the internal representation of X i X_{i} produced by f f , and d 𝒳 d^{\mathcal{X}} is the output dimensionality of the model.
The use of angular distance ensures that d d defines a valid Lawvere cost function.
Given this cost function, the Indra representation of each sample X i ∈ 𝒳 X_{i}\in\mathcal{X} is the covariant Hom-functor:

|
| 𝒞 ⁡ ( X i , − ) : 𝒳 → [ 0 , ∞ ] , X j ↦ d ⁡ ( X i , X j ) . \mathcal{C}(X_{i},-):\mathcal{X}\to[0,\infty],\ \ \ X_{j}\mapsto d(X_{i},X_{j}).
|
| (4)

This distance-based embedding forms a principled and interpretable representation.
In the finite case, this can be written as 𝒞 ⁡ ( X i , − ) = [ d ⁡ ( X i , X 1 ) , … , d ⁡ ( X i , X n ) ] \mathcal{C}(X_{i},-)=\left[d(X_{i},X_{1}),\ldots,d(X_{i},X_{n})\right] , which captures the relational profile of X i X_{i} with respect to all other samples.
Notably, when the relational profile of each sample is computed using only a small set of representative landmarks, this formulation closely aligns with the approach proposed in Moschella et al. (2023) .

### 3.4 Relational Matching across Modalities

Our hypothesis in Section 3.1 posits that unimodal foundation models learn convergent representations that capture the shared relational structure underlying reality.
The proposed Indra representations are designed to reflect this inherent structure and can thus be leveraged to improve cross-modal understanding.
To demonstrate how Indra representations facilitate relational matching across modalities, we consider a dataset 𝒟 = { ( U i , Q i ) } i = 1 n \mathcal{D}=\{(U_{i},{Q}_{i})\}_{i=1}^{n} of n n samples, where U i ∈ 𝒰 {U}_{i}\in\mathcal{U} and Q i ∈ 𝒬 {Q}_{i}\in\mathcal{Q} correspond to instances from two distinct modalities, and 𝒟 ⊆ 𝒰 × 𝒬 \mathcal{D}\subseteq\mathcal{U}\times\mathcal{Q} .
For single-modality scenarios, we define U i = Q i , ∀ i ∈ { 1 , … , n } U_{i}=Q_{i},\ \forall i\in\{1,\ldots,n\} .
We use two pretrained foundation models f : 𝒰 → ℝ d 𝒳 f:\mathcal{U}\rightarrow\mathbb{R}^{d^{\mathcal{X}}} and g : 𝒬 → ℝ d 𝒴 g:\mathcal{Q}\rightarrow\mathbb{R}^{d^{\mathcal{Y}}} to extract modality-specific embeddings, where d 𝒳 d^{\mathcal{X}} and d 𝒴 d^{\mathcal{Y}} are the embedding dimensionalities of the two models, respectively.
Based on these embeddings, we construct the Indra representations 𝐈 𝒰 \mathbf{I}^{\mathcal{U}} and 𝐈 𝒬 \mathbf{I}^{\mathcal{Q}} for each modality as follows:

|
| 𝐈 i ​ j 𝒰 = d ( U i , U j ) , 𝐈 i ​ j 𝒬 = d ( Q i , Q j ) , ∀ i , j ∈ { 1 , … , n } , \mathbf{I}_{ij}^{\mathcal{U}}=d(U_{i},U_{j}),\quad\mathbf{I}_{ij}^{\mathcal{Q}}=d(Q_{i},Q_{j}),\quad\forall i,j\in\{1,\ldots,n\},
|
| (5)

where the cost function d d is defined in Equation 3 .
In practice, we may apply post-processing operations such as sparsification and normalization to enhance robustness:

|
| 𝐈 ^ 𝒰 = 𝚘𝚙𝚎𝚛𝚊𝚝𝚘𝚛 ⁡ ( 𝐈 𝒰 ) , 𝐈 ^ 𝒬 = 𝚘𝚙𝚎𝚛𝚊𝚝𝚘𝚛 ⁡ ( 𝐈 𝒬 ) , \hat{\mathbf{I}}^{\mathcal{U}}=\mathtt{operator}(\mathbf{I}^{\mathcal{U}}),\quad\hat{\mathbf{I}}^{\mathcal{Q}}=\mathtt{operator}(\mathbf{I}^{\mathcal{Q}}),
|
| (6)

where 𝚘𝚙𝚎𝚛𝚊𝚝𝚘𝚛 ⁡ ( ⋅ ) \mathtt{operator}(\cdot) denotes the chosen post-processing function.
Unlike traditional representation approaches that reflect only internal characteristics of samples, the proposed Indra representations act as external representations, where each vector captures interdependencies by encoding the sample’s relative profile within the dataset.

## 4 Experiments

To comprehensively assess the effectiveness of our Indra representation, we perform evaluations across a range of settings, including unimodal vision, vision–language, and speech–language tasks.

### 4.1 Evaluation on Single Modality

Table 1: Accuracy (%) on CIFAR-10 and CIFAR-100 under different Gaussian noise levels.

| CIFAR-10
| σ = 0.0 \sigma{=}0.0
| σ = 3.0 \sigma{=}3.0
| σ = 5.0 \sigma{=}5.0
| σ = 7.0 \sigma{=}7.0

| ViT
| 93.98 93.98
| 87.75 87.75
| 79.77 79.77
| 68.15 68.15

| Indra
| 94.84 94.84
| 89.51 89.51
| 80.84 80.84
| 68.71 68.71

| Convnext
| 97.00 97.00
| 85.89 85.89
| 80.10 80.10
| 65.85 65.85

| Indra
| 97.21 97.21
| 92.86 92.86
| 81.59 81.59
| 66.64 66.64

| Dinov2
| 99.19 99.19
| 95.21 95.21
| 85.57 85.57
| 76.54 76.54

| Indra
| 99.14 99.14
| 96.87 96.87
| 89.73 89.73
| 77.92 77.92

| CIFAR-100
| σ = 0.0 \sigma{=}0.0
| σ = 3.0 \sigma{=}3.0
| σ = 5.0 \sigma{=}5.0
| σ = 7.0 \sigma{=}7.0

| ViT
| 79.45 79.45
| 54.69 54.69
| 35.76 35.76
| 27.45 27.45

| Indra
| 80.09 80.09
| 69.00 69.00
| 51.59 51.59
| 32.74 32.74

| Convnext
| 85.77 85.77
| 62.79 62.79
| 34.39 34.39
| 21.28 21.28

| Indra
| 85.64 85.64
| 72.16 72.16
| 51.51 51.51
| 30.25 30.25

| Dinov2
| 91.97 91.97
| 82.21 82.21
| 63.06 63.06
| 40.16 40.16

| Indra
| 91.93 91.93
| 84.83 84.83
| 74.29 74.29
| 58.67 58.67

Table 2: Accuracy (%) on Office-Home dataset under different Gaussian noise levels.

| Art
| σ = 0.0 \sigma{=}0.0
| σ = 3.0 \sigma{=}3.0
| σ = 5.0 \sigma{=}5.0
| σ = 7.0 \sigma{=}7.0

| ViT
| 80.25 80.25
| 64.40 64.40
| 44.03 44.03
| 22.63 22.63

| Indra
| 79.63 79.63
| 65.02 {65.02}
| 43.62 43.62
| 27.57 {27.57}

| Convnext
| 89.71 89.71
| 62.76 62.76
| 27.98 27.98
| 12.14 12.14

| Indra
| 87.86 87.86
| 59.88 59.88
| 28.81 {28.81}
| 14.20 {14.20}

| Dinov2
| 87.65 87.65
| 73.05 73.05
| 46.91 46.91
| 27.78 27.78

| Indra
| 87.04 87.04
| 70.99 70.99
| 47.53 {47.53}
| 27.37 27.37

| Product
| σ = 0.0 \sigma{=}0.0
| σ = 3.0 \sigma{=}3.0
| σ = 5.0 \sigma{=}5.0
| σ = 7.0 \sigma{=}7.0

| ViT
| 92.34 92.34
| 80.74 80.74
| 61.15 61.15
| 35.25 35.25

| Indra
| 89.75 {89.75}
| 81.53 {81.53}
| 64.08 64.08
| 40.77 40.77

| Convnext
| 96.62 96.62
| 84.91 84.91
| 44.26 44.26
| 19.37 19.37

| Indra
| 96.73 {96.73}
| 85.92 {85.92}
| 45.61 {45.61}
| 22.18 22.18

| Dinov2
| 96.73 96.73
| 93.24 93.24
| 83.33 83.33
| 60.70 60.70

| Indra
| 96.40 96.40
| 92.79 92.79
| 84.46 {84.46}
| 60.59 {60.59}

| Clipart
| σ = 0.0 \sigma{=}0.0
| σ = 3.0 \sigma{=}3.0
| σ = 5.0 \sigma{=}5.0
| σ = 7.0 \sigma{=}7.0

| ViT
| 73.20 73.20
| 50.40 50.40
| 28.64 28.64
| 15.23 15.23

| Indra
| 69.76 69.76
| 54.98 54.98
| 33.10 {33.10}
| 18.21 18.21

| Convnext
| 83.62 83.62
| 54.07 54.07
| 20.85 20.85
| 09.74 09.74

| Indra
| 82.70 82.70
| 57.85 57.85
| 25.09 {25.09}
| 11.34 11.34

| Dinov2
| 88.43 88.43
| 75.14 75.14
| 51.09 51.09
| 31.04 31.04

| Indra
| 87.29 87.29
| 76.63 76.63
| 54.75 {54.75}
| 33.56 {33.56}

| Real
| σ = 0.0 \sigma{=}0.0
| σ = 3.0 \sigma{=}3.0
| σ = 5.0 \sigma{=}5.0
| σ = 7.0 \sigma{=}7.0

| ViT
| 89.22 89.22
| 82.11 82.11
| 60.09 60.09
| 35.32 35.32

| Indra
| 87.16 87.16
| 83.49 {83.49}
| 63.65 63.65
| 40.48 40.48

| Convnext
| 93.46 93.46
| 82.11 82.11
| 38.30 38.30
| 17.78 17.78

| Indra
| 93.35 93.35
| 84.63 84.63
| 40.71 {40.71}
| 19.61 {19.61}

| Dinov2
| 92.78 92.78
| 87.39 87.39
| 71.44 71.44
| 48.51 48.51

| Indra
| 92.89 92.89
| 88.53 {88.53}
| 73.17 73.17
| 49.89 {49.89}

Datasets.
We first conduct classification tasks on the CIFAR-10 Krizhevsky (2009) , CIFAR-100 Krizhevsky (2009) , and Office-Home Venkateswara et al. (2017) datasets. For CIFAR-10 and CIFAR-100, we use the standard data splits. For Office-Home, we evaluate classification accuracy across four distinct domains: Art, Clipart, Product, and Real-World, using an 80 / 20 80/20 split for training and testing. Each domain exhibits unique visual styles and distribution shifts, making the dataset a widely used benchmark for evaluating the robustness and generalization of vision models in object recognition tasks. Across all datasets, we adopt logistic regression (i.e., linear probing) to assess the quality of the extracted representations.

Foundation Models. For vision models, we evaluate ViT Dosovitskiy et al. (2021) , Convnext Woo et al. (2023) , and Dinov2 Oquab et al. (2023) .

Evaluation Metrics.
We assess model classification accuracy (%) using ground-truth labels.
To investigate the robustness of Indra representations, we inject Gaussian noise into the features with varying standard deviations σ ∈ { 0.0 , 3.0 , 5.0 , 7.0 } \sigma\in\{0.0,3.0,5.0,7.0\} . For each noise level, we perturb the features accordingly and train a linear classifier on the noisy representations. This allows us to assess how classification performance degrades as the feature representations are increasingly corrupted by noise.

Analysis.
In Tables 1 and 2 , we report the classification results on the CIFAR-10, CIFAR-100, and Office-Home datasets.
The results clearly show that stronger backbone models (e.g., Dinov2 ) lead to better performance for Indra representations across all noise levels. For instance, on CIFAR-100 with σ = 0.0 \sigma{=}0.0 , our Indra representations achieve 91.93 % 91.93\% accuracy using Dinov2 as the backbone, compared to 85.64 % 85.64\% with Convnext and 80.09 % 80.09\% with ViT .
This performance gap persists and even widens under higher noise: at σ = 7.0 \sigma=7.0 , Indra representations with Dinov2 maintain 58.67 % 58.67\% , while Convnext and ViT drop to 30.25 % 30.25\% and 32.74 % 32.74\% , respectively.
In addition, as Gaussian noise increases, our Indra representations consistently retain higher classification accuracy compared to the original representations, highlighting their robustness in the classification tasks.
The performance gains of Indra representations hold across multiple backbone architectures (i.e., ViT , Convnext , and Dinov2 ), indicating the broad applicability of the proposed method.

Table 3: Performance on image-text datasets 𝒟 \mathcal{D} using different representations ℛ \mathcal{R} ( Orign : original, Indra : Indra representation) with various vision ( Vis-E ) and language ( Lan-E ) models.

| 𝒟 \mathcal{D}
| Vis-E
| Lan-E
| ℛ \mathcal{R}
| Top- 5 5
| Top- 10 10
| Top- 30 30
| Top- 50 50

| T → \rightarrow I
| I → \rightarrow T
| T → \rightarrow I
| I → \rightarrow T
| T → \rightarrow I
| I → \rightarrow T
| T → \rightarrow I
| I → \rightarrow T

|

MS-COCO

| CLIP-I
| CLIP-T
| Orign
| 1.420 1.420
| 1.381 1.381
| 2.734 2.734
| 2.661 2.661
| 7.634 7.634
| 7.470 7.470
| 12.212 12.212
| 11.986 11.986

| ViT
| BERT
| Orign
| 0.482 0.482
| 0.483 0.483
| 0.967 0.967
| 0.966 0.966
| 2.911 2.911
| 2.905 2.905
| 4.863 4.863
| 4.846 4.846

| Indra
| 0.663 0.663
| 0.832 0.832
| 1.303 1.303
| 1.613 1.613
| 3.787 3.787
| 4.426 4.426
| 6.199 6.199
| 7.036 7.036

| ViT
| Roberta
| Orign
| 0.486 0.486
| 0.491 0.491
| 0.970 0.970
| 0.981 0.981
| 2.912 2.912
| 2.927 2.927
| 4.853 4.853
| 4.874 4.874

| Indra
| 1.048 1.048
| 0.880 0.880
| 2.065 2.065
| 1.749 1.749
| 5.970 5.970
| 5.149 5.149
| 9.702 9.702
| 8.446 8.446

| Convnext
| BERT
| Orign
| 0.396 0.396
| 0.474 0.474
| 0.837 0.837
| 0.950 0.950
| 2.603 2.603
| 2.851 2.851
| 4.412 4.412
| 4.755 4.755

| Indra
| 0.612 0.612
| 0.537 0.537
| 1.127 1.127
| 1.022 1.022
| 3.182 3.182
| 2.875 2.875
| 5.242 5.242
| 4.783 4.783

| Convnext
| Roberta
| Orign
| 0.492 0.492
| 0.480 0.480
| 0.985 0.985
| 0.964 0.964
| 2.962 2.962
| 2.889 2.889
| 4.940 4.940
| 4.824 4.824

| Indra
| 1.005 1.005
| 0.616 0.616
| 1.930 1.930
| 1.217 1.217
| 5.247 5.247
| 3.538 3.538
| 8.267 8.267
| 5.790 5.790

| Dinov2
| BERT
| Orign
| 0.496 0.496
| 0.473 0.473
| 0.991 0.991
| 0.947 0.947
| 2.969 2.969
| 2.852 2.852
| 4.936 4.936
| 4.760 4.760

| Indra
| 0.540 0.540
| 0.539 0.539
| 1.123 1.123
| 1.022 1.022
| 3.194 3.194
| 2.872 2.872
| 5.277 5.277
| 4.804 4.804

| Dinov2
| Roberta
| Orign
| 0.468 0.468
| 0.490 0.490
| 0.945 0.945
| 0.982 0.982
| 2.859 2.859
| 2.949 2.949
| 4.779 4.779
| 4.914 4.914

| Indra
| 1.016 1.016
| 0.949 0.949
| 1.978 1.978
| 1.863 1.863
| 5.603 5.603
| 5.370 5.370
| 9.021 9.021
| 8.766 8.766

|

NOCAPS

| CLIP-I
| CLIP-T
| Orign
| 1.357 1.357
| 1.325 1.325
| 2.556 2.556
| 2.499 2.499
| 6.860 6.860
| 6.717 6.717
| 10.795 10.795
| 10.584 10.584

| ViT
| BERT
| Orign
| 0.479 0.479
| 0.474 0.474
| 0.956 0.956
| 0.947 0.947
| 2.864 2.864
| 2.844 2.844
| 4.769 4.769
| 4.742 4.742

| Indra
| 0.701 0.701
| 0.667 0.667
| 1.375 1.375
| 1.293 1.293
| 3.960 3.960
| 3.712 3.712
| 6.449 6.449
| 6.069 6.069

| ViT
| Roberta
| Orign
| 0.484 0.484
| 0.483 0.483
| 0.966 0.966
| 0.963 0.963
| 2.891 2.891
| 2.886 2.886
| 4.814 4.814
| 4.805 4.805

| Indra
| 0.924 0.924
| 0.727 0.727
| 1.792 1.792
| 1.419 1.419
| 5.014 5.014
| 4.102 4.102
| 8.011 8.011
| 6.700 6.700

| Convnext
| BERT
| Orign
| 0.449 0.449
| 0.451 0.451
| 0.904 0.904
| 0.910 0.910
| 2.754 2.754
| 2.752 2.752
| 4.640 4.640
| 4.604 4.604

| Indra
| 0.415 0.415
| 0.485 0.485
| 0.906 0.906
| 0.971 0.971
| 2.911 2.911
| 2.907 2.907
| 4.821 4.821
| 4.845 4.845

| Convnext
| Roberta
| Orign
| 0.472 0.472
| 0.462 0.462
| 0.944 0.944
| 0.926 0.926
| 2.833 2.833
| 2.781 2.781
| 4.721 4.721
| 4.647 4.647

| Indra
| 0.764 0.764
| 0.557 0.557
| 1.472 1.472
| 1.102 1.102
| 4.079 4.079
| 3.338 3.338
| 6.494 6.494
| 5.526 5.526

| Dinov2
| BERT
| Orign
| 0.465 0.465
| 0.439 0.439
| 0.928 0.928
| 0.883 0.883
| 2.701 2.701
| 2.674 2.674
| 4.500 4.500
| 4.485 4.485

| Indra
| 0.566 0.566
| 0.485 0.485
| 1.065 1.065
| 0.971 0.971
| 3.179 3.179
| 2.907 2.907
| 5.238 5.238
| 4.845 4.845

| Dinov2
| Roberta
| Orign
| 0.497 0.497
| 0.467 0.467
| 0.993 0.993
| 0.936 0.936
| 2.969 2.969
| 2.822 2.822
| 4.938 4.938
| 4.712 4.712

| Indra
| 0.830 0.830
| 0.774 0.774
| 1.604 1.604
| 1.513 1.513
| 4.549 4.549
| 4.324 4.324
| 7.335 7.335
| 7.019 7.019

### 4.2 Evaluation on Vision & Language Modalities

Datasets. We adopt two widely used image-text datasets: MS-COCO Lin et al. (2014) and NOCAPS Agrawal et al. (2019) to evaluate performance on vision and language modalities.
MS-COCO serves as a standard benchmark for image captioning and retrieval tasks, while NOCAPS poses a greater challenge due to its focus on novel object categories. We use the validation sets of both datasets for evaluation.

Foundation Models. We use the same vision models as in Section  4.1 .
For language models, we evaluate BERT Devlin et al. (2019) and Roberta Liu et al. (2019) , both of which are pretrained independently on unimodal data without cross-modal alignment. We include CLIP Radford et al. (2021) as the aligned baseline for evaluation.

Evaluation Metrics :
We adopt CLIPScore Hessel et al. (2021) as the evaluation metric, which measures semantic alignment between image and text based on the cosine similarity of their embeddings within the CLIP multimodal space.
We report Top- k k matching accuracy ( k ∈ { 5 , 10 , 30 , 50 } k\in\{5,10,30,50\} ) in both text-to-image (T → \rightarrow I) and image-to-text (I → \rightarrow T) tasks.

Analysis.
Table 3 compares the performance of original embeddings versus Indra representations across both datasets using various combinations of vision and language models.
The results clearly demonstrate that Indra representations lead to consistent performance gains across different architectures and modalities.
Significant improvements are observed in both T → \rightarrow I and I → \rightarrow T retrieval, highlighting the effectiveness of our method for cross-modal alignment. These findings suggest that the Indra representation offers a generalizable mechanism to improve vision-language matching, independent of model architecture or dataset. Nonetheless, there remains a noticeable gap in performance compared to the fully aligned CLIP model, indicating further room for improvement.

Table 4: Performance on audio-text dataset 𝒟 \mathcal{D} using different representations ℛ \mathcal{R} ( Orign : original, Indra : Indra representation) with various audio ( Aud-E ) and language ( Lan-E ) models, where *-b and *-l refer to the base and large versions of model * , respectively.

| 𝒟 \mathcal{D}
| Aud-E
| Lan-E
| ℛ \mathcal{R}
| Top- 5 5
| Top- 10 10
| Top- 30 30
| Top- 50 50

| T → \rightarrow A
| A → \rightarrow T
| T → \rightarrow A
| A → \rightarrow T
| T → \rightarrow A
| A → \rightarrow T
| T → \rightarrow A
| A → \rightarrow T

|

TIMIT

| CLAP-I
| CLAP-T
| Orign
| 1.062 1.062
| 1.836 1.836
| 2.046 2.046
| 3.611 3.611
| 5.726 5.726
| 10.146 10.146
| 9.204 9.204
| 16.225 16.225

| wav2vec-b
| Roberta
| Orign
| 0.319 0.319
| 0.072 0.072
| 0.670 0.670
| 0.276 0.276
| 2.214 2.214
| 1.409 1.409
| 3.655 3.655
| 2.750 2.750

| Indra
| 0.413 0.413
| 0.578 0.578
| 0.819 0.819
| 1.209 1.209
| 2.506 2.506
| 2.953 2.953
| 4.225 4.225
| 4.692 4.692

| wav2vec-l
| Roberta
| Orign
| 0.418 0.418
| 0.308 0.308
| 0.864 0.864
| 0.634 0.634
| 2.548 2.548
| 2.194 2.194
| 4.200 4.200
| 3.738 3.738

| Indra
| 0.363 0.363
| 0.578 0.578
| 0.908 0.908
| 1.243 1.243
| 2.783 2.783
| 2.956 2.956
| 4.640 4.640
| 4.318 4.318

| wavlm-b
| Roberta
| Orign
| 0.328 0.328
| 0.328 0.328
| 0.659 0.659
| 0.648 0.648
| 2.080 2.080
| 1.920 1.920
| 3.518 3.518
| 3.208 3.208

| Indra
| 0.436 0.436
| 0.578 0.578
| 0.913 0.913
| 1.241 1.241
| 2.493 2.493
| 2.976 2.976
| 4.227 4.227
| 4.338 4.338

| wavlm-l
| Roberta
| Orign
| 0.472 0.472
| 0.426 0.426
| 0.912 0.912
| 0.876 0.876
| 2.450 2.450
| 2.518 2.518
| 3.997 3.997
| 4.148 4.148

| Indra
| 0.504 0.504
| 0.578 0.578
| 0.971 0.971
| 1.229 1.229
| 2.693 2.693
| 2.967 2.967
| 4.387 4.387
| 4.335 4.335

| hubert-b
| Roberta
| Orign
| 0.280 0.280
| 0.431 0.431
| 0.553 0.553
| 0.793 0.793
| 1.978 1.978
| 2.069 2.069
| 3.282 3.282
| 3.240 3.240

| Indra
| 0.322 0.322
| 0.578 0.578
| 0.697 0.697
| 1.232 1.232
| 2.230 2.230
| 2.971 2.971
| 3.880 3.880
| 4.369 4.369

| hubert-l
| Roberta
| Orign
| 0.449 0.449
| 0.308 0.308
| 0.861 0.861
| 0.638 0.638
| 2.475 2.475
| 1.949 1.949
| 4.119 4.119
| 3.286 3.286

| Indra
| 0.454 0.454
| 0.578 0.578
| 0.878 0.878
| 1.248 1.248
| 2.610 2.610
| 3.003 3.003
| 4.461 4.461
| 4.255 4.255

### 4.3 Evaluation on Speech & Language Modalities

Datasets. We adopt the TIMIT dataset Garofolo et al. (1993) for audio and language modality experiments.
TIMIT contains recordings from 630 630 speakers representing eight major dialect regions of American English, each reading ten phonetically rich sentences. The dataset provides time-aligned phonetic and word-level transcriptions along with 16 ​ k 16k Hz audio recordings.

Foundation Models. For audio models, we evaluate wav2vec Schneider et al. (2019) , wavlm Chen et al. (2022) , and hubert Hsu et al. (2021) , using both base and large variants.
For the language modality, we use Roberta Liu et al. (2019) .
All models are pretrained independently on unimodal data, without any cross-modal alignment.
As an aligned baseline, we include CLAP Wu et al. (2023) , an audio-language model jointly trained on paired audio-text data.

Evaluation Metrics :
Similarly, we adopt CLAPScore Wu et al. (2023) as the evaluation metric.
We report Top- k k matching ( k ∈ { 5 , 10 , 30 , 50 } k\in\{5,10,30,50\} ) in both text-to-audio (T → \rightarrow A) and audio-to-text (A → \rightarrow T) tasks.

Analysis.
Table 4 presents the audio-text matching results using different audio models. As shown, the Indra representations consistently improve matching performance in both directions across all model configurations.
However, compared to the vision-language setting, the improvements in the audio-language modality are relatively modest. This is likely due to the comparatively weaker capacity of the audio models used. Nonetheless, we observe that larger audio models yield better matching accuracy, further supporting the notion that model capacity positively influences cross-modal alignment.

## 5 Related Work

Instance-Level Representation Learning.
In conventional deep learning paradigms, each data instance is independently encoded into a fixed-dimensional vector, optimized using either supervised signals or unsupervised objectives. In supervised learning, representations are shaped by categorical labels Krizhevsky et al. (2012) ; He et al. (2016) , whereas in unsupervised settings, objectives such as reconstruction (e.g., autoencoders Vincent et al. (2010) ) or self-prediction (e.g., BERT Devlin et al. (2019) ) guide the learning of instance-level representations. These approaches primarily focus on learning standalone embeddings and do not explicitly model the relationships between samples, either in training or inference stages.

Contrastive Representation Learning.
Contrastive methods such as SimCLR Chen et al. (2020) , MoCo He et al. (2020a) , and BYOL Grill et al. (2020)
introduce pairwise relational inductive bias during training by encouraging the embeddings of similar (augmented) views to be close, while pushing apart dissimilar ones.
Building on this framework, CLIP Radford et al. (2021) extends contrastive learning to vision-language pretraining by aligning paired image-text embeddings within a shared multimodal space, enabling strong zero-shot performance. BLIP Li et al. (2022) further integrates contrastive and generative objectives to enhance cross-modal representation learning, achieving state-of-the-art results on several vision-language benchmarks.
Despite these advances, the sample representations in contrastive frameworks remain inherently instance-centric during inference.
Additionally, contrastive learning approaches typically require large-scale datasets to achieve satisfactory performance, as the training objectives rely heavily on diverse and abundant positive and negative pairs.

Graph-based Representation Learning.
Graph neural networks Kipf and Welling (2017) offer a framework for encoding relational information through message passing across graph-structured data. Variants such as GraphSAGE Hamilton et al. (2017) and GAT Velickovic et al. (2017) have improved flexibility and representation ability. However, most graph-based approaches rely on predefined adjacency structures or proximity assumptions Lu et al. (2023) , which may embed inductive biases that are misaligned with the latent semantics of the data. They typically operate over local k k -hop neighborhoods Lu et al. (2025a) , limiting their ability to capture long-range dependencies unless deeply stacked, which can lead to oversmoothing and degraded performance Li et al. (2018) .

Attention-based Representation Learning.
Transformer-based architectures Vaswani et al. (2017) offer a powerful alternative by leveraging global self-attention to aggregate contextual information. Numerous studies Devlin et al. (2019) ; Liu et al. (2019) ; Dosovitskiy et al. (2021) ; He et al. (2020b) ; Oquab et al. (2023) ; Radford et al. (2023) have demonstrated that attention-based architectures can be highly effective even in non-sequential domains. While attention enables global interactions both during training and inference, the resulting token-level representations are determined through dynamic mixing rather than explicitly encoding pairwise or global sample-to-sample relationships, making their geometric interpretation less transparent.

In contrast to the above paradigms, our approach explicitly constructs each representation as a relational profile, specifically, a distance-based reflection of its relationship to all other samples.
This design is motivated by the philosophical ontology of Indra’s Net, where each entity reflects and is reflected by all others, forming a holistic network of interdependence.
More importantly, our Indra representation is formally grounded in the Yoneda Lemma from enriched category theory, offering a theoretically sound and interpretable framework for relational representation learning.

## 6 Conclusion

In this paper, we present a theoretical and empirical investigation into the convergent behavior of unimodal foundation models. Motivated by the philosophical metaphor of Indra’s Net and grounded in enriched category theory, we introduce the Indra representation as a relational encoding that reflects each sample through its relationships with all others. We demonstrate that this representation can be derived via the 𝒱 \mathcal{V} -enriched Yoneda embedding and instantiated practically using angular distance. Our theoretical analysis proves that the proposed Indra representations are unique, complete, and structure-preserving, offering a principled basis for training-free alignment of various foundation models. Through extensive experiments across single-modality, vision-language, and speech-language settings, we demonstrate that Indra representations improve robustness and latent cross-modal capabilities of unimodal foundation models, offering a general framework for training-free multimodal alignment. Our findings suggest a new perspective for bridging modalities, which emphasizes the importance of the intrinsic relational structure of data or reality.

Limitations . Constructing exact Indra representations requires a computational complexity of 𝒪 ⁡ ( n 2 ​ d ) \mathcal{O}(n^{2}d) and a memory complexity of 𝒪 ⁡ ( n 2 ) \mathcal{O}(n^{2}) for a dataset with n n samples and embedding dimension d d .
This quadratic scaling potentially limits the direct applicability of the exact Indra representations to large-scale settings.
However, the scalability concern is addressable in practice. In the literature, there exists a rich body of work on approximating pairwise distances efficiently. For example, approximate nearest neighbor search Jegou et al. (2010) ; Johnson et al. (2019) , landmark-based approximation Lu et al. (2021) ; Moschella et al. (2023) , hashing-based methods Chen et al. (2021) , and sparse graph constructions Wang et al. (2026) . From an application perspective, these techniques can be readily adapted to approximate the Indra representation at scale without sacrificing its robustness.
From a theoretical view, Kan extensions Dubuc (2006) provide a principled framework for extrapolating from partial distance information to a full representation, preserving maximal structural fidelity.

## References

Agrawal et al. (2019)
H. Agrawal, K. Desai, Y. Wang, X. Chen, R. Jain, M. Johnson, D. Batra, D. Parikh, S. Lee, and P. Anderson

Nocaps: novel object captioning at scale .

In Proceedings of the IEEE/CVF international conference on computer vision ,

pp. 8948–8957 .

Cited by: §4.2 .

Baevski et al. (2020)
A. Baevski, Y. Zhou, A. Mohamed, and M. Auli

Wav2vec 2.0: a framework for self-supervised learning of speech representations .

Advances in neural information processing systems 33 , pp. 12449–12460 .

Cited by: §1 .

Bai et al. (2023)
J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al.

Qwen technical report .

arXiv preprint arXiv:2309.16609 .

Cited by: §2.2 .

Beringer et al. (2012)
J. Beringer, J. Arguin, R. Barnett, K. Copic, O. Dahl, D. Groom, C. Lin, J. Lys, H. Murayama, C. Wohl, et al.

Review of particle physics .

Physical Review D—Particles, Fields, Gravitation, and Cosmology 86 ( 1 ), pp. 010001 .

Cited by: §2.1 .

Bürger et al. (2024)
L. Bürger, F. A. Hamprecht, and B. Nadler

Truth is universal: robust detection of lies in llms .

Advances in Neural Information Processing Systems 37 .

Cited by: §2.2 .

Chen et al. (2022)
S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao, et al.

Wavlm: large-scale self-supervised pre-training for full stack speech processing .

IEEE Journal of Selected Topics in Signal Processing 16 ( 6 ), pp. 1505–1518 .

Cited by: §4.3 .

Chen et al. (2020)
T. Chen, S. Kornblith, M. Norouzi, and G. Hinton

A simple framework for contrastive learning of visual representations .

In International Conference on Machine Learning ,

pp. 1597–1607 .

Cited by: §5 .

Chen et al. (2021)
Y. Chen, S. Wang, J. Lu, Z. Chen, Z. Zhang, and Z. Huang

Local graph convolutional networks for cross-modal hashing .

In Proceedings of the 29th ACM international conference on multimedia ,

pp. 1921–1928 .

Cited by: §6 .

Cleary (1993)
T. Cleary

The flower ornament scripture: a translation of the avatamsaka sutra .

Shambhala Publications , Boston .

External Links: ISBN 978-0877739401

Cited by: §2.1 .

Cook (1977)
F.H. Cook

Hua-yen buddhism: the jewel net of indra .

Iaswr Series , Pennsylvania State University Press .

External Links: ISBN 9780271021904 ,
LCCN 76043288 ,
Link

Cited by: §2.1 .

Devlin et al. (2019)
J. Devlin, M. Chang, K. Lee, and K. Toutanova

Bert: pre-training of deep bidirectional transformers for language understanding .

In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 ,

pp. 4171–4186 .

Cited by: §1 ,
§4.2 ,
§5 ,
§5 .

Dosovitskiy et al. (2021)
A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby

An image is worth 16x16 words: transformers for image recognition at scale .

In International Conference on Learning Representations ,

Cited by: §1 ,
§4.1 ,
§5 .

Dubuc (2006)
E. J. Dubuc

Kan extensions in enriched category theory .

Vol. 145 , Springer .

Cited by: §6 .

Faraday (1859)
M. Faraday

Experimental researches in electricity .

Bernard Quaritch , London .

Note: Originally published as a series of papers between 1831 and 1855

Cited by: §2.1 .

J. R. Firth (Ed.) (1957)
J. R. Firth (Ed.)

Studies in linguistic analysis .

Philological Society , Oxford .

Note: Special volume of the Philological Society

Cited by: §2.1 .

Garofolo et al. (1993)
J. S. Garofolo, L. F. Lamel, W. M. Fisher, D. S. Pallett, N. L. Dahlgren, V. Zue, and J. G. Fiscus

TIMIT acoustic-phonetic continuous speech corpus .

(No Title) .

Cited by: §4.3 .

Gergen (2009)
K. J. Gergen

Relational being: beyond self and community .

Oxford University Press , New York .

External Links: ISBN 978-0-19-530538-8

Cited by: §2.1 .

Griffiths (2020)
D. Griffiths

Introduction to elementary particles .

John Wiley & Sons .

Cited by: §2.1 .

Grill et al. (2020)
J. Grill, F. Strub, F. Altmann, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. A. Pires, Z. Guo, M. G. Azar, et al.

Bootstrap your own latent-a new approach to self-supervised learning .

In Advances in Neural Information Processing Systems ,

Vol. 33 , pp. 21271–21283 .

Cited by: §5 .

Gurnee et al. (2024)
W. Gurnee, T. Horsley, Z. C. Guo, T. R. Kheirkhah, Q. Sun, W. Hathaway, N. Nanda, and D. Bertsimas

Universal neurons in gpt2 language models .

arXiv preprint arXiv:2401.12181 .

Cited by: §2.2 .

Hamilton et al. (2017)
W. Hamilton, Z. Ying, and J. Leskovec

Inductive representation learning on large graphs .

In Advances in Neural Information Processing Systems ,

Cited by: §5 .

Harvard FAS CAMLab (2022)
Harvard FAS CAMLab

Indra’s net .

External Links: Link

Cited by: §2.1 .

He et al. (2020a)
K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick

Momentum contrast for unsupervised visual representation learning .

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition ,

pp. 9729–9738 .

Cited by: §5 .

He et al. (2016)
K. He, X. Zhang, S. Ren, and J. Sun

Deep residual learning for image recognition .

In Proceedings of the IEEE conference on computer vision and pattern recognition ,

pp. 770–778 .

Cited by: §5 .

He et al. (2020b)
P. He, X. Liu, J. Gao, and W. Chen

Deberta: decoding-enhanced bert with disentangled attention .

arXiv preprint arXiv:2006.03654 .

Cited by: §1 ,
§5 .

Hessel et al. (2021)
J. Hessel, A. Holtzman, M. Forbes, R. L. Bras, and Y. Choi

Clipscore: a reference-free evaluation metric for image captioning .

arXiv preprint arXiv:2104.08718 .

Cited by: §4.2 .

Hosseini et al. (2024)
E. Hosseini, C. Casto, N. Zaslavsky, C. Conwell, M. Richardson, and E. Fedorenko

Universality of representation in biological and artificial neural networks .

bioRxiv .

Cited by: §1 ,
§2.2 .

Hsu et al. (2021)
W. Hsu, B. Bolte, Y. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed

Hubert: self-supervised speech representation learning by masked prediction of hidden units .

IEEE/ACM transactions on audio, speech, and language processing 29 , pp. 3451–3460 .

Cited by: §4.3 .

Huh et al. (2024)
M. Huh, B. Cheung, T. Wang, and P. Isola

The platonic representation hypothesis .

In Proceedings of the 41st International Conference on Machine Learning ,

Proceedings of Machine Learning Research .

Cited by: §1 ,
§2.2 .

Jegou et al. (2010)
H. Jegou, M. Douze, and C. Schmid

Product quantization for nearest neighbor search .

IEEE transactions on pattern analysis and machine intelligence 33 ( 1 ), pp. 117–128 .

Cited by: §6 .

Johnson et al. (2019)
J. Johnson, M. Douze, and H. Jégou

Billion-scale similarity search with gpus .

IEEE Transactions on Big Data 7 ( 3 ), pp. 535–547 .

Cited by: §6 .

Kasulis (2018)
T. P. Kasulis

Engaging japanese philosophy: a short history .

University of Hawai’i Press , Honolulu .

External Links: ISBN 978-0824836841

Cited by: §2.1 .

Kelly (1982)
G. M. Kelly

Basic concepts of enriched category theory .

London Mathematical Society Lecture Note Series , Vol. 64 , Cambridge University Press , Cambridge .

Cited by: §1 ,
Corollary 1 ,
Lemma 1 .

Khosla et al. (2024)
M. Khosla, A. H. Williams, J. McDermott, and N. Kanwisher

Privileged representational axes in biological and artificial neural networks .

bioRxiv , pp. 2024–06 .

Cited by: §1 ,
§2.2 .

Kipf and Welling (2017)
T. N. Kipf and M. Welling

Semi-supervised classification with graph convolutional networks .

In International Conference on Learning Representations ,

Cited by: §5 .

Koh et al. (2023)
J. Y. Koh, R. Salakhutdinov, and D. Fried

Grounding language models to images for multimodal inputs and outputs .

In International Conference on Machine Learning ,

pp. 17283–17300 .

Cited by: §1 .

Krizhevsky et al. (2012)
A. Krizhevsky, I. Sutskever, and G. E. Hinton

Imagenet classification with deep convolutional neural networks .

Advances in neural information processing systems 25 .

Cited by: §5 .

Krizhevsky (2009)
A. Krizhevsky

Learning multiple layers of features from tiny images .

Technical report

University of Toronto .

Note: Technical Report

Cited by: §4.1 .

Lawvere (1973)
F. W. Lawvere

Metric spaces, generalized logic, and closed categories .

Rendiconti del seminario matématico e fisico di Milano 43 , pp. 135–166 .

Cited by: §3.2 .

Lee et al. (2025)
A. Lee, M. Weber, F. Viégas, and M. Wattenberg

Shared global and local geometry of language model embeddings .

arXiv preprint arXiv:2503.21073 .

Cited by: §2.2 .

Li et al. (2023)
J. Li, D. Li, S. Savarese, and S. Hoi

Blip-2: bootstrapping language-image pre-training with frozen image encoders and large language models .

In International conference on machine learning ,

pp. 19730–19742 .

Cited by: §1 .

Li et al. (2022)
J. Li, D. Li, C. Xiong, and S. Hoi

Blip: bootstrapping language-image pre-training for unified vision-language understanding and generation .

In International conference on machine learning ,

pp. 12888–12900 .

Cited by: §1 ,
§5 .

Li et al. (2018)
Q. Li, Z. Han, and X. Wu

Deeper insights into graph convolutional networks for semi-supervised learning .

In Proceedings of the AAAI conference on artificial intelligence ,

Vol. 32 .

Cited by: §5 .

Lin et al. (2014)
T. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick

Microsoft coco: common objects in context .

In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13 ,

pp. 740–755 .

Cited by: §4.2 .

Liu et al. (2023)
H. Liu, C. Li, Q. Wu, and Y. J. Lee

Visual instruction tuning .

Advances in neural information processing systems 36 , pp. 34892–34916 .

Cited by: §1 ,
§1 .

Liu et al. (2019)
Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov

Roberta: a robustly optimized bert pretraining approach .

arXiv preprint arXiv:1907.11692 .

Cited by: §1 ,
§4.2 ,
§4.3 ,
§5 .

Liu et al. (2022)
Z. Liu, H. Mao, C. Wu, C. Feichtenhofer, T. Darrell, and S. Xie

A convnet for the 2020s .

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) .

Cited by: §1 .

Lu et al. (2025a)
J. Lu, Y. Liu, Y. Zhang, and Y. Fu

Scale-free graph-language models .

In The Thirteenth International Conference on Learning Representations ,

Cited by: §5 .

Lu et al. (2025b)
J. Lu, H. Wang, Y. Xu, Y. Wang, K. Yang, and Y. Fu

Representation potentials of foundation models for multimodal alignment: a survey .

In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing ,

pp. 16669–16684 .

Cited by: §1 ,
§2.2 .

Lu et al. (2021)
J. Lu, H. Wang, J. Zhou, Y. Chen, Z. Lai, and Q. Hu

Low-rank adaptive graph embedding for unsupervised feature extraction .

Pattern Recognition 113 , pp. 107758 .

Cited by: §6 .

Lu et al. (2023)
J. Lu, Y. Xu, H. Wang, Y. Bai, and Y. Fu

Latent graph inference with limited supervision .

In Advances in Neural Information Processing Systems ,

Cited by: §5 .

Maniparambil et al. (2024)
M. Maniparambil, R. Akshulakov, Y. A. D. Djilali, M. El Amine Seddik, S. Narayan, K. Mangalam, and N. E. O’Connor

Do vision and language encoders represent the world similarly? .

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) ,

Cited by: §1 .

Markus and Kitayama (2014)
H. R. Markus and S. Kitayama

Culture and the self: implications for cognition, emotion, and motivation .

In College student development and academic life ,

pp. 264–293 .

Cited by: §2.1 .

Maxwell (1873)
J. C. Maxwell

A treatise on electricity and magnetism .

First edition , Clarendon Press , Oxford .

Cited by: §2.1 .

Merullo et al. (2023)
J. Merullo, L. Castricato, C. Eickhoff, and E. Pavlick

Linearly mapping from image to text space .

In The Eleventh International Conference on Learning Representations ,

Cited by: §1 ,
§1 .

Mikolov et al. (2013)
T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean

Distributed representations of words and phrases and their compositionality .

Advances in neural information processing systems 26 .

Cited by: §2.1 .

Moschella et al. (2023)
L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, and E. Rodolà

Relative representations enable zero-shot latent space communication .

In The Eleventh International Conference on Learning Representations ,

Cited by: §3.3 ,
§6 .

Ng et al. (2023)
E. Ng, S. Subramanian, D. Klein, A. Kanazawa, T. Darrell, and S. Ginosar

Can language models learn to listen? .

In Proceedings of the IEEE/CVF International Conference on Computer Vision ,

pp. 10083–10093 .

Cited by: §1 .

Ngo and Kim (2024)
J. Ngo and Y. Kim

What do language models hear? probing for auditory representations in language models .

In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics ,

pp. 5435–5448 .

Cited by: §1 .

OpenAI (2024)
OpenAI

GPT-4 technical report .

External Links: 2303.08774

Cited by: §1 .

Oquab et al. (2023)
M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al.

Dinov2: learning robust visual features without supervision .

arXiv preprint arXiv:2304.07193 .

Cited by: §1 ,
§4.1 ,
§5 .

Radford et al. (2021)
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al.

Learning transferable visual models from natural language supervision .

In International conference on machine learning ,

pp. 8748–8763 .

Cited by: §1 ,
§4.2 ,
§5 .

Radford et al. (2023)
A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever

Robust speech recognition via large-scale weak supervision .

In International conference on machine learning ,

pp. 28492–28518 .

Cited by: §1 ,
§5 .

Riehl (2017)
E. Riehl

Category theory in context .

Courier Dover Publications .

Cited by: Corollary 1 ,
Lemma 1 .

Roeder et al. (2021)
G. Roeder, L. Metz, and D. Kingma

On linear identifiability of learned representations .

In International Conference on Machine Learning ,

pp. 9030–9039 .

Cited by: §2.2 .

Schneider et al. (2019)
S. Schneider, A. Baevski, R. Collobert, and M. Auli

Wav2vec: unsupervised pre-training for speech recognition .

arXiv preprint arXiv:1904.05862 .

Cited by: §1 ,
§4.3 .

Sharma et al. (2024)
P. Sharma, T. R. Shaham, M. Baradad, S. Fu, A. Rodriguez-Munoz, S. Duggal, P. Isola, and A. Torralba

A vision check-up for language models .

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition ,

pp. 14410–14419 .

Cited by: §1 .

Tan et al. (2024)
D. Tan, D. Chanin, A. Lynch, B. Paige, D. Kanoulas, A. Garriga-Alonso, and R. Kirk

Analysing the generalisation and reliability of steering vectors .

Advances in Neural Information Processing Systems 37 .

Cited by: §2.2 .

Touvron et al. (2023)
H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al.

Llama 2: open foundation and fine-tuned chat models .

arXiv preprint arXiv:2307.09288 .

Cited by: §2.2 .

Vaswani et al. (2017)
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin

Attention is all you need .

Advances in neural information processing systems .

Cited by: §5 .

Velickovic et al. (2017)
P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, Y. Bengio, et al.

Graph attention networks .

stat 1050 ( 20 ), pp. 10–48550 .

Cited by: §5 .

Venkateswara et al. (2017)
H. Venkateswara, J. Eusebio, S. Chakraborty, and S. Panchanathan

Deep hashing network for unsupervised domain adaptation .

In Proceedings of the IEEE conference on computer vision and pattern recognition ,

pp. 5018–5027 .

Cited by: §4.1 .

Vincent et al. (2010)
P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, P. Manzagol, and L. Bottou

Stacked denoising autoencoders: learning useful representations in a deep network with a local denoising criterion. .

Journal of machine learning research 11 ( 12 ).

Cited by: §5 .

Wang et al. (2025)
J. Wang, X. Ge, W. Shu, Q. Tang, Y. Zhou, Z. He, and X. Qiu

Towards universality: studying mechanistic similarity across language model architectures .

In The Thirteenth International Conference on Learning Representations ,

Cited by: §2.2 .

Wang et al. (2026)
Y. Wang, J. Zhou, M. Song, Y. Guo, and J. Lu

Fuzzy multi-subspace clustering .

IEEE Transactions on Fuzzy Systems ( ), pp. 1–14 .

Cited by: §6 .

Wentworth (2021)
J. Wentworth

Testing the natural abstraction hypothesis .

AI Alignment Forum .

Cited by: §2.2 .

Woo et al. (2023)
S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie

ConvNeXt v2: co-designing and scaling convnets with masked autoencoders .

In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) ,

Cited by: §1 ,
§4.1 .

Wu et al. (2023)
Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov

Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation .

In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) ,

pp. 1–5 .

Cited by: §4.3 ,
§4.3 .
