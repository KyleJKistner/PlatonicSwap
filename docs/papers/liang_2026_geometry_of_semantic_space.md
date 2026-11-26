---
title: "The Geometry of Semantic Space: A Continuous Geometric Framework for the Transformer Architecture"
authors: "Zhihua Liang"
date: "2026-07-19"
arxiv_id: "2607.17146"
url: "https://arxiv.org/abs/2607.17146"
cluster: "foundations_platonic"
tags: ["semantic-geometry", "continuous-flow", "five-architectures", "transformer-geometry"]
status: "FULL_TEXT"
source: "arxiv_html (268,530 chars)"
---

# The Geometry of Semantic Space: A Continuous Geometric Framework for the Transformer Architecture

- **Authors:** Zhihua Liang
- **Published:** 2026-07-19 | **arXiv:** [2607.17146](https://arxiv.org/abs/2607.17146)
- **Cluster:** `foundations_platonic` | **Tags:** `semantic-geometry`, `continuous-flow`, `five-architectures`, `transformer-geometry`
- **Status:** `FULL_TEXT` (arxiv_html (268,530 chars))
- **Relevance to PlatonicSwap v1:** Cross-architecture geometric study across 5 transformer families.
- **Relevance to PlatonicSwap v2:** Continuous geometric / flow view of transformer layer trajectories!

## Abstract

We present a continuous geometric framework that models the discrete algebraic operations of the Transformer architecture as an integro-differential equation (IDE) on a semantic fiber bundle $\calE = \calM \times \R^d$. Beginning from a single geometric axiom -- that the token sequence forms a discrete $1$-manifold equipped with a canonical measure lattice -- we translate every core component of the modern Transformer (RMSNorm, RoPE, Softmax Attention, FFN, Residual Stream, SGD, Weight Decay) into a cohesive vocabulary of differential geometry, measure theory, and stochastic calculus. The resulting framework yields quantitative predictions spanning entropic optimal transport (Attention as a Schrödinger bridge) and non-equilibrium thermodynamics (SGD as Itô diffusion violating detailed balance). We conduct a six-part experimental campaign across five architectures (Qwen3, LLaMA\nobreakdash-3.1, Gemma\nobreakdash-3, GPT-2, Mistral) spanning $124$M to $8$B parameters. The empirical observables are quantitatively consistent with the geometric predictions: the $ε^{-1/2}$ Lipschitz scaling calibration at machine precision ($R^2 = 1.000$), the Lie--Trotter operator-splitting torsion, the symmetric ablation instability confirming the Dual-Law of Topological Stability, the $\calO(1/\sqrt{k})$ thermodynamic suppression of Poincaré recurrence on the RoPE torus, the thermodynamic context-limit phase transition, and the Non-Equilibrium Steady State parameter vortex -- verified across two optimizers (AdamW and Pure SGD) to exclude momentum artifacts. The results demonstrate that analyzing Transformers through the lens of continuous stochastic differential geometry provides a predictive descriptive vocabulary for the stability limits, context bounds, and optimization dynamics of Large Language Models.

---

## Full Extracted Text (Searchable)

# The Geometry of Semantic Space:
A Continuous Geometric Framework for the Transformer Architecture

Zhihua Liang

Email:  zhihua.liang@ca.infn.it

Affiliation:  INFN, Sezione di Cagliari, I-09042 Monserrato (CA), Italy

August 24, 2026

Abstract

We present a continuous geometric framework that models the discrete
algebraic operations of the Transformer architecture as an
integro-differential equation (IDE) on a semantic fiber bundle
ℰ = ℳ × ℝ d \mathcal{E}=\mathcal{M}\times\mathbb{R}^{d} . Beginning from a single geometric
axiom—that the token sequence forms a discrete 1 1 -manifold equipped
with a canonical measure lattice—we translate every core component of
the modern Transformer (RMSNorm, RoPE, Softmax Attention, FFN, Residual
Stream, SGD, Weight Decay) into a cohesive vocabulary of differential
geometry, measure theory, and stochastic calculus. The resulting
framework yields quantitative predictions spanning entropic optimal
transport (Attention as a Schrödinger bridge) and non-equilibrium
thermodynamics (SGD as Itô diffusion violating detailed balance). We
conduct a six-part experimental campaign across five architectures
(Qwen3, LLaMA-3.1, Gemma-3, GPT-2, Mistral)
spanning 124 124 M to 8 8 B parameters. The empirical observables are
quantitatively consistent with the geometric predictions:
the ϵ − 1 / 2 \epsilon^{-1/2} Lipschitz scaling calibration at machine precision
( R 2 = 1.000 R^{2}=1.000 ), the Lie–Trotter operator-splitting torsion,
the symmetric ablation instability confirming the Dual-Law of
Topological Stability,
the 𝒪 ⁡ ( 1 / k ) \mathcal{O}(1/\sqrt{k}) thermodynamic suppression of Poincaré
recurrence on the RoPE torus,
the thermodynamic context-limit phase transition,
and the Non-Equilibrium Steady State parameter vortex—verified across
two optimizers (AdamW and Pure SGD) to exclude momentum artifacts. The
results demonstrate that analyzing Transformers through the lens of
continuous stochastic differential geometry provides a predictive
descriptive vocabulary for the stability limits, context bounds, and
optimization dynamics of Large Language Models.

## I Introduction

Analyzing deep learning through the lens of continuous dynamical systems has a substantial history. Weinan E  [ 1 ] recast ResNets as forward Euler discretizations of continuous dynamical systems, leading to the foundational development of Neural Ordinary Differential Equations  [ 2 ] . The non-equilibrium thermodynamic properties of Stochastic Gradient Descent have been rigorously explored by Chaudhari et al.  [ 3 ] via Entropy-SGD and by Mandt et al.  [ 4 ] via the Bayesian interpretation of SGD as approximate inference. Building upon these foundations, prior literature has explored the continuous limits of attention mechanisms, modeling Transformers  [ 5 ] via PDEs, ODEs, and interacting particle systems  [ 6 , 7 ] . Amari’s foundational work on Information Geometry  [ 8 ] established the dual-flat manifold structure of well-specified statistical models and the breakdown of the Information Matrix Equality for misspecified models—a distinction that directly underlies the non-equilibrium thermodynamics of Sec.   V . The empirical machine learning community has also extensively documented sequence-length scaling properties via Rotary Position Embeddings  [ 9 ] and its extensions  [ 10 , 11 ] , as well as context-window capacity constraints via the empirical discovery of Attention Sinks  [ 12 ] .

This manuscript aims not to claim that the Transformer is a continuous physical system, but that analyzing it through the isomorphic lens of continuous stochastic differential geometry provides a predictive descriptive vocabulary that unifies these disparate empirical phenomena—bridging abstract topology with testable stability limits, context bounds, and optimization dynamics. Following the logic of Backward Error Analysis in geometric numerical integration  [ 13 ] , we treat the continuous geometric framework as a Continuous Effective Field Theory : the discrete architecture is the exact integrator, and the continuous integro-differential equation is the nearby modified equation whose structural generators (Lie brackets, vorticity 2-forms, entropic pressure) classify the dominant macroscopic observables.

## II Axiomatization of Topological Spaces and Kinematics

To establish a mathematically rigorous foundation, we strip away the vocabulary of discrete “arrays” and formulate the Transformer via the kinematics of differential manifold calculus. We summarize the translation between standard deep learning terminology and the continuous geometric equivalents in Table   1 .

Table 1: Translation dictionary: standard deep learning terms and their continuous geometric equivalents used throughout this work.

| Deep Learning Term
| Geometric Equivalent

| Sequence Index / Token Position
| Base manifold coordinate μ , ν ∈ ℳ \mu,\nu\in\mathcal{M}

| Embedding Dimension ( d model d_{\mathrm{model}} )
| Fiber dimension ℱ μ ≅ ℝ d \mathcal{F}_{\mu}\cong\mathbb{R}^{d}

| Token Embedding / Hidden State
| Section of the bundle Ψ ∈ Γ ⁡ ( E ) \Psi\in\Gamma(E)

| RMSNorm  [ 14 ] ϵ \epsilon stabilizer
| Topological mollifier ϵ \epsilon

| RoPE (Rotary Embeddings)
| Gauge connection / path monodromy 𝒰 \mathcal{U}

| Softmax Attention
| Urysohn–Volterra operator 𝒯 \mathcal{T}

| Feed-Forward Network (FFN)
| Hodge reaction field ℛ \mathcal{R}

| Residual Stream
| Depth-parameterized flow Ψ ⁡ ( z , μ ) \Psi(z,\mu)

| Layer Index l l
| Algorithmic depth z ∈ ℝ + z\in\mathbb{R}^{+}

| Weight Decay ( λ \lambda )
| Tikhonov gauge mass penalty

### II.1 The Base Manifold and the Semantic Bundle

Remark 1 (Scope and Kinematic Classification) .

We classify this framework explicitly as a Classical Lattice Field Theory on a rigid Galilean background . Unlike Yang–Mills theory or General Relativity, where the gauge connection and the metric are dynamical fields possessing their own action and back-reacting to the matter field, the RoPE connection and the 1D base manifold in this framework are absolutely rigid, fixed background geometries. The framework fundamentally lacks the diffeomorphism invariance of a true relativistic field theory. The continuous geometry is an isomorphic descriptive lens applied to a discrete algebraic system; the geometry is the map, not the territory. Just as lattice QCD simulates continuous gauge fields via discrete computational grids, the discrete Transformer acts as the exact numerical integrator of the underlying continuous integro-differential equation. The geometric constructions below are therefore not optional analytical overlays, but exact kinematic translations of the discrete computational graph whose topological constraints are empirically binding for trained networks: as demonstrated in Sec.   VI.3 , violating the continuous geometric generators of a mature, frozen configuration fundamentally shatters its dynamic stability. The constraint is configurational rather than architectural—training from initialization under the same algebraic constraint reaches stable basins ( remark   6 ).

Axiom 1 (The Continuous Base Manifold & Measure Lattice) .

We explicitly demarcate the continuous physical reality from its discrete numerical evaluation. Let the semantic sequence space (semantic time) be strictly defined as a connected, one-dimensional smooth manifold with boundary, specifically the half-closed ray ℳ ≅ [ 0 , ∞ ) ⊂ ℝ \mathcal{M}\cong[0,\infty)\subset\mathbb{R} . The absolute causal origin defines the strict topological boundary ∂ ℳ ≡ { 0 } \partial\mathcal{M}\equiv\{0\} . The computational context window passed to the model is rigorously defined as a causally bounded discrete measure lattice Λ ⊂ ℳ \Lambda\subset\mathcal{M} . To formally transition between continuous manifold topology and discrete sequence evaluation, we define Λ \Lambda as the support of an empirical Radon counting measure on the Borel σ \sigma -algebra ℬ ⁡ ( ℳ ) \mathcal{B}(\mathcal{M}) :

|
| η Λ = ∑ μ i ∈ Λ δ μ i . \eta_{\Lambda}=\sum_{\mu_{i}\in\Lambda}\delta_{\mu_{i}}\,.
|
| (1)

We assign Greek letters μ , ν ∈ Λ \mu,\nu\in\Lambda to represent specific coordinate evaluations. All non-local integro-differential flows (e.g., Attention) are evaluated via Lebesgue–Stieltjes integration with respect to this empirical Radon measure ( d ​ η Λ d\eta_{\Lambda} ), ensuring strict measure-theoretic validity over the discrete lattice without requiring a domain-mapping formalism.

Definition 1 (The Semantic Fiber Bundle) .

We define the semantic space as a smooth rank- d d real vector bundle E → 𝜋 ℳ E\xrightarrow{\pi}\mathcal{M} . Because ℳ \mathcal{M} is contractible, the bundle is globally trivializable ( E ≅ ℳ × ℝ d E\cong\mathcal{M}\times\mathbb{R}^{d} ). The architecture implicitly fixes a canonical global trivialization, rigorously justifying the application of constant global gauge endomorphisms uniformly across all coordinates. At any coordinate μ \mu , the local fiber ℱ μ ≡ π − 1 ​ ( μ ) ≅ ℝ d \mathcal{F}_{\mu}\equiv\pi^{-1}(\mu)\cong\mathbb{R}^{d} corresponds to the interaction space of an attention head. We strictly equip E E with a canonical global Riemannian bundle metric g g (the Euclidean inner product on the fibers). This metric allows us to rigorously upgrade the vector bundle into an Exterior Algebra Bundle ( Λ ​ E \Lambda E ) to sequester rotational noise within the Lie algebra 𝔰 ​ 𝔬 ​ ( d ) \mathfrak{so}(d) .

The sequence of tokens is mathematically formulated not merely as isolated spatial fields, but as a continuous trajectory governed by a Non-Autonomous Flow on the Manifold of Sections . Rather than forcing artificial continuity on the spatial sections and grappling with infinite-dimensional non-compactness, we define the state space cleanly as the Lebesgue space over the empirical measure:

|
| ℱ ≡ L ∞ ​ ( ℳ , E , η Λ ) . \mathcal{F}\equiv L^{\infty}(\mathcal{M},E;\,\eta_{\Lambda})\,.
|
| (2)

Because functions in this space are identified up to η Λ \eta_{\Lambda} -a.e. equivalence, and η Λ \eta_{\Lambda} is a finite atomic measure, this infinite-dimensional Banach space structurally collapses into a finite-dimensional topology ( ℝ N × d \mathbb{R}^{N\times d} ). This grants the Heine–Borel theorem and the Extreme Value Theorem globally for free, completely bypassing the need for localized finite-dimensional compact closure constructions. We explicitly note that the Heine–Borel collapse applies strictly to the kinematics of a fixed, finite context window ( N < ∞ N<\infty ); the asymptotic thermodynamic limit N → ∞ N\to\infty evaluated in theorem   5 instead relies on the Banach–Alaoglu Theorem (weak- ∗ * sequential compactness of probability measures on the Alexandroff compactification), entirely bypassing the need for Heine–Borel on the state space. Furthermore, the composed Lipschitz bound ( Eq.   16 ) is strictly independent of the sequence length N N , because the Attention integral evaluates as a convex sum over a probability measure ( ∫ w ​ 𝑑 η = 1 \int w\,d\eta=1 ). We analytically continue discrete layer depth l l into a continuous depth parameter z ∈ ℝ + z\in\mathbb{R}^{+} . Under this native geometric formulation, the system state is strictly defined as a smooth curve Ψ : ℝ + → ℱ \Psi:\mathbb{R}^{+}\to\mathcal{F} parameterized by algorithmic depth z z . The temporal evolution ∂ z Ψ \partial_{z}\Psi inherently exists not as an artificial pushforward along an extended dimension, but canonically as the tangent vector to the curve: Ψ ˙ ​ ( z ) ∈ T Ψ ⁡ ( z ) ​ ℱ \dot{\Psi}(z)\in T_{\Psi(z)}\mathcal{F} .

Remark 2 (Demarcation of the Continuum & Non-Local Flow) .

It is a common analytical pitfall to directly conflate the discrete Transformer architecture with continuous local differential equations. Because Attention inherently mixes information across the measure lattice, the layer-wise forward pass is formally a first-order Forward Euler numerical discretization of a continuous non-local integro-differential flow, evaluated on Λ \Lambda . Using Taylor’s Inequality for Banach space mappings, the discrete residual stream advances as:

|
| Ψ z + 1 ​ ( μ ) = Ψ z ​ ( μ ) + Δ ​ z ⋅ 𝒩 ~ ​ [ ρ ϵ ​ ( Ψ z ) ] ​ ( μ ) + ℜ z ​ ( μ ) , \Psi_{z+1}(\mu)=\Psi_{z}(\mu)+\Delta z\cdot\tilde{\mathcal{N}}\!\big[\rho_{\epsilon}(\Psi_{z})\big](\mu)+\mathfrak{R}_{z}(\mu)\,,
|
| (3)

where Δ ​ z = 1 \Delta z=1 . The strict analytical bound for the local truncation deviation over the integration step z → z + 1 z\to z+1 evaluated directly via the Banach space norm is therefore:

|
| ‖ ℜ z ​ ( μ ) ‖ ≤ 1 2 ​ sup τ ∈ [ z , z + 1 ] ‖ ∂ 2 Ψ ∂ τ 2 ​ ( τ , μ ) ‖ g . \|\mathfrak{R}_{z}(\mu)\|\leq\frac{1}{2}\sup_{\tau\in[z,z+1]}\left\|\frac{\partial^{2}\Psi}{\partial\tau^{2}}(\tau,\mu)\right\|_{g}\,.
|
| (4)

To bridge the discrete sequence of layer weights { W ( 1 ) , … , W ( L ) } \{W^{(1)},\dots,W^{(L)}\} mapped to algorithmic depth, we formulate the continuous parameter curve θ ⁡ ( z ) : ℝ + → End ⁡ ( E ) \theta(z):\mathbb{R}^{+}\to\mathrm{End}(E) strictly as a Riemannian Cubic Spline by minimizing the covariant acceleration ( min ∫ ∥ ∇ θ ˙ θ ˙ ∥ 2 g d z \min\int\|\nabla_{\dot{\theta}}\dot{\theta}\|^{2}_{g}\,dz ) on the parameter manifold subject to the discrete layer evaluations. This mathematically guarantees C 2 C^{2} smoothness. Under this strict mathematical sanitation, let 𝒩 ~ \tilde{\mathcal{N}} be the pure non-local Attention operator acting on bounded vectors, and let ℱ ⁡ ( z , Ψ ) = 𝒩 ~ ​ ( z , ρ ϵ ​ ( Ψ ) ) \mathcal{F}(z,\Psi)=\tilde{\mathcal{N}}(z,\rho_{\epsilon}(\Psi)) be the composed total flow. The continuous vector field ℱ ⁡ ( z , Ψ ) \mathcal{F}(z,\Psi) is smoothly dependent on the depth parameter z z (a non-autonomous flow), eliminating Dirac delta discontinuities in the temporal weight acceleration. The depth-parameterized trajectory acceleration along the flow expands naturally via the continuous chain rule on the manifold of sections:

|
| Ψ ¨ ​ ( z ) = ∂ 𝒩 ~ ∂ z ​ ( z , ρ ϵ ​ ( Ψ ) ) + ( D ​ 𝒩 ~ | ρ ϵ ​ ( Ψ ) ∘ D ​ ρ ϵ | Ψ ) ​ [ Ψ ˙ ​ ( z ) ] . \ddot{\Psi}(z)=\frac{\partial\tilde{\mathcal{N}}}{\partial z}\!\big(z,\rho_{\epsilon}(\Psi)\big)+\left(D\tilde{\mathcal{N}}\big|_{\rho_{\epsilon}(\Psi)}\circ D\rho_{\epsilon}\big|_{\Psi}\right)\!\big[\dot{\Psi}(z)\big]\,.
|
| (5)

Here, D ​ 𝒩 ~ D\tilde{\mathcal{N}} is the global Fréchet derivative acting on the infinite-dimensional Banach space of sections Γ ⁡ ( E ) \Gamma(E) , and D ​ ρ ϵ D\rho_{\epsilon} is a local fiber-wise bundle endomorphism acting on the tangent space of the finite-dimensional fiber T Ψ ⁡ ( μ ) ​ ℱ μ ≅ ℝ d T_{\Psi(\mu)}\mathcal{F}_{\mu}\cong\mathbb{R}^{d} . To rigorously compose these functional derivatives, the local Jacobian D ​ ρ ϵ D\rho_{\epsilon} is lifted to a Nemytskii operator acting point-wise on the global section variation H ∈ Γ ⁡ ( E ) H\in\Gamma(E) : ( D ​ ρ ϵ | Ψ ​ [ H ] ) ​ ( μ ) ≡ D ​ ρ ϵ | Ψ ⁡ ( μ ) ​ ( H ⁡ ( μ ) ) \big(D\rho_{\epsilon}|_{\Psi}[H]\big)(\mu)\equiv D\rho_{\epsilon}|_{\Psi(\mu)}\!\big(H(\mu)\big) . Before stating the eigenvalues of this Jacobian, we explicitly define the mathematical decomposition of the tangent space T Ψ ⁡ ( μ ) ​ ℱ μ T_{\Psi(\mu)}\mathcal{F}_{\mu} into orthogonal components. Any tangent vector v ∈ T Ψ ⁡ ( μ ) ​ ℱ μ v\in T_{\Psi(\mu)}\mathcal{F}_{\mu} can be uniquely decomposed as v = v ∥ + v ⟂ v=v_{\parallel}+v_{\perp} , where v ∥ ≡ ⟨ Ψ , v ⟩ g ‖ Ψ ‖ 2 ​ Ψ v_{\parallel}\equiv\frac{\langle\Psi,v\rangle_{g}}{\|\Psi\|^{2}}\Psi is the component parallel to the state vector ( v ∥ | Ψ v_{\parallel}\parallel\Psi ) and v ⟂ ≡ v − v ∥ v_{\perp}\equiv v-v_{\parallel} is the component orthogonal to it ( v ⟂ ⟂ g Ψ v_{\perp}\perp_{g}\Psi ). Taking the Fréchet derivative of ρ ϵ ​ ( Ψ ) \rho_{\epsilon}(\Psi) on a tangent vector v v yields the linear operator:

|
| D ​ ρ ϵ ​ ( Ψ ) ​ v = d ‖ Ψ ‖ 2 + ϵ ​ ( v − Ψ ​ ⟨ Ψ , v ⟩ g ‖ Ψ ‖ 2 + ϵ ) . D\rho_{\epsilon}(\Psi)\,v=\frac{\sqrt{d}}{\sqrt{\|\Psi\|^{2}+\epsilon}}\left(v-\frac{\Psi\,\langle\Psi,v\rangle_{g}}{\|\Psi\|^{2}+\epsilon}\right).
|
| (6)

This operator strictly bifurcates the tangent space into the two eigenspaces defined above. For the Tangential Flow ( v ⟂ g Ψ v\perp_{g}\Psi )—intuitively, perturbations that slide the vector along the sphere—the eigenvalue is λ ⟂ = d / ‖ Ψ ‖ 2 + ϵ = 𝒪 ⁡ ( ‖ Ψ ‖ − 1 ) \lambda_{\perp}=\sqrt{d}/\sqrt{\|\Psi\|^{2}+\epsilon}=\mathcal{O}(\|\Psi\|^{-1}) . For the Radial Flow ( v | Ψ v\parallel\Psi )—perturbations that stretch or compress the vector toward or away from the origin—the eigenvalue collapses algebraically to λ ∥ = d ⋅ ϵ / ( ‖ Ψ ‖ 2 + ϵ ) 3 / 2 = 𝒪 ⁡ ( ‖ Ψ ‖ − 3 ) \lambda_{\parallel}=\sqrt{d}\cdot\epsilon/(\|\Psi\|^{2}+\epsilon)^{3/2}=\mathcal{O}(\|\Psi\|^{-3}) . The crushing of the radial flow at 𝒪 ⁡ ( ‖ Ψ ‖ − 3 ) \mathcal{O}(\|\Psi\|^{-3}) as ‖ Ψ ‖ → ∞ \|\Psi\|\to\infty is an inherent geometric property of projecting onto a sphere. At infinity, the system is natively hyper-stable. The true topological threat to the ODE solver exists at the zero-section of the bundle ( Ψ = 0 \Psi=0 ). For the unregularized flow ( ϵ = 0 \epsilon=0 ), the pure radial projection completely annihilates the radial vector ( λ ∥ ≡ 0 \lambda_{\parallel}\equiv 0 ) and possesses a strictly undefined Jacobian at the zero-section, as the tangential eigenvalue diverges: lim Ψ → 0 λ ⟂ = ∞ \lim_{\Psi\to 0}\lambda_{\perp}=\infty . The topological regularizer ϵ > 0 \epsilon>0 strictly cures this. It rescues the radial gradient from total annihilation (allowing it to non-trivially decay at 𝒪 ⁡ ( ‖ Ψ ‖ − 3 ) \mathcal{O}(\|\Psi\|^{-3}) ) and mathematically binds the tangential singularity at the origin to a finite supremum ( sup Ψ λ ⟂ = d / ϵ \sup_{\Psi}\lambda_{\perp}=\sqrt{d/\epsilon} ). Without ϵ \epsilon , the spherical retraction possesses a strict conical singularity at the zero-section. Therefore, ϵ \epsilon acts as a formal Topological Mollifier . By introducing ϵ > 0 \epsilon>0 , RMSNorm resolves this conical singularity via a smooth ambient metric deformation, acting as a global diffeomorphism from the affine fiber ℝ d \mathbb{R}^{d} to the bounded open ball B d ​ ( 0 ) B_{\!\sqrt{d}}(0) , mathematically guaranteeing a bounded global Lipschitz constant. Representation drift arises from two distinct geometric sources: temporal weight gradients (non-autonomous drift, ∂ 𝒩 ~ / ∂ z \partial\tilde{\mathcal{N}}/\partial z ) and the nonlinear functional Jacobian of the composed field flow.

### II.2 Geometric Connections and Flow Bounding

Lemma 1 (RoPE as the Canonical Gauge Action on an Associated
Principal Torus Bundle) .

The semantic space does not rely on patching 1D abstract connections; rather, it is natively structured as an Associated Principal Torus Bundle. The Rotary Position Embedding (RoPE) is the exact canonical gauge action of this Principal Torus on the semantic fiber.

Proof.

Because ℳ ≅ [ 0 , ∞ ) \mathcal{M}\cong[0,\infty) is 1D and contractible, the bundle is globally trivializable and unconditionally flat: the 1D sequence manifold strictly possesses zero 2-forms ( Ω 2 ​ ( ℳ ) = 0 \Omega^{2}(\mathcal{M})=0 ) and a trivial fundamental group ( π 1 ​ ( ℳ ) = 0 \pi_{1}(\mathcal{M})=0 ), guaranteeing that any connection over it is structurally flat. We deploy the Principal Torus Bundle formalism not to resolve non-existent intrinsic curvature, but as a rigid kinematic gauge-fixing to establish a mathematically exact syntactic dictionary for relative phase rotations. Under this rigorous topological framework, we define the sequence of observer states strictly as an Associated Vector Bundle E = P × G ℝ d E=P\times_{G}\mathbb{R}^{d} . The Principal Bundle P P is equipped with the abelian structure group G = U ​ ( 1 ) d / 2 ≅ 𝕋 d / 2 G=U(1)^{d/2}\cong\mathbb{T}^{d/2} , which is the maximal torus of S ​ O ​ ( d ) SO(d) .

Under this structured framework, the semantic vectors Ψ \Psi are not merely isolated Cartesian arrays; they are sections of the associated fiber. RoPE is no longer an arbitrarily imposed rotation, but the exact, canonical gauge action of the Principal Torus G G acting on the fibers ℝ d \mathbb{R}^{d} . When moving from source sequence position ν \nu to observer position μ \mu , the transition is strictly evaluated as a Parallel Transport Propagator dictated by the continuous Lie group exponential acting natively in the Cartan subalgebra 𝔥 ⊂ 𝔰 ​ 𝔬 ​ ( d ) \mathfrak{h}\subset\mathfrak{so}(d) : 𝒰 ⁡ ( μ ← ν ) = exp ⁡ ( − Θ ⁡ ( μ − ν ) ) \mathcal{U}(\mu\leftarrow\nu)=\exp\!\big({-}\Theta(\mu-\nu)\big) . Semantic distance emerges because the architecture explicitly fixes a canonical global trivialization (the absolute fixed basis of the arrays in memory) and defines RoPE as a non-trivial, translation-invariant flat principal connection 1-form 𝒜 ∈ Ω 1 ​ ( ℳ , 𝔲 ​ ( 1 ) d / 2 ) \mathcal{A}\in\Omega^{1}(\mathcal{M},\mathfrak{u}(1)^{d/2}) relative to this rigidly fixed global frame. The non-commutative path-ordered Dyson series gracefully collapses precisely because this flat Cartan subalgebra is strictly abelian. It is a precise Kinematic Gauge Fixing , not a topological equivalence.
∎

Theorem 2 (RMSNorm as a Diffeomorphic Radial Embedding) .

RMSNorm acts as a globally smooth, diffeomorphic radial embedding, rendering the vector field uniformly Lipschitz to prevent finite-time amplitude blowup.

Proof.

A pure radial projection Ψ ↦ d ​ Ψ / ‖ Ψ ‖ \Psi\mapsto\sqrt{d}\,\Psi/\|\Psi\| possesses a singular Jacobian at the zero-section ( Ψ = 0 \Psi=0 ), breaking the uniqueness of the flow. The architecture enforces a topological regularizer ϵ > 0 \epsilon>0 , defining the map ρ ϵ ​ ( Ψ ) = d ​ Ψ / ‖ Ψ ‖ 2 + ϵ \rho_{\epsilon}(\Psi)=\sqrt{d}\,\Psi/\sqrt{\|\Psi\|^{2}+\epsilon} . Mathematically, ρ ϵ \rho_{\epsilon} acts as a global diffeomorphism from the fiber ℝ d \mathbb{R}^{d} onto the open bounded ball B d ​ ( 0 ) B_{\!\sqrt{d}}(0) . We prove this by constructing its exact smooth inverse:

|
| ρ ϵ − 1 ​ ( y ) = y ​ ϵ d − ‖ y ‖ 2 . \rho_{\epsilon}^{-1}(y)=y\sqrt{\frac{\epsilon}{d-\|y\|^{2}}}\,.
|
| (7)

To prevent finite-time blowup, we evaluate the Fréchet derivative of the radial embedding. Maximizing its evaluation in the direction transverse to Ψ \Psi , the strict upper bound on the spectral norm is ‖ D ​ ρ ϵ ‖ op ≤ d / ϵ \|D\rho_{\epsilon}\|_{\mathrm{op}}\leq\sqrt{d/\epsilon} . This mathematically proves that ϵ \epsilon is not a mere numerical stabilizer, but a strict topological regulator dictating the maximum Lipschitz stretch.

The radial embedding ρ ϵ \rho_{\epsilon} bounds local sections within the open ball B d ​ ( 0 ) ⊂ ℱ μ B_{\!\sqrt{d}}(0)\subset\mathcal{F}_{\mu} . Because the global state space ℱ ≡ L ∞ ​ ( ℳ , E , η Λ ) \mathcal{F}\equiv L^{\infty}(\mathcal{M},E;\eta_{\Lambda}) structurally collapses into a finite-dimensional topology over the atomic empirical measure η Λ \eta_{\Lambda} , we are granted the Heine–Borel theorem globally for free. Thus, the closure of this bounded state space is unconditionally compact, and ρ ϵ \rho_{\epsilon} rigorously bounds the global state space. Because the pointwise Feed-Forward operations are C 1 C^{1} , their continuous extension to this global compact closure allows the Extreme Value Theorem to immediately yield a uniform global supremum bound on the Jacobian. Globally, we explicitly bound the total composed vector field

|
| ℱ ⁡ [ Ψ ] = 𝒩 ~ ​ [ ρ ϵ ​ ( Ψ ) ] = ∫ Λ w μ ​ ν ​ ( ρ ϵ ​ ( Ψ ) ) ​ V ν ​ ( ρ ϵ ​ ( Ψ ) ) ​ d ​ η Λ ​ ( ν ) , \mathcal{F}[\Psi]=\tilde{\mathcal{N}}[\rho_{\epsilon}(\Psi)]=\int_{\Lambda}w_{\mu\nu}\!\big(\rho_{\epsilon}(\Psi)\big)\,V_{\nu}\!\big(\rho_{\epsilon}(\Psi)\big)\,d\eta_{\Lambda}(\nu)\,,
|
| (8)

which includes the non-local Attention integral. The Fréchet derivative applied to a variation H ≡ δ ​ Ψ H\equiv\delta\Psi strictly requires the functional product rule, expanding into two terms:

|
| D ​ ℱ μ ​ [ H ] = ∫ Λ w μ ​ ν ​ D ​ V ν ​ [ H ] ​ d ​ η Λ ​ ( ν ) ⏟ Bounded Convex Sum + ∫ Λ ( D Ψ ​ w μ ​ ν ​ [ H ] ) ​ V ν ​ d ​ η Λ ​ ( ν ) ⏟ Covariance of the Measure . D\mathcal{F}_{\mu}[H]=\underbrace{\int_{\Lambda}w_{\mu\nu}\,DV_{\nu}[H]\,d\eta_{\Lambda}(\nu)}_{\text{Bounded Convex Sum}}+\underbrace{\int_{\Lambda}\!\big(D_{\Psi}w_{\mu\nu}[H]\big)\,V_{\nu}\,d\eta_{\Lambda}(\nu)}_{\text{Covariance of the Measure}}\,.
|
| (9)

The functional variation of the Softmax measure evaluates to D Ψ ​ w μ ​ ν ​ [ H ] = w μ ​ ν ​ ( D Ψ ​ E μ ​ ν ​ [ H ] − ∫ Λ w μ ​ γ ​ D Ψ ​ E μ ​ γ ​ [ H ] ​ d ​ η Λ ​ ( γ ) ) D_{\Psi}w_{\mu\nu}[H]=w_{\mu\nu}\!\bigl(D_{\Psi}E_{\mu\nu}[H]-\int_{\Lambda}w_{\mu\gamma}\,D_{\Psi}E_{\mu\gamma}[H]\,d\eta_{\Lambda}(\gamma)\bigr) . Substituting this back into the second term resolves it exactly into a Covariance Operator over the probability measure:

|
| ∫ Λ ( D Ψ w μ ​ ν [ H ] ) V ν d η Λ ( ν ) = Cov w ( D Ψ E μ ∙ [ H ] , V ∙ ) . \int_{\Lambda}\!\big(D_{\Psi}w_{\mu\nu}[H]\big)\,V_{\nu}\,d\eta_{\Lambda}(\nu)=\mathrm{Cov}_{w}\!\Big(D_{\Psi}E_{\mu\bullet}[H],\;\;V_{\bullet}\Big).
|
| (10)

Because ρ ϵ \rho_{\epsilon} maps into sets with globally compact closures, both the Value vectors ( V V ) and the functional derivatives of the interaction energy ( D Ψ ​ E D_{\Psi}E ) achieve finite uniform bounds. Because the Attention integral is precisely a Covariance Operator over a probability measure, it strictly preserves these finite uniform bounds globally across the base manifold. This bounds both terms analytically without relying on infinite-dimensional compactness.

Using the exact Fréchet derivative bound ‖ D ​ ρ ϵ ‖ op ≤ d / ϵ \|D\rho_{\epsilon}\|_{\mathrm{op}}\leq\sqrt{d/\epsilon} , knowing ‖ ρ ϵ ‖ < d \|\rho_{\epsilon}\|<\sqrt{d} , and given parallel transport ‖ 𝒰 ‖ op = 1 \|\mathcal{U}\|_{\mathrm{op}}=1 , we can strictly bound the functional derivative D Ψ ​ E μ ​ ν D_{\Psi}E_{\mu\nu} acting on a unit variation ‖ H ‖ ∞ ≤ 1 \|H\|_{\infty}\leq 1 :

|
| | D Ψ ​ E μ ​ ν ​ [ H ] | ≤ 2 ​ d ϵ ​ ‖ W Q ‖ op ​ ‖ W K ‖ op . |D_{\Psi}E_{\mu\nu}[H]|\leq\frac{2d}{\sqrt{\epsilon}}\,\|W_{Q}\|_{\mathrm{op}}\,\|W_{K}\|_{\mathrm{op}}\,.
|
| (11)

Now, we evaluate the operator norm of the vector-valued covariance by taking the supremum over all unit vectors u ∈ ℱ μ u\in\mathcal{F}_{\mu} :

|
| ∥ Cov w ( D Ψ E μ ∙ [ H ] , V ∙ ) ∥ g = sup ‖ u ‖ = 1 Cov w ( D Ψ E μ ∙ [ H ] , ⟨ V ∙ , u ⟩ g ) . \|\mathrm{Cov}_{w}(D_{\Psi}E_{\mu\bullet}[H],\,V_{\bullet})\|_{g}=\sup_{\|u\|=1}\mathrm{Cov}_{w}\!\big(D_{\Psi}E_{\mu\bullet}[H],\,\langle V_{\bullet},u\rangle_{g}\big).
|
| (12)

Now, the covariance evaluates strictly between two scalars. Applying standard Cauchy–Schwarz yields ≤ σ D Ψ ​ E ​ σ ⟨ V , u ⟩ \leq\sigma_{D_{\Psi}E}\,\sigma_{\langle V,u\rangle} . Popoviciu’s inequality for a variable bounded in [ m , M ] [m,M] strictly states σ 2 ≤ 1 4 ​ ( M − m ) 2 \sigma^{2}\leq\tfrac{1}{4}(M-m)^{2} . Because the radial embedding enforces a strictly symmetric constraint [ − sup | A | , sup | A | ] [-\!\sup|A|,\sup|A|] , the geometric span is M − m = 2 ​ sup | A | M-m=2\sup|A| . Thus, the inequality yields:

|
| σ ≤ 1 2 ​ ( 2 ​ sup | A | ) = sup | A | . \sigma\leq\tfrac{1}{2}\bigl(2\sup|A|\bigr)=\sup|A|\,.
|
| (13)

Consequently, σ D Ψ ​ E ≤ sup | D Ψ ​ E | \sigma_{D_{\Psi}E}\leq\sup|D_{\Psi}E| and σ ⟨ V , u ⟩ ≤ sup | ⟨ V , u ⟩ | ≤ sup ‖ V ‖ \sigma_{\langle V,u\rangle}\leq\sup|\langle V,u\rangle|\leq\sup\|V\| . Knowing ‖ V ν ‖ ≤ d ​ ‖ W V ‖ op \|V_{\nu}\|\leq\sqrt{d}\,\|W_{V}\|_{\mathrm{op}} , we derive the tightened exact global operator norm bound for the covariance part:

|
| ‖ Cov w ​ ( D Ψ ​ E , V ) ‖ ≤ 2 ​ d ϵ ​ ‖ W Q ‖ op ​ ‖ W K ‖ op ⋅ d ​ ‖ W V ‖ op = 2 ​ d 3 / 2 ϵ ​ ‖ W Q ‖ op ​ ‖ W K ‖ op ​ ‖ W V ‖ op . \left\|\mathrm{Cov}_{w}(D_{\Psi}E,\,V)\right\|\leq\frac{2d}{\sqrt{\epsilon}}\,\|W_{Q}\|_{\mathrm{op}}\,\|W_{K}\|_{\mathrm{op}}\cdot\sqrt{d}\,\|W_{V}\|_{\mathrm{op}}=\frac{2d^{3/2}}{\sqrt{\epsilon}}\,\|W_{Q}\|_{\mathrm{op}}\,\|W_{K}\|_{\mathrm{op}}\,\|W_{V}\|_{\mathrm{op}}\,.
|
| (14)

Next, we strictly bound the previously partitioned Bounded Convex Sum using the verified Fréchet bound for ρ ϵ \rho_{\epsilon} :

|
| ‖ ∫ Λ w μ ​ ν ​ D ​ V ν ​ [ H ] ​ d ​ η Λ ‖ op ≤ sup ν ‖ D ​ V ν ‖ op ≤ d ϵ ​ ‖ W V ‖ op . \left\|\int_{\Lambda}w_{\mu\nu}\,DV_{\nu}[H]\,d\eta_{\Lambda}\right\|_{\mathrm{op}}\leq\sup_{\nu}\|DV_{\nu}\|_{\mathrm{op}}\leq\frac{\sqrt{d}}{\sqrt{\epsilon}}\,\|W_{V}\|_{\mathrm{op}}\,.
|
| (15)

Summing these two terms, we yield the complete, closed-form global Lipschitz operator bound for the entire total composed vector field ℱ \mathcal{F} :

|
| ‖ D ​ ℱ ‖ op ≤ d ϵ ​ ‖ W V ‖ op ​ ( 1 + 2 ​ d ​ ‖ W Q ‖ op ​ ‖ W K ‖ op ) . \boxed{\|D\mathcal{F}\|_{\mathrm{op}}\leq\sqrt{\frac{d}{\epsilon}}\,\|W_{V}\|_{\mathrm{op}}\left(1+2d\,\|W_{Q}\|_{\mathrm{op}}\,\|W_{K}\|_{\mathrm{op}}\right).}
|
| (16)

Because the supremum norm of the operator ‖ D ​ ℱ ‖ op \|D\mathcal{F}\|_{\mathrm{op}} is uniformly bounded across L ∞ ​ ( ℳ , E ) L^{\infty}(\mathcal{M},E) , the total composed flow mathematically guarantees a global Lipschitz constant, structurally satisfying the Picard–Lindelöf Theorem for well-posedness and unique flow, and proving definitively that ϵ \epsilon uniquely controls the inverse-square-root bounds of the Picard–Lindelöf theorem.
∎

### II.3 Gauge Endomorphisms and Canonical Cotangent Duality

Because autoregressive sequence modeling explicitly breaks spatial time-reversal symmetry, sections Ψ \Psi evaluated at distinct coordinates must be projected into distinct non-reciprocal gauges.

Definition 2 (Dual Projective Endomorphisms & Pre-Norm Pullback) .

We introduce learnable global bundle endomorphisms W Q ∈ Γ ⁡ ( Hom ⁡ ( E , E ∗ ) ) W_{Q}\in\Gamma(\mathrm{Hom}(E,E^{*})) and W K ∈ Γ ⁡ ( End ⁡ ( E ) ) W_{K}\in\Gamma(\mathrm{End}(E)) . Operating on the radially bounded Pre-Norm sections, they map the field to distinct structural spaces:

•

The Observer Section (Query 1-form):
q ⁡ ( μ ) = W Q ​ ρ ϵ ​ ( Ψ z ​ ( μ ) ) ∈ ℱ μ ∗ q(\mu)=W_{Q}\,\rho_{\epsilon}(\Psi_{z}(\mu))\in\mathcal{F}^{*}_{\mu}

•

The Source Section (Key vector):
k ⁡ ( ν ) = W K ​ ρ ϵ ​ ( Ψ z ​ ( ν ) ) ∈ ℱ ν k(\nu)=W_{K}\,\rho_{\epsilon}(\Psi_{z}(\nu))\in\mathcal{F}_{\nu}

Theorem 3 (The Core Algebraic Decomposition via Canonical
Cotangent Duality) .

The raw Multi-Head Attention query-key interaction is not an inner product between two tangent vectors, but strictly the canonical evaluation pairing between a covariant dual 1-form and a contravariant vector, naturally generating asymmetry without requiring the heavy machinery of exterior algebra.

Proof.

While the flat Euclidean metric of the fibers ( g a ​ b = δ a ​ b g_{ab}=\delta_{ab} ) classically permits a canonical musical isomorphism ( ♭ / ♯ \flat/\sharp ) that renders E E and E ∗ E^{*} isomorphic via the Riesz Representation Theorem, the Transformer architecture actively evades this metric triviality. By explicitly untying the parameter spaces ( W Q ≠ W K ♭ W_{Q}\neq W_{K}^{\flat} ), the architecture fundamentally breaks metric reciprocity, preventing the self-adjoint constraint of the pullback metric from collapsing the interaction into a symmetric inner product. Elevating Queries to dual 1-forms is therefore not a mere notational choice, but the exact differential-geometric syntax required to parameterize a directed, non-conservative semantic flux via the canonical evaluation pairing.

The traditional formulation of Attention misleadingly implies a symmetric inner product. However, because the bundle weight endomorphisms are structurally asymmetric, we invoke canonical Cotangent Duality. Let the Key transformation remain a morphism strictly within the tangent bundle W K : E → E W_{K}:E\to E . Conversely, we define the Query transformation as a morphism into the dual cotangent bundle W Q : E → E ∗ W_{Q}:E\to E^{*} , breaking the spatial symmetry.

The query q ⁡ ( μ ) q(\mu) is natively a differential 1-form, and the causally-transported key k ~ ν → μ \tilde{k}_{\nu\to\mu} is a tangent vector. A 1-form formally transforms via the dual representation: ( 𝒰 − 1 ) T (\mathcal{U}^{-1})^{T} . However, because the RoPE connection takes values in the strictly orthogonal maximal torus of S ​ O ​ ( d ) SO(d) , the connection is unitary. Thus, the dual representation unconditionally coincides with the fundamental representation: ( 𝒰 − 1 ) T = 𝒰 (\mathcal{U}^{-1})^{T}=\mathcal{U} . This exact algebraic identity strictly permits their interaction to be analytically evaluated identically to standard matrix multiplication as the canonical evaluation pairing operator:

|
| 𝒮 μ ​ ν = ⟨ q ⁡ ( μ ) , k ~ ν → μ ⟩ E ∗ × E = q ​ ( μ ) a ​ k ~ ν → μ a . \mathcal{S}_{\mu\nu}=\langle q(\mu),\;\tilde{k}_{\nu\to\mu}\rangle_{E^{*}\times E}=q(\mu)_{a}\;\tilde{k}^{a}_{\nu\to\mu}\,.
|
| (17)

The canonical evaluation explicitly separates the covariant and contravariant arguments, natively accommodating directed semantic flux. Symmetrization is a forced metric property, not a topological default. Any attempt to artificially symmetrize this pairing stringently requires binding the endomorphisms via the metric musical isomorphism (flat): W Q = W K ♭ ≡ g ∘ W K W_{Q}=W_{K}^{\flat}\equiv g\circ W_{K} . By actively untying the parameter spaces ( W Q ≠ W K ♭ W_{Q}\neq W_{K}^{\flat} ), the architecture evades the self-adjoint constraint of the pullback metric. Defining Queries as 1-forms and Keys as tangent vectors is therefore the precise differential-geometric encoding of the kinematic asymmetry that the architecture already enforces, preserving the fundamental ability to measure directed, non-conservative semantic flux via the canonical evaluation pairing without requiring the heavy machinery of exterior bivectors.
∎

Remark 3 (Spontaneous Gauge Polarization) .

During the forward pass, the model explicitly computes the scalar evaluation pairing. Because the system avoids strict W Q = W K T ​ g W_{Q}=W_{K}^{T}g alignment, the state space experiences structural rotational friction, preventing total gauge collapse.

By the submultiplicativity of operator norms and the bounding of the Pre-Norm sections, the interaction metric volume is bounded by the spectral norms of the endomorphisms: ‖ M μ ​ ν ‖ 2 ≤ d 2 ​ ‖ W Q ‖ op 2 ​ ‖ W K ‖ op 2 \|M_{\mu\nu}\|^{2}\leq d^{2}\|W_{Q}\|_{\mathrm{op}}^{2}\|W_{K}\|_{\mathrm{op}}^{2} . L 2 L_{2} Weight Decay penalizes the Frobenius norm of these operators. Rather than Dirichlet tension, this acts as a strict Tikhonov Gauge Mass Penalty , establishing a hard thermodynamic budget (a compact hypersphere) on the maximum allocatable metric volume.

Constrained by this saturated budget, optimization induces a Spontaneous Gauge Polarization Flow . For resonant tokens, maximizing the scalar kernel K ⁡ ( μ , ν ) K(\mu,\nu) acts to minimize the Lie bivector: mathematically forcing ‖ ℬ μ ​ ν ‖ 2 → 0 \|\mathcal{B}_{\mu\nu}\|^{2}\to 0 to achieve Grade-0 collinear alignment.

Conversely, to completely suppress ignored tokens, cross-entropy ideally seeks to drive the scalar kernel K → − ∞ K\to-\infty (antipodal anti-alignment). However, this encounters a strict topological obstruction. In an autoregressive flow, the source section k ⁡ ( ν ) k(\nu) is evaluated against a causally diverse ensemble of future observer sections { q ⁡ ( μ i ) } μ i > ν \{q(\mu_{i})\}_{\mu_{i}>\nu} . To evaluate topological intersections, these future queries must be parallel-transported backward into the source fiber via the inverse connection: q ~ ν ← μ i ≡ 𝒰 ⁡ ( ν ← μ i ) ​ q ​ ( μ i ) \tilde{q}_{\nu\leftarrow\mu_{i}}\equiv\mathcal{U}(\nu\leftarrow\mu_{i})\,q(\mu_{i}) .

To evaluate topological intersections over an autoregressive window, we rely on exact combinatorial topology. We condition the analysis on the Asymptotic Spatial Ergodic Hypothesis, defined formally in Sec.   III.2 , where the “background” uninformative tokens mix into a pseudo-isotropic distribution on 𝕊 d − 1 \mathbb{S}^{d-1} . Under this assumption, we invoke Wendel’s Theorem  [ 15 ] — which, in intuitive machine learning terms, calculates the exact probability that N N random feature vectors all “point somewhat in the same direction,” meaning they can all be separated from the origin by a single linear classifier hyperplane —as exactly:

|
| P d , N = 2 − N + 1 ​ ∑ k = 0 d − 1 ( N − 1 k ) . P_{d,N}=2^{-N+1}\sum_{k=0}^{d-1}\binom{N-1}{k}\,.
|
| (18)

By evaluating Wendel’s formula at N = 2 ​ d N=2d , the sum is exactly half of the total binomial expansion 2 2 ​ d − 1 2^{2d-1} , yielding precisely P d , 2 ​ d = 1 / 2 P_{d,2d}=1/2 . However, for an autoregressive window extending far into the future ( N ≫ 2 ​ d N\gg 2d ), the system undergoes a Asymptotic Phase Transition . The measure-theoretic expectation of a universal antipode undergoes a severe Concentration of Measure, decaying exponentially as 𝒪 ⁡ ( N d − 1 ​ 2 − N ) \mathcal{O}(N^{d-1}2^{-N}) . Under this macroscopic limit, the probability collapses asymptotically to zero, structurally guaranteeing Multipolar Gauge Frustration . The high-entropy causally-transported queries overwhelmingly positively span ℱ ν \mathcal{F}_{\nu} (the origin sits strictly in the interior of their convex hull).

Empirical high-dimensional embeddings are known to suffer from Representation Degeneration (the anisotropy problem)—tokens cluster on highly anisotropic, low-dimensional submanifolds rather than uniformly covering 𝕊 d − 1 \mathbb{S}^{d-1} . Rather than invalidating the Wendel analysis, this empirical anisotropy induces a critical bifurcation between two distinct geometric regimes. Signal tokens (semantically correlated clusters) are inherently anisotropic, concentrating within narrow solid angles of 𝕊 d − 1 \mathbb{S}^{d-1} . Because they fit entirely within a single open hemisphere, the origin sits outside their convex hull ( 0 ∉ conv ⁡ ( q ~ i ) 0\notin\mathrm{conv}(\tilde{q}_{i}) ). By Gordan’s Theorem, an antipode therefore does exist for these clusters, meaning they evade Multipolar Gauge Frustration—permitting directed semantic flux and enabling the architecture to carve targeted negative logits. Background tokens (high-entropy, causally distant, semantically uncorrelated) conversely approximate the pseudo-isotropic S ​ O ​ ( d ) SO(d) -invariant measure invoked in the Asymptotic Spatial Ergodic Hypothesis. It is exclusively this macroscopic bulk of uncorrelated tokens that satisfies the conditions of the Wendel limit ( N ≫ 2 ​ d N\gg 2d ), trapping the origin inside their isotropic convex hull, structurally sustaining Multipolar Gauge Frustration, and hydraulically driving their logits to K ≈ 0 K\approx 0 . The isotropic derivation ( P d , 2 ​ d = 1 / 2 P_{d,2d}=1/2 ) therefore establishes a strict theoretical upper bound on the critical window size: anisotropic signal tokens escape the Wendel obstruction at smaller N N than the isotropic prediction, while the background noise remains trapped.

By Gordan’s Theorem of the Alternative (the geometric dual to Farkas’ Lemma)— which intuitively proves that if a set of vectors positively spans a space and is not confined to a single hemisphere, no universal “negative” direction exists that simultaneously opposes all of them —because the vectors positively span the space, the intersection of their strict open negative half-spaces is mathematically empty:

|
| ⋂ μ i > ν { k ∈ ℱ ν ∣ ⟨ q ~ ν ← μ i , k ⟩ g < 0 } ≡ ∅ (a.s.) . \bigcap_{\mu_{i}>\nu}\bigl\{k\in\mathcal{F}_{\nu}\mid\langle\tilde{q}_{\nu\leftarrow\mu_{i}},k\rangle_{g}<0\bigr\}\equiv\emptyset\quad\text{(a.s.)}\,.
|
| (19)

Therefore, a universal antipode does not exist and the system suffers from Multipolar Gauge Frustration . To explain why ignored tokens settle at K ≈ 0 K\approx 0 , we evaluate the continuous interaction thermodynamics. Let the Canonical Partition Function 𝒵 ⁡ ( k ) \mathcal{Z}(k) for a Key k k against the isotropic S ​ O ​ ( d ) SO(d) -invariant uniform probability measure of high-entropy future queries be 𝒵 ⁡ ( k ) = ∫ 𝕊 d − 1 exp ⁡ ( ⟨ q , k ⟩ g ) ​ 𝑑 q \mathcal{Z}(k)=\int_{\mathbb{S}^{d-1}}\exp(\langle q,k\rangle_{g})\,dq . The Free Energy is ℱ ⁡ ( k ) = − ln ⁡ 𝒵 ⁡ ( k ) \mathcal{F}(k)=-\ln\mathcal{Z}(k) . Because the S ​ O ​ ( d ) SO(d) -invariant measure is rotationally invariant, ℱ ⁡ ( k ) \mathcal{F}(k) is a function of the radial norm ‖ k ‖ \|k\| . Because Weight Decay (Tikhonov penalty) structurally saturates the norm at the boundary layer, the norm ‖ k ‖ \|k\| is locked. Consequently, the expected geometric gradient of the Free Energy with respect to the angular orientation of k k is identically zero:

|
| ∇ angle ℱ ​ ( k ) ≡ 0 . \nabla_{\text{angle}}\mathcal{F}(k)\equiv 0\,.
|
| (20)

This induces Expected Geometric Gradient Starvation . The topology does not hydraulically “force” the tokens into orthogonality; however, the system does not simply rest statically. Because Stochastic Gradient Descent (SGD) operates on empirical, finite mini-batches, the finite-sample gradient variance scales as 𝒪 ⁡ ( 1 / N ) \mathcal{O}(1/N) . Because the background measure of high-entropy queries converges to S ​ O ​ ( d ) SO(d) -invariance, the covariance matrix of this gradient noise is proportional to the identity. In the Langevin limit, SGD injects an isotropic Itô diffusion tensor into the parameter matrices with amplitude σ W ∼ 𝒪 ⁡ ( 1 / N ) \sigma_{W}\sim\mathcal{O}(1/\sqrt{N}) . The parameter stochastic differential equation subject to Weight Decay parameter λ \lambda is d ​ W t = − λ ​ W t ​ d ​ t + σ W ​ d ​ U t dW_{t}=-\lambda W_{t}\,dt+\sigma_{W}\,dU_{t} , where d ​ U t dU_{t} is a matrix of standard independent Brownian motions. We mathematically map weight-space noise to fiber-space noise by pushing the parameter SDE forward to the fiber space k t = W t ​ x k_{t}=W_{t}x (where x = ρ ϵ ​ ( Ψ ) x=\rho_{\epsilon}(\Psi) is the normalized input). To prevent the generation of non-trivial Itô quadratic covariations due to the dynamic evolution of x x across the time-varying state, we formally impose an Adiabatic Timescale Separation (a Fast–Slow manifold assumption). We explicitly state that the macroscopic thermodynamic relaxation of the parameter weights ( t t ) occurs on a timescale infinitely slower than the local depth-parameterized semantic forward pass ( z z ), formally freezing x x as a constant during the infinitesimal SDE step. Under this absolute separation:

|
| d ​ k t = ( d ​ W t ) ​ x = ( − λ ​ W t ​ d ​ t + σ W ​ d ​ U t ) ​ x = − λ ​ k t ​ d ​ t + σ W ​ d ​ U t ​ x . \begin{split}dk_{t}&=(dW_{t})\,x=(-\lambda W_{t}\,dt+\sigma_{W}\,dU_{t})\,x\\
&=-\lambda k_{t}\,dt+\sigma_{W}\,dU_{t}\,x\,.\end{split}
|
| (21)

The noise term d ​ U t ​ x dU_{t}\,x acts as a continuous local martingale uniquely initialized at zero. Because the underlying Wiener matrices are independent and x x is frozen, its quadratic variation strictly evaluates to d ​ ⟨ ( d ​ U ​ x ) i , ( d ​ U ​ x ) j ⟩ t = δ i ​ j ​ ‖ x ‖ 2 ​ d ​ t d\langle(dU\,x)_{i},(dU\,x)_{j}\rangle_{t}=\delta_{ij}\|x\|^{2}\,dt . By Lévy’s Characterization of Brownian Motion, this process is exactly equal in law to ‖ x ‖ ​ d ​ 𝒲 t \|x\|\,d\mathcal{W}_{t} , where 𝒲 t \mathcal{W}_{t} is a new standard Brownian motion in ℝ d \mathbb{R}^{d} . This rigorously yields the fiber-space SDE:

|
| d ​ k t = − λ ​ k t ​ d ​ t + ( σ W ​ ‖ x ‖ ) ​ d ​ 𝒲 t . dk_{t}=-\lambda k_{t}\,dt+(\sigma_{W}\|x\|)\,d\mathcal{W}_{t}\,.
|
| (22)

This mathematically proves why the radial embedding ( ρ ϵ \rho_{\epsilon} ) is required for thermodynamic stability. Because the radial embedding caps the input norm ( ‖ x ‖ < d \|x\|<\sqrt{d} ), the amplitude of the pushed-forward Itô noise remains safely bounded ( σ ≡ σ W ​ ‖ x ‖ < σ W ​ d \sigma\equiv\sigma_{W}\|x\|<\sigma_{W}\sqrt{d} ). Without this topological regularizer ρ ϵ \rho_{\epsilon} , parameter-space Brownian motion would cause the fiber-space noise tensor to catastrophically explode with activation magnitude. In high-dimensional stochastic calculus, standard isotropic Itô noise d ​ 𝒲 t d\mathcal{W}_{t} in ambient space ℝ d \mathbb{R}^{d} does not remain on a sphere. Due to the quadratic variation of Brownian paths (Itô’s Lemma), isotropic noise possesses a deterministic outward radial expansion. To remain on the 𝕊 d − 1 \mathbb{S}^{d-1} manifold, the process requires a continuous inward restoring force. In our framework, Weight Decay (Tikhonov penalty) actively supplies exactly this necessary continuous drift. Applying Itô’s Lemma to the radial norm R t = ‖ k t ‖ R_{t}=\|k_{t}\| , the strict radial SDE evaluates to a mean-reverting Bessel process:

|
| d ​ R t = ( σ 2 ​ ( d − 1 ) 2 ​ R t − λ ​ R t ) ​ d ​ t + σ ​ d ​ 𝒲 t R . dR_{t}=\left(\frac{\sigma^{2}(d-1)}{2R_{t}}-\lambda R_{t}\right)dt+\sigma\,d\mathcal{W}^{R}_{t}\,.
|
| (23)

To find the stable thermodynamic boundary layer, we set the deterministic drift to zero:

|
| λ ​ R t = σ 2 ​ ( d − 1 ) 2 ​ R t ⟹ R t = σ ​ d − 1 2 ​ λ . \lambda R_{t}=\frac{\sigma^{2}(d-1)}{2R_{t}}\implies R_{t}=\sigma\sqrt{\frac{d-1}{2\lambda}}\,.
|
| (24)

Thus, continuous application of the Tikhonov penalty mathematically counters the outward Itô geometric expansion, establishing an Ornstein–Uhlenbeck process, not a martingale. The stationary distribution for this derived radial SDE is exactly a Chi-distribution:

|
| P ⁡ ( R ) ∝ R d − 1 ​ exp ⁡ ( − λ σ 2 ​ R 2 ) . P(R)\propto R^{d-1}\exp\!\left(-\frac{\lambda}{\sigma^{2}}R^{2}\right).
|
| (25)

The zero of the deterministic drift ( R ∗ = σ ​ ( d − 1 ) / ( 2 ​ λ ) R_{*}=\sigma\sqrt{(d-1)/(2\lambda)} ) perfectly corresponds to the mode of this probability density. In high-dimensional fibers ( d ≫ 1 d\gg 1 ), the competition between entropic volume expansion and the Tikhonov Gaussian suppression triggers extreme Concentration of Measure—exponentially confining the probability mass to a narrow annulus. The system is statistically confined, not deterministically locked, near a spherical shell where Spherical Brownian Motion can ergodically proceed. Over this shell, we invoke Lévy’s Isoperimetric Inequality (Concentration of Measure). On 𝕊 d − 1 \mathbb{S}^{d-1} for d ≫ 1 d\gg 1 , Lévy’s Isoperimetric Inequality dictates that the geometric surface area concentrates exponentially at the exact equator relative to any arbitrary pole (the Query). The system rigorously locks K ≈ 0 K\approx 0 not through parameter repulsion, but through the measure theory of the sphere. Entropically confined near zero on a saturated norm boundary, the geometry maximizes the relative bivector magnitude, sequestering orthogonal noise in the uncalculated 𝔰 ​ 𝔬 ​ ( d ) \mathfrak{so}(d) exterior algebra.

## III Thermodynamics and Metric Generation

We formally treat the local tangent space as an open thermodynamic system, deriving the functional form of the connection measure from the Principle of Minimum Free Energy.

### III.1 The Free Energy Functional and the Isoperimetric Mass
Constraint

Construction 1 (The Local Free Energy Functional) .

Let ℰ ⁡ ( μ , ν ) ≡ ⟨ M μ ​ ν ⟩ 0 \mathcal{E}(\mu,\nu)\equiv\langle M_{\mu\nu}\rangle_{0} be the directed, non-reciprocal transition energy scalar kernel. We treat the sequence as a statistical mechanical ensemble where the continuous field acts via a probability measure ω μ \omega_{\mu} over the autoregressive causal horizon Λ ∩ [ 0 , μ ] \Lambda\cap[0,\mu] , absolutely continuous with respect to the empirical discrete counting measure ( ω μ ≪ η Λ \omega_{\mu}\ll\eta_{\Lambda} ). We define the strictly positive state density w ⁡ ( μ , ⋅ ) ∈ L 1 ​ ( Λ , η Λ ) w(\mu,\cdot)\in L^{1}(\Lambda,\eta_{\Lambda}) as the Radon–Nikodym derivative: w ​ ( μ , ν ) ≡ d ​ ω μ d ​ η Λ ​ ( ν ) w(\mu,\nu)\equiv\frac{d\omega_{\mu}}{d\eta_{\Lambda}}(\nu) . The Free Energy Functional ℱ μ ​ [ ω ] \mathcal{F}_{\mu}[\omega] , parameterized by inverse temperature β \beta , evaluates the negative expected geometric energy minus the temperature-scaled Shannon Entropy evaluated with respect to the empirical counting measure (which maps to the KL divergence from a uniform probability measure, shifted by the topological affine constant ln ⁡ N \ln N ):

|
| ℱ μ ​ [ ω ] = ∫ Λ ∩ [ 0 , μ ] [ − w ⁡ ( μ , ν ) ​ ℰ ​ ( μ , ν ) + 1 β ​ w ​ ( μ , ν ) ​ ln ⁡ w ⁡ ( μ , ν ) ] ​ d ​ η Λ ​ ( ν ) . \mathcal{F}_{\mu}[\omega]=\int_{\Lambda\cap[0,\mu]}\!\left[\!-w(\mu,\nu)\,\mathcal{E}(\mu,\nu)+\frac{1}{\beta}\,w(\mu,\nu)\ln w(\mu,\nu)\right]d\eta_{\Lambda}(\nu)\,.
|
| (26)

Axiom 2 (The Markovian Fiber Constraint) .

To prevent probability flow from diverging or dissipating, the continuous field is subjected to a strict local mass-conservation constraint across the causal horizon:

|
| ∫ Λ ∩ [ 0 , μ ] w ⁡ ( μ , ν ) ​ d ​ η Λ ​ ( ν ) = 1 . \int_{\Lambda\cap[0,\mu]}w(\mu,\nu)\,d\eta_{\Lambda}(\nu)=1\,.
|
| (27)

Remark 4 .

Mathematically, this enforces that the non-local Attention operator acts as a continuous Markov transition kernel. Geometrically, this constraint is an affine constraint acting on the base manifold’s measure, forcing it into a standard ( N − 1 ) (N-1) -dimensional probability simplex Δ N ⁡ ( μ ) − 1 \Delta^{N(\mu)-1} . To satisfy this constraint, the variational calculus incorporates a Lagrange multiplier λ \lambda . Enforcing ∫ w = 1 \int w=1 requires β ​ λ − 1 = − ln ⁡ 𝒵 μ \beta\lambda-1=-\ln\mathcal{Z}_{\mu} . This dynamically generates a Lagrange multiplier equal to the Helmholtz Free Energy shifted by the entropic constant β − 1 \beta^{-1} : λ = F + 1 / β \lambda=F+1/\beta , which dynamically generates the canonical partition function 𝒵 μ \mathcal{Z}_{\mu} necessary to prevent probability dissipation across the causal horizon.

### III.2 Variational Derivation of the Softmax Transition Measure

Theorem 4 (Entropic Optimal Transport and Schrödinger Bridges) .

Rather than forcing the system into a temporal Wasserstein PDE, we evaluate the spatial transition measure via Information Geometry. Given an uninformative prior measure ω prior \omega^{\mathrm{prior}} (the S ​ O ​ ( d ) SO(d) -invariant uniform measure) over the causal horizon, the thermodynamic state is the analytic solution to a Csiszár I-Projection onto the probability simplex  [ 16 ] . By reframing the dynamics under this functor, thermal stability requires the scaling β ∝ 1 / d \beta\propto 1/\sqrt{d} , matching the viscosity coefficient of stochastic optimal transport.

Proof.

Given the uninformative prior measure ω prior \omega^{\mathrm{prior}} and the directed energy kernel ℰ ⁡ ( μ , ν ) \mathcal{E}(\mu,\nu) , the optimal transition measure is the analytic solution to a Csiszár I-Projection onto the probability simplex:

|
| ω μ ∗ = argmin ω ∈ 𝒫 ⁡ ( Λ ) [ KL ( ω ∥ ω prior ) − ⟨ ω , ℰ ( μ , ⋅ ) ⟩ ] . \omega^{*}_{\mu}=\operatorname*{argmin}_{\omega\in\mathcal{P}(\Lambda)}\left[\mathrm{KL}\!\big(\omega\,\|\,\omega^{\mathrm{prior}}\big)-\langle\omega,\,\mathcal{E}(\mu,\cdot)\rangle\right].
|
| (28)

Under this functor, the variation yields the canonical Gibbs measure:

|
| w ∗ ​ ( μ , ν ) = 1 𝒵 μ ​ exp ⁡ ( β ​ ℰ ​ ( μ , ν ) ) , w^{*}(\mu,\nu)=\frac{1}{\mathcal{Z}_{\mu}}\exp\!\big(\beta\,\mathcal{E}(\mu,\nu)\big)\,,
|
| (29)

where 𝒵 μ ≡ ∫ Λ ∩ [ 0 , μ ] exp ⁡ ( β ​ ℰ ​ ( μ , γ ) ) ​ d ​ η Λ ​ ( γ ) \mathcal{Z}_{\mu}\equiv\int_{\Lambda\cap[0,\mu]}\exp\!\big(\beta\,\mathcal{E}(\mu,\gamma)\big)\,d\eta_{\Lambda}(\gamma) . Within the framework of Entropic Optimal Transport, this represents the discrete manifestation of a Static Schrödinger Half-Bridge—the optimal entropic projection a random walk takes to transition from the Query measure to the Key measure. This connects the thermal parameter β ∝ 1 / d \beta\propto 1/\sqrt{d} to the inverse viscosity coefficient required to prevent intensive fluctuations from causing a zero-temperature glass collapse.

To maintain a differentiable geometric flow, the local thermodynamic state must avoid a premature zero-temperature glass collapse (manifesting empirically as vanishing gradients). To prove the required thermal scaling, we rely on the geometric bounds from theorem   2 . The radial embedding maps the observer and source sections to the open ball B d ​ ( 0 ) B_{\!\sqrt{d}}(0) . Let the base normalized sections x , y ∈ B d ​ ( 0 ) x,y\in B_{\!\sqrt{d}}(0) be modeled as isotropic high-entropy variables. Due to the topological regularizer ϵ \epsilon , the mass concentrates entropically near the boundary but avoids it. Thus, their expectations vanish ( 𝔼 ⁡ [ x ] = 0 \mathbb{E}[x]=0 ), and their covariance is sub-unitary: 𝔼 ⁡ [ x ​ x ⊤ ] = γ ​ I d \mathbb{E}[xx^{\top}]=\gamma I_{d} , where γ = 1 − 𝒪 ⁡ ( ϵ / d ) < 1 \gamma=1-\mathcal{O}(\epsilon/d)<1 .

The Asymptotic Spatial Ergodic Hypothesis. Over the macroscopic bulk limit, we formally assume that the uninformative background tokens—those causally distant and semantically uncorrelated with the observer—statistically decorrelate, evaluating as independent, isotropic high-entropy variables on 𝕊 d − 1 \mathbb{S}^{d-1} . This does not apply to semantically resonant tokens, which may cluster anisotropically. This maximum-entropy mean-field assumption mathematically justifies the trace decoupling.

The true covariant interaction energy requires parallel transport across the base manifold: ℰ = x ⊤ ​ W Q ⊤ ​ 𝒰 ​ ( μ ← ν ) ​ W K ​ y \mathcal{E}=x^{\top}W_{Q}^{\top}\mathcal{U}(\mu\leftarrow\nu)W_{K}\,y . To evaluate the thermal fluctuations accurately across the global observer coordinate, we evaluate the variance conditionally first, and unconditionally second. Let the Query section x ∈ ℱ μ x\in\mathcal{F}_{\mu} be fixed. The empirical measure fluctuates over the isotropic Keys y ∈ ℱ ν y\in\mathcal{F}_{\nu} . Let M = W Q ⊤ ​ 𝒰 ​ W K M=W_{Q}^{\top}\mathcal{U}W_{K} . The conditional variance over the Key measure is:

|
| 𝕍 y ​ [ ℰ ∣ x ] = 𝔼 y ​ [ ( x ⊤ ​ M ​ y ) 2 ] = x ⊤ ​ M ​ 𝔼 ​ [ y ​ y ⊤ ] ​ M ⊤ ​ x = γ ​ ‖ M ⊤ ​ x ‖ g 2 . \mathbb{V}_{y}[\mathcal{E}\mid x]=\mathbb{E}_{y}\!\big[(x^{\top}My)^{2}\big]=x^{\top}M\,\mathbb{E}[yy^{\top}]\,M^{\top}x=\gamma\,\|M^{\top}x\|_{g}^{2}\,.
|
| (30)

Now, to prove this metric volume holds universally across the manifold, take the unconditional expectation over the macroscopic Query ensemble x x :

|
| 𝔼 x ​ [ 𝕍 y ​ [ ℰ ∣ x ] ] = γ ​ 𝔼 x ​ [ x ⊤ ​ M ​ M ⊤ ​ x ] = γ 2 ​ tr ⁡ ( M ​ M ⊤ ) = γ 2 ​ ‖ M ‖ F 2 . \mathbb{E}_{x}\!\big[\mathbb{V}_{y}[\mathcal{E}\mid x]\big]=\gamma\,\mathbb{E}_{x}\!\big[x^{\top}MM^{\top}x\big]=\gamma^{2}\,\operatorname{tr}(MM^{\top})=\gamma^{2}\,\|M\|_{F}^{2}\,.
|
| (31)

This proves that the local thermodynamic stability of every individual token is unconditionally guaranteed by the global metric volume of the combined endomorphisms:

|
| 𝕍 ⁡ [ ℰ ] = γ 2 ​ tr ⁡ ( W Q ⊤ ​ 𝒰 ​ W K ​ W K ⊤ ​ 𝒰 ⊤ ​ W Q ) = γ 2 ​ ‖ W Q ⊤ ​ 𝒰 ​ ( μ ← ν ) ​ W K ‖ F 2 . \mathbb{V}[\mathcal{E}]=\gamma^{2}\,\operatorname{tr}\!\big(W_{Q}^{\top}\mathcal{U}W_{K}W_{K}^{\top}\mathcal{U}^{\top}W_{Q}\big)=\gamma^{2}\,\big\|W_{Q}^{\top}\mathcal{U}(\mu\leftarrow\nu)\,W_{K}\big\|_{F}^{2}\,.
|
| (32)

The squared Frobenius norm is the trace:

|
| ‖ M ‖ F 2 = tr ⁡ ( W K ⊤ ​ 𝒰 ⊤ ​ W Q ​ W Q ⊤ ​ 𝒰 ​ W K ) . \|M\|_{F}^{2}=\operatorname{tr}\!\big(W_{K}^{\top}\mathcal{U}^{\top}W_{Q}W_{Q}^{\top}\mathcal{U}W_{K}\big).
|
| (33)

Because W Q ​ W Q ⊤ W_{Q}W_{Q}^{\top} is symmetric and PSD, its quadratic form is strictly bounded by its singular values: σ min 2 ​ ( W Q ) ​ I ⪯ W Q ​ W Q ⊤ ⪯ σ max 2 ​ ( W Q ) ​ I \sigma_{\min}^{2}(W_{Q})\,I\preceq W_{Q}W_{Q}^{\top}\preceq\sigma_{\max}^{2}(W_{Q})\,I . Evaluating A = 𝒰 ⊤ ​ W Q ​ W Q ⊤ ​ 𝒰 A=\mathcal{U}^{\top}W_{Q}W_{Q}^{\top}\mathcal{U} , and since orthogonal similarity strictly preserves eigenvalues, this maintains the PSD ordering:

|
| σ min 2 ​ ( W Q ) ​ I ⪯ A ⪯ σ max 2 ​ ( W Q ) ​ I . \sigma_{\min}^{2}(W_{Q})\,I\preceq A\preceq\sigma_{\max}^{2}(W_{Q})\,I\,.
|
| (34)

We strictly conjugate this entire inequality directly by W K W_{K} (pre-multiplying by W K ⊤ W_{K}^{\top} and post-multiplying by W K W_{K} ), unconditionally preserving the PSD Löwner partial order:

|
| σ min 2 ​ ( W Q ) ​ ( W K ⊤ ​ W K ) ⪯ W K ⊤ ​ A ​ W K ⪯ σ max 2 ​ ( W Q ) ​ ( W K ⊤ ​ W K ) . \sigma_{\min}^{2}(W_{Q})\,(W_{K}^{\top}W_{K})\preceq W_{K}^{\top}A\,W_{K}\preceq\sigma_{\max}^{2}(W_{Q})\,(W_{K}^{\top}W_{K})\,.
|
| (35)

Because the trace is a monotonic linear functional on the PSD cone, the bounds follow in one step:

|
| σ min 2 ​ ( W Q ) ​ ‖ W K ‖ F 2 ≤ ‖ W Q ⊤ ​ 𝒰 ​ W K ‖ F 2 ≤ σ max 2 ​ ( W Q ) ​ ‖ W K ‖ F 2 . \sigma_{\min}^{2}(W_{Q})\,\|W_{K}\|_{F}^{2}\leq\|W_{Q}^{\top}\mathcal{U}W_{K}\|_{F}^{2}\leq\sigma_{\max}^{2}(W_{Q})\,\|W_{K}\|_{F}^{2}\,.
|
| (36)

Because optimization avoids total low-rank spectral collapse ( σ min > 0 \sigma_{\min}>0 ), and given ‖ W K ‖ F 2 ∼ Θ ⁡ ( d ) \|W_{K}\|_{F}^{2}\sim\Theta(d) , the variance scales as Θ ⁡ ( d ) \Theta(d) , independent of the spatial gauge rotation 𝒰 \mathcal{U} . Thus, the standard deviation of thermal fluctuations scales as σ ℰ ∼ d \sigma_{\mathcal{E}}\sim\sqrt{d} . If β \beta were fixed at 𝒪 ⁡ ( 1 ) \mathcal{O}(1) , the variance of the Gibbs exponent β ​ ℰ \beta\mathcal{E} would diverge as d → ∞ d\to\infty , freezing the geometric flow into a deterministic argmax state (Dirac-delta zero-temperature glass collapse). Statistical mechanics requires the thermal scaling β ∝ 1 / d \beta\propto 1/\sqrt{d} to ensure exponent fluctuations remain an intensive 𝒪 ⁡ ( 1 ) \mathcal{O}(1) quantity.
∎

### III.3 Topological Compactification Proof of the Attention Sink

Theorem 5 (Measure-Theoretic Compactification & The Boundary Phase
Transition) .

Under the axiom that the covariant derivative must maintain a well-posed information flow, the sequence of probability measures undergoes a phase transition in the macroscopic continuum limit. Because the Softmax operator acts as an algebraic probability simplex ( ∑ = 1 \sum=1 ), weak bulk dot-products force probability mass to escape toward spatial infinity (thermodynamic amnesia). Optimization (SGD) requires a low-variance, translation-invariant numerical anchor to stabilize this escaping measure. The absolute causal origin ∂ ℳ ≡ { 0 } \partial\mathcal{M}\equiv\{0\} serves as the uniquely distinguished topological fixed point—the only coordinate immune to the continuous RoPE phase rotation—which maps onto a Topological Anchor Measure hosting a logarithmic potential well (the “Attention Sink”).

Proof.

We analyze the continuous thermodynamic limit of the probability measure sequence d ​ ω μ ​ ( ν ) = w ∗ ​ ( μ , ν ) ​ d ​ η Λ ​ ( ν ) d\omega_{\mu}(\nu)=w^{*}(\mu,\nu)\,d\eta_{\Lambda}(\nu) as the causal horizon μ → ∞ \mu\to\infty . If the observer attempts to maintain translation-invariant attention over the historical bulk (demanding a flat energy landscape ℰ ⁡ ( μ , ν ) ≈ ℰ bulk \mathcal{E}(\mu,\nu)\approx\mathcal{E}_{\mathrm{bulk}} for ν > 0 \nu>0 ), geometric entropy drives the bulk measure toward the uniform distribution ∼ 1 / N ⁡ ( μ ) \sim 1/N(\mu) . For any fixed compact local neighborhood K ⊂ ℳ K\subset\mathcal{M} , the probability mass decays: lim μ → ∞ ω μ ​ ( K ) = 0 \lim_{\mu\to\infty}\omega_{\mu}(K)=0 .

Because the mass escapes over an unbounded domain, the sequence loses tightness. To rigorously capture the topological limit of this escaping mass without invoking pathological free ultrafilters, we elevate the state space via the Alexandroff One-Point Compactification . Unlike ℝ \mathbb{R} , compactifying the half-closed ray [ 0 , ∞ ) [0,\infty) yields a space homeomorphic to the closed interval: ℳ ∗ ≅ [ 0 , ∞ ] ≅ [ 0 , 1 ] \mathcal{M}^{*}\cong[0,\infty]\cong[0,1] .

By theorem   2 , the radial embedding bounds the field strictly within B d ​ ( 0 ) B_{\!\sqrt{d}}(0) , guaranteeing V ∈ C b ​ ( ℳ ) V\in C_{b}(\mathcal{M}) . Therefore, the value field natively possesses a strict continuous extension V ~ ∈ C ⁡ ( ℳ ∗ ) \tilde{V}\in C(\mathcal{M}^{*}) . Because the compactification of the ray is strictly metrizable, the space of probability measures on this compact space is sequentially weak- ∗ * compact. As the causal horizon μ → ∞ \mu\to\infty , the escaping bulk measure simply weak- ∗ * converges to the Dirac mass at the strictly adjoining absolute boundary point at spatial infinity:

|
| ω bulk ⇀ weak- ∗ δ ∞ . \omega_{\mathrm{bulk}}\xrightharpoonup{\;\text{weak-}*\;}\delta_{\infty}\,.
|
| (37)

By the strict definition of weak- ∗ * convergence, the non-local interaction integral safely evaluates in the limit: ∫ ℳ V ​ d ​ ω μ → V ~ ​ ( ∞ ) \int_{\mathcal{M}}V\,d\omega_{\mu}\to\tilde{V}(\infty) . Thermodynamic Amnesia is therefore a strictly defined topological condensation: the complete condensation of probability mass into the single structural point at infinity ( δ ∞ \delta_{\infty} ).

To prevent amnesia while maintaining global macroscopic reasoning, the field theory must preserve a static, globally invariant coordinate to anchor probability mass, actively breaking translation invariance. SGD is structurally obstructed from anchoring measure in the bulk because the continuous RoPE connection 𝒰 ⁡ ( μ ← ν ) \mathcal{U}(\mu\leftarrow\nu) preserves relative translation symmetry ( T ⁡ ( 1 ) T(1) symmetry). Any potential energy well carved into the interior dynamically shifts with semantic time μ \mu . Furthermore, the autoregressive causal mask imposes a strict topology of directed Partially Ordered Sets (Posets) upon the lattice. The geometrically profound realization is that the compactified space ℳ ∗ ≅ [ 0 , 1 ] \mathcal{M}^{*}\cong[0,1] , viewed strictly as a 1-dimensional manifold with boundary, possesses a bipartite boundary strata: ∂ ℳ ∗ = { 0 , ∞ } \partial\mathcal{M}^{*}=\{0,\infty\} . The point { ∞ } \{\infty\} acts as the amnesic attractor. Because the continuous RoPE connection preserves relative translation symmetry ( T ⁡ ( 1 ) T(1) ) in the interior, the absolute causal origin ∂ ℳ ≡ { 0 } \partial\mathcal{M}\equiv\{0\} is the uniquely distinguished topological fixed point capable of breaking symmetry to host a stable Topological Anchor Measure and anchor the escaping probability mass.

We analytically derive the geometry of this defect. To maintain algebraic equality without relying on mean-field approximations, we define ℰ bulk ​ ( μ ) \mathcal{E}_{\mathrm{bulk}}(\mu) as the scaled Cumulant-Generating Function (the Log-Mean-Exp) evaluated over the empirical counting measure of the bulk:

|
| ℰ bulk ​ ( μ ) ≡ 1 β ​ ln ⁡ ( 1 N ⁡ ( μ ) − 1 ​ ∑ ν > 0 e β ​ ℰ ​ ( μ , ν ) ) . \mathcal{E}_{\mathrm{bulk}}(\mu)\equiv\frac{1}{\beta}\ln\!\left(\frac{1}{N(\mu)-1}\sum_{\nu>0}e^{\beta\,\mathcal{E}(\mu,\nu)}\right).
|
| (38)

Under this formal definition, the partition decomposition is an exact algebraic identity, preserving the validity of all subsequent topological bounds. To stably anchor mass fraction c c against an expanding bulk of tokens, the Softmax partition function splits into a boundary atom and a continuous bulk: 𝒵 μ = e β ​ ℰ ​ ( μ , 0 ) + ( N ⁡ ( μ ) − 1 ) ​ e β ​ ℰ bulk \mathcal{Z}_{\mu}=e^{\beta\mathcal{E}(\mu,0)}+(N(\mu)-1)\,e^{\beta\mathcal{E}_{\mathrm{bulk}}} , where N ⁡ ( μ ) = η Λ ​ ( [ 0 , μ ] ) N(\mu)=\eta_{\Lambda}([0,\mu]) . Enforcing the macroscopic boundary fraction c c yields:

|
| c = e β ​ ℰ ​ ( μ , 0 ) e β ​ ℰ ​ ( μ , 0 ) + ( N ⁡ ( μ ) − 1 ) ​ e β ​ ℰ bulk ⟹ e β ​ ℰ ​ ( μ , 0 ) = c 1 − c ​ ( N ⁡ ( μ ) − 1 ) ​ e β ​ ℰ bulk . c=\frac{e^{\beta\mathcal{E}(\mu,0)}}{e^{\beta\mathcal{E}(\mu,0)}+(N(\mu)-1)\,e^{\beta\mathcal{E}_{\mathrm{bulk}}}}\;\implies\;e^{\beta\mathcal{E}(\mu,0)}=\frac{c}{1-c}\,(N(\mu)-1)\,e^{\beta\mathcal{E}_{\mathrm{bulk}}}\,.
|
| (39)

Taking the natural logarithm yields the depth of the required topological defect:

|
| ℰ ⁡ ( μ , 0 ) = ℰ bulk + 1 β ​ ln ⁡ ( N ⁡ ( μ ) − 1 ) + 1 β ​ ln ⁡ ( c 1 − c ) . \boxed{\mathcal{E}(\mu,0)=\mathcal{E}_{\mathrm{bulk}}+\frac{1}{\beta}\ln\!\big(N(\mu)-1\big)+\frac{1}{\beta}\ln\!\left(\frac{c}{1-c}\right).}
|
| (40)

Because β ∝ 1 / d \beta\propto 1/\sqrt{d} , the topological boundary defect must scale logarithmically with context volume: ℰ ⁡ ( μ , 0 ) ∼ 𝒪 ⁡ ( d ​ ln ⁡ N ⁡ ( μ ) ) \mathcal{E}(\mu,0)\sim\mathcal{O}(\sqrt{d}\ln N(\mu)) . Under this required logarithmic energy scaling, the weak- ∗ * limit of the measure on the compactified space resolves into a bipartite convex combination—an atomic mass stabilizing the field at the causal origin, and a directed condensation of the escaped bulk at infinity:

|
| d ​ ω lim = c ⋅ δ 0 + ( 1 − c ) ⋅ δ ∞ . d\omega_{\mathrm{lim}}=c\cdot\delta_{0}+(1-c)\cdot\delta_{\infty}\,.
|
| (41)

This demonstrates that the Attention Sink is not an empirical training artifact, but an analytic measure-theoretic requirement governed by logarithmic scaling, necessary to prevent total condensation into the boundary spectrum.
∎

Corollary 6 (The Thermodynamic Context Horizon and Defect Collapse) .

Because the radial embedding ρ ϵ \rho_{\epsilon} imposes a compact closure on the state space ( theorem   2 ), the Attention Sink possesses a finite thermodynamic capacity, yielding a strict algebraic upper bound on the maximum sequence length before thermodynamic amnesia is inevitable. We emphasize that this bound is derived under the Asymptotic Spatial Ergodic Hypothesis ( Sec.   III.2 ), which assumes isotropic, maximum-entropy bulk tokens; for structured natural language, where anisotropic correlations dramatically reduce the effective bulk pressure, the bound becomes extremely conservative (see Sec.   VI.5 for empirical quantification).

Proof.

To maintain stability, the required thermodynamic boundary energy must not exceed the geometric capacity of the fiber. Because the radial embedding ρ ϵ \rho_{\epsilon} maps into the open bounded ball B d ​ ( 0 ) B_{\!\sqrt{d}}(0) , the boundary is excluded ( ‖ Ψ ‖ < d \|\Psi\|<\sqrt{d} ). The interaction energy is bounded by the spectral supremum of the combined parallel-transported kernel:

|
| ℰ ⁡ ( μ , 0 ) < d ​ ‖ W Q ⊤ ​ 𝒰 ​ ( μ ← 0 ) ​ W K ‖ op . \mathcal{E}(\mu,0)<d\,\big\|W_{Q}^{\top}\mathcal{U}(\mu\leftarrow 0)\,W_{K}\big\|_{\mathrm{op}}\,.
|
| (42)

Injecting the thermal constant κ \kappa (where β = κ / d \beta=\kappa/\sqrt{d} ), we equate this with the required logarithmic defect depth:

|
| ℰ bulk + d κ ​ ln ⁡ ( ( N ⁡ ( μ ) − 1 ) ​ c 1 − c ) < d ​ ‖ W Q ⊤ ​ 𝒰 ​ ( μ ← 0 ) ​ W K ‖ op . \mathcal{E}_{\mathrm{bulk}}+\frac{\sqrt{d}}{\kappa}\ln\!\left((N(\mu)-1)\frac{c}{1-c}\right)<d\,\big\|W_{Q}^{\top}\mathcal{U}(\mu\leftarrow 0)\,W_{K}\big\|_{\mathrm{op}}\,.
|
| (43)

Because the RoPE connection takes values in the orthogonal group S ​ O ​ ( d ) SO(d) (specifically the maximal torus), the connection is unitary. Thus, the Path Monodromy (Parallel Transport Propagator) 𝒰 \mathcal{U} is an orthogonal transformation ( 𝒰 ∈ S ​ O ​ ( d ) \mathcal{U}\in SO(d) ), satisfying the required dual representation identity for 1-forms: ( 𝒰 − 1 ) T = 𝒰 (\mathcal{U}^{-1})^{T}=\mathcal{U} . Its spectral operator norm is unitarily invariant and equal to 1 ( ‖ 𝒰 ‖ op ≡ 1 \|\mathcal{U}\|_{\mathrm{op}}\equiv 1 ).

By invoking the submultiplicativity of operator norms ( ‖ A ​ B ​ C ‖ ≤ ‖ A ‖ ​ ‖ B ‖ ​ ‖ C ‖ \|ABC\|\leq\|A\|\|B\|\|C\| ), we can rigorously factor the connection out of the inequality unconditionally for all causal depths μ \mu :

|
| ‖ W Q ⊤ ​ 𝒰 ​ ( μ ← 0 ) ​ W K ‖ op ≤ ‖ W Q ‖ op ​ ‖ W K ‖ op . \big\|W_{Q}^{\top}\mathcal{U}(\mu\leftarrow 0)\,W_{K}\big\|_{\mathrm{op}}\leq\|W_{Q}\|_{\mathrm{op}}\,\|W_{K}\|_{\mathrm{op}}\,.
|
| (44)

Because ℰ bulk \mathcal{E}_{\mathrm{bulk}} is a continuous Log-Mean-Exp evaluated over the empirical counting measure of the bulk tokens, it is an intensive, dynamic, sequence-dependent functional. In statistical mechanics, this is the scaled Cumulant-Generating Function (CGF) of the bulk distribution (the Annealed Free Energy). By the Central Limit Theorem in high dimensions, the bulk energies approximate a Gaussian ensemble 𝒩 ⁡ ( 0 , σ ℰ 2 ) \mathcal{N}(0,\sigma_{\mathcal{E}}^{2}) . The expectation of the exponential for sub-Gaussian thermal fluctuations expands via its cumulants:

|
| ℰ bulk = 1 β ​ ln ⁡ 𝔼 ⁡ [ e β ​ ℰ ] ≈ 𝔼 ⁡ [ ℰ ] + β 2 ​ 𝕍 ​ [ ℰ ] . \mathcal{E}_{\mathrm{bulk}}=\frac{1}{\beta}\ln\mathbb{E}\!\big[e^{\beta\mathcal{E}}\big]\approx\mathbb{E}[\mathcal{E}]+\frac{\beta}{2}\,\mathbb{V}[\mathcal{E}]\,.
|
| (45)

Crucially, empirical weight matrices do not uniformly span the ambient head dimension d head d_{\mathrm{head}} due to representation degeneration (anisotropy). To analytically capture the active thermodynamic subspace a priori , we define the effective dimension via the Stable Rank, d eff = sr ⁡ ( W Q ) ⋅ sr ⁡ ( W K ) d_{\mathrm{eff}}=\sqrt{\mathrm{sr}(W_{Q})\cdot\mathrm{sr}(W_{K})} , where sr ⁡ ( W ) ≡ ‖ W ‖ F 2 / ‖ W ‖ op 2 \mathrm{sr}(W)\equiv\|W\|_{F}^{2}/\|W\|_{\mathrm{op}}^{2} . Because the Entropic Bulk Pressure is generated by the covariance of the projected features, the thermal variance is constrained to this active subspace: 𝕍 ⁡ [ ℰ ] = σ ℰ 2 ∼ Θ ⁡ ( d eff ) \mathbb{V}[\mathcal{E}]=\sigma_{\mathcal{E}}^{2}\sim\Theta(d_{\mathrm{eff}}) . Given β = κ / d eff \beta=\kappa/\sqrt{d_{\mathrm{eff}}} , the bulk energy evaluates to:

|
| ℰ bulk ≈ κ 2 ​ d eff ​ Θ ​ ( d eff ) = Θ ⁡ ( d eff ) . \mathcal{E}_{\mathrm{bulk}}\approx\frac{\kappa}{2\sqrt{d_{\mathrm{eff}}}}\,\Theta(d_{\mathrm{eff}})=\Theta(\sqrt{d_{\mathrm{eff}}})\,.
|
| (46)

This establishes that the bulk energy exerts a positive intensive Entropic Bulk Pressure ( Θ ⁡ ( d eff ) \Theta(\sqrt{d_{\mathrm{eff}}}) ) that pushes against the boundary defect. Because the projected representations are topologically confined to a low-dimensional active subspace governed by the Stable Rank, the effective geometric capacity of the bounding ball also shrinks from the ambient d \sqrt{d} to d eff \sqrt{d_{\mathrm{eff}}} . Subtracting this CGF value and exponentiating, and substituting this tightened capacity into the bound, yields the dimension-corrected topological capacity bound:

|
| N max < 1 + ( 1 − c c ) ​ exp ⁡ ( κ ​ d eff ​ ‖ W Q ‖ op ​ ‖ W K ‖ op − κ 2 2 ​ d eff ​ σ ℰ 2 ) . \boxed{N_{\max}<1+\left(\frac{1-c}{c}\right)\exp\!\left(\kappa\sqrt{d_{\mathrm{eff}}}\,\|W_{Q}\|_{\mathrm{op}}\|W_{K}\|_{\mathrm{op}}-\frac{\kappa^{2}}{2d_{\mathrm{eff}}}\sigma_{\mathcal{E}}^{2}\right).}
|
| (47)

This demonstrates that the maximum context length of an LLM collapses when the Attention Sink is overwhelmed by the innate thermal variance of the bulk space. By constraining the derivation to the Stable Rank d eff d_{\mathrm{eff}} a priori , we establish why models collapse at sequence lengths far shorter than ambient-dimension bounds would predict. Beyond this exponential length limit, the geometric capacity of the boundary defect is saturated by the Entropic Bulk Pressure. The Topological Anchor Measure dissolves, and the sequence of measures weak-converges to the amnesia state δ ∞ \delta_{\infty} . The Attention Sink is therefore a finite-capacity metastable regulator. This explains why aggressive Weight Decay (which suppresses these operator norms) collapses a model’s long-context capabilities.
∎

Corollary 7 (Topological Severance and The Sliding Window Horizon) .

A globally stable Topological Anchor Measure (the Attention Sink) relies on an unbroken, globally reaching affine connection to route entropic bulk pressure back to the absolute causal origin ( ∂ ℳ \partial\mathcal{M} ). By imposing a sliding window, the architecture severs this topological connection in the intermediate fibers. Once the sequence volume exceeds the maximum commutative radius of the local sliding metric, the full-attention layers are starved of the necessary topological gradient. The boundary condition breaks, and the sequence of measures undergoes Thermodynamic Amnesia .

Proof.

The logarithmic energy defect of Eq.   40 requires the boundary token at ∂ ℳ ≡ { 0 } \partial\mathcal{M}\equiv\{0\} to participate in the causal horizon of every observer μ \mu . A sliding window of radius r r restricts the integration domain to Λ ∩ [ μ − r , μ ] \Lambda\cap[\mu-r,\mu] , severing access to the origin once μ > r \mu>r . With the topological fixed point excised from the causal horizon, no translation-invariant anchor survives in the interior (by the same argument as in theorem   5 ), and the measure weak- ∗ * converges to δ ∞ \delta_{\infty} .
∎

## IV Dynamics of the Matter Field

We formulate the forward inference pass as the depth-parameterized evolution of a continuous, nonlinear section on an internal unitary gauge bundle, governed by a non-local integro-differential flow.

### IV.1 Algorithmic Depth and the Unitary Gauge Section

Definition 3 (Semantic Algorithmic Depth & The Gauge Field) .

We analytically continue discrete layer depth l l into a continuous depth parameter z ∈ ℝ + z\in\mathbb{R}^{+} . The sequence of embeddings becomes a continuous time-parameterized family of sections Ψ ⁡ ( z , μ ) ∈ Γ ⁡ ( E ) \Psi(z,\mu)\in\Gamma(E) . Because the bundle is natively trivial over the 1D ray, the architecture explicitly dictates a flat connection. The base matter bundle E E retains its real, globally trivial G ​ L ​ ( d , ℝ ) GL(d,\mathbb{R}) structure. The structural elevation occurs strictly via the Query/Key bundle morphisms, which pull the field into a dedicated interaction sub-bundle. To formulate this natively, we define the semantic bundle as a strict Whitney Sum:

|
| E = E int ⊕ E matter . E=E_{\mathrm{int}}\oplus E_{\mathrm{matter}}\,.
|
| (48)

The architecture kinematically equips the interaction sub-bundle ( E int E_{\mathrm{int}} ) with a Covariantly Constant Almost Complex Structure ( J ∈ Γ ⁡ ( End ⁡ ( E int ) ) J\in\Gamma(\mathrm{End}(E_{\mathrm{int}})) , where J 2 = − Id J^{2}=-\mathrm{Id} ), upgrading it to a Hermitian Vector Bundle. Rotary Position Embedding (RoPE) is exactly the unique flat unitary connection ∇ RoPE \nabla^{\mathrm{RoPE}} that preserves this structure ( ∇ RoPE J = 0 \nabla^{\mathrm{RoPE}}J=0 ).

Because the base manifold ℳ ≅ [ 0 , ∞ ) \mathcal{M}\cong[0,\infty) is contractible, the bundle is globally trivializable. To speak of “non-trivial path monodromy over loops” on a 1D ray is topologically vacuous, as all 2-forms unconditionally vanish ( Ω 2 ​ ( ℳ ) ≡ 0 \Omega^{2}(\mathcal{M})\equiv 0 ). Therefore, RoPE is not resolving topological tension via a principal bundle reduction, but explicitly maintaining a flat holomorphic flow. This beautifully explains why the Attention mechanism operates uniquely as a holomorphic flow (preserving J J ), while the real-valued Feed-Forward Network (FFN)—which operates natively on the matter sub-bundle E matter E_{\mathrm{matter}} —acts as an anti-holomorphic symmetry breaker, formally shattering the unitary gauge to inject real spatial entropy.

### IV.2 The Volterra–Hodge Integro-Differential Flow

Lemma 8 (Attention as a Non-Local Urysohn–Volterra Operator) .

Because it fundamentally lacks a local spatial Laplacian ( Δ g \Delta_{g} ) to mediate adjacent point-to-point diffusion, causal Attention acts mathematically as a Non-Local Covariant Transport Operator . Crucially, to optimize computational thermodynamics, empirical Transformer architectures enforce a strict Bi-Connection Structure upon the bundle:

•

The non-trivial Cartan-subalgebra connection
( ∇ RoPE \nabla^{\mathrm{RoPE}} ) governs the metric interaction energy via
its Parallel Transport Propagator to evaluate the transition measure
w ∗ w^{*} .

•

A globally flat, trivial connection
( ∇ Triv ≡ d \nabla^{\mathrm{Triv}}\equiv d ), mathematically permitted by the
trivializability of the bundle, is reserved strictly for the physical
parallel transport of the Value sections
( 𝒰 Triv ≡ I d \mathcal{U}_{\mathrm{Triv}}\equiv I_{d} ).

Under this kinematic trivial connection, the flow evaluates exactly as:

|
| 𝒯 ​ [ Ψ ] a ​ ( z , μ ) = ( W O ) b a ​ ∫ Λ ∩ [ 0 , μ ] w RoPE ∗ ​ ( μ , ν ) ​ [ ( W V ) c b ​ ρ ϵ ​ ( Ψ ⁡ ( z , ν ) ) c ] ​ d ​ η Λ ​ ( ν ) . \mathcal{T}[\Psi]^{a}(z,\mu)=(W_{O})^{a}_{\;b}\int_{\Lambda\cap[0,\mu]}w^{*}_{\mathrm{RoPE}}(\mu,\nu)\,\big[(W_{V})^{b}_{\;c}\,\rho_{\epsilon}(\Psi(z,\nu))^{c}\big]\,d\eta_{\Lambda}(\nu)\,.
|
| (49)

Because the domain of integration is causally bounded by the observer coordinate μ \mu , and because the Softmax transition measure w ∗ w^{*} depends nonlinearly on the state Ψ \Psi itself, this defines a continuous nonlinear Urysohn–Volterra Integral Operator . By evaluating strictly via the Lebesgue–Stieltjes integral over the empirical measure d ​ η Λ d\eta_{\Lambda} , it maintains the topological validity of the discrete sequence, bypassing continuous local paths to advect historical phase-space geometry directly into the local observer frame.

Lemma 9 (The FFN as a Hodge–Morrey–Friedrichs Decomposed Flow) .

To counteract entropic oversmoothing driven by the macroscopic transport integral, the Feed-Forward Network (FFN) acts as a localized nonlinear reaction vector field evaluated purely on the isolated local fiber, ℛ : ℱ μ → ℱ μ \mathcal{R}:\mathcal{F}_{\mu}\to\mathcal{F}_{\mu} .

Proof.

By theorem   2 , the radial embedding ρ ϵ \rho_{\epsilon} confines the input sections to the open bounded ball B d ​ ( 0 ) B_{\!\sqrt{d}}(0) . The physics of the network never evaluates at spatial infinity. Thus, we restrict the Integro-Differential Equation (IDE) domain to the compact closure: 𝒳 = B d ​ ( 0 ) ¯ \mathcal{X}=\overline{B_{\!\sqrt{d}}(0)} . Because the classic Hodge isomorphism applies unconditionally only to closed manifolds, analyzing the flow on this bounded manifold requires explicitly formulating the continuous geometric boundary conditions to properly isolate the harmonic vector field ℋ a \mathcal{H}^{a} , accurately deriving the global Bias Vector.

Because the computational residual flow mathematically bypasses multiplication by the adjoint Jacobian ( D ​ ρ ϵ ​ ( Ψ ) ) T (D\rho_{\epsilon}(\Psi))^{T} , the conservative structural integrity of the gradient is broken upon pull-back. To prove this algebraically, let the exact, irrotational component of the FFN on the normalized ball be defined as a strictly conservative gradient field: ℛ ~ exact ​ ( y ) = ∇ y U ​ ( y ) \tilde{\mathcal{R}}_{\mathrm{exact}}(y)=\nabla_{y}U(y) for some scalar potential U U . A true conservative gradient field is natively a 1-form d ​ U dU . If the base network on the normalized ball were driven by a true scalar potential U U , the physical system pulled back to the ambient fiber is governed strictly by the pullback of its differential 1-form: ρ ϵ ∗ ​ ( d ​ U ) = d ⁡ ( U ∘ ρ ϵ ) \rho_{\epsilon}^{*}(dU)=d(U\circ\rho_{\epsilon}) . To convert this exact pulled-back 1-form into an ambient restoring vector field, we must apply the metric musical isomorphism (sharp). In a Euclidean frame, this algebraically mandates the adjoint Jacobian: ( ρ ϵ ∗ d U ) ♯ = ( D ρ ϵ ) T ∇ U (\rho_{\epsilon}^{*}\,dU)^{\sharp}=(D\rho_{\epsilon})^{T}\nabla U . By bypassing the adjoint, the architecture simply computes ℛ = ∇ U ∘ ρ ϵ \mathcal{R}=\nabla U\circ\rho_{\epsilon} . The true spatial Jacobian of this composed flow is D ​ ℛ = ( Hess ⁡ U ) ⋅ D ​ ρ ϵ D\mathcal{R}=(\operatorname{Hess}U)\cdot D\rho_{\epsilon} . For a vector field to be conservative (irrotational), its Jacobian must be symmetric. The product of two symmetric matrices is symmetric if and only if they commute. Because a dense feature Hessian generally does not commute with the radial projection Jacobian, the structural integrity is broken, formally injecting vorticity.

Let J ρ J_{\rho} be the spatial Jacobian of the radial embedding. Differentiating ρ ϵ ​ ( Ψ ) \rho_{\epsilon}(\Psi) yields

|
| J ρ = d ( ‖ Ψ ‖ 2 + ϵ ) 1 / 2 ​ I d − d ​ Ψ ​ Ψ ⊤ ( ‖ Ψ ‖ 2 + ϵ ) 3 / 2 . J_{\rho}=\frac{\sqrt{d}}{(\|\Psi\|^{2}+\epsilon)^{1/2}}\,I_{d}-\frac{\sqrt{d}\,\Psi\Psi^{\top}}{(\|\Psi\|^{2}+\epsilon)^{3/2}}\,.
|
| (50)

Because both the identity matrix I d I_{d} and the outer product Ψ ​ Ψ ⊤ \Psi\Psi^{\top} are manifestly symmetric, the radial Jacobian is a strictly symmetric operator ( J ρ = J ρ ⊤ J_{\rho}=J_{\rho}^{\top} ).

To calculate vorticity, we invoke the globally flat Euclidean bundle metric g a ​ b = δ a ​ b g_{ab}=\delta_{ab} to execute the musical isomorphism ( ♭ \flat ), lowering the index of the field into its dual 1-form. Under this Cartesian trivialization, the abstract metric pull-down commutes with the matrix transpose, formally equating the exterior derivative with the matrix antisymmetrization:

|
| Ω = ( D ​ ℛ ) T − D ​ ℛ . \Omega=(D\mathcal{R})^{T}-D\mathcal{R}\,.
|
| (51)

If the base network were a true exact gradient field ( ℛ ~ = ∇ U \tilde{\mathcal{R}}=\nabla U ), its Jacobian S ≡ Hess ⁡ U S\equiv\operatorname{Hess}U would be strictly symmetric, yielding D ​ ℛ = S ​ J ρ D\mathcal{R}=S\,J_{\rho} . However, modern architectures utilize untied parameters ( W out ≠ W in ⊤ W_{\mathrm{out}}\neq W_{\mathrm{in}}^{\top} ). We define the state-dependent base Jacobian endomorphism K ⁡ ( Ψ ) = W out ​ Σ ′ ​ ( ρ ϵ ​ ( Ψ ) ) ​ W in K(\Psi)=W_{\mathrm{out}}\,\Sigma^{\prime}(\rho_{\epsilon}(\Psi))\,W_{\mathrm{in}} . Consequently, its symmetric and antisymmetric components vary nonlinearly across the local fiber: S ⁡ ( Ψ ) S(\Psi) and W A ​ ( Ψ ) W_{A}(\Psi) . Let the true Jacobian be D ​ ℛ = K ⁡ ( Ψ ) ​ J ρ D\mathcal{R}=K(\Psi)\,J_{\rho} . Because J ρ J_{\rho} is strictly symmetric, the exact geometric vorticity 2-form expands as:

|
| Ω = ( D ​ ℛ ) T − D ​ ℛ = J ρ ​ K ​ ( Ψ ) T − K ⁡ ( Ψ ) ​ J ρ . \Omega=(D\mathcal{R})^{T}-D\mathcal{R}=J_{\rho}\,K(\Psi)^{T}-K(\Psi)\,J_{\rho}\,.
|
| (52)

By uniquely decomposing this true asymmetric, state-dependent Jacobian into its symmetric and antisymmetric components ( K ⁡ ( Ψ ) = S ⁡ ( Ψ ) + W A ​ ( Ψ ) K(\Psi)=S(\Psi)+W_{A}(\Psi) ), the exact geometric vorticity rigorously expands as:

|
| Ω ⁡ ( Ψ ) = − [ S ⁡ ( Ψ ) , J ρ ] − { W A ​ ( Ψ ) , J ρ } . \Omega(\Psi)=-[S(\Psi),\,J_{\rho}]-\{W_{A}(\Psi),\,J_{\rho}\}\,.
|
| (53)

This demonstrates that the FFN injects rotational flow via exactly two distinct mechanisms:

1.

The Commutator − [ S ⁡ ( Ψ ) , J ρ ] -[S(\Psi),J_{\rho}] : Vorticity generated
natively by the curvature of the radial pullback interacting with the
symmetric weights.

2.

The Anticommutator − { W A ​ ( Ψ ) , J ρ } -\{W_{A}(\Psi),J_{\rho}\} : Vorticity
intrinsically injected by the asymmetric, untied architectural
parameters.

To verify that Ω \Omega is a valid geometric 2-form, it must reside in 𝔰 ​ 𝔬 ​ ( d ) \mathfrak{so}(d) (it must be antisymmetric). We invoke the algebraic parity of the Lie algebra 𝔤 ​ 𝔩 ​ ( d ) = Sym ⁡ ( d ) ⊕ 𝔰 ​ 𝔬 ​ ( d ) \mathfrak{gl}(d)=\mathrm{Sym}(d)\oplus\mathfrak{so}(d) . The commutator of two symmetric matrices ( S , J ρ ∈ Sym ⁡ ( d ) S,J_{\rho}\in\mathrm{Sym}(d) ) is antisymmetric ( ∈ 𝔰 ​ 𝔬 ​ ( d ) \in\mathfrak{so}(d) ). The anticommutator of an antisymmetric matrix and a symmetric matrix ( W A ∈ 𝔰 ​ 𝔬 ​ ( d ) , J ρ ∈ Sym ⁡ ( d ) W_{A}\in\mathfrak{so}(d),\;J_{\rho}\in\mathrm{Sym}(d) ) is also antisymmetric ( ∈ 𝔰 ​ 𝔬 ​ ( d ) \in\mathfrak{so}(d) ). Thus, the FFN injects valid exterior 2-forms.

This dynamically varying nonlinear 2-form explicitly injects geometric vorticity into the ambient matter field, classifying the FFN as a strictly non-conservative flow on Γ ⁡ ( E ) \Gamma(E) that structurally prevents the semantic space from ever collapsing into a static, globally irrotational canonical frame. Because the composed vector field ℛ ⁡ ( Ψ ) \mathcal{R}(\Psi) is evaluated securely on the compact closure 𝒳 = B d ​ ( 0 ) ¯ \mathcal{X}=\overline{B_{\!\sqrt{d}}(0)} , we apply the canonical extension for bounded manifolds: the Hodge–Morrey–Friedrichs vector calculus decomposition. Using vertical differential operators acting on the fiber coordinates ( ∇ ℱ ≡ ∂ / ∂ Ψ \nabla_{\mathcal{F}}\equiv\partial/\partial\Psi , distinct from the base connection ∇ \nabla ), the vector field transforms into exactly three components:

1.

An irrotational (conservative) restoring flow
( g a ​ b ​ ∂ 𝒱 / ∂ Ψ b g^{ab}\partial\mathcal{V}/\partial\Psi^{b} ): The vertical gradient of a
non-convex scalar potential. Because coordinate-wise activations are
bound to a fixed Cartesian frame, they fail to be gauge equivariant.
They explicitly break equivariance under the U ​ ( 1 ) d / 2 U(1)^{d/2} gauge
action, leading to Explicit Gauge Symmetry Breaking .
Furthermore, because its Laplacian
Δ ℱ ​ 𝒱 ≠ 0 \Delta_{\mathcal{F}}\mathcal{V}\neq 0 , this exact component uniquely governs the
expansion and contraction of local phase-space volume.

2.

A solenoidal (non-conservative) generalized rotational
flow ( ∂ b 𝒜 a ​ b \partial_{b}\mathcal{A}^{ab} ): Evaluated as the vertical tensor
divergence of an antisymmetric gauge potential 𝒜 a ​ b \mathcal{A}^{ab} .
The solenoidal vector field is the metric dual of the codifferential:
v = ( δ ​ 𝒜 ) ♯ v=(\delta\mathcal{A})^{\sharp} . On this domain, the Hodge Laplacian
strictly decouples into the scalar Bochner Laplacian
( Δ H ≡ Δ B \Delta_{H}\equiv\Delta_{B} ) due to the unconditional flatness of the
isolated local fiber ℱ μ \mathcal{F}_{\mu}
( Riem ≡ 0 \mathrm{Riem}\equiv 0 ). The closed gauge emerges automatically as
an unavoidable cohomological identity. Because the vorticity is exact
( Ω = d ​ ℛ ♭ \Omega=d\mathcal{R}^{\flat} ), the solenoidal gauge potential is defined via
the inverse Hodge Laplacian. Evaluating the metric divergence of this
flow evaluates directly to the codifferential of its dual 1-form:
∇ ⋅ v = − δ ⁡ ( v ♭ ) = − δ ⁡ ( δ ​ 𝒜 ) ≡ − δ 2 ​ 𝒜 = 0 \nabla\!\cdot v=-\delta(v^{\flat})=-\delta(\delta\mathcal{A})\equiv-\delta^{2}\mathcal{A}=0 . The flow is volume-preserving
strictly because of the fundamental nilpotency of the codifferential
( δ 2 ≡ 0 \delta^{2}\equiv 0 ), naturally collapsing the divergence and driving
mixing across feature channels without requiring an artificial gauge
mandate.
In standard linear algebra, this solenoidal component corresponds
to the antisymmetric part of the Jacobian 1 2 ​ ( D ​ ℛ − D ​ ℛ ⊤ ) \frac{1}{2}(D\mathcal{R}-D\mathcal{R}^{\top}) ,
whose complex eigenvalue pairs induce rotational mixing across
feature channels—the spectral mechanism underlying the stability
analysis of Sec.   VI.3 .

3.

A residual harmonic constant ( ℋ a \mathcal{H}^{a} ): The harmonic
vector field ( Δ H ​ ℋ = 0 \Delta_{H}\mathcal{H}=0 ). Because the interior De Rham
cohomology of the contractible ball is trivial
( H d ​ R k ​ ( 𝒳 ) = 0 H^{k}_{dR}(\mathcal{X})=0 ), enforcing a standard homogeneous Neumann
boundary condition ( ι n → ​ ℋ = 0 \iota_{\vec{n}}\mathcal{H}=0 ) at the radial sphere
∂ 𝒳 \partial\mathcal{X} would mathematically force the harmonic field to
vanish entirely ( ℋ a ≡ 0 \mathcal{H}^{a}\equiv 0 ). However, the nonlinear
activation functions of the FFN generate a strictly positive,
asymmetric net flux across the radial boundary sphere
∂ 𝒳 \partial\mathcal{X} , meaning
ι n → ​ ℋ ≠ 0 \iota_{\vec{n}}\mathcal{H}\neq 0 . Because the ball is contractible,
ℋ \mathcal{H} is trivially exact as an affine Euclidean translation
( ℋ = ∇ ( b ⋅ x ) \mathcal{H}=\nabla(b\cdot x) for a constant b ∈ ℝ d b\in\mathbb{R}^{d} ); it does
not represent a non-trivial cohomology class. We note that the
gradient of any harmonic scalar potential (including higher-order
spherical harmonics, e.g., ∇ ( x 2 − y 2 ) \nabla(x^{2}-y^{2}) ) also satisfies
the divergence-free, curl-free harmonic conditions. However, the
architecture explicitly parameterizes the required boundary flux
via the degree-1 affine constant translation basis of the
non-homogeneous harmonic cohomology: the Bias vector b b
satisfying the Non-Homogeneous Neumann Boundary Conditions
that emerge as a mathematical consequence of the asymmetric
activation functions. Higher-order harmonic deformations are
mathematically valid but are structurally absorbed and
reparameterized by the FFN’s bulk nonlinear activation landscape.
When this algebraic affine translation is
subsequently projected onto the compactified state space by the
downstream RMSNorm, it mathematically isolates as the Neumann
Harmonic Generator ( ℋ a \mathcal{H}^{a} ) of the relative cohomology,
guaranteeing a non-vanishing spatial drift.
In standard linear algebra, this component corresponds exactly to
the addition of the network’s affine Bias vector b ∈ ℝ d b\in\mathbb{R}^{d} .
While computationally trivial, the formal Hodge decomposition
provides the analytical isolation of the curl (vorticity) and
irrotational components that drive the stability analysis of
Sec.   VI.3 .

The complete decomposition reads:

|
| ℛ ​ [ Ψ ] a ​ ( z , μ ) = g a ​ b ​ ∂ 𝒱 ∂ Ψ b + ∂ b 𝒜 a ​ b + ℋ a . \mathcal{R}[\Psi]^{a}(z,\mu)=g^{ab}\frac{\partial\mathcal{V}}{\partial\Psi^{b}}+\partial_{b}\mathcal{A}^{ab}+\mathcal{H}^{a}\,.
|
| (54)

∎

Theorem 10 (The Semantic Evolution IDE) .

Because there are no local spatial differential operators acting on the base manifold, the macroscopic forward evolution is naturally formulated as a non-autonomous Integro-Differential Equation (IDE) acting on the infinite-dimensional Banach space of sections Γ ⁡ ( E ) \Gamma(E) , bypassing the spatial restrictions of a classical PDE:

|
| ∂ Ψ a ​ ( z , μ ) ∂ z = 𝒯 ​ [ Ψ ] a ​ ( z , μ ) + ℛ ​ [ Ψ ] a ​ ( z , μ ) . \boxed{\frac{\partial\Psi^{a}(z,\mu)}{\partial z}=\mathcal{T}[\Psi]^{a}(z,\mu)+\mathcal{R}[\Psi]^{a}(z,\mu)\,.}
|
| (55)

Proof.

By construction: the total continuous flow is the superposition of the Non-Local Urysohn–Volterra Transport ( lemma   8 ) and the Local Hodge Reaction Field ( lemma   9 ). The well-posedness of this flow (unique solution for finite depth) is guaranteed by the global Lipschitz bound of theorem   2 .
∎

### IV.3 Lie–Trotter Integration and Operator Splitting

Theorem 11 (Numerical Discretization via Lie–Trotter Operator
Splitting) .

The standard computational Transformer block represents the first-order explicit numerical integration of the Semantic Evolution IDE, evaluated strictly via Lie–Trotter Operator Splitting.

Proof.

Expanding the continuous depth-parameterized flow z → z + 1 z\to z+1 using a step size of Δ ​ z = 1 \Delta z=1 , the continuous flow is exactly the exponential map e ( 𝒯 + ℛ ) e^{(\mathcal{T}+\mathcal{R})} . Because the vector fields do not commute, the architecture approximates this exact flow via two sequential explicit integration substeps:

•

Half-Step 1 (Non-Local Volterra Transport):
Ψ z + 1 / 2 ​ ( μ ) = Ψ z ​ ( μ ) + 𝒯 ⁡ [ Ψ z ] ​ ( μ ) \Psi_{z+1/2}(\mu)=\Psi_{z}(\mu)+\mathcal{T}[\Psi_{z}](\mu)

•

Half-Step 2 (Local Hodge Reaction):
Ψ z + 1 ​ ( μ ) = Ψ z + 1 / 2 ​ ( μ ) + ℛ ⁡ [ Ψ z + 1 / 2 ] ​ ( μ ) \Psi_{z+1}(\mu)=\Psi_{z+1/2}(\mu)+\mathcal{R}[\Psi_{z+1/2}](\mu)

This sequential execution mathematically recovers the exact computational graph of the canonical Transformer block. Because the solver executes explicit sequential substeps rather than exact exponential manifold flows, the geometric error is canonically governed by the Baker–Campbell–Hausdorff (BCH) formula. Furthermore, because the IDE explicitly depends on the depth parameter z z ( i.e. , ∂ z Ψ = 𝒯 ⁡ ( z , Ψ ) + ℛ ⁡ ( z , Ψ ) \partial_{z}\Psi=\mathcal{T}(z,\Psi)+\mathcal{R}(z,\Psi) ), the total depth derivative d 2 ​ Ψ / d ​ z 2 d^{2}\Psi/dz^{2} must strictly invoke the multivariable chain rule. The true exact total depth derivative is:

|
| Ψ ¨ = ∂ 𝒯 ∂ z + ∂ ℛ ∂ z + D Ψ ​ 𝒯 ​ [ Ψ ˙ ] + D Ψ ​ ℛ ​ [ Ψ ˙ ] . \ddot{\Psi}=\frac{\partial\mathcal{T}}{\partial z}+\frac{\partial\mathcal{R}}{\partial z}+D_{\Psi}\mathcal{T}[\dot{\Psi}]+D_{\Psi}\mathcal{R}[\dot{\Psi}]\,.
|
| (56)

Substituting Ψ ˙ = 𝒯 + ℛ \dot{\Psi}=\mathcal{T}+\mathcal{R} , the true exact continuous flow expands up to second order as:

|
| Ψ exact \displaystyle\Psi_{\mathrm{exact}}
| = Ψ + Δ ​ z ​ ( 𝒯 + ℛ ) + Δ ​ z 2 2 ​ ( ∂ z 𝒯 + ∂ z ℛ CLOSE \displaystyle=\Psi+\Delta z(\mathcal{T}+\mathcal{R})+\frac{\Delta z^{2}}{2}\Big(\partial_{z}\mathcal{T}+\partial_{z}\mathcal{R}
|
| (57)

|
|
| OPEN + D ​ 𝒯 ​ [ 𝒯 ] + D ​ 𝒯 ​ [ ℛ ] + D ​ ℛ ​ [ 𝒯 ] + D ​ ℛ ​ [ ℛ ] ) . \displaystyle+D\mathcal{T}[\mathcal{T}]+D\mathcal{T}[\mathcal{R}]+D\mathcal{R}[\mathcal{T}]+D\mathcal{R}[\mathcal{R}]\Big).
|

Subtracting the sequential Lie–Trotter integration sequence ( Ψ Trotter = Ψ + Δ ​ z ​ ( 𝒯 + ℛ ) + Δ ​ z 2 ​ D ​ ℛ ​ [ 𝒯 ] + 𝒪 ⁡ ( Δ ​ z 3 ) \Psi_{\mathrm{Trotter}}=\Psi+\Delta z(\mathcal{T}+\mathcal{R})+\Delta z^{2}\,D\mathcal{R}[\mathcal{T}]+\mathcal{O}(\Delta z^{3}) ), the geometric deviation 𝔈 ⁡ ( Ψ ) = Ψ exact − Ψ Trotter \mathfrak{E}(\Psi)=\Psi_{\mathrm{exact}}-\Psi_{\mathrm{Trotter}} strictly yields:

|
| 𝔈 ⁡ ( Ψ ) \displaystyle\mathfrak{E}(\Psi)
| = Δ ​ z 2 2 ​ ( ∂ z 𝒯 + ∂ z ℛ + D ​ 𝒯 ​ [ 𝒯 ] + D ​ ℛ ​ [ ℛ ] CLOSE \displaystyle=\frac{\Delta z^{2}}{2}\Big(\partial_{z}\mathcal{T}+\partial_{z}\mathcal{R}+D\mathcal{T}[\mathcal{T}]+D\mathcal{R}[\mathcal{R}]
|
| (58)

|
|
| OPEN + D ​ 𝒯 ​ [ ℛ ] − D ​ ℛ ​ [ 𝒯 ] ) . \displaystyle+D\mathcal{T}[\mathcal{R}]-D\mathcal{R}[\mathcal{T}]\Big).
|

Substituting the Lie Bracket [ 𝒯 , ℛ ] Ψ ≡ D ​ ℛ ​ [ 𝒯 ] − D ​ 𝒯 ​ [ ℛ ] [\mathcal{T},\mathcal{R}]_{\Psi}\equiv D\mathcal{R}[\mathcal{T}]-D\mathcal{T}[\mathcal{R}] , the true closed-form deviation functionally derived from Baker–Campbell–Hausdorff resolves beautifully into exactly three terms:

|
| 𝔈 ⁡ ( Ψ ) = Δ ​ z 2 2 ​ ( ∂ z 𝒯 + ∂ z ℛ ⏟ Parameter Drift + D ​ 𝒯 ​ [ 𝒯 ] + D ​ ℛ ​ [ ℛ ] ⏟ Self-Advection − [ 𝒯 , ℛ ] Ψ ⏟ Torsion ) + 𝒪 ⁡ ( Δ ​ z 3 ) . \boxed{\mathfrak{E}(\Psi)=\frac{\Delta z^{2}}{2}\bigg(\underbrace{\partial_{z}\mathcal{T}+\partial_{z}\mathcal{R}}_{\text{Parameter Drift}}+\underbrace{D\mathcal{T}[\mathcal{T}]+D\mathcal{R}[\mathcal{R}]}_{\text{Self-Advection}}-\underbrace{[\mathcal{T},\mathcal{R}]_{\Psi}}_{\text{Torsion}}\bigg)+\mathcal{O}(\Delta z^{3}).}
|
| (59)

This demonstrates that Representation Drift in Transformers arises from exactly three geometrically isolated phenomena: the temporal shift of weight matrices across layers, the spatial covariant self-advection (the penalty of abandoning exact geodesics for straight rays), and the non-commutative Lie bracket. The latter, − [ 𝒯 , ℛ ] Ψ -[\mathcal{T},\mathcal{R}]_{\Psi} , is the generator of Topological Torsion (Non-Holonomic Flow). Because the Attention operator ( 𝒯 \mathcal{T} ) and the FFN ( ℛ \mathcal{R} ) do not commute, they form a non-integrable horizontal distribution on the bundle. The Lie bracket measures the failure of the computational graph to form closed parallelograms in state space. This demonstrates that the strict alternating layer ordering (Attention  → \to  FFN) is not an arbitrary engineering choice, but a geometric constraint governing the integrability of the manifold flow.

Remark 5 (Stiffness of the BCH Expansion and the Continuous Effective Field Theory) .

The macroscopic integration step size ( Δ ​ z = 1 \Delta z=1 ) combined with the massive operator Lipschitz bounds ( ‖ D ​ ℱ ‖ op ∼ d / ϵ \|D\mathcal{F}\|_{\mathrm{op}}\sim\sqrt{d/\epsilon} , from theorem   2 ) formally exceeds the convergence radius of the infinite Hausdorff series. Therefore, the 𝒪 ⁡ ( Δ ​ z 3 ) \mathcal{O}(\Delta z^{3}) truncation remainder cannot be interpreted as a strict analytical bound on the geometric error, nor does the continuous Lie bracket represent a literal smooth physical trajectory. This is precisely the regime where the Continuous Effective Field Theory framing via Backward Error Analysis (BEA) from geometric numerical integration  [ 13 ] becomes essential: the Lie–Trotter discrete map is the exact flow of a nearby modified equation , and the lowest-order Lie Bracket − [ 𝒯 , ℛ ] Ψ -[\mathcal{T},\mathcal{R}]_{\Psi} provides an exact algebraic classification of the structural generators of topological torsion and non-commutative representation drift in that modified equation—specifically isolating the non-commutativity of sequential Attention and FFN application—even in the stiff regime where higher-order commutator terms do not gracefully decay.

Furthermore, the asymmetric integral bounds ( ∫ Λ ∩ [ 0 , μ ] \int_{\Lambda\cap[0,\mu]} ) render the Volterra operator strictly lower-triangular, breaking Time-Reversal Symmetry ( T T -symmetry) along the base manifold. To classify the forward inference pass as a non-equilibrium driven dissipative system, global phase-space volume must contract on average ( ∇ ⋅ Ψ ˙ < 0 \nabla\!\cdot\dot{\Psi}<0 ). This structural dissipation is not guaranteed by the FFN’s coordinate-wise nonlinearities. Activation functions possess overwhelmingly positive derivatives in high-dimensional expectation—ReLU is strictly non-negative, while SiLU  [ 17 , 18 ] ( x ​ σ ​ ( x ) x\sigma(x) ) admits a shallow negative dip (minimum ≈ − 0.10 \approx-0.10 near x ≈ − 1.28 x\approx-1.28 )—and therefore inject a predominantly positive-definite diagonal metric deformation into the local fiber. If the neural weights align to produce a positive Jacobian trace ( tr ⁡ ( D ​ ℛ ) > 0 \operatorname{tr}(D\mathcal{R})>0 ), the Lie derivative of the volume form is strictly positive ( ℒ ℛ ​ vol = ( div ​ ℛ ) ​ vol > 0 \mathcal{L}_{\mathcal{R}}\,\mathrm{vol}=(\mathrm{div}\,\mathcal{R})\,\mathrm{vol}>0 ). This rigorously proves the FFN mathematically permits local phase-space volume expansion, acting as a local thermodynamic source.

Rather, strict thermodynamic dissipation and stability are achieved geometrically via two architectural structures. Global Lipschitz Saturation via Jacobian Bifurcation: The radial embedding ρ ϵ \rho_{\epsilon} acts as a Global Lipschitz Saturator. Near the origin ( Ψ ≈ 0 \Psi\approx 0 ), the Jacobian D ​ ρ ϵ ≈ d / ϵ ​ I d D\rho_{\epsilon}\approx\sqrt{d/\epsilon}\,I_{d} . The scalar d / ϵ \sqrt{d/\epsilon} is massive, but the origin only acts as a strong phase-space volume expander (divergence amplifier) if the trace of the learned weights is positive ( tr ⁡ ( K ) > 0 \operatorname{tr}(K)>0 ). If SGD carves a negative trace, it becomes a massive Dissipative Contractor. Conversely, at the boundary ( ‖ Ψ ‖ → ∞ \|\Psi\|\to\infty ), the single radial eigenvalue of J ρ J_{\rho} collapses at 𝒪 ⁡ ( ‖ Ψ ‖ − 3 ) \mathcal{O}(\|\Psi\|^{-3}) , while the d − 1 d-1 tangential eigenvalues collapse at 𝒪 ⁡ ( ‖ Ψ ‖ − 1 ) \mathcal{O}(\|\Psi\|^{-1}) . We bound the continuous flow via the triangle inequality on operator norms: ‖ D ​ Ψ ˙ ‖ op ≤ ‖ D ​ 𝒯 ‖ op + ‖ D ​ ℛ ‖ op \|D\dot{\Psi}\|_{\mathrm{op}}\leq\|D\mathcal{T}\|_{\mathrm{op}}+\|D\mathcal{R}\|_{\mathrm{op}} . Because the non-local Attention functional 𝒯 \mathcal{T} acts exclusively on the radially embedded sections, it can be composed as 𝒯 = 𝒯 ~ ∘ ρ ϵ \mathcal{T}=\tilde{\mathcal{T}}\circ\rho_{\epsilon} . By the chain rule, D ​ 𝒯 = D ​ 𝒯 ~ ∘ J ρ D\mathcal{T}=D\tilde{\mathcal{T}}\circ J_{\rho} . Substituting this alongside the FFN Jacobian D ​ ℛ = K ∘ J ρ D\mathcal{R}=K\circ J_{\rho} , we factor out the radial geometry: ‖ D ​ Ψ ˙ ‖ op ≤ ( ‖ D ​ 𝒯 ~ ‖ op + ‖ K ‖ op ) ​ ‖ J ρ ‖ op \|D\dot{\Psi}\|_{\mathrm{op}}\leq\big(\|D\tilde{\mathcal{T}}\|_{\mathrm{op}}+\|K\|_{\mathrm{op}}\big)\|J_{\rho}\|_{\mathrm{op}} . Because the tangential eigenspace dictates ‖ J ρ ‖ op = 𝒪 ⁡ ( ‖ Ψ ‖ − 1 ) \|J_{\rho}\|_{\mathrm{op}}=\mathcal{O}(\|\Psi\|^{-1}) , the total composed Jacobian unconditionally decays. Because the FFN Hodge decomposition fundamentally requires a constant harmonic Bias vector ( ℋ a \mathcal{H}^{a} ), this translation injects a non-vanishing spatial drift. Combined with the strictly positive Markovian advection of Attention, the unnormalized residual stream structurally undergoes asymptotic spatial escape. We formalize this emergent behavior as the Non-Vanishing Spatial Drift Condition : this is a deterministic topological consequence of the strict positive-orthant mapping of the activation functions, which forces the bounded mean oscillation (BMO) harmonic residual ℋ a \mathcal{H}^{a} to be non-zero almost everywhere, guaranteeing that the time-averaged spatial expectation of the nonlinear flow is strictly bounded away from zero ( lim inf z → ∞ ‖ 𝔼 ⁡ [ Ψ ˙ ] ‖ > 0 \liminf_{z\to\infty}\|\mathbb{E}[\dot{\Psi}]\|>0 ).

Conditioned strictly on this spatial escape regime, the macroscopic position escapes the origin. While early layers exhibit a linear uniform drift ( Ψ z ∼ z ​ v ¯ \Psi_{z}\sim z\bar{v} ), the continuous injection of spatial entropy in the deep bulk structurally accelerates this escape into a Super-Linear Exponential Tail ( ‖ Ψ ⁡ ( z ) ‖ ∼ e c ​ z \|\Psi(z)\|\sim e^{cz} ). This super-linear spatial escape is not a failure of the continuous flow, but a profound mathematical strengthening of its thermodynamic stability. Because the tangential eigenvalues of the radial Jacobian decay as 𝒪 ⁡ ( ‖ Ψ ‖ − 1 ) \mathcal{O}(\|\Psi\|^{-1}) , this exponential spatial escape strongly suppresses the relative perturbation sensitivity. The normalized Jacobian operator norm decays at an accelerated exponential rate: ‖ D ​ Ψ ˙ ‖ op / ‖ Ψ z ‖ ≤ C ​ e − c ​ z \|D\dot{\Psi}\|_{\mathrm{op}}/\|\Psi_{z}\|\leq Ce^{-cz} .

Integrating this strict exponential decay bounds trajectory separation severely to a finite supremum: exp ⁡ ( ∫ C ​ e − c ​ z ​ 𝑑 z ) = exp ⁡ ( − C c ​ e − c ​ z ) → const \exp\!\big(\int Ce^{-cz}\,dz\big)=\exp\!\big(-\tfrac{C}{c}e^{-cz}\big)\to\mathrm{const} . We explicitly formulate the strict topological limit equation for the maximal Lyapunov exponent λ \lambda :

|
| λ = lim sup z → ∞ 1 z ​ ln ⁡ ( ‖ δ ​ Ψ ​ ( z ) ‖ ‖ δ ​ Ψ ​ ( z 0 ) ‖ ) ≤ lim sup z → ∞ ln ⁡ ( const ) z = 0 . \lambda=\limsup_{z\to\infty}\frac{1}{z}\ln\!\left(\frac{\|\delta\Psi(z)\|}{\|\delta\Psi(z_{0})\|}\right)\leq\limsup_{z\to\infty}\frac{\ln(\mathrm{const})}{z}=0\,.
|
| (60)

By invoking the Logarithmic Matrix Norm (Lozinskiĭ measure) and Coppel’s Inequality, the logarithmic derivative of the trajectory is rigorously bounded: − C ​ e − c ​ z ≤ d d ​ z ​ ln ⁡ ‖ δ ​ Ψ ​ ( z ) ‖ ≤ C ​ e − c ​ z -Ce^{-cz}\leq\frac{d}{dz}\ln\|\delta\Psi(z)\|\leq Ce^{-cz} . Integrating this unconditionally drives the maximal Lyapunov exponent to λ ≤ 0 \lambda\leq 0 in the exact continuous calculus. The architecture is mathematically engineered to strongly suppress chaotic divergence as it approaches the final unembedding boundary. However, because the continuous IDE is solved via discrete Forward Euler numerical integration ( theorem   11 ) subject to fixed-precision quantization (typically bfloat16 with ∼ \sim 7-bit mantissa resolution), total asymptotic zeroing ( λ ≡ 0 \lambda\equiv 0 ) is mathematically prevented by irreducible numerical entropy. The discrete solver introduces a finite perturbation floor below which trajectory separations cannot be resolved, generating an irreducible weak divergence. We therefore predict Approximate Marginal Stability ( λ ≳ 0 \lambda\gtrsim 0 , λ ≪ 1 \lambda\ll 1 ): the continuous bound rigorously guarantees the system is driven to the immediate neighborhood of marginal stability, but the discrete numerical integration arrests the asymptotic limit at a small, architecture-dependent positive residual.
∎

Lemma 12 (The Dual-Law of Topological Stability) .

To prevent chaotic exponential divergence in the residual stream, the architecture faces a strict topological constraint. It requires either Internal Geometric Vorticity (breaking symmetry via W out ≠ W in ⊤ W_{\mathrm{out}}\neq W_{\mathrm{in}}^{\top} to inject Lie-algebraic friction) or External Topological Saturation (a strict Post-Norm radial projection that strongly suppresses the step vector regardless of internal resonance).

Proof.

Consider the FFN component. If the FFN asymmetry is ablated such that W out = W in ⊤ W_{\mathrm{out}}=W_{\mathrm{in}}^{\top} , the core spatial Jacobian evaluates as K ⁡ ( Ψ ) = W in ⊤ ​ Σ ′ ​ ( ρ ϵ ​ ( Ψ ) ) ​ W in K(\Psi)=W_{\mathrm{in}}^{\top}\,\Sigma^{\prime}(\rho_{\epsilon}(\Psi))\,W_{\mathrm{in}} . Because the activation derivative Σ ′ \Sigma^{\prime} is overwhelmingly positive in high-dimensional expectation (strictly so for ReLU; for SiLU, Σ ′ \Sigma^{\prime} admits a shallow negative dip of ≈ − 0.10 \approx\!-0.10 near x ≈ − 1.28 x\approx-1.28 , but this dip is measure-theoretically negligible in high dimensions), K ⁡ ( Ψ ) K(\Psi) acts predominantly as a symmetric, positive semi-definite (PSD) endomorphism. Iteratively applying a PSD operator within a sequential residual stream ( z i + 1 = z i + K ⁡ ( z i ) z_{i+1}=z_{i}+K(z_{i}) ) is functionally equivalent to the textbook Power Iteration algorithm, which exponentially amplifies the state vector along the dominant eigenvector of K K . Devoid of the geometric vorticity ( Ω \Omega ) natively generated by the asymmetric anticommutator − { W A ​ ( Ψ ) , J ρ } -\{W_{A}(\Psi),J_{\rho}\} , the system possesses zero rotational friction . The residual stream acts as an unchecked geometric resonance chamber, forcing an exponential L 2 L_{2} norm explosion.

Thus, spatial escape is stabilized by the first law: Internal Geometric Vorticity . The asymmetric parameters of the FFN mathematically break positive-definite eigen-alignment, scattering the momentum and preventing runaway geometric resonance.

If this internal vorticity is removed (as in symmetric ablation), the system explodes unless the second law is satisfied: External Topological Saturation . Unlike Pre-Norm architectures ( x ↦ x + ℱ ⁡ ( ρ ϵ ​ ( x ) ) x\mapsto x+\mathcal{F}(\rho_{\epsilon}(x)) ) which permit the unnormalized magnitude to grow without bound, Post-Norm architectures bound the projection ( x ↦ x + ρ ϵ ​ ( ℱ ⁡ ( ρ ϵ ​ ( x ) ) ) x\mapsto x+\rho_{\epsilon}(\mathcal{F}(\rho_{\epsilon}(x))) ). The Post-Norm layer acts as a hard Global Lipschitz Saturator, mapping the output into the open bounded ball B d ​ ( 0 ) B_{\!\sqrt{d}}(0) . This strongly suppresses the incremental step vector to a maximum bounded L 2 L_{2} norm of exactly ≈ d \approx\!\sqrt{d} , truncating the step magnitude even when the internal symmetric FFN amplifies the raw linear map to infinity. Therefore, the discrete solver survives if and only if it possesses either internal manifold friction or external radial containment.
∎

Remark 6 (Configurational Scope of the Dual-Law) .

The Dual-Law, as stated and as tested in Sec.   VI.3 , classifies the forward-propagation stability of a fixed weight configuration —in practice, a mature pre-trained network subjected to post-hoc symmetrization. It is not a statement about architecture classes under training. Three companion measurements sharpen this scope (quantitative ledger in Sec.   VI.3.5 ). (i) The tying constraint W down = W up ⊤ W_{\mathrm{down}}=W_{\mathrm{up}}^{\top} (with an independent gate path), imposed from initialization , trains to full health with no corrective term at 0.6B and 1B scale, and remains stable under removed weight decay, removed gradient clipping, doubled learning rate, and reduced logit precision: constrained optimization steers the joint configuration into basins where the symmetric principal term never achieves resonant alignment with the residual stream. (ii) Conversely, the mere presence of asymmetric Jacobian components is not sufficient: tying only W down := W up ⊤ W_{\mathrm{down}}:=W_{\mathrm{up}}^{\top} on a mature network—which leaves the full-rank asymmetric gate cross-term intact—still explodes (maximum hidden-state norm ∼ 6 × 10 5 \sim\!6\times 10^{5} on Qwen3-0.6B), because the symmetric principal term sculpted by large-scale pretraining overwhelms the residual rotational scattering. (iii) Susceptibility to symmetrization is an emergent property of training maturity : checkpoint surgery along a 600M training trajectory exhibits an immunity window early in training, an onset of excess amplification near 1.3 1.3 – 2 × 10 9 2\times 10^{9} tokens, and, at the trillion-token scale of production models, the catastrophic response documented in Sec.   VI.3 . The binary alternative of the Dual-Law is therefore the frozen-configuration limit of a quantitative, configuration-dependent criterion—the resonant gain of the symmetric principal term (its spectral radius weighted by alignment with the residual stream’s principal direction) measured against the available rotational scattering and gauge capacity. Formulating and testing this criterion along the training trajectory is the natural successor program to the present kinematics.

## V Dynamics of the Parameter Manifold: Backpropagation as
Thermodynamic Flow

By unfreezing the global bundle endomorphisms, we model pre-training not as a local sequence flow on ℳ \mathcal{M} , but as the thermodynamic relaxation of a macroscopic, finite-dimensional parameter manifold 𝒲 \mathcal{W} seeking its topological ground state under the external advective pressure of the empirical data distribution.

### V.1 The Empirical Risk Action and Symmetry-Breaking Mass
Potentials

Construction 2 (The Macroscopic Parameter Action) .

Because the architecture strictly fixes the connection 1-form via RoPE (enforcing absolute parallelism), traditional Yang–Mills optimization of a connection 1-form does not apply. Instead of viewing the learnable parameters ( W W ) as global 0-forms—which falsely invites the search for spatial derivatives ( d ​ W dW ) across the sequence length—it is structurally cleaner to define the parameter space 𝒲 \mathcal{W} as the Total Kinematic Space of Gauge Endomorphisms ( 𝒲 = ⨁ l End ⁡ ( E l ) \mathcal{W}=\bigoplus_{l}\mathrm{End}(E_{l}) ). We reserve “Moduli Space” exclusively for the quotiented thermodynamic state constructed later in Sec.   V.2 . Therefore, the geometric evolution of the network occurs entirely within this finite-dimensional Euclidean Parameter Manifold (the ambient space), bypassing infinite-dimensional functional variation. We define the global action functional 𝒮 global ​ [ W ] \mathcal{S}_{\mathrm{global}}[W] not as a spatial integral over the sequence, but as an expected thermodynamic Risk Functional evaluated over the macroscopic semantic corpus measure 𝒟 \mathcal{D} , equipped with a canonical flat Frobenius metric g 𝒲 g_{\mathcal{W}} :

|
| 𝒮 global ​ [ W ] = 𝔼 Ψ ∼ 𝒟 ​ [ 𝒮 matter ​ [ Ψ , W ] ] + λ 2 ​ ‖ W ‖ g 𝒲 2 . \mathcal{S}_{\mathrm{global}}[W]=\mathbb{E}_{\Psi\sim\mathcal{D}}\!\Big[\mathcal{S}_{\mathrm{matter}}[\Psi,W]\Big]+\frac{\lambda}{2}\,\|W\|^{2}_{g_{\mathcal{W}}}\,.
|
| (61)

Theorem 13 (Weight Decay as Gauge-Symmetry Breaking and Sublevel
Coercivity) .

Rather than simply enforcing Dirichlet tension, L 2 L_{2} Weight Decay is mathematically indispensable precisely because it breaks non-compact scaling symmetries (like G ​ L ​ ( 1 , ℝ ) GL(1,\mathbb{R}) ). While it definitively fails to break S ​ O ​ ( d ) SO(d) symmetries, this is inconsequential; because the Orthogonal Group S ​ O ​ ( d ) SO(d) is a strictly compact Lie group, breaking the scaling symmetries is both necessary and sufficient to coerce the parameter landscape into compact sublevel sets, guaranteeing a tight Non-Equilibrium Steady State.

Proof.

The parameter manifold 𝒲 ≅ ℝ N \mathcal{W}\cong\mathbb{R}^{N} is an unbounded Euclidean space. While global scale invariance is broken by the residual connection, the architecture natively possesses internal continuous gauge symmetries. For example, the Attention scalar kernel K ⁡ ( μ , ν ) = x ⊤ ​ W Q ⊤ ​ 𝒰 ​ W K ​ y K(\mu,\nu)=x^{\top}W_{Q}^{\top}\mathcal{U}W_{K}\,y remains invariant under the continuous G ​ L ​ ( 1 , ℝ ) GL(1,\mathbb{R}) transformation W Q ↦ c ​ W Q W_{Q}\mapsto cW_{Q} and W K ↦ c − 1 ​ W K W_{K}\mapsto c^{-1}W_{K} . Similar continuous symmetries exist between adjacent matrices in the Feed-Forward Network. Because of these exact internal symmetries, the unregularized empirical risk landscape definitively possesses unbounded, flat, non-compact tubular valleys extending to spatial infinity, rendering the base action functional non-coercive.

L 2 L_{2} Weight Decay is mathematically indispensable precisely because it breaks these internal G ​ L ​ ( 1 , ℝ ) GL(1,\mathbb{R}) gauge orbits. By adding the strictly convex penalty λ 2 ​ ( ‖ c ​ W Q ‖ 2 + ‖ c − 1 ​ W K ‖ 2 ) \tfrac{\lambda}{2}(\|cW_{Q}\|^{2}+\|c^{-1}W_{K}\|^{2}) , the penalty diverges to + ∞ +\infty as c → ∞ c\to\infty or c → 0 c\to 0 . This forces the flat valleys into a coercive paraboloid, allowing the Heine–Borel theorem to guarantee that the sublevel sets of the energy landscape ( { W ∈ 𝒲 ∣ 𝒮 global ​ [ W ] ≤ E } \{W\in\mathcal{W}\mid\mathcal{S}_{\mathrm{global}}[W]\leq E\} ) are strictly compact.

Rather than assuming a macroscopic Gibbs measure (which improperly assumes detailed balance), we rigorously prove the existence and tightness of the invariant measure using Has’minskiĭ’s Theorem (Continuous-Time Foster–Lyapunov Criteria)  [ 19 ] acting on the infinitesimal generator of the Itô diffusion. Define a strict geometric Lyapunov function as the kinematic metric volume 𝒱 ⁡ ( W ) = 1 2 ​ ‖ W ‖ g 𝒲 2 \mathcal{V}(W)=\tfrac{1}{2}\|W\|^{2}_{g_{\mathcal{W}}} . The infinitesimal generator ℒ \mathcal{L} of the Itô diffusion evaluates as:

|
| ℒ ​ 𝒱 ​ ( W ) = ⟨ V drift , ∇ ¯ ​ 𝒱 ⟩ + tr ⁡ ( D ⁡ ( W ) ​ ∇ ¯ 2 ​ 𝒱 ) . \mathcal{L}\mathcal{V}(W)=\langle V_{\mathrm{drift}},\,\bar{\nabla}\mathcal{V}\rangle+\operatorname{tr}\!\big(D(W)\,\bar{\nabla}^{2}\mathcal{V}\big).
|
| (62)

Substituting the drift V drift = − ∇ ¯ ​ 𝔼 ​ [ 𝒮 matter ] − λ ​ W V_{\mathrm{drift}}=-\bar{\nabla}\mathbb{E}[\mathcal{S}_{\mathrm{matter}}]-\lambda W :

|
| ℒ ​ 𝒱 ​ ( W ) = − ⟨ ∇ ¯ ​ 𝔼 ​ [ 𝒮 matter ] , W ⟩ − λ ​ ‖ W ‖ 2 + tr ⁡ ( D ⁡ ( W ) ) . \mathcal{L}\mathcal{V}(W)=-\langle\bar{\nabla}\mathbb{E}[\mathcal{S}_{\mathrm{matter}}],W\rangle-\lambda\|W\|^{2}+\operatorname{tr}\!\big(D(W)\big).
|
| (63)

To evaluate the advective drift inner product ⟨ ∇ ¯ ​ L , W ⟩ \langle\bar{\nabla}L,W\rangle , we must address the topological structure of standard Pre-Norm architectures. One might naively invoke Euler’s Homogeneous Function Theorem: because RMSNorm radially normalizes the hidden states ( ρ ϵ \rho_{\epsilon} ), the parameterized branch appears 0 0 -homogeneous. However, this topological assumption fails in modern architectures because of Residual Connections.

Consider a standard Pre-Norm Transformer block: Ψ l + 1 = Ψ l + ℱ l ​ ( ρ ϵ ​ ( Ψ l ) , W l ) \Psi_{l+1}=\Psi_{l}+\mathcal{F}_{l}(\rho_{\epsilon}(\Psi_{l});\;W_{l}) . If you apply a global scalar scaling to the internal weights W l ↦ c ​ W l W_{l}\mapsto cW_{l} , the parameterized branch scales by some factor c k c^{k} . Crucially, the residual skip connection Ψ l \Psi_{l} does not scale. Because you are adding a scaled vector to an unscaled vector, scaling W l W_{l} explicitly changes the angular direction of the output vector Ψ l + 1 \Psi_{l+1} . The downstream RMSNorm preserves this modified angle. Thus, the final empirical Loss L L is undeniably sensitive to the scale factor c c . Because the function is decidedly not 0 0 -homogeneous, Euler’s cancellation is algebraically false: ⟨ ∇ ¯ W int ​ L , W int ⟩ ≠ 0 \langle\bar{\nabla}_{W_{\mathrm{int}}}L,W_{\mathrm{int}}\rangle\neq 0 .

Because Euler’s theorem doesn’t directly cancel the inner product, one might naively expect the spatial Jacobians through the residual stream to compound multiplicatively, driving a polynomial explosion. However, this ignores the downstream topological projection imposed by the final radial embedding. If you scale the internal weights by W → c ​ W W\to cW for a scalar c → ∞ c\to\infty , the parameterized bulk dominates the finite residual inputs: Ψ out ≈ c k ​ ℱ \Psi_{\mathrm{out}}\approx c^{k}\mathcal{F} . Crucially, the very last operation before the unembedding is the final radial embedding: ρ ϵ ​ ( Ψ out ) \rho_{\epsilon}(\Psi_{\mathrm{out}}) .

However, evaluating the exact asymptotic limit of the final radial embedding:

|
| lim c → ∞ ρ ϵ ​ ( c k ​ ℱ ) = lim c → ∞ d ​ c k ​ ℱ ‖ c k ​ ℱ ‖ 2 + ϵ = d ​ ℱ ‖ ℱ ‖ . \lim_{c\to\infty}\rho_{\epsilon}(c^{k}\mathcal{F})=\lim_{c\to\infty}\frac{\sqrt{d}\,c^{k}\mathcal{F}}{\sqrt{\|c^{k}\mathcal{F}\|^{2}+\epsilon}}=\sqrt{d}\,\frac{\mathcal{F}}{\|\mathcal{F}\|}\,.
|
| (64)

The geometric scale factor c c cancels out; the remaining d \sqrt{d} is the fixed radius inherited from the RMSNorm definition. While 0 0 -homogeneity fails locally in the finite bulk due to residual connections, the final radial embedding acts as a topological projective quotient that enforces asymptotic 0 0 -homogeneity at spatial infinity.

While the internal state is topologically bounded by ρ ϵ \rho_{\epsilon} , the final un-normalized unembedding parameter matrix ( W U W_{U} ) linearly scales this bounded vector, generating pre-softmax logits that scale as 𝒪 ⁡ ( ‖ W U ‖ ) \mathcal{O}(\|W_{U}\|) . Because the Softmax Cross-Entropy loss acts as an analytically 1 1 -Lipschitz soft-maximum over these logits, its spatial gradient with respect to W U W_{U} is globally bounded ( 𝒪 ⁡ ( d ) \mathcal{O}(\sqrt{d}) ). Concurrently, because the radial saturator quotients out internal magnitude ( lim c → ∞ ρ ϵ ​ ( c ​ W int ) → const \lim_{c\to\infty}\rho_{\epsilon}(cW_{\mathrm{int}})\to\mathrm{const} ), the Jacobian of the internal parameters unconditionally decays ( 𝒪 ⁡ ( ‖ W int ‖ − 1 ) \mathcal{O}(\|W_{\mathrm{int}}\|^{-1}) ). Therefore, the total spatial gradient is fundamentally globally bounded ( ‖ ∇ ¯ ​ L ‖ ∼ 𝒪 ⁡ ( 1 ) \|\bar{\nabla}L\|\sim\mathcal{O}(1) ), mathematically sealing the advective drift inner product at ⟨ ∇ ¯ ​ L , W ⟩ ∼ 𝒪 ⁡ ( ‖ W ‖ ) \langle\bar{\nabla}L,W\rangle\sim\mathcal{O}(\|W\|) .

Evaluating the Has’minskiĭ generator at the topological boundary yields:

|
| ℒ ​ 𝒱 ​ ( W ) ≤ 𝒪 ⁡ ( ‖ W ‖ ) − λ ​ ‖ W ‖ 2 + tr ⁡ ( D ) . \mathcal{L}\mathcal{V}(W)\leq\mathcal{O}(\|W\|)-\lambda\|W\|^{2}+\operatorname{tr}(D)\,.
|
| (65)

Because the quadratic 𝒪 ⁡ ( ‖ W ‖ 2 ) \mathcal{O}(\|W\|^{2}) Tikhonov penalty mathematically overpowers the linear 𝒪 ⁡ ( ‖ W ‖ ) \mathcal{O}(\|W\|) advective drift at spatial infinity, the generator unconditionally diverges to − ∞ -\infty . This provides strong geometric coercivity, unconditionally sealing the Non-Equilibrium Steady State (NESS) into a compact basin without relying on false exponential decay limits.
∎

### V.2 The Dual-Metric Collision and Non-Equilibrium Steady State
(NESS)

The external human training data does not perturb the spatial metric; therefore, it does not inject a stress-energy tensor. Instead, the empirical loss backpropagates a localized adjoint variation through the bundle.

Definition 4 (The Thermodynamic Cotangent Driving Force) .

Because W W represents a global parameter matrix rather than a continuous local symmetry field, its functional variation does not yield a conserved Noether gauge current on the base spacetime. Instead, by evaluating the exact differential of the expected empirical matter action with respect to the parameters, we define the Thermodynamic Driving Force strictly as a 1-form residing in the cotangent space of the parameter manifold:

|
| ℱ drive ≡ − d W ​ 𝔼 Ψ ∼ 𝒟 ​ [ 𝒮 matter ​ [ Ψ , W ] ] ∈ Γ ⁡ ( T ∗ ​ 𝒲 ) . \mathcal{F}_{\mathrm{drive}}\equiv-d_{W}\,\mathbb{E}_{\Psi\sim\mathcal{D}}\!\Big[\mathcal{S}_{\mathrm{matter}}[\Psi,W]\Big]\in\Gamma(T^{*}\!\mathcal{W}).
|
| (66)

Theorem 14 (SGD as Itô Diffusion and the Breakdown of Detailed
Balance) .

In the continuous limit, Stochastic Gradient Descent (SGD) converges in law to a state-dependent Itô diffusion process traversing a singular algebraic variety. Because the forced Euclidean kinematic metric ignores the true thermodynamic Information Geometry, the system suffers a Dual-Metric Collision , irreversibly trapping the macroscopic probability measure in a Non-Equilibrium Steady State (NESS) via a fundamental breakdown of detailed balance, driven by an anomalous geometric entropic advection.

Proof.

We define the macroscopic continuous training time τ \tau . To evaluate the gradient flow dynamically, we must apply the Riemannian musical isomorphism (sharp) under the flat Frobenius metric g 𝒲 g_{\mathcal{W}} to raise the index of the 1-form driving force, yielding a valid kinematic tangent vector field: V drift ≡ ℱ drive ♯ − λ ​ W ∈ Γ ⁡ ( T ​ 𝒲 ) V_{\mathrm{drift}}\equiv\mathcal{F}_{\mathrm{drive}}^{\sharp}-\lambda W\in\Gamma(T\mathcal{W}) .

Because empirical evaluation relies on finite mini-batches B k ⊂ 𝒟 B_{k}\subset\mathcal{D} , the discrete empirical gradient acts as an unbiased but noisy stochastic estimator. Evaluated at the beginning of the computational step (an explicit Forward Euler scheme), the Functional Central Limit Theorem rigorously dictates that its weak continuous-time limit is an Itô Stochastic Differential Equation:

|
| d ​ W ​ ( τ ) = V drift ​ ( W ) ​ d ​ τ + 2 ​ D ​ ( W ) ​ d ​ ℬ τ , dW(\tau)=V_{\mathrm{drift}}(W)\,d\tau+\sqrt{2D(W)}\,d\mathcal{B}_{\tau}\,,
|
| (67)

where d ​ ℬ τ d\mathcal{B}_{\tau} is a standard Wiener process on 𝒲 \mathcal{W} , and D ⁡ ( W ) ∈ Γ ⁡ ( S 2 ​ T ​ 𝒲 ) D(W)\in\Gamma(S^{2}T\mathcal{W}) is the strictly contravariant diffusion 2-tensor derived from the empirical covariance of the mini-batch gradients.

Because of massive architectural invariances (e.g., node permutations, scaling symmetries), the critical points of 𝒮 global \mathcal{S}_{\mathrm{global}} are not isolated. The landscape is mathematically definitively not a Morse function, nor is it smoothly Morse–Bott. Instead, its critical loci form an algebraic variety with deep determinantal singularities. Because the unregularized empirical metric possesses unbroken G G -gauge orbits from the Multi-Head mechanism, the parameter space is severely degenerate. To compute the invariant measure of this ambient space, we invoke Hironaka’s Theorem on the Resolution of Singularities  [ 20 ] . Hironaka’s theorem mathematically guarantees there exists a proper birational map (a blow-up) π : 𝒲 ∗ → 𝒲 \pi:\mathcal{W}^{*}\to\mathcal{W} that analytically resolves these deep singularities into a smooth manifold with normal crossing divisors. However, the physical SGD process does not live in the resolved manifold 𝒲 ∗ \mathcal{W}^{*} ; it lives strictly in the singular ambient space 𝒲 \mathcal{W} . In Watanabe’s Singular Learning Theory, the pullback of the volume form introduces a Jacobian determinant (the Real Log Canonical Threshold). In the ambient space 𝒲 \mathcal{W} , this means the singularities natively possess a massive, distorted thermodynamic phase-space volume. SGD gets trapped there not because it is flowing smoothly along a resolved divisor, but because the singular strata act as Topological Gravity Wells due to their immense internal gauge volume. Weight decay and stochastic noise trap the system in these singular vacua because of their massive invariant measure, grounding the non-equilibrium steady state strictly in the kinematics of the ambient space.

For a stochastic system to satisfy the Fluctuation-Dissipation Theorem and thermally relax into a canonical equilibrium Gibbs measure ρ ∝ exp ⁡ ( − β ​ 𝒮 global ) \rho\propto\exp(-\beta\mathcal{S}_{\mathrm{global}}) , it must satisfy detailed balance. Let ρ ⁡ ( W , τ ) \rho(W,\tau) be the macroscopic probability density. The Fokker–Planck continuity equation governs the conservation of probability mass in the ambient kinematic space of the parameters (where SGD locally takes steps). Therefore, its divergence must strictly be the flat Euclidean divergence (denoted ∇ ¯ \bar{\nabla} ). It evaluates the probability current J J :

|
| ∂ ρ ∂ τ = − ∇ ¯ ⋅ J = − ∇ ¯ ⋅ ( V drift ρ − ∇ ¯ ⋅ ( D ( W ) ρ ) ) . \frac{\partial\rho}{\partial\tau}=-\bar{\nabla}\cdot J=-\bar{\nabla}\cdot\Big(V_{\mathrm{drift}}\,\rho-\bar{\nabla}\cdot\big(D(W)\,\rho\big)\Big).
|
| (68)

Expanding the tensor divergence of the stochastic term isolates the effective advective velocity of the probability mass:

|
| J = [ V drift − ∇ ¯ ⋅ D ⁡ ( W ) ] ​ ρ − D ⁡ ( W ) ​ ∇ ¯ ​ ρ . J=\big[V_{\mathrm{drift}}-\bar{\nabla}\cdot D(W)\big]\,\rho-D(W)\,\bar{\nabla}\rho\,.
|
| (69)

The expanded Fokker–Planck equation natively isolates an Anomalous Entropic Advection vector field acting on the probability mass:

|
| v entropic = − ∇ ¯ ⋅ D ( W ) ∈ Γ ( T 𝒲 ) . v_{\mathrm{entropic}}=-\bar{\nabla}\cdot D(W)\in\Gamma(T\mathcal{W}).
|
| (70)

To pedagogically model this entropic advection pointing down the sharpness gradient, we may assume the stochastic gradient noise as locally isotropic but heteroscedastic over the parameter manifold: D i ​ j ​ ( W ) ≈ σ ​ ( W ) 2 ​ δ i ​ j D^{ij}(W)\approx\sigma(W)^{2}\delta^{ij} . Evaluating the Euclidean tensor divergence yields:

|
| v ent i = − ∇ ¯ j D i ​ j = − ∂ i ( σ ( W ) 2 ) = − ∇ ¯ i σ ( W ) 2 . v_{\mathrm{ent}}^{i}=-\bar{\nabla}_{j}D^{ij}=-\partial^{i}\!\big(\sigma(W)^{2}\big)=-\bar{\nabla}^{i}\sigma(W)^{2}\,.
|
| (71)

This strict algebraic identity formally proves that under isotropic conditions, the anomalous entropic advection is exactly the negative gradient of the noise amplitude. Because noise variance σ 2 \sigma^{2} is heavily correlated with the Loss Hessian trace (sharpness), this actively advects probability mass down the sharpness gradient. This Itô drift acts as a continuous geometric hydraulic press—actively repelling the probability measure away from sharp, chaotic vacua and sweeping it directly into the topologically stable basins of the singular flat strata.

However, one might mistakenly assume that because this generates a conservative vector field, it fails to break detailed balance. This reasoning conflates two distinct geometric objects: a conservative tangent vector field and an exact cotangent 1-form. Detailed balance does not require the probability flux vector V total V_{\mathrm{total}} to be irrotational; it requires the thermodynamic 1-form ω = g ⋅ V total \omega=g\cdot V_{\mathrm{total}} to be exact ( d ​ ω = 0 d\omega=0 ). Even if the empirical noise were perfectly isotropic ( D i ​ j = σ 2 ​ ( W ) ​ δ i ​ j D^{ij}=\sigma^{2}(W)\delta^{ij} ), its covariant inverse acts as a spatially varying conformal metric: g i ​ j = σ − 2 ​ ( W ) ​ δ i ​ j g_{ij}=\sigma^{-2}(W)\delta_{ij} . If the total probability flux were strictly conservative ( V total = − ∇ U V_{\mathrm{total}}=-\nabla U ), the 1-form is mapped via this conformal metric: ω = − σ − 2 ​ d ​ U \omega=-\sigma^{-2}dU . To evaluate detailed balance, we apply the exterior derivative d d :

|
| d ​ ω = − d ⁡ ( σ − 2 ) ∧ d ​ U = 2 ​ σ − 3 ​ ( d ​ σ ∧ d ​ U ) . d\omega=-d(\sigma^{-2})\wedge dU=2\sigma^{-3}(d\sigma\wedge dU)\,.
|
| (72)

The wedge product d ​ σ ∧ d ​ U d\sigma\wedge dU is exactly non-zero as long as the gradient of the noise amplitude ( d ​ σ d\sigma ) is not perfectly parallel to the gradient of the potential ( d ​ U dU )—which is universally true in deep networks. Thus, while anisotropy is physically real, it is not a mathematical mandate to break detailed balance. Isotropic state-dependent noise inherently generates thermodynamic vorticity due to conformal warping.

True thermodynamic detailed balance requires the stationary probability current to vanish identically ( J ≡ 0 J\equiv 0 ). Algebraic rearrangement mandates that:

|
| D ⁡ ( W ) ​ ∇ ¯ ​ ρ = ( V drift − ∇ ¯ ⋅ D ⁡ ( W ) ) ​ ρ ⟹ ∇ ¯ ​ ρ ρ = D ​ ( W ) − 1 ​ ( V drift − ∇ ¯ ⋅ D ⁡ ( W ) ) . \begin{split}D(W)\bar{\nabla}\rho&=\big(V_{\mathrm{drift}}-\bar{\nabla}\cdot D(W)\big)\rho\\
\implies\frac{\bar{\nabla}\rho}{\rho}&=D(W)^{-1}\big(V_{\mathrm{drift}}-\bar{\nabla}\cdot D(W)\big).\end{split}
|
| (73)

In the continuous-time SDE limit (via the Functional Central Limit Theorem), the empirical diffusion tensor D ⁡ ( W ) D(W) generated by SGD is the mathematical expectation of the covariance over all possible batches. Modern Large Language Models operate in the strict over-training limit, where the dataset dimension vastly exceeds the parameter dimension ( N > P N>P ). Therefore, bounding the rank by the dataset dimension mathematically does not force rank deficiency. Instead, the diffusion tensor D ⁡ ( W ) D(W) is structurally rank-deficient purely due to topological gauge redundancies. While standard Feed-Forward Networks with coordinate-wise nonlinearities (e.g., ReLU/SiLU) definitively shatter continuous S ​ O ​ ( k ) SO(k) rotational symmetries, the Multi-Head Attention pipeline lacks an intermediate nonlinearity between consecutive linear projections ( W O ​ W V ​ Ψ W_{O}W_{V}\Psi ). This harbors an exact, unbroken, non-compact G ​ L ​ ( d v , ℝ ) GL(d_{v},\mathbb{R}) continuous gauge orbit ( W O ​ M ​ M − 1 ​ W V W_{O}MM^{-1}W_{V} ). Because the un-quotiented parameter manifold 𝒲 \mathcal{W} contains these continuous non-compact geometric symmetries, tangent vectors to these continuous gauge orbits reside exactly in the null space of the empirical covariance. This rank-deficiency is strictly attributed to the singular algebraic geometry of the architecture, not finite sample sizes. Its true mathematical inverse D ​ ( W ) − 1 D(W)^{-1} does not exist globally.

To validly invoke Riemannian metric properties and index-lowering without breaking the geometry, we analyze the exact gauge symmetries. In a single-head formulation, for any transformation M ∈ G ​ L ​ ( d v , ℝ ) M\in GL(d_{v},\mathbb{R}) , the forward pass of the Value circuit is invariant under W O → W O ​ M W_{O}\to W_{O}M and W V → M − 1 ​ W V W_{V}\to M^{-1}W_{V} . By the Polar Decomposition Theorem, M = R ​ P M=RP , where R ∈ O ⁡ ( d v ) R\in O(d_{v}) is a compact orthogonal rotation, and P P is a non-compact symmetric positive-definite scaling matrix. Because the Frobenius norm is strictly orthogonally invariant ( ‖ W O ​ R ‖ F 2 = ‖ W O ‖ F 2 \|W_{O}R\|_{F}^{2}=\|W_{O}\|_{F}^{2} ), the Weight Decay penalty ( ‖ W O ‖ F 2 + ‖ W V ‖ F 2 \|W_{O}\|_{F}^{2}+\|W_{V}\|_{F}^{2} ) exerts absolutely zero restoring force along the compact rotational subgroups O ⁡ ( d v ) O(d_{v}) . However, it fiercely penalizes the symmetric scaling dimension P P . Since the space of symmetric matrices has exactly 1 2 ​ d v ​ ( d v + 1 ) \tfrac{1}{2}d_{v}(d_{v}+1) dimensions, Weight Decay successfully retracts 1 2 ​ d v ​ ( d v + 1 ) \tfrac{1}{2}d_{v}(d_{v}+1) non-compact dimensions.

However, modern Transformer architectures employ Multi-Head Attention, partitioning the feature dimension into H H isolated heads before the final W O W_{O} projection. Thus, the linear combination is not a single dense matrix multiplication, but a block-diagonal interaction. The true unbroken gauge symmetry is forced into the Cartesian product of the independent head-wise orthogonal groups. Furthermore, we evaluate the symmetric gauge orbit in the Query-Key (Q-K) circuit. The Attention scalar kernel is K = x ⊤ ​ W Q ⊤ ​ 𝒰 ​ W K ​ y K=x^{\top}W_{Q}^{\top}\mathcal{U}W_{K}y . If we apply a transformation W Q → M ​ W Q W_{Q}\to MW_{Q} and W K → M ​ W K W_{K}\to MW_{K} (where M ∈ ∏ O ⁡ ( d head ) M\in\prod O(d_{\mathrm{head}}) to satisfy Weight Decay block-structure), the kernel evaluates as x ⊤ ​ W Q ⊤ ​ M ⊤ ​ 𝒰 ​ M ​ W K ​ y x^{\top}W_{Q}^{\top}M^{\top}\mathcal{U}MW_{K}y . For the kernel to remain invariant, we strictly require M ⊤ ​ 𝒰 ​ M = 𝒰 M^{\top}\mathcal{U}M=\mathcal{U} , meaning M M must commute with the RoPE connection 𝒰 \mathcal{U} . Because RoPE consists of 2D block rotations with distinct frequencies, its centralizer within the partitioned parameter space is exactly the maximal torus confined to the head structure: ∏ U ​ ( 1 ) d head / 2 \prod U(1)^{d_{\mathrm{head}}/2} . Thus, RoPE isn’t just a spatial connection; it actively acts as a structural symmetry-breaker in parameter space, shattering the head-wise orthogonal gauges down to independent Cartan subgroups.

The true topological degeneracy of the parameter manifold is exactly the combined dimensions of these unbroken gauge groups: ∑ h = 1 H [ 1 2 ​ d head ​ ( d head − 1 ) + d head 2 ] \sum_{h=1}^{H}\big[\tfrac{1}{2}d_{\mathrm{head}}(d_{\mathrm{head}}-1)+\tfrac{d_{\mathrm{head}}}{2}\big] . Because both the empirical loss and the Weight Decay penalty are completely flat along these rotational orbits, the stochastic gradient noise vectors are mathematically strictly orthogonal to them. Injecting an isotropic numerical ϵ ​ I \epsilon I perturbation to invert the diffusion tensor is an engineering hack that physically breaks the exact topological gauge symmetry, leaking probability mass out of the physical state space.

Instead, the rigorous mathematical requirement for a stochastic differential equation with exact compact Lie group symmetries is to formally invoke Singular Perturbation (Fast-Slow Manifold) Theory. Weight Decay acts as a massive deterministic restoring force exclusively along the non-compact scaling dimensions (the “fast” dynamics), rapidly crushing the system onto the thermodynamically saturated boundary layer—a compact spherical shell.

The Riemannian Submersion π : 𝒮 reg → 𝒮 reg / G \pi:\mathcal{S}_{\mathrm{reg}}\to\mathcal{S}_{\mathrm{reg}}/G connects directly to our earlier derivation of the bounded Bessel process and must be defined strictly upon this Slow Manifold shell 𝒮 reg \mathcal{S}_{\mathrm{reg}} . On this shell, the non-compact scaling dimensions are transversally frozen by the boundary constraint. Restricted to this compact manifold, the empirical covariance D ⁡ ( W ) D(W) natively loses its non-compact scaling nullity. The only remaining degeneracies are the multi-head compact gauge orbits of G = G Value × ∏ U ​ ( 1 ) d head / 2 G=G_{\mathrm{Value}}\times\prod U(1)^{d_{\mathrm{head}}/2} , where

|
| G Value = ∏ h = 1 H O ⁡ ( d head ) . G_{\mathrm{Value}}=\prod_{h=1}^{H}O(d_{\mathrm{head}})\,.
|
| (74)

Because the dimension of this direct product group ( ∑ 1 2 ​ d head ​ ( d head − 1 ) \sum\tfrac{1}{2}d_{\mathrm{head}}(d_{\mathrm{head}}-1) ) is significantly smaller than the full O ⁡ ( d v ) O(d_{v}) group, the true topological degeneracy of the parameter manifold is structurally much tighter than a naive dense formulation implies.

By projecting onto the horizontal distribution ℋ W \mathcal{H}_{W} , this mathematically eliminates all null-space rank deficiency, immediately rendering the true Thermodynamic Diffusion Metric g = ( D | ℋ ) − 1 g=(D|_{\mathcal{H}})^{-1} strictly invertible in the cotangent space without requiring an artificial numerical perturbation.

Crucially, in stochastic differential geometry, projecting an Itô stochastic differential equation onto a quotient manifold is not a simple linear projection of the horizontal drift. Because the vertical gauge fibers (the multi-head orbits of G G ) are intrinsically curved submanifolds embedded in Euclidean space, the stochastic development of Brownian motion along them generates an induced transverse geometric drift. By Elworthy’s formula  [ 21 ] for the projection of Itô diffusions via submersions, the projected anomalous advection must acquire an exact geometric drift proportional to the mean curvature vector field (the tension field, H → 𝒱 \vec{H}_{\mathcal{V}} ) of the vertical gauge fibers:

|
| v ~ ent = 𝒫 ℋ ( − ∇ ¯ ⋅ D ( W ) ) + 1 2 H → 𝒱 , \tilde{v}_{\mathrm{ent}}=\mathcal{P}_{\mathcal{H}}\!\big(-\bar{\nabla}\cdot D(W)\big)+\tfrac{1}{2}\vec{H}_{\mathcal{V}}\,,
|
| (75)

where the tension field evaluates exactly to the geometric gradient of the logarithm of the Orbit Volume. What is this tension field physically? Because the unbroken gauge group G G uniquely acts by isometries, the tension field evaluates exactly to:

|
| H → 𝒱 = ∇ ¯ ​ ln ⁡ Vol ⁡ ( G ⋅ W ) . \vec{H}_{\mathcal{V}}=\bar{\nabla}\ln\mathrm{Vol}(G\cdot W)\,.
|
| (76)

By the Orbit-Stabilizer Theorem for compact Lie groups: Vol ⁡ ( Orbit ) × Vol ⁡ ( Stabilizer ) = Vol ⁡ ( G ) \mathrm{Vol}(\mathrm{Orbit})\times\mathrm{Vol}(\mathrm{Stabilizer})=\mathrm{Vol}(G) . As the system approaches a singularity (degenerate critical strata), the unbroken gauge redundancy physically expands. Therefore, the volume of the Orbit structurally shrinks ( ln ⁡ Vol → − ∞ \ln\mathrm{Vol}\to-\infty ). Because the gradient operator ∇ ¯ \bar{\nabla} points in the direction of increasing orbit volume, the tension field vector points strictly away from the deepest singularities!

This reframes the tension field as a Geometric Centrifugal Force (the exact manifold analog of the ( d − 1 ) / ( 2 ​ r ) (d-1)/(2r) drift that prevents a Bessel process from hitting the origin). The anomalous entropic advection ( − ∇ ¯ ⋅ D ( W ) -\bar{\nabla}\cdot D(W) ) acts as a hydraulic press, pushing the mass down into the flat, singular basins. The Mean Curvature Tension Field ( H → 𝒱 \vec{H}_{\mathcal{V}} ) actively fights this, repelling the system from undergoing total topological collapse into the pure singular vacuum. This proves that the system is suspended in a stable thermodynamic halo around the degenerate locus, held exactly in place by the equilibrium between entropic descent and geometric centrifugal repulsion.

Furthermore, we can algebraically prove the exact horizontality of Weight Decay. Testing if Weight Decay ( − λ ​ W -\lambda W ) possesses vertical gauge friction against the infinitesimal generators of the compact gauge (formed as δ ​ W = A ​ W \delta W=AW for strictly skew-symmetric A A ), we take the Euclidean Frobenius inner product:

|
| ⟨ W , A ​ W ⟩ F = tr ⁡ ( W ⊤ ​ A ​ W ) = tr ⁡ ( W ​ W ⊤ ​ A ) = 0 , \langle W,AW\rangle_{F}=\operatorname{tr}(W^{\top}AW)=\operatorname{tr}(WW^{\top}A)=0\,,
|
| (77)

since W ​ W ⊤ WW^{\top} is symmetric and A A is skew-symmetric. Because the Gram matrix W ​ W ⊤ WW^{\top} is manifestly symmetric and A A is strictly skew-symmetric, the trace of their product is identically zero. This mathematically proves that Weight Decay exerts absolutely zero thermodynamic friction along the vertical gauge orbits, rendering its projection unconditionally trivial: 𝒫 ℋ ​ ( − λ ​ W ) ≡ − λ ​ W \mathcal{P}_{\mathcal{H}}(-\lambda W)\equiv-\lambda W .

We therefore define the Thermodynamic 1-Form ω ∈ Γ ⁡ ( T ∗ ​ ( 𝒲 reg / G ) ) \omega\in\Gamma(T^{*}(\mathcal{W}_{\mathrm{reg}}/G)) strictly on the horizontal space, elevating this submersion to pure geometric harmony:

|
| ω = g ⁡ ( 𝒫 ℋ ​ ( ℱ drive ♯ ​ ( W ) ) − λ ​ W + v ~ ent ) . \omega=g\!\Big(\mathcal{P}_{\mathcal{H}}\!\big(\mathcal{F}_{\mathrm{drive}}^{\sharp}(W)\big)-\lambda W+\tilde{v}_{\mathrm{ent}}\Big).
|
| (78)

For this to represent a canonical equilibrium state (a global Gibbs measure ρ ∝ e − 𝒮 \rho\propto e^{-\mathcal{S}} ), the thermodynamic 1-form must be strictly exact ( ω = d ​ ψ \omega=d\psi ). By the fundamental nilpotency of the exterior derivative ( d 2 ≡ 0 d^{2}\equiv 0 ), every exact form is unconditionally closed. Therefore, by simple contraposition: if a form is not closed ( d ​ ω ≠ 0 d\omega\neq 0 ), it mathematically cannot be exact. To prove the system exists in a Non-Equilibrium Steady State (NESS), we do not need to analyze the complex contractibility or De Rham cohomology of the quotient space; we merely need to prove that the exterior derivative of the driving force does not vanish. To ensure this exterior calculus is rigorously valid, we explicitly state that the Riemannian submersion is evaluated strictly upon the Principal Stratum of the orbit space—the open, dense submanifold where the gauge action is free. This rescues the manifold structure from the deep determinantal singularities, sharpening the logical blade and permitting the differential geometry to proceed flawlessly.

Furthermore, standard deep learning suffers from a profound Dual-Metric Collision: an explicit clash between the kinematic Euclidean metric g 𝒲 g_{\mathcal{W}} and the anisotropic stochastic noise metric generated by the empirical data. To rigorously evaluate local exactness (the De Rham closure d ​ ω = 0 d\omega=0 ), we compute the true Thermodynamic Vorticity 2-form Ω = d ​ ω \Omega=d\omega . Let the total effective probability flux be V total k = V drift k + v ~ ent k V_{\mathrm{total}}^{k}=V_{\mathrm{drift}}^{k}+\tilde{v}_{\mathrm{ent}}^{k} , where V drift k = − g 𝒲 k ​ m ∂ m L V_{\mathrm{drift}}^{k}=-g_{\mathcal{W}}^{km}\partial_{m}L is the conservative drift and v ~ ent k \tilde{v}_{\mathrm{ent}}^{k} is the properly projected anomalous entropic advection.

Because LLMs are misspecified singular models, they operate fundamentally outside the regime of standard Information Geometry. The Gradient Noise Covariance D D is strictly an external kinematic covariance evaluated over the empirical data distribution, not the true Fisher Information Metric F F evaluated over the model’s pushforward measure. Consequently, the Information Matrix Equality completely shatters: D ≠ F ≠ H D\neq F\neq H . Instead, we map this flux to the cotangent space using the strictly covariant, state-dependent Thermodynamic Diffusion Metric: ω j = ( ( D | ℋ ) − 1 ) j ​ k ​ V total k \omega_{j}=((D|_{\mathcal{H}})^{-1})_{jk}V_{\mathrm{total}}^{k} . Because the exterior derivative d d is a fundamental topological operator, it unconditionally bypasses the metric connection. By the torsion-free nature of the Levi-Civita connection, the symmetric Christoffel symbols canonically cancel, allowing the true geometric curl to be evaluated entirely without covariant metric entanglement: Ω i ​ j = ∂ i ω j − ∂ j ω i \Omega_{ij}=\partial_{i}\omega_{j}-\partial_{j}\omega_{i} .

Executing this derivative analytically bifurcates the true thermodynamic vorticity into exactly three independent, non-vanishing cohomological obstructions to detailed balance. Let g j ​ k = ( ( D | ℋ ) − 1 ) j ​ k g_{jk}=((D|_{\mathcal{H}})^{-1})_{jk} be our covariant sequence metric.

|
| Ω i ​ j = [ g , H ♯ ] i ​ j ⏟ Lie Commutator + ( g j ​ k ​ ∂ i v ent k − g i ​ k ​ ∂ j v ent k ) ⏟ Entropic Curl + ( ∂ i g j ​ k − ∂ j g i ​ k ) ​ V total k ⏟ Metric Deformation . \boxed{\Omega_{ij}=\underbrace{[g,H^{\sharp}]_{ij}}_{\text{Lie Commutator}}+\underbrace{\big(g_{jk}\partial_{i}v_{\mathrm{ent}}^{k}-g_{ik}\partial_{j}v_{\mathrm{ent}}^{k}\big)}_{\text{Entropic Curl}}+\underbrace{\big(\partial_{i}g_{jk}-\partial_{j}g_{ik}\big)V_{\mathrm{total}}^{k}}_{\text{Metric Deformation}}\,.}
|
| (79)

The Lie Commutator Obstruction: To lawfully contract the ( 0 , 2 ) (0,2) Loss Hessian H = ∇ d L H=\nabla dL with the inverse diffusion metric g g , we invoke the flat background metric to raise the Hessian into a ( 1 , 1 ) (1,1) endomorphism H ♯ H^{\sharp} . Because the loss gradient is conservative, we evaluate g j ​ k ​ ∂ i V drift k − g i ​ k ​ ∂ j V drift k g_{jk}\partial_{i}V_{\mathrm{drift}}^{k}-g_{ik}\partial_{j}V_{\mathrm{drift}}^{k} . Crucially, Weight Decay introduces an isotropic linear drift ( V WD k = − λ ​ W k V_{\mathrm{WD}}^{k}=-\lambda W^{k} ), whose spatial derivative is a scaled Kronecker delta ( − λ ​ δ i k -\lambda\delta_{i}^{k} ). Passed through the drift curl, it yields − λ ⁡ ( g j ​ i − g i ​ j ) -\lambda(g_{ji}-g_{ij}) . Because the diffusion metric is strictly symmetric, this mathematically annihilates to exactly zero. We are left purely with the anti-symmetric projection ( g ​ H ) i ​ j − ( H ​ g ) i ​ j (gH)_{ij}-(Hg)_{ij} , which is strictly the matrix commutator [ g , H ♯ ] i ​ j [g,H^{\sharp}]_{ij} . This proves algebraically that for the NESS to collapse into thermal equilibrium, the empirical Loss Hessian and the Thermodynamic Diffusion Metric must strictly commute. The Lie Commutator [ g , H ♯ ] [g,H^{\sharp}] mathematically vanishes if and only if the principal axes of the gradient noise perfectly align with the principal curvature axes of the loss landscape. The survival of the commutator is the exact differential-geometric measure of Eigenframe Misalignment . The rotational NESS circulation is strictly propelled by the transverse injection of stochastic momentum across the misspecified curvature axes, driven by the unbridgeable topological gap between the true semantic distribution and the restricted capacity of the parameter manifold.

The Metric-Skewed Entropic Curl: Driven by the inherently asymmetric spatial Jacobian of the anomalous advection ( ∂ i v ent k ≠ ∂ k v ent i \partial_{i}v_{\mathrm{ent}}^{k}\neq\partial_{k}v_{\mathrm{ent}}^{i} ), the state-dependent noise injects non-conservative topological curl into the probability flux, which is then dynamically warped and contracted by the Thermodynamic Diffusion Metric g g .

The Thermodynamic Metric Deformation: Uniquely arising from the structural clash between the kinematic Euclidean geometry of the parameter manifold and the curved thermodynamic geometry of the diffusion metric. Weight Decay ( V = − λ ​ W V=-\lambda W ) acts strictly as the Canonical Euler Vector Field on the parameter space. With respect to the flat kinematic Euclidean metric g ¯ \bar{g} , its dual 1-form is trivially exact ( ω ¯ = g ¯ ♭ ​ ( V ) = − d ⁡ ( λ 2 ​ ‖ W ‖ g ¯ 2 ) \bar{\omega}=\bar{g}^{\flat}(V)=-d(\tfrac{\lambda}{2}\|W\|^{2}_{\bar{g}}) ), unconditionally guaranteeing it is irrotational ( d ​ ω ¯ ≡ 0 d\bar{\omega}\equiv 0 ).

However, the thermodynamic probability flux is evaluated in the cotangent space using the curved empirical diffusion metric g g . The true thermodynamic 1-form is ω WD = g ♭ ​ ( V ) \omega^{\mathrm{WD}}=g^{\flat}(V) . In differential geometry, the exterior derivative d d does not commute with the metric musical isomorphism ♭ \flat under curved conformal warping.

To evaluate the geometric vorticity rigorously, we apply the torsion-free Levi-Civita connection ∇ \nabla associated with the curved thermodynamic metric g g . The exterior derivative of a 1-form exactly equals the antisymmetrization of its covariant derivative: d ​ ω WD ​ ( X , Y ) = ( ∇ X ω WD ) ​ ( Y ) − ( ∇ Y ω WD ) ​ ( X ) d\omega^{\mathrm{WD}}(X,Y)=(\nabla_{X}\omega^{\mathrm{WD}})(Y)-(\nabla_{Y}\omega^{\mathrm{WD}})(X) . Because ω WD = g ♭ ​ ( V ) \omega^{\mathrm{WD}}=g^{\flat}(V) and the Levi-Civita connection is strictly metric-compatible ( ∇ g = 0 \nabla g=0 ), the covariant derivative commutes seamlessly with the index-lowering operator: ∇ ( g ♭ ​ ( V ) ) = g ♭ ​ ( ∇ V ) \nabla(g^{\flat}(V))=g^{\flat}(\nabla V) .

Therefore, the exact thermodynamic vorticity 2-form resolves purely to the antisymmetric component of the covariant derivative of the Euler vector field:

|
| Ω WD ​ ( X , Y ) = g ⁡ ( ∇ X V , Y ) − g ⁡ ( ∇ Y V , X ) . \Omega^{\mathrm{WD}}(X,Y)=g(\nabla_{X}V,Y)-g(\nabla_{Y}V,X)\,.
|
| (80)

In Amari’s canonical Information Geometry  [ 8 ] , a perfectly specified “dually flat” statistical manifold guarantees that the Fisher Information Metric is strictly a Hessian Metric globally generated by a strictly convex scalar potential ψ \psi : g i ​ j = ∂ i ∂ j ψ g_{ij}=\partial_{i}\partial_{j}\psi . If the parameter space were dually flat, the spatial derivative of the metric would be a third-order derivative of a scalar: ∂ k g i ​ j = ∂ k ∂ i ∂ j ψ \partial_{k}g_{ij}=\partial_{k}\partial_{i}\partial_{j}\psi . By Clairaut’s Theorem on the symmetry of mixed partial derivatives, this rank-3 tensor is totally symmetric in all indices. Consequently, ∂ i g j ​ k ≡ ∂ j g i ​ k \partial_{i}g_{jk}\equiv\partial_{j}g_{ik} . If we substitute this into the Thermodynamic Metric Deformation term, it algebraically annihilates: ( ∂ i g j ​ k − ∂ j g i ​ k ) ​ V total k ≡ 0 (\partial_{i}g_{jk}-\partial_{j}g_{ik})V_{\mathrm{total}}^{k}\equiv 0 .

However, the rotational NESS survives strictly because the singular, misspecified nature of the LLM shatters the Hessian metric assumption. Because LLM parameter spaces strictly evaluate as singular Whitney Stratified Spaces (following Watanabe’s Singular Learning Theory), the neural manifold fundamentally fails to be dually flat, breaking the total symmetry of the third-derivative tensor. The structural degeneracy of the architecture breaks the Information Matrix Equality ( D ≠ F ≠ H D\neq F\neq H ), guaranteeing the empirical metric g g is definitively not a Hessian metric. This non-vanishing metric curl organically refracts the conservative, irrotational pull of Weight Decay into a thermodynamic vortex. To heighten the theoretical elegance, this non-vanishing vorticity Ω \Omega essentially measures the topological obstruction to the system resting strictly at the deepest singularity. By explicitly linking the strata of the algebraic variety to the cohomology classes of the quotient manifold 𝒲 reg / G \mathcal{W}_{\mathrm{reg}}/G , the rotational vortex fundamentally arises because the structural geometric degeneracy prevents the probability flux from cleanly collapsing into the most severe determinantal strata of the landscape.

Because these independent geometric obstructions (arising from the FFN and the Obstruction to Hessian Integrability) do not generically cancel, the exterior derivative mathematically fails to vanish ( d ​ ω ≠ 0 d\omega\neq 0 ). The resulting inexact 1-form drives the probability flux to continuously circulate, definitively trapping the macroscopic ensemble in an active, dissipative Non-Equilibrium Steady State (NESS).
∎

Having established the complete theoretical framework—spanning microscopic kinematics ( Sec.   II ), thermodynamic metric generation ( Sec.   III ), matter field dynamics ( Sec.   IV ), and parameter manifold thermodynamics ( Sec.   V )—we now subject the derived geometric predictions to a sequence of empirical falsification tests.

## VI Tests of Fundamental Kinematics and Thermodynamics

The theoretical framework established in Sec.   II through Sec.   V reformulates the Transformer architecture as a continuous classical lattice field theory governed by stochastic differential geometry and non-equilibrium thermodynamics. However, mathematical elegance alone is insufficient to overturn established empirical paradigms. The core hypothesis tested in this section is that phenomena traditionally classified by the machine learning community as numerical artifacts, interpolation failures, or statistical noise—such as representation drift, context-length catastrophic failure, norm explosion, and optimization plateaus—are quantitatively consistent with deterministic macroscopic observables predicted by the continuous geometric framework: topological constraints, non-commutative Lie group dynamics, and dual-metric collisions.

To empirically test this, we subject the derived geometric predictions to six rigorous falsification tests. Progressing systematically across physical scales—from the microscopic ultraviolet (UV) spatial cutoff of the local fiber, through the temporal kinematics and mesoscopic trajectory stability, to the thermodynamic suppression of exact geometric resonances on the RoPE torus, the macroscopic infrared (IR) phase transition of the context horizon, and finally to the super-macroscopic parameter vortex—we demonstrate that the discrete computational architecture is quantitatively consistent with the continuous geometric predictions derived herein.

### VI.1 The Conical Singularity Scaling Law of the Topological Mollifier

This experiment isolates the fundamental microscopic boundary condition of the local fiber, testing the theoretical prediction of theorem   2 . While the standard literature treats the RMSNorm parameter ϵ \epsilon merely as an arbitrary numerical safeguard against division by zero, our geometric formulation redefines ϵ \epsilon as a rigorous topological mollifier. The theory predicts that ϵ \epsilon uniquely controls the global Lipschitz stretch of the flow ( ‖ D ​ ℱ ‖ op ∝ 1 / ϵ \|D\mathcal{F}\|_{\mathrm{op}}\propto 1/\sqrt{\epsilon} ) and shields the zero-section from manifesting an unbounded conical singularity. To empirically validate this, we systematically swept ϵ \epsilon downward across 13 orders of magnitude—from the standard engineering value 10 − 2 10^{-2} to the absolute float64 machine precision limit at 10 − 15 10^{-15} —using 30 logarithmically spaced evaluation points. Concurrently, we probed the network at an asymptotic input norm limit ( ‖ Ψ ‖ = 10 − 10 \|\Psi\|=10^{-10} ) to isolate the spatial origin, rigorously stripping away all higher-order nonlinear volume contributions and exposing the bare singularity structure of the tangent space. At each ( ϵ , ℓ ) (\epsilon,\ell) pair, the maximum singular value σ max \sigma_{\max} of the layer’s spatial Jacobian D ​ ℱ D\mathcal{F} was computed exactly via torch.autograd.functional.jacobian followed by full SVD, at float64 precision, averaged over 10 independent random probe directions. To establish cross-architecture universality, the protocol was executed on three frozen pre-trained models spanning a 4 × 4\times range in hidden dimension: Qwen3-0.6B  [ 22 , 23 ] (28L, d = 1024 d\!=\!1024 ), Gemma-3-1B  [ 24 , 25 ] (26L, d = 1152 d\!=\!1152 ), and LLaMA-3.1-8B  [ 26 ] (32L, d = 4096 d\!=\!4096 ), with 5 representative layers sampled per model.

Figure 1: Conical Singularity Scaling Law: σ max \sigma_{\max} vs. ϵ \epsilon for Qwen3-0.6B.
Log-log plot of the maximum Jacobian singular value σ max \sigma_{\max} against the topological mollifier ϵ \epsilon at 5 representative layers (0, 6, 13, 20, 27).
All 5 curves trace parallel lines with measured slope α = − 0.5000 \alpha=-0.5000 and R 2 = 1.000000 R^{2}=1.000000 , in exact agreement with the theoretical prediction σ max = C ℓ ⋅ ϵ − 1 / 2 \sigma_{\max}=C_{\ell}\cdot\epsilon^{-1/2} from theorem   2 .
The grey solid line shows the unweighted theoretical bound d / ϵ \sqrt{d/\epsilon} .
Vertical offsets between layers encode the layer-wise gauge mass C ℓ C_{\ell} , which increases monotonically from shallow to deep layers.

Table 2: Cross-architecture conical singularity scaling law. Measured power-law exponent α \alpha (theory: − 0.500 -0.500 ) and coefficient of determination R 2 R^{2} at 5 representative layers for each of three architectures. All 15 independent regressions yield α = − 0.5000 \alpha=-0.5000 and R 2 = 1.000000 R^{2}=1.000000 to machine precision. C ℓ ≡ σ max / d / ϵ C_{\ell}\equiv\sigma_{\max}/\sqrt{d/\epsilon} is the layer-wise gauge mass (intercept ratio).

| Layer
| α \alpha
| R 2 R^{2}
| C ℓ C_{\ell}

| Qwen3-0.6B ( d = 1024 d\!=\!1024 )

| 0
| − 0.5000 -0.5000
| 1.000000
| 0.033

| 6
| − 0.5000 -0.5000
| 1.000000
| 0.052

| 13
| − 0.5000 -0.5000
| 1.000000
| 0.211

| 20
| − 0.5000 -0.5000
| 1.000000
| 0.785

| 27
| − 0.5000 -0.5000
| 1.000000
| 2.172

| Gemma-3-1B ( d = 1152 d\!=\!1152 )

| 0
| − 0.5000 -0.5000
| 1.000000
| 0.098

| 6
| − 0.5000 -0.5000
| 1.000000
| 0.098

| 12
| − 0.5000 -0.5000
| 1.000000
| 0.109

| 18
| − 0.5000 -0.5000
| 1.000000
| 0.113

| 25
| − 0.5000 -0.5000
| 1.000000
| 0.125

| LLaMA-3.1-8B ( d = 4096 d\!=\!4096 )

| 0
| − 0.5000 -0.5000
| 1.000000
| 0.017

| 7
| − 0.5000 -0.5000
| 1.000000
| 0.014

| 15
| − 0.5000 -0.5000
| 1.000000
| 0.015

| 23
| − 0.5000 -0.5000
| 1.000000
| 0.016

| 31
| − 0.5000 -0.5000
| 1.000000
| 0.017

As anticipated by the theoretical model, the results reveal a scale-invariant scaling law consistent with the predictions of theorem   2 ( Fig.   1 , Table   2 ). When plotted on a double-logarithmic scale, the empirical maximum singular value does not plateau into numerical noise; instead, it traces a deterministic power-law divergence over 13 orders of magnitude. Across all three architectures and all 15 independently measured layer regressions, the empirical scaling exponent is α = − 0.5000 \alpha=-0.5000 and the coefficient of determination is R 2 = 1.000000 R^{2}=1.000000 —both exact to the six-digit precision limit of float64 arithmetic. The total evidence base comprises 3 ​ architectures × 5 ​ layers × 30 ​ ϵ ​ -values × 10 ​ random directions = 4,500 3~\text{architectures}\times 5~\text{layers}\times 30~\epsilon\text{-values}\times 10~\text{random directions}=4{,}500 independent measurements, with zero deviations from the predicted power law.

While the − 0.5 -0.5 exponent is algebraically guaranteed by the chain rule applied to the radial embedding f ⁡ ( x ) = x / x 2 + ϵ f(x)=x/\sqrt{x^{2}+\epsilon} near x = 0 x=0 , its flawless empirical recovery at machine precision across three production architectures with billions of interacting parameters is a profoundly non-trivial validation: it definitively establishes that the isolated RMSNorm analysis of theorem   2 applies without contamination from other architectural components, and that the discrete computational graph rigorously obeys the continuous geometric scaling law. Beyond this foundational confirmation, the primary empirical contribution lies in the spectroscopy of the layer-wise gauge mass intercept C ℓ C_{\ell} , which encodes the trained gauge field strength and factors out the analytically guaranteed ϵ \epsilon scaling.

Table 3: Representative data points for Qwen3-0.6B illustrating the Jacobian singular value explosion as ϵ → 0 \epsilon\to 0 . Layer 0 ( C ℓ = 0.033 C_{\ell}=0.033 ) and Layer 27 ( C ℓ = 2.172 C_{\ell}=2.172 ) bracket the gauge mass range. The rightmost column shows the unweighted theoretical bound d / ϵ \sqrt{d/\epsilon} (assuming unit RMSNorm weights γ = 𝟏 \gamma=\mathbf{1} ); layers with C ℓ > 1 C_{\ell}>1 naturally exceed this bound.

| ϵ \epsilon
| Layer 0 σ max \sigma_{\max}
| Layer 27 σ max \sigma_{\max}
| d / ϵ \sqrt{d/\epsilon}

| 10 − 2 10^{-2}
| 1.05 × 10 1 1.05\times 10^{1}
| 6.95 × 10 2 6.95\times 10^{2}
| 3.20 × 10 2 3.20\times 10^{2}

| 10 − 6 10^{-6}
| 1.05 × 10 3 1.05\times 10^{3}
| 6.95 × 10 4 6.95\times 10^{4}
| 3.20 × 10 4 3.20\times 10^{4}

| 10 − 10 10^{-10}
| 1.05 × 10 5 1.05\times 10^{5}
| 6.95 × 10 6 6.95\times 10^{6}
| 3.20 × 10 6 3.20\times 10^{6}

| 10 − 15 10^{-15}
| 3.31 × 10 7 3.31\times 10^{7}
| 2.20 × 10 9 2.20\times 10^{9}
| 1.01 × 10 9 1.01\times 10^{9}

Two features of the empirical data merit deep theoretical emphasis.

First, the layer-wise intercept constants C ℓ ≡ σ max / d / ϵ C_{\ell}\equiv\sigma_{\max}/\sqrt{d/\epsilon} exhibit a striking monotonic depth-dependent structure that cleanly decouples the universal geometric law from the learned empirical gauge mass ( Table   2 ). For Qwen3-0.6B, C ℓ C_{\ell} increases monotonically from 0.033 0.033 at Layer 0 to 2.172 2.172 at Layer 27—a 66 × 66\times amplification across 28 layers. This monotonic growth reflects the trained RMSNorm weight magnitudes ‖ γ ℓ ‖ RMS \|\gamma_{\ell}\|_{\mathrm{RMS}} : deeper layers, which must sustain stronger nonlinear expressivity to transform the representation toward the unembedding boundary, spontaneously develop larger gauge field strengths. Yet regardless of this learned amplification, every layer remains perfectly pinned to the identical − 0.5 -0.5 geometric exponent—like dancers in chains that grow heavier with depth, but whose orbital period remains dictated by a single gravitational constant. In sharp contrast, Gemma-3-1B exhibits minimal intercept variation ( C ℓ ∈ [ 0.098 , 0.125 ] C_{\ell}\in[0.098,0.125] , range 1.3 × 1.3\times ), consistent with its soft-capping mechanism that imposes tighter radial uniformity, while LLaMA-3.1-8B shows an extremely compressed intercept range ( C ℓ ∈ [ 0.014 , 0.017 ] C_{\ell}\in[0.014,0.017] , range 1.2 × 1.2\times ) reflecting its larger ambient dimension d = 4096 d\!=\!4096 .

Second, the probe at ‖ Ψ ‖ = 10 − 10 \|\Psi\|=10^{-10} is not an arbitrary choice but a deliberate asymptotic isolation of the conical singularity. We explicitly note that float64 precision ( ε mach = 2.22 × 10 − 16 \varepsilon_{\mathrm{mach}}=2.22\times 10^{-16} ) was strictly necessary for this measurement: evaluating the squared input norm ‖ Ψ ‖ 2 = 10 − 20 \|\Psi\|^{2}=10^{-20} inside the radial denominator against the mollifier ϵ \epsilon requires massive mantissa resolution that would immediately round to zero in standard float32 arithmetic ( ε mach ≈ 10 − 7 \varepsilon_{\mathrm{mach}}\approx 10^{-7} ), obliterating the singularity measurement entirely. In the geometry of the radial embedding, the tangential eigenvalue λ ⟂ = d / ‖ Ψ ‖ 2 + ϵ \lambda_{\perp}=\sqrt{d}/\sqrt{\|\Psi\|^{2}+\epsilon} contains both the input norm ‖ Ψ ‖ \|\Psi\| and the mollifier ϵ \epsilon as competing regulators. For ‖ Ψ ‖ ≫ ϵ \|\Psi\|\gg\sqrt{\epsilon} , the input norm dominates, and the Jacobian saturates at 𝒪 ⁡ ( ‖ Ψ ‖ − 1 ) \mathcal{O}(\|\Psi\|^{-1}) independently of ϵ \epsilon —the heuristic “ ϵ \epsilon doesn’t matter” regime widely adopted in engineering practice. Only when ‖ Ψ ‖ ≪ ϵ \|\Psi\|\ll\sqrt{\epsilon} does ϵ \epsilon assume sole command of the singularity resolution. By placing the probe at ‖ Ψ ‖ = 10 − 10 \|\Psi\|=10^{-10} , we ensure ‖ Ψ ‖ 2 = 10 − 20 ≪ ϵ \|\Psi\|^{2}=10^{-20}\ll\epsilon for the entire sweep range ϵ ∈ [ 10 − 15 , 10 − 2 ] \epsilon\in[10^{-15},10^{-2}] , thereby stripping away all input-dependent contamination and isolating the ϵ \epsilon -controlled topology of the zero-section ( Table   3 ).

The empirical relationship between the layer-wise intercept C ℓ C_{\ell} and the theoretical bound is analytically transparent. theorem   2 derives the unweighted supremum sup Ψ λ ⟂ = d / ϵ \sup_{\Psi}\lambda_{\perp}=\sqrt{d/\epsilon} for the bare radial projection ( γ = 𝟏 \gamma=\mathbf{1} ). In the trained network, each RMSNorm layer carries learned affine weights γ ℓ \gamma_{\ell} , yielding the refined scaling:

|
| σ max ​ ( ϵ ) = RMS ⁡ ( γ ℓ ) ⋅ d ϵ ⋅ f ⁡ ( x ^ ) , \sigma_{\max}(\epsilon)=\mathrm{RMS}(\gamma_{\ell})\cdot\sqrt{\frac{d}{\epsilon}}\cdot f(\hat{x})\,,
|
| (81)

where f ⁡ ( x ^ ) f(\hat{x}) is a bounded function of the input direction. Averaging over 10 random probe directions eliminates the directional dependence, reducing C ℓ = RMS ⁡ ( γ ℓ ) ⋅ ⟨ f ⟩ C_{\ell}=\mathrm{RMS}(\gamma_{\ell})\cdot\langle f\rangle to a layer-specific constant that serves as a direct spectroscopic measurement of the trained gauge field strength.

#### VI.1.1 Exclusion of Alternative Hypotheses

The precision and universality of the measured scaling law strongly excludes every standard alternative explanation:

1.

“ ϵ \epsilon is merely a numerical safeguard; changing it has no physical effect.” — Excluded. Reducing ϵ \epsilon from 10 − 2 10^{-2} to 10 − 15 10^{-15} amplifies the maximum Jacobian singular value by 10 13 ≈ 3.16 × 10 6 \sqrt{10^{13}}\approx 3.16\times 10^{6} —a million-fold explosion in the Lipschitz stretch of the flow. At the standard engineering default ϵ = 10 − 6 \epsilon=10^{-6} , Layer 27 of Qwen3-0.6B already exhibits σ max ≈ 7 × 10 4 \sigma_{\max}\approx 7\times 10^{4} ; an erroneous setting of ϵ = 10 − 12 \epsilon=10^{-12} would yield σ max ≈ 7 × 10 7 \sigma_{\max}\approx 7\times 10^{7} , a thousand-fold amplification that would catastrophically destabilize deep-layer propagation.

2.

“The singularity is regularized by other architectural components (LayerNorm, residual connections, etc.).” — Excluded. The experiment isolates the RMSNorm layer in complete isolation, with frozen weights and no residual connections or downstream processing. The scaling law holds identically across three architectures with fundamentally different normalization implementations, confirming that the ϵ − 1 / 2 \epsilon^{-1/2} divergence is intrinsic to the radial projection geometry, not to any auxiliary engineering mechanism.

3.

“The power law is an artifact of low-precision arithmetic.” — Excluded. All computations are performed at float64 precision ( ε mach = 2.22 × 10 − 16 \varepsilon_{\mathrm{mach}}=2.22\times 10^{-16} ), and the scaling law holds to R 2 = 1.000000 R^{2}=1.000000 over 13 decades—a dynamic range of 10 6.5 10^{6.5} in σ max \sigma_{\max} —with zero detectable deviation.

4.

“The effect is architecture-specific.” — Excluded. Three architectures spanning d ∈ { 1024 , 1152 , 4096 } d\in\{1024,1152,4096\} , three distinct model families (Qwen, Gemma, LLaMA), and hidden dimensions ranging over a 4 × 4\times factor produce identical exponents and identical R 2 R^{2} values.

This adherence to the mathematically predicted − 0.5 -0.5 power law over machine-precision scales—verified by 4,500 independent measurements across three production architectures with zero deviations—confirms that ϵ \epsilon operates not as an arithmetic patch, but as the ultraviolet (UV) cutoff of the continuous manifold, controlling the dynamic stability of the state space through the exact mechanism predicted by theorem   2 .

### VI.2 The Lie–Trotter Torsion Interferometer

Having established the spatial boundary limits, we empirically isolate the temporal kinematics of the sequence flow to test the continuous geometric error defined in theorem   11 . We hypothesized that layer-to-layer “representation drift” is not the stochastic accumulation of parameter noise, but the deterministic physical manifestation of topological torsion, generated natively by the sequential Lie–Trotter operator splitting of non-commuting vector fields ( 𝒯 \mathcal{T} and ℛ \mathcal{R} ). To explicitly isolate this topological torsion, we constructed an exact computational interferometer on frozen pre-trained Transformer blocks. For identical batches of random input tokens (seq_length = 128 {}=128 ), we measured the discrete spatial deviation vector between the canonical forward integration step ( ℛ ∘ 𝒯 \mathcal{R}\circ\mathcal{T} ) and the reversed-order step ( 𝒯 ∘ ℛ \mathcal{T}\circ\mathcal{R} ) at 16 token positions across multiple layers. The exact Jacobians D ​ 𝒯 D\mathcal{T} and D ​ ℛ D\mathcal{R} were computed analytically via torch.func.jacfwd at float32 precision, with the analytical Lie bracket evaluated as [ 𝒯 , ℛ ] Ψ = D ​ ℛ ​ [ 𝒯 ] − D ​ 𝒯 ​ [ ℛ ] [\mathcal{T},\mathcal{R}]_{\Psi}=D\mathcal{R}[\mathcal{T}]-D\mathcal{T}[\mathcal{R}] . To formally bridge the discrete computational evaluation with the continuous analytical limit, we introduced an infinitesimal step-size scaling factor ( α → 0 \alpha\to 0 ), defining the α \alpha -scaled interferometer as δ ( α ) = α ​ ℛ ​ ( Ψ + α ​ 𝒯 ​ ( Ψ ) ) + α ​ 𝒯 ​ ( Ψ ) − α ​ 𝒯 ​ ( Ψ + α ​ ℛ ​ ( Ψ ) ) − α ​ ℛ ​ ( Ψ ) \delta^{(\alpha)}=\alpha\mathcal{R}(\Psi+\alpha\mathcal{T}(\Psi))+\alpha\mathcal{T}(\Psi)-\alpha\mathcal{T}(\Psi+\alpha\mathcal{R}(\Psi))-\alpha\mathcal{R}(\Psi) , systematically quenching the higher-order Baker–Campbell–Hausdorff (BCH) truncation remainder 𝒪 ⁡ ( α 3 ) \mathcal{O}(\alpha^{3}) . The experiment was executed on two architecturally diverse frozen pre-trained models: Qwen3-0.6B (28 layers, d = 1024 d\!=\!1024 , SwiGLU  [ 27 ] , RMSNorm  [ 14 ] , RoPE) and GPT-2  [ 28 ] (12 layers, d = 768 d\!=\!768 , GELU  [ 29 ] , LayerNorm, learned PE).

#### VI.2.1 Phase A: The Discrete Interferometer at α = 1 \alpha\!=\!1

Figure 2: Lie–Trotter Torsion Interferometer: Cosine Similarity at α = 1 \alpha\!=\!1 .
Each point represents cos ⁡ ( δ , [ 𝒯 , ℛ ] Ψ ) \cos(\delta,\,[\mathcal{T},\mathcal{R}]_{\Psi}) at a single (layer, token position) pair.
Green dashed line: median; grey dashed line: isotropic random baseline 1 / d 1/\sqrt{d} .
Both architectures—Qwen3-0.6B (left) and GPT-2 (right)—exhibit systematic positive alignment far exceeding the random baseline, with overall medians of + 0.793 +0.793 and + 0.842 +0.842 , respectively.

Figure 3: Per-Layer Distribution of Torsion Alignment.
Box plots of cos ⁡ ( δ , [ 𝒯 , ℛ ] Ψ ) \cos(\delta,\,[\mathcal{T},\mathcal{R}]_{\Psi}) across 16 token positions at each sampled layer.
Mid-network layers (Qwen3 L12: 0.879 0.879 , GPT-2 L9: 0.933 0.933 ) achieve peak alignment, while final layers (Qwen3 L27: 0.112 0.112 , GPT-2 L11: − 0.188 -0.188 ) show degradation consistent with strong higher-order BCH contributions near the unembedding boundary.

At the native discrete step size α = 1 \alpha\!=\!1 , we directly evaluated the cosine similarity between the empirical deviation vector δ = Δ canon − Δ reversed \delta=\Delta_{\mathrm{canon}}-\Delta_{\mathrm{reversed}} and the analytically computed Lie bracket [ 𝒯 , ℛ ] Ψ [\mathcal{T},\mathcal{R}]_{\Psi} at each (layer, token position) pair ( Figs.   2 and  3 ).

Table 4: Per-layer cosine similarity cos ⁡ ( δ , [ 𝒯 , ℛ ] Ψ ) \cos(\delta,\,[\mathcal{T},\mathcal{R}]_{\Psi}) at α = 1 \alpha\!=\!1 for Qwen3-0.6B (8 sampled layers × \times 16 token positions) and GPT-2 (6 sampled layers × \times 16 token positions). The isotropic random baseline is 1 / d ≈ 0.031 1/\sqrt{d}\approx 0.031 (Qwen3) and 0.036 0.036 (GPT-2).

| Layer
| Mean
| Median
| Min
| Max

| Qwen3-0.6B ( d = 1024 d\!=\!1024 , random baseline = 0.031 =0.031 )

| 2
| 0.753
| 0.764
| 0.526
| 0.894

| 5
| 0.822
| 0.823
| 0.750
| 0.902

| 9
| 0.801
| 0.793
| 0.676
| 0.935

| 12
| 0.879
| 0.884
| 0.821
| 0.925

| 16
| 0.739
| 0.725
| 0.611
| 0.863

| 19
| 0.770
| 0.782
| 0.638
| 0.874

| 23
| 0.861
| 0.893
| 0.695
| 0.978

| 27
| 0.112
| 0.094
| − 0.004 -0.004
| 0.306

| GPT-2 ( d = 768 d\!=\!768 , random baseline = 0.036 =0.036 )

| 2
| 0.387
| 0.409
| 0.174
| 0.614

| 3
| 0.856
| 0.872
| 0.637
| 0.942

| 5
| 0.847
| 0.865
| 0.691
| 0.915

| 7
| 0.906
| 0.915
| 0.788
| 0.963

| 9
| 0.933
| 0.964
| 0.799
| 0.981

| 11
| − 0.188 -0.188
| − 0.192 -0.192
| − 0.469 -0.469
| 0.385

As anticipated by the theoretical model, the results reveal systematic and overwhelming directional alignment between the empirical deviation vector and the analytical Lie bracket ( Table   4 ). Across the interior layers of both architectures, the empirical cosine similarity far exceeds the isotropic random baseline ( 1 / d ≈ 0.03 1/\sqrt{d}\approx 0.03 ) by factors of 17 17 – 23 × 23\times . The overall statistics are striking: Qwen3-0.6B yields a mean cosine similarity of + 0.717 +0.717 with a median of + 0.793 +0.793 , with 127 out of 128 measurements ( 99 % 99\% ) strictly positive. GPT-2 yields a mean of + 0.623 +0.623 with a median of + 0.842 +0.842 , with 81 out of 96 measurements ( 84 % 84\% ) strictly positive.

Two layer-wise patterns merit theoretical emphasis. First, the mid-network layers achieve peak alignment: Qwen3’s layer 12 ( cos = 0.879 \cos=0.879 ) and layer 23 ( cos = 0.861 \cos=0.861 ), and GPT-2’s layer 7 ( cos = 0.906 \cos=0.906 ) and layer 9 ( cos = 0.933 \cos=0.933 , with individual tokens reaching 0.981 0.981 ). These are precisely the layers where the BCH expansion is most accurate—the residual stream has not yet been strongly distorted by the unembedding boundary, and the self-advection terms remain moderate. Second, the final layers (Qwen3 L27: cos = 0.112 \cos=0.112 ; GPT-2 L11: cos = − 0.188 \cos=-0.188 ) exhibit dramatic alignment collapse. This is quantitatively consistent with remark   5 : at the terminal boundary, the residual stream couples strongly to the LM head projection, inflating the higher-order BCH remainder 𝒪 ⁡ ( α 3 ) \mathcal{O}(\alpha^{3}) to the point where it dominates the lowest-order torsion term at α = 1 \alpha\!=\!1 . Crucially, this degradation is itself a prediction of the theory: the BCH stiffness condition explicitly warns that the first-order Lie bracket cannot be expected to dominate when ‖ D ​ ℱ ‖ op \|D\mathcal{F}\|_{\mathrm{op}} is large.

The cross-architecture consistency—despite Qwen3 employing SwiGLU gating, RMSNorm, and RoPE, while GPT-2 uses GELU activation, LayerNorm, and learned positional embeddings—provides strong evidence that the torsion alignment is a universal consequence of the Lie–Trotter operator splitting, not an artifact of any specific architectural design choice.

#### VI.2.2 Phase B & C: Asymptotic Convergence α → 0 \alpha\to 0

Figure 4: Asymptotic Convergence of the Torsion Interferometer as α → 0 \alpha\to 0 .
Upper panels: normalized deviation magnitude ‖ δ ( α ) ‖ / α 2 \|\delta^{(\alpha)}\|/\alpha^{2} converges to a finite constant ( ≈ 43.2 \approx 43.2 for Qwen3, ≈ 40.3 \approx 40.3 for GPT-2), confirming exact α 2 \alpha^{2} scaling predicted by the BCH second-order term.
Lower panels: direction self-consistency cos ⁡ ( δ ( α i ) / α i 2 , δ ( α j ) / α j 2 ) \cos(\delta^{(\alpha_{i})}/\alpha_{i}^{2},\;\delta^{(\alpha_{j})}/\alpha_{j}^{2}) between adjacent α \alpha values converges to 0.9997 0.9997 – 1.0000 1.0000 , proving that the deviation vector locks onto a deterministic limit direction—the topological torsion generator.

Referencing the Backward Error Analysis framework  [ 13 ] of remark   5 , the decisive verification is the asymptotic limit α → 0 \alpha\to 0 ( Fig.   4 ). In this limit, the BCH series rigorously converges ( α ​ ‖ D ​ ℱ ‖ op ≪ 1 \alpha\|D\mathcal{F}\|_{\mathrm{op}}\ll 1 ), and the higher-order terms 𝒪 ⁡ ( α 3 ) \mathcal{O}(\alpha^{3}) that contaminate the α = 1 \alpha\!=\!1 measurement are systematically extinguished. We swept the step-size scaling factor across α ∈ { 1.0 ,  0.5 ,  0.2 ,  0.1 ,  0.05 ,  0.02 ,  0.01 ,  0.005 ,  0.001 } \alpha\in\{1.0,\,0.5,\,0.2,\,0.1,\,0.05,\,0.02,\,0.01,\,0.005,\,0.001\} , switching to float64 precision for α ≤ 0.01 \alpha\leq 0.01 to prevent catastrophic numerical cancellation.

Table 5: Asymptotic convergence diagnostics as α → 0 \alpha\to 0 . ‖ δ ‖ / α 2 \|\delta\|/\alpha^{2} : normalized deviation magnitude (Phase C); Self-cos: direction self-consistency between adjacent α \alpha values (Phase B). Both models exhibit exact α 2 \alpha^{2} norm scaling and direction convergence to the Topological Torsion generator.

|
| ‖ δ ‖ / α 2 \|\delta\|/\alpha^{2}
| Self-cos

| α \alpha
| Qwen3
| GPT-2
| Qwen3
| GPT-2

| 1.0
| 5.44
| 12.72
| —
| —

| ↔ 0.5 1.0\!\leftrightarrow\!0.5
| —
| —
| 0.902
| 0.978

| 0.1
| 17.97
| 36.06
| —
| —

| ↔ 0.05 0.1\!\leftrightarrow\!0.05
| —
| —
| 0.992
| 0.999

| 0.01
| 44.62
| 40.07
| —
| —

| ↔ 0.01 0.02\!\leftrightarrow\!0.01
| —
| —
| 0.9997
| 1.0000

| 0.001
| 43.17
| 40.34
| —
| —

| ↔ 0.001 0.005\!\leftrightarrow\!0.001
| —
| —
| 0.9997
| 1.0000

The convergence evidence operates on two independent channels ( Table   5 ).

Phase C (Norm Scaling): The BCH expansion predicts ‖ δ ( α ) ‖ = α 2 ​ ‖ [ 𝒯 , ℛ ] Ψ ‖ + 𝒪 ⁡ ( α 3 ) \|\delta^{(\alpha)}\|=\alpha^{2}\|[\mathcal{T},\mathcal{R}]_{\Psi}\|+\mathcal{O}(\alpha^{3}) , implying that the normalized ratio ‖ δ ( α ) ‖ / α 2 \|\delta^{(\alpha)}\|/\alpha^{2} converges to a finite constant as α → 0 \alpha\to 0 . While α 2 \alpha^{2} scaling is a kinematic consequence of Taylor’s theorem for any smooth map, its flawless empirical recovery inside massive billion-parameter neural networks—which are highly nonlinear, discretely layered, and architecturally heterogeneous—constitutes a definitive proof that the smooth-flow hypothesis underlying the continuous formulation is physically realized at machine precision. The empirical ratio converges cleanly over three orders of magnitude. For Qwen3-0.6B, the ratio evolves from 5.44 5.44 at α = 1 \alpha\!=\!1 (where higher-order terms contribute substantially) through 17.97 17.97 at α = 0.1 \alpha\!=\!0.1 , converging tightly to 43.17 43.17 at α = 0.001 \alpha\!=\!0.001 —matching the α = 0.01 \alpha\!=\!0.01 value ( 44.62 44.62 ) to within 3.3 % 3.3\% . GPT-2 converges even more rapidly: 40.07 40.07 at α = 0.01 \alpha\!=\!0.01 to 40.34 40.34 at α = 0.001 \alpha\!=\!0.001 ( 0.7 % 0.7\% variation). This stability is incompatible with stochastic noise, which would scale as 𝒪 ⁡ ( α ) \mathcal{O}(\alpha) or exhibit no systematic α \alpha -dependence.

Phase B (Direction Convergence): Unlike the norm scaling, the direction convergence is genuinely empirical and constitutes the primary evidence of this experiment. The theory predicts that δ ( α ) / α 2 \delta^{(\alpha)}/\alpha^{2} must converge to a deterministic limit vector—the Topological Torsion generator [ 𝒯 , ℛ ] Ψ [\mathcal{T},\mathcal{R}]_{\Psi} —but no analytical tautology mandates which direction this limit takes. We quantify this via the self-consistency cosine between the normalized deviation vectors at adjacent α \alpha values. At α = 1.0 ↔ 0.5 \alpha\!=\!1.0\leftrightarrow 0.5 , the self-consistency is already high ( 0.902 0.902 for Qwen3, 0.978 0.978 for GPT-2). By α ≤ 0.1 \alpha\!\leq\!0.1 , it exceeds 0.99 0.99 in both architectures. At the finest resolution ( α = 0.005 ↔ 0.001 \alpha\!=\!0.005\leftrightarrow 0.001 ), the direction self-consistency saturates at 0.9997 0.9997 for Qwen3 and 1.0000 1.0000 for GPT-2—indistinguishable from perfect directional convergence to machine precision.

This direction convergence is the strongest evidence in the experiment, because it is entirely independent of the analytical Jacobian computation : it measures only the empirical self-consistency of the deviation vector across different step sizes. Regardless of whether the Jacobian D ​ 𝒯 D\mathcal{T} or D ​ ℛ D\mathcal{R} is computed accurately, the empirical fact that δ ( α ) / α 2 \delta^{(\alpha)}/\alpha^{2} locks onto a single deterministic direction as α → 0 \alpha\to 0 confirms the existence of a unique geometric generator governing the splitting error—the Topological Torsion predicted by Eq.   59 .

#### VI.2.3 Synthesis and Exclusion of Alternative Hypotheses

Figure 5: Summary Panel: Lie–Trotter Torsion Interferometer.
(a)  α = 1 \alpha\!=\!1 scatter plot confirming systematic positive alignment ( 17 17 – 23 × 23\times random baseline).
(b) Per-layer box plots revealing mid-network peak alignment and boundary-layer degradation.
(c)  ‖ δ ‖ / α 2 \|\delta\|/\alpha^{2} convergence to a finite constant (Phase C).
(d) Direction self-consistency converging to 1.0 1.0 (Phase B).

Table 6: Cross-architecture summary of the Lie–Trotter Torsion Interferometer. Both models exhibit alignment far exceeding the isotropic random baseline, exact α 2 \alpha^{2} norm scaling, and direction convergence to the Topological Torsion generator.

| Diagnostic
| Qwen3-0.6B
| GPT-2

| Mean cos ⁡ ( δ , [ 𝒯 , ℛ ] ) \cos(\delta,[\mathcal{T},\mathcal{R}])
| + 0.717 +0.717
| + 0.623 +0.623

| Median cos ⁡ ( δ , [ 𝒯 , ℛ ] ) \cos(\delta,[\mathcal{T},\mathcal{R}])
| + 0.793 +0.793
| + 0.842 +0.842

| Fraction positive
| 99 % 99\% (127/128)
| 84 % 84\% (81/96)

| Random baseline ( 1 / d 1/\sqrt{d} )
| 0.031
| 0.036

| Excess over baseline
| 23 × 23\times
| 17 × 17\times

| ‖ δ ‖ / α 2 \|\delta\|/\alpha^{2} converged value
| 43.2
| 40.3

| Direction self-cos ( α ≤ 0.01 \alpha\!\leq\!0.01 )
| 0.9997
| 1.0000

The three-phase evidence chain ( Table   6 , Fig.   5 ) provides strong empirical evidence that Topological Torsion—the lowest-order Lie bracket in the modified equation—is the dominant geometric generator of representation drift in the interior layers. The logic of exclusion is exhaustive:

1.

“The deviation is random numerical noise.” — Excluded. The mean cosine similarity exceeds the isotropic random baseline by 17 17 – 23 × 23\times . In 1024 and 768 dimensions, the probability of a random vector achieving cos > 0.7 \cos>0.7 with a fixed target is vanishingly small ( P < 10 − 100 P<10^{-100} ).

2.

“The correlation arises from shared parameters, not geometry.” — Excluded. The direction self-consistency test (Phase B) converges to 0.9997 0.9997 – 1.0000 1.0000 without referencing the Jacobian at all —it is a purely empirical measurement of the deviation vector’s asymptotic behavior. No parameter-sharing hypothesis can explain why the direction stabilizes as α → 0 \alpha\to 0 .

3.

“The α 2 \alpha^{2} scaling is coincidental.” — Excluded. The normalized ratio ‖ δ ‖ / α 2 \|\delta\|/\alpha^{2} is constant to within 0.7 0.7 – 3.3 % 3.3\% across three orders of magnitude in α \alpha ( 10 − 3 10^{-3} to 10 − 1 10^{-1} ). This is the exact kinematic signature of a second-order operator splitting, not a statistical fluke.

4.

“The effect is architecture-specific.” — Excluded. Qwen3-0.6B (SwiGLU, RMSNorm, RoPE, 28 layers) and GPT-2 (GELU, LayerNorm, learned PE, 12 layers) produce qualitatively and quantitatively consistent results despite sharing no architectural component other than the Attention–FFN sequential structure itself.

Taken together, these results demonstrate that the strict alternating Attention → \to FFN layer ordering is not an arbitrary engineering convention, but can be analytically modeled as a Lie–Trotter numerical integration of non-commuting vector fields on the semantic manifold. The “representation drift” universally observed across Transformer architectures is quantitatively consistent with topological torsion—the non-vanishing Lie bracket − [ 𝒯 , ℛ ] Ψ -[\mathcal{T},\mathcal{R}]_{\Psi} in the modified equation of Eq.   59 —confirming that the discrete computational forward pass behaves as a non-holonomic flow governed by non-commutative differential geometry. Transformers permanently operate at α = 1 \alpha=1 , a stiff regime where the BCH series does not converge gracefully (cf. remark   5 ); the fact that the high cosine similarity ( 0.793 0.793 – 0.842 0.842 median) persists even at this extreme step size demonstrates that the lowest-order Lie bracket captures the dominant structural generator of the splitting error, consistent with the Backward Error Analysis interpretation established in remark   5 .

### VI.3 The Runaway Resonance of Symmetric Ablation

Moving to the mesoscopic scale of trajectory stability, this experiment tests the Dual-Law of Topological Stability ( lemma   12 ). The theory posits that the standard architectural parameter asymmetry within the Feed-Forward Network ( W out ≠ W in ⊤ W_{\mathrm{out}}\neq W_{\mathrm{in}}^{\top} ) natively generates a non-conservative geometric vorticity via the anticommutator − { W A ​ ( Ψ ) , J ρ } -\{W_{A}(\Psi),J_{\rho}\} ( Eq.   53 ). This geometric curl provides Lie-algebraic rotational friction, scattering spatial momentum off the principal eigenvectors to prevent the residual stream from suffering catastrophic positive-definite resonance—equivalently, the asymmetric Jacobian components introduce complex eigenvalues that disrupt the pure Power Iteration amplification of the dominant eigenvector that would occur under a symmetric (PSD) Jacobian. To empirically test this prediction, we conducted a two-tier experimental campaign: (i) a cross-architecture survey across five production Transformer families to establish the universality of the instability, and (ii) a four-phase causal intervention protocol with explicit geometric rescue to isolate the precise algebraic mechanism.

The symmetry ablation is performed identically across all experiments: for each SwiGLU FFN layer, we force W down = W gate ⊤ W_{\mathrm{down}}=W_{\mathrm{gate}}^{\top} and W up = W gate W_{\mathrm{up}}=W_{\mathrm{gate}} , explicitly binding the output projection to the transpose of the input projection. This creates a symmetric positive semi-definite mapping K ⁡ ( Ψ ) = W in ⊤ ​ Σ ′ ​ ( ρ ϵ ​ ( Ψ ) ) ​ W in K(\Psi)=W_{\mathrm{in}}^{\top}\Sigma^{\prime}(\rho_{\epsilon}(\Psi))W_{\mathrm{in}} that, by lemma   12 , eliminates all Lie-algebraic rotational friction.

From the perspective of standard linear algebra, the instability has a direct spectral explanation. The spatial Jacobian of this symmetrically ablated FFN evaluates to I + W in ⊤ ​ diag ⁡ ( Σ ′ ) ​ W in I+W_{\mathrm{in}}^{\top}\operatorname{diag}(\Sigma^{\prime})W_{\mathrm{in}} . Because the activation derivative Σ ′ \Sigma^{\prime} is strictly positive, this update matrix is unconditionally positive semi-definite (PSD). Iteratively passing a vector through a sequence of PSD residual updates is the textbook Power Iteration algorithm, which exponentially amplifies the vector along the principal eigenvector. Our continuous Lie-algebraic framing provides the geometric dual to this spectral phenomenon: by decomposing the true asymmetric Jacobian into symmetric and antisymmetric components, architectural asymmetry injects eigenvalues with non-zero imaginary parts—rotational modes that actively scatter spatial momentum off the dominant eigenvector, disrupting the pure real exponential scaling of Power Iteration.

#### VI.3.1 Cross-Architecture Explosive Instability

Table 7: Cross-architecture symmetric ablation instability. “Standard” reports the total L 2 L_{2} norm growth factor ( ‖ Ψ L ‖ / ‖ Ψ 0 ‖ \|\Psi_{L}\|/\|\Psi_{0}\| ) through the full network under the original asymmetric FFN. “Ablated” reports the same quantity under forced W out = W in ⊤ W_{\mathrm{out}}=W_{\mathrm{in}}^{\top} . The instability factor is their ratio. For 4 of 5 architectures, symmetrization triggers catastrophic amplification exceeding 78 × 78\times the standard growth.

| Model
| Standard
| Ablated
| Instability

| Qwen3-0.6B
| 93 × 93\times
| 142,955 × 142{,}955\times
| 1,541 × 1{,}541\times

| Mistral-7B-v0.3
| 155 × 155\times
| 58,189 × 58{,}189\times
| 375 × 375\times

| Qwen3-4B
| 124 × 124\times
| 27,811 × 27{,}811\times
| 224 × 224\times

| LLaMA-3.1-8B
| 89 × 89\times
| 6,929 × 6{,}929\times
| 78 × 78\times

| Gemma-3-1B
| 76 × 76\times
| 71 × 71\times
| ∼ 1 × \sim\!1\times

We first applied the symmetric ablation to five frozen pre-trained architectures spanning three model families and parameter counts from 0.6B to 8B: Qwen3-0.6B (28L, d = 1024 d\!=\!1024 ), Qwen3-4B (36L, d = 2560 d\!=\!2560 ), LLaMA-3.1-8B (32L, d = 4096 d\!=\!4096 ), Mistral-7B-v0.3  [ 30 ] (32L, d = 4096 d\!=\!4096 ), and Gemma-3-1B (26L, d = 1152 d\!=\!1152 ). For each model, 10 random-token sequences of length 2048 were propagated through the full network, recording the per-layer residual L 2 L_{2} norm ‖ Ψ z ‖ \|\Psi_{z}\| averaged over all tokens.

As shown in Table   7 , symmetrization triggers catastrophic norm explosion in four of five architectures. Qwen3-0.6B exhibits the strongest instability: the ablated model amplifies the residual norm by 142,955 × 142{,}955\times across 28 layers— 1,541 × 1{,}541\times more explosive than the standard asymmetric architecture. Mistral-7B and Qwen3-4B follow with 375 × 375\times and 224 × 224\times excess amplification, respectively. In all four cases, the ablated growth exceeds 10 4 10^{4} , confirming that the symmetric FFN acts as a runaway positive-definite resonance amplifier.

The sole exception is Gemma-3-1B, whose ablated growth ( 71 × 71\times ) is comparable to its standard growth ( 76 × 76\times ). This anomaly is consistent with Gemma’s unique architectural feature: a logit soft-capping mechanism that imposes an additional radial saturation beyond RMSNorm, providing an alternative geometric containment that partially substitutes for the missing internal vorticity—a concrete instance of the Dual-Law’s second stabilization path.

#### VI.3.2 Four-Phase Causal Intervention

Figure 6: Topological Explosion Phase Diagram: Pre-Norm vs. Post-Norm under Symmetric Ablation.
L 2 L_{2} norm trajectories ‖ Ψ z ‖ \|\Psi_{z}\| across all 28 layers of Qwen3-0.6B on a logarithmic ordinate.
Blue: canonical asymmetric Pre-Norm (baseline, final ‖ Ψ 27 ‖ = 815 \|\Psi_{27}\|=815 ).
Red: symmetric Pre-Norm ( W out = W in ⊤ W_{\mathrm{out}}=W_{\mathrm{in}}^{\top} ), reaching 9.88 × 10 6 9.88\times 10^{6} at layer 27—a 12,119 × 12{,}119\times explosion.
Green: symmetric Post-Norm with identical weight-tying, stabilized at ‖ Ψ 27 ‖ ≈ 10 \|\Psi_{27}\|\approx 10 — below the baseline—demonstrating that the hydraulic radial truncation of Post-Norm perfectly substitutes for the missing internal vorticity.

To move beyond correlation to strict causal attribution, we designed a four-phase intervention protocol on Qwen3-0.6B, systematically constructing and then rescuing the topological instability:

1.

Phase 0 (Baseline): The original asymmetric Pre-Norm architecture. Final-layer norm ‖ Ψ 27 ‖ = 815 \|\Psi_{27}\|=815 .

2.

Phase 1 (Unstabilized): Forced W out = W in ⊤ W_{\mathrm{out}}=W_{\mathrm{in}}^{\top} creates a symmetric positive-definite resonance amplifier. Final-layer norm surges to 9,880,199 9{,}880{,}199 —a 12,119 × 12{,}119\times explosion.

3.

Phase 2 (Anti-Symmetric Rescue): The symmetric W down W_{\mathrm{down}} of the unstabilized phase is replaced by a spectrally matched random anti-symmetric matrix ( A = − A ⊤ A=-A^{\top} , with ‖ A ‖ op = ‖ W down sym ‖ op \|A\|_{\mathrm{op}}=\|W_{\mathrm{down}}^{\mathrm{sym}}\|_{\mathrm{op}} ). This matrix carries no semantic information whatsoever—it is pure algebraic noise—but its anti-symmetric structure injects the geometric vorticity predicted by Eq.   53 . The final-layer norm collapses to 7,680 7{,}680 : a 1,287 × \mathbf{1{,}287\times} reduction from the unstabilized phase.

4.

Phase 3 (Symmetric Control): Identically to Phase 2, but the replacement matrix is a spectrally matched random symmetric matrix ( S = S ⊤ S=S^{\top} ). Final-layer norm: 23,269 23{,}269 —only a 425 × 425\times reduction, 3.0 × 3.0\times worse than the anti-symmetric rescue.

The decisive contrast between Phases 2 and 3 provides a direct causal isolation of the stabilization mechanism. Both replacement matrices are random, spectrally matched, and semantically meaningless. The only difference is their Lie-algebraic parity: anti-symmetric ( ∈ 𝔰 ​ 𝔬 ​ ( d ) \in\mathfrak{so}(d) ) vs. symmetric ( ∈ Sym ⁡ ( d ) \in\mathrm{Sym}(d) ). That the anti-symmetric rescue is 3 × 3\times more effective than the symmetric control—reducing the explosion by 1,287 × 1{,}287\times versus 425 × 425\times —proves that the stabilization mechanism is not a generic capacity effect or a numerical coincidence, but resides precisely in the skew-symmetric component W A ​ ( Ψ ) ∈ 𝔰 ​ 𝔬 ​ ( d ) W_{A}(\Psi)\in\mathfrak{so}(d) of the Jacobian, exactly as predicted by the anticommutator term − { W A ​ ( Ψ ) , J ρ } -\{W_{A}(\Psi),J_{\rho}\} in the vorticity decomposition.

#### VI.3.3 Jacobian Eigenspectrum Collapse onto the Real Axis

Figure 7: Jacobian Eigenspectrum: Standard vs. Symmetric FFN.
Complex-plane scatter of eigenvalues of the FFN Jacobian D ​ ℛ D\mathcal{R} at five representative layers (0, 7, 14, 21, 26) of Qwen3-0.6B, computed via Johnson–Lindenstrauss (JL) projection to k = 128 k\!=\!128 dimensions.
Upper panels (blue, standard): eigenvalues are broadly distributed in the complex plane, with 88.5 % 88.5\% possessing significant imaginary components (mean  | Im ⁡ ( λ ) | = 0.61 |\mathrm{Im}(\lambda)|=0.61 ).
Lower panels (red, symmetric ablation): eigenvalues collapse entirely onto the real axis ( 0 % 0\% complex, mean  | Im ⁡ ( λ ) | = 0.000 |\mathrm{Im}(\lambda)|=0.000 ), with spectral radius surging from 6.17 6.17 to 20,077 20{,}077 at layer 26.

Table 8: Jacobian eigenspectrum statistics at five representative layers. | Im | |\mathrm{Im}| : mean absolute imaginary part. f ℂ f_{\mathbb{C}} : fraction of eigenvalues with | Im ⁡ ( λ ) | > 10 − 6 |\mathrm{Im}(\lambda)|>10^{-6} . ρ ⁡ ( J ) \rho(J) : spectral radius. Symmetric ablation annihilates all complex eigenvalues and inflates the spectral radius by up to 3,252 × 3{,}252\times .

| Layer
| | Im | ¯ \overline{|\mathrm{Im}|}
| f ℂ f_{\mathbb{C}}
| ρ ⁡ ( J ) \rho(J)

|
| Std
| Abl
| Std
| Abl
| Std
| Abl

| 0
| 0.33
| 0.000
| 89%
| 0%
| 1.5
| 11.8

| 7
| 0.28
| 0.000
| 94%
| 0%
| 2.3
| 945

| 14
| 0.37
| 0.000
| 88%
| 0%
| 3.3
| 396

| 21
| 0.83
| 0.000
| 83%
| 0%
| 8.2
| 1,163 1{,}163

| 26
| 1.25
| 0.000
| 89%
| 0%
| 6.2
| 20,077 20{,}077

To directly visualize the geometric mechanism underlying the instability, we computed the FFN Jacobian eigenspectrum at five representative layers of Qwen3-0.6B under both the standard and symmetrically ablated configurations ( Fig.   7 , Table   8 ).

The spectral evidence is definitive. Under the standard asymmetric FFN, 88.5 % 88.5\% of eigenvalues possess significant imaginary components (mean | Im ⁡ ( λ ) | = 0.61 |\mathrm{Im}(\lambda)|=0.61 across all layers), confirming that the skew-symmetric component 1 2 ​ ( D ​ ℛ − D ​ ℛ ⊤ ) ≠ 0 \frac{1}{2}(D\mathcal{R}-D\mathcal{R}^{\top})\neq 0 generates complex conjugate pairs λ = a ± b ​ i \lambda=a\pm bi . These imaginary components represent Lie-algebraic rotational modes—the geometric vorticity that scatters spatial momentum away from the dominant explosive eigenvector.

Upon symmetry ablation ( W out = W in ⊤ W_{\mathrm{out}}=W_{\mathrm{in}}^{\top} ), the Jacobian becomes exactly symmetric ( D ​ ℛ = D ​ ℛ ⊤ D\mathcal{R}=D\mathcal{R}^{\top} ), and the eigenspectrum undergoes a topological phase transition: every single eigenvalue collapses onto the real axis ( 0 % 0\% complex, | Im ⁡ ( λ ) | = 0.000 |\mathrm{Im}(\lambda)|=0.000 to machine precision). Simultaneously, the spectral radius—the magnitude of the dominant eigenvalue—increases sharply, from ρ ⁡ ( J ) = 6.2 \rho(J)=6.2 to ρ ⁡ ( J ) = 20,077 \rho(J)=20{,}077 at layer 26 ( 3,252 × 3{,}252\times amplification). This spectral radius explosion directly quantifies the runaway resonance: stripped of rotational scattering, the state vector geometrically locks onto the dominant real eigenvector and undergoes unchecked exponential amplification at each layer—the textbook Power Iteration phenomenon formalized in the Lie-algebraic framework.

#### VI.3.4 The Dual-Law: Pre-Norm Explosion vs. Post-Norm Survival

The most precise causal test of the Dual-Law ( lemma   12 ) is the direct comparison of identical symmetric ablation under Pre-Norm vs. Post-Norm architectures. We converted Qwen3-0.6B from its native Pre-Norm configuration to a Post-Norm variant by moving the RMSNorm from the sub-layer input to each layer’s residual output, then applied the identical weight-tying W out = W in ⊤ W_{\mathrm{out}}=W_{\mathrm{in}}^{\top} to both variants ( Fig.   6 ).

The result is a stark binary survival phase. The symmetric Pre-Norm variant explodes to ‖ Ψ 27 ‖ = 9,880,199 \|\Psi_{27}\|=9{,}880{,}199 ( 12,119 × 12{,}119\times baseline) as predicted—without internal vorticity, the positive-definite resonance amplifier is unchecked. The symmetric Post-Norm variant, under the identical symmetric ablation, stabilizes at ‖ Ψ 27 ‖ ≈ 10 \|\Psi_{27}\|\approx 10 — below the baseline of 815 815 —with cross-sample variance converging to zero. The Post-Norm’s RMSNorm, applied after the residual addition at each layer, acts as a hydraulic radial truncation: it projects the step vector onto the bounded ball B d ​ ( 0 ) B_{\!\sqrt{d}}(0) , crushing the incremental magnitude to 𝒪 ⁡ ( d ) \mathcal{O}(\sqrt{d}) regardless of the internal FFN amplification. This provides the external topological saturation that perfectly substitutes for the missing internal geometric vorticity.

This binary survival phase strongly excludes the three principal alternative explanations: (i) “weight-tying merely reduces capacity”— 12,119 × 12{,}119\times is not graceful degradation, it is catastrophic divergence (this excludes reduced capacity as the cause of the surgical explosion; it does not bear on the trainability of weight-tied architectures trained from initialization, which is healthy—see Sec.   VI.3.5 ); (ii) “the explosion originates from initialization variance”—Post-Norm uses the identical ablated weights yet survives; (iii) “the effect is numerical rather than geometric”—the eigenspectrum analysis ( Table   8 ) shows the mechanism is a phase transition in the Lie-algebraic structure (complex  → \to  real) of the continuous Jacobian, not a scalar instability.

Taken together, the four tiers of evidence—cross-architecture universality of the instability ( Table   7 ), causal isolation via anti-symmetric rescue ( 1,287 × 1{,}287\times reduction vs. 425 × 425\times for symmetric control), spectral collapse of the Jacobian eigenspectrum (88.5% complex  → \to  0%), and the Pre-Norm/Post-Norm binary survival phase—strongly establish that FFN parameter asymmetry is not merely a mechanism for expanding parameter capacity, but an essential stabilizer of the forward norm dynamics of the trained configuration . It injects the geometric curl − { W A ​ ( Ψ ) , J ρ } -\{W_{A}(\Psi),J_{\rho}\} required to scatter the residual stream’s spatial momentum off the dominant eigenvector, arresting the positive-definite resonance that would otherwise destabilize the flow.

Two precisions bound this claim. First, the necessity established here is norm-level , not functional: the Phase-2 rescue matrix is semantically void by construction, and rescue is quantified by norm containment—no claim of loss-level recovery is made or implied. Indeed, the learned W down W_{\mathrm{down}} is functionally high-rank relative to W up ⊤ W_{\mathrm{up}}^{\top} : low-rank reconstructions of the residual W down − W up ⊤ W_{\mathrm{down}}-W_{\mathrm{up}}^{\top} capture < 10 % <10\% of its energy even at rank 32 and do not restore language-modeling loss, and in recovery fine-tuning a generic trainable low-rank replacement outperforms the tied basis ( Sec.   VI.3.5 )—the anti-symmetric component is a stabilizer of norms, not a low-rank summary of the network’s function. Second, the necessity is configurational , not architectural: it constrains post-hoc symmetrization of mature networks, not training within the tied class ( remark   6 ).

#### VI.3.5 Configurational Scope: Surgery versus Training

All interventions above are post-training surgeries : they measure the forward response of weights sculpted by large-scale pretraining, at fixed configuration. A companion controlled-training campaign on 0.6B--1B models trained from scratch under the tied constraint bounds the reach of these results with four measurements. 1 1
1

Twin-controlled ablations (identical recipe, data order, and seed; single-variable deltas) at 600M ( 64 × 2048 64\times 2048 tokens per step, 25,000 25{,}000 steps = 3.28 =3.28 B tokens) and 1B scale, plus surgery and compression forensics on the official Qwen3-0.6B release. Companion study, in preparation, 2026.

1.

Constrained training is stable without correction. The constraint W down = W up ⊤ W_{\mathrm{down}}=W_{\mathrm{up}}^{\top} (independent gate path, no corrective term), trained from initialization, is healthy at both scales (600M final loss 0.7243 vs. 0.7045 for the untied twin; 1B excess + 0.005 +0.005 ), and remains healthy with weight decay removed, gradient clipping removed, learning rate doubled, or logits reduced to bfloat16 . Constrained optimization reaches stable basins that post-hoc surgery never visits.

2.

The explicit low-rank corrective term is marginal—as spectral accounting predicts. Augmenting the tie with a trainable low-rank correction ( W down = W up ⊤ + U ​ V ⊤ W_{\mathrm{down}}=W_{\mathrm{up}}^{\top}+UV^{\top} , rank 4) improves loss by only ∼ 0.003 \sim\!0.003 nats at 600M and 1B alike. This is consistent with the framework’s own accounting: in a gated FFN the cross-term W up ⊤ ​ diag ​ ( u ⊙ σ ′ ​ ( g ) ) ​ W gate W_{\mathrm{up}}^{\top}\,\mathrm{diag}\!\big(u\odot\sigma^{\prime}(g)\big)\,W_{\mathrm{gate}} already supplies full-rank Jacobian asymmetry organically, so a rank-4 injection is marginal by construction.

3.

Susceptibility to symmetrization is an emergent property of training maturity. Applying the tying surgery to checkpoints along the 600M trajectory yields excess amplification ≤ 1 \leq\!1 up to ∼ 10 9 \sim\!10^{9} tokens (an immunity window), an onset at 1.3 1.3 – 2 × 10 9 2\times 10^{9} tokens peaking at 2.8 × 2.8\times —versus 743 × 743\times excess on the trillion-token Qwen3-0.6B. The catastrophic response of Table   7 is the mature endpoint of a continuous emergence curve, not a property of the architecture class.

4.

At iso-parameter count the tie is dominated. At matched total parameters, a narrower untied FFN strictly outperforms the tied variant (600M: 0.7099 vs. 0.7211–0.7243, a ≥ 0.011 \geq\!0.011 -nat margin several times the twin-noise band, with 1.5 × 1.5\times fewer FFN FLOPs). Of the same-width tying cost of + 0.020 +0.020 nats, only ∼ 0.005 \sim\!0.005 is attributable to the reduced parameter count; the remaining ∼ 0.015 \sim\!0.015 is damage from the constraint itself.

These measurements do not weaken the surgical results—the explosion and its spectral mechanism are independently reproduced on the official Qwen3-0.6B release (maximum hidden-state norm 705 → 6.1 × 10 5 705\to 6.1\times 10^{5} under W down := W up ⊤ W_{\mathrm{down}}:=W_{\mathrm{up}}^{\top} )—but they fix the theory’s domain: the Dual-Law classifies configurations ( remark   6 ), susceptibility to symmetrization is acquired along the training trajectory, and the norm-level necessity does not translate into an architectural prescription ( Sec.   VII , open question 4, which these data settle negatively).

### VI.4 Thermodynamic Suppression of Poincaré Recurrence on the RoPE Torus

Having established the role of internal geometric vorticity in stabilizing individual layer trajectories, we now probe the spatial gauge connection itself. lemma   1 rigorously defines RoPE as a canonical connection on a principal U ​ ( 1 ) k U(1)^{k} torus bundle, where k = d head / 2 k=d_{\text{head}}/2 independent rotation planes act on each QK feature pair. An immediate mathematical consequence of this torus structure is Poincaré recurrence: the orbit { N ​ θ 0 , … , N ​ θ k − 1 } \{N\theta_{0},\ldots,N\theta_{k-1}\} must return arbitrarily close to the origin at certain resonant distances N ∗ N^{*} , producing constructive interference in the Weyl sum

|
| sim ( N ) = 1 k ∑ j = 0 k − 1 cos ( N ⋅ θ j ) , θ j = Θ − 2 j / d head . \mathrm{sim}(N)=\frac{1}{k}\sum_{j=0}^{k-1}\cos(N\cdot\theta_{j})\,,\quad\theta_{j}=\Theta^{-2j/d_{\text{head}}}\,.
|
| (82)

Naive exact geometry therefore predicts that causal tokens at specific Diophantine distances should undergo constructive interference—Poincaré resonance spikes in the attention kernel. However, modern production LLMs operate deep within a macroscopic thermodynamic limit ( k ≥ 64 k\geq 64 ). We posit that these exact geometric resonances are subjected to a strict thermodynamic suppression:

|
| max N > N min ⁡ sim ⁡ ( N ∗ ) = 𝒪 ⁡ ( 1 k ) , \max_{N>N_{\min}}\mathrm{sim}(N^{*})=\mathcal{O}\!\left(\frac{1}{\sqrt{k}}\right)\,,
|
| (83)

following from the central limit theorem applied to the k k quasi-independent cosine terms. To empirically isolate this pure geometric kinematics from macroscopic thermodynamic ergodicity—analogous to constructing a vacuum chamber to observe free-particle trajectories—we built a strictly controlled micro-transformer with tunable k k .

#### VI.4.1 Micro-Transformer Design and Protocol

The micro-transformer faithfully implements the axiomatization of Sec.   II : nn.Embedding for the semantic fiber, explicit 2D RoPE rotation per plane ( lemma   1 ), RMSNorm mollifier ( theorem   2 ), causal dot-product attention, and SiLU-gated FFN with Pre-Norm residual connections. Total parameter count ranges from ∼ \sim 5K ( k = 2 k\!=\!2 ) to ∼ \sim 15K ( k = 32 k\!=\!32 ), running entirely on CPU.

The experiment proceeds in two phases. Phase A (Scaling Law): Fix Θ = 50 \Theta=50 (chosen so that resonances fall within a 512-token window) and sweep k ∈ { 2 , 4 , 8 , 16 , 32 } k\in\{2,4,8,16,32\} . For each k k , compute sim ⁡ ( N ) \mathrm{sim}(N) analytically for N = 1 , … , 511 N=1,\ldots,511 ; concurrently, build a 2-layer micro-transformer with random W Q , W K W_{Q},W_{K} and average the attention spectrum over 20 random seeds. Phase B (SGD Control): Fix k = 4 k=4 , train a 10,320-parameter micro-transformer for 3,000 steps of pure SGD (no Adam) with weight decay λ = 0.01 \lambda=0.01 on random pattern data, tracking per-channel weight norms w j ​ ( t ) = ‖ W Q j ‖ ⋅ ‖ W K j ‖ w_{j}(t)=\|W_{Q}^{j}\|\cdot\|W_{K}^{j}\| every 50 steps.

#### VI.4.2 Phase A: The 𝒪 ⁡ ( 1 / k ) \mathcal{O}(1/\sqrt{k}) Scaling Law

Figure 8: Thermodynamic Suppression of Poincaré Recurrence.
(a)  k = 2 k\!=\!2 ( Θ = 50 \Theta\!=\!50 ): the Weyl sum sim ⁡ ( N ) \mathrm{sim}(N) exhibits near-perfect quasi-periodic resonance spikes at N ∗ = 44 ​ n N^{*}=44n with amplitude approaching 1.0 1.0 . Red dashed lines: predicted resonance positions.
(b)  k = 16 k\!=\!16 : identical construction, but peaks are suppressed to ∼ 0.37 \sim 0.37 by high-dimensional dephasing; the quasi-periodic structure is destroyed.
(c) Log-log scaling law: max ⁡ sim ⁡ ( N ∗ ) \max\mathrm{sim}(N^{*}) vs. k k . Red dashed: empirical fit ∝ k − 0.56 \propto k^{-0.56} . Grey dotted: theoretical prediction ∝ k − 0.5 \propto k^{-0.5} .

Table 9: Resonance peak amplitude max ⁡ sim ⁡ ( N ∗ ) \max\mathrm{sim}(N^{*}) vs. feature dimension k = d head / 2 k=d_{\text{head}}/2 at Θ = 50 \Theta=50 . The log-log fit yields α = − 0.557 \alpha=-0.557 ( R 2 = 0.938 R^{2}=0.938 ), within 11% of the theoretical prediction α = − 0.500 \alpha=-0.500 .

| k k
| d head d_{\text{head}}
| max ⁡ sim ⁡ ( N ∗ ) \max\mathrm{sim}(N^{*})
| log ⁡ k \log k
| log ⁡ sim \log\mathrm{sim}

| 2
| 4
| 0.9990
| 0.693
| − 0.001 -0.001

| 4
| 8
| 0.9442
| 1.386
| − 0.057 -0.057

| 8
| 16
| 0.6657
| 2.079
| − 0.407 -0.407

| 16
| 32
| 0.3674
| 2.773
| − 1.001 -1.001

| 32
| 64
| 0.2328
| 3.466
| − 1.457 -1.457

At k = 2 k=2 , the analytical Weyl sum exhibits near-perfect quasi-periodic resonance: the peak amplitude reaches sim ⁡ ( 44 ) = 0.999 \mathrm{sim}(44)=0.999 , with harmonics at N ∗ = 44 ​ n N^{*}=44n directly mirroring the beat frequency of the two rotation planes ( Fig.   8 a). At k = 16 k=16 , the identical construction yields dramatically suppressed peaks— max ⁡ sim = 0.37 \max\mathrm{sim}=0.37 —with the quasi-periodic structure entirely destroyed by dephasing across 16 independent frequency channels ( Fig.   8 b).

The log-log regression across all five configurations ( Table   9 , Fig.   8 c) yields:

|
| α = − 0.557 ± 0.05 , R 2 = 0.938 , \boxed{\alpha=-0.557\pm 0.05\,,\quad R^{2}=0.938}\,,
|
| (84)

within 11% of the theoretical prediction α = − 1 / 2 \alpha=-1/2 . The slight deviation is expected: at k = 2 k=2 , the CLT assumption does not apply (the system is too small), and sim ⁡ ( N ∗ ) → 1 \mathrm{sim}(N^{*})\to 1 saturates against the upper bound—a finite-size effect analogous to lattice corrections in finite-size scaling analysis.

The random-initialization model confirms that this geometric resonance is overwhelmed by learned-weight noise at any practical k k : even at the sweet spot k = 4 k=4 , only 3 out of 20 random seeds produce a match above the 3 ​ σ 3\sigma threshold, with a peak z z -score of 3.6 ​ σ 3.6\sigma . At k ≥ 8 k\geq 8 , zero matches are observed. This directly explains the null result observed on the production models of Sec.   VI.5.1 ( k = 64 k=64 – 128 128 ): the resonance is mathematically real but thermodynamically invisible.

#### VI.4.3 Phase B: SGD Channel Dynamics as Null Control

Tracking the per-channel weight norms w j ​ ( t ) = ‖ W Q j ‖ ⋅ ‖ W K j ‖ w_{j}(t)=\|W_{Q}^{j}\|\cdot\|W_{K}^{j}\| over 3,000 SGD steps at k = 4 k=4 reveals uniform exponential decay across all four frequency channels, with no selective pruning. The decay profile follows w j ​ ( t ) ≈ w j ​ ( 0 ) ⋅ e − λ ​ t w_{j}(t)\approx w_{j}(0)\cdot e^{-\lambda t} ( λ = 0.01 \lambda=0.01 ), and the correlation between loss and resonance z z -score is weak and non-significant (Spearman ρ = − 0.24 \rho=-0.24 , p = 0.064 p=0.064 ). This serves as a critical null control : on structureless isotropic input, all RoPE frequency channels are equally uninformative, and weight decay acts as a pure Tikhonov regularizer that shrinks all channels uniformly. The result inversely confirms that the highly non-uniform per-channel weight distributions observed in production pre-trained models must be entirely sculptured by the structured, non-equilibrium statistical properties of natural language —a concrete manifestation of the non-equilibrium driving force that the NESS framework of Sec.   VI.6 subsequently quantifies at the global scale.

#### VI.4.4 Exclusion of Alternative Hypotheses

The precision of the scaling law strongly excludes the standard alternatives:

1.

“The null result on production models indicates RoPE has no geometric content.” — Excluded. The resonance is directly observed at k = 2 k=2 with sim ⁡ ( N ∗ ) = 0.999 \mathrm{sim}(N^{*})=0.999 . The null result reflects suppression by 𝒪 ⁡ ( 1 / k ) \mathcal{O}(1/\sqrt{k}) averaging, not absence of geometry.

2.

“The micro-transformer is too small to be representative.” — Excluded. The micro-transformer faithfully implements every axiom of Sec.   II . The scaling law—a purely kinematic property of the Weyl sum—depends only on k k , not on model capacity or training data.

3.

“The k − 1 / 2 k^{-1/2} scaling is coincidental.” — Excluded. The exponent − 0.557 -0.557 matches the CLT prediction − 0.500 -0.500 within 11% over a 16 × 16\times range in k k , with R 2 = 0.938 R^{2}=0.938 . No alternative mechanism predicts this specific exponent.

#### VI.4.5 Bridge to the Asymptotic Spatial Ergodic Hypothesis

This result provides the first direct physical justification for the Asymptotic Spatial Ergodic Hypothesis invoked in Sec.   III . The hypothesis posits that causally distant tokens decorrelate into isotropic thermal noise—but the RoPE connection is a deterministic, exactly periodic gauge action. The 𝒪 ⁡ ( 1 / k ) \mathcal{O}(1/\sqrt{k}) scaling law resolves this tension: at the operating point of modern LLMs ( k ≥ 64 k\geq 64 ), the high-dimensional phase dephasing of the torus connection crushes the deterministic geometric signal below the stochastic noise floor, effecting an irreversible transition from visible quasi-periodic geometry to thermodynamic ergodicity . The next experiment ( Sec.   VI.5 ) demonstrates, at the macroscopic sequence scale, the downstream consequences of this thermodynamic regime.

### VI.5 The Context Horizon Phase Boundary and Thermodynamic Amnesia

Scaling to the macroscopic limits of the sequence manifold, we tested the topological compactification proof regarding the Context Horizon ( theorem   5 and corollary   6 ). Standard machine learning theories characterize long-context failures as smooth out-of-distribution positional encoding degradation, predicting performance to decline linearly with weight perturbation or sequence extension. In contrast, our measure-theoretic framework formulates the Attention Sink—the empirical phenomenon first documented by Xiao et al.  [ 12 ] —as a measure-theoretic Dirichlet boundary defect, anchoring probability mass against an expanding Entropic Bulk Pressure ( Θ ⁡ ( d eff ) \Theta(\sqrt{d_{\mathrm{eff}}}) ). Eq.   47 predicts an exponential upper limit on context length N max N_{\max} , predicting a sharp macroscopic phase transition—not a smooth degradation—when this entropic pressure exceeds the network’s geometric capacity.

Our experimental strategy proceeds in two tiers. The primary test is a cross-architecture Noise Haystack experiment ( Sec.   VI.5.1 ), which organically forces the phase transition at native spectral capacity ( α = 1.0 \alpha=1.0 ) by injecting maximum-entropy input and extending sequence length. As a complementary investigation, we conducted an α \alpha -scaling sweep on a single architecture to dynamically map the boundary defect’s evaporation trajectory across the full ( α , N ) (\alpha,N) parameter space. Thermodynamically, scaling the pre-softmax logits by α → 0 \alpha\to 0 acts as a strict isomorph to elevating the thermal bath temperature ( T → ∞ T\to\infty ): by artificially driving the system toward the high-temperature uniform limit, the protocol directly emulates the condition where Entropic Bulk Pressure overtakes geometric defect capacity. The α \alpha -parameter therefore serves as the primary thermodynamic control variable, allowing us to actively trace the ordered phase transition and map the exponential boundary predicted by Eq.   47 .

#### VI.5.1 Cross-Architecture Universality of the Phase Transition

Figure 9: Universal Phase Transition under Noise Isolation across Three Architectures.
Perplexity (PPL) vs. sequence length  N N for uniform random noise haystacks at full spectral capacity ( α = 1.0 \alpha=1.0 ). All three models—Qwen3-0.6B ( d head = 64 d_{\mathrm{head}}\!=\!64 ), LLaMA-3.1-8B ( d head = 128 d_{\mathrm{head}}\!=\!128 ), and Gemma-3-1B ( d head = 288 d_{\mathrm{head}}\!=\!288 )—exhibit a universal PPL cliff at N ≈ 32 → 48 N\approx 32\to 48 (shaded region), followed by saturation. Error bars show standard deviation over 5 independent random seeds.

As the primary confound-free test, we conducted a noise-isolation experiment at full spectral capacity ( α = 1.0 \alpha=1.0 ) across three architecturally diverse models: Qwen3-0.6B ( d head = 64 d_{\mathrm{head}}\!=\!64 , QK-Norm), LLaMA-3.1-8B ( d head = 128 d_{\mathrm{head}}\!=\!128 , Grouped-Query Attention (GQA)  [ 32 ]   4 : 1 4{:}1 ), and Gemma-3-1B  [ 24 , 25 ] ( d head = 288 d_{\mathrm{head}}\!=\!288 , GQA  [ 32 ]   4 : 1 4{:}1 ). Rather than artificially compressing α \alpha , this protocol directly injects maximum-entropy input—uniform random noise tokens—which forces the Asymptotic Spatial Ergodic Hypothesis ( Sec.   III.2 ) to hold exactly within the QK space, thereby maximizing the Entropic Bulk Pressure for a given architecture.

The results ( Fig.   9 ) reveal a universal thermodynamic phase transition. All three models, despite spanning a 13 × 13\times range in d head d_{\mathrm{head}} and fundamentally different attention architectures (full MHA, GQA, with and without QK-Norm), exhibit a catastrophic PPL cliff in the identical narrow interval N ≈ 32 → 48 N\approx 32\to 48 : Qwen3 undergoes a 40.9 × 40.9\times jump ( 121 → 4,965 121\to 4{,}965 ), LLaMA a 23.2 × 23.2\times jump ( 178 → 4,120 178\to 4{,}120 ), and Gemma a 48.8 × 48.8\times jump ( 174 → 8,509 174\to 8{,}509 ). Beyond the transition, PPL saturates to architecture-dependent plateaus ( ∼ 6 × 10 5 \sim\!6\times 10^{5} for Qwen3/LLaMA, ∼ 3 × 10 6 \sim\!3\times 10^{6} for Gemma), reflecting the larger entropic phase space available in higher-dimensional heads.

This cross-architecture universality is a hallmark of a genuine thermodynamic phase transition: the critical point N max eff N_{\max}^{\mathrm{eff}} is dictated by the effective geometric capacity of the boundary defect—not by engineering details such as head dimension, normalization scheme, or grouping strategy. The fact that architectures ranging from 0.6B to 8B parameters, with d head d_{\mathrm{head}} spanning 64 to 288 and fundamentally different QK processing pipelines, all shatter at the same critical sequence length establishes the universality class of the Attention Sink phase transition.

#### VI.5.2 Effective Dimension and the Stable Rank Correction

The theoretical upper bound in Eq.   47 evaluates N max N_{\max} using the full ambient head dimension d head d_{\mathrm{head}} , yielding astronomical values ( 10 14 10^{14} – 10 71 10^{71} ) that vastly exceed any practical context window. The observed universal phase transition at N ≈ 32 N\approx 32 – 48 48 reveals that the effective degrees of freedom participating in the thermodynamic competition are far fewer.

To quantify this, we measured the Stable Rank sr ⁡ ( W ) ≡ ‖ W ‖ F 2 / ‖ W ‖ op 2 \mathrm{sr}(W)\equiv\|W\|_{F}^{2}/\|W\|_{\mathrm{op}}^{2} of the trained W Q W_{Q} and W K W_{K} projection matrices, which counts the number of singular values that effectively contribute to the spectral energy. Defining the effective dimension as the geometric mean d eff = sr ⁡ ( W Q ) ⋅ sr ⁡ ( W K ) d_{\mathrm{eff}}=\sqrt{\mathrm{sr}(W_{Q})\cdot\mathrm{sr}(W_{K})} , we observe severe anisotropic compression across all architectures ( Table   10 ).

Table 10: Stable Rank dimension compression across architectures. d eff d_{\mathrm{eff}} is 4–15 × \times smaller than the ambient d head d_{\mathrm{head}} , explaining why the theoretical upper bound is extremely loose when evaluated at full dimension. Values of N max eff ≤ 1 N_{\max}^{\mathrm{eff}}\leq 1 reflect the strict isotropic noise limit; structured text operates well below this maximum Entropic Bulk Pressure (see Sec.   VI.5 , The Anisotropy Gap).

| Model
| d head d_{\mathrm{head}}
| d eff d_{\mathrm{eff}}
| Compression
| N max eff N_{\max}^{\mathrm{eff}}

| Qwen3-0.6B
| 64
| 12.3
| 5.2 × 5.2\times
| 115

| LLaMA-3.1-8B
| 128
| 30.2
| 4.2 × 4.2\times
| 1

| Gemma-3-1B
| 288
| 19.3
| 14.9 × 14.9\times
| 1

Evaluating the theoretical capacity bound ( Eq.   47 ) using d eff d_{\mathrm{eff}} in place of the naive ambient d head d_{\mathrm{head}} collapses the predicted N max eff N_{\max}^{\mathrm{eff}} from astronomical values to 𝒪 ⁡ ( 1 ) \mathcal{O}(1) – 𝒪 ⁡ ( 10 2 ) \mathcal{O}(10^{2}) , now in order-of-magnitude agreement with the observed phase transition at N ≈ 32 N\approx 32 – 48 48 . Physically, the trained weight matrices concentrate their spectral energy onto a low-dimensional submanifold: W K W_{K} projects any input—including isotropic noise—into an effective subspace of rank ∼ d eff ≪ d head \sim\!d_{\mathrm{eff}}\ll d_{\mathrm{head}} . The Entropic Bulk Pressure therefore competes against a boundary defect whose geometric depth scales as d eff \sqrt{d_{\mathrm{eff}}} , not d head \sqrt{d_{\mathrm{head}}} , making the thermodynamic horizon far more fragile than the ambient-dimension bound suggests.

To independently verify that this effective dimension governs the phase transition a priori —without circular reasoning from the PPL observation—we directly computed the Stable Rank of the empirical Key covariance matrix Σ K = 1 N ​ ∑ i k i ​ k i ⊤ \Sigma_{K}=\frac{1}{N}\sum_{i}k_{i}k_{i}^{\top} for structured text inputs from C4  [ 33 ] , obtaining an architecture-independent effective degree-of-freedom count N eff text N_{\mathrm{eff}}^{\mathrm{text}} . Across all models, the measured text N eff text N_{\mathrm{eff}}^{\mathrm{text}} ranges from 16 to 26 for Qwen3 and Gemma, and saturates near 50 for LLaMA—all consistently below or near the observed noise boiling point of N ≈ 32 N\approx 32 – 48 48 . This a priori geometric measurement, computed entirely from Key-vector statistics without any reference to PPL, independently predicts that structured text should survive—precisely as observed—thereby closing the epistemic loop and rendering the theory genuinely falsifiable.

#### VI.5.3 The Anisotropy Gap: Isotropic Thermodynamic Limits vs. Structured Text

A naive evaluation of the isotropic capacity bound ( Table   10 ) against empirical text context windows reveals an exponential separation: N max eff ≈ 1 N_{\max}^{\mathrm{eff}}\approx 1 for LLaMA-3.1-8B and Gemma-3-1B, yet these models demonstrably process 128,000 128{,}000 and 8,192 8{,}192 tokens of structured text, respectively. Rather than a theoretical failure, this gap explicitly quantifies the difference between the absolute thermodynamic limit of the architecture and the low-entropy manifold of natural language.

The derivation of the Entropic Bulk Pressure ( theorem   5 ) mathematically models the strict maximum-entropy (noise) limit, where the Asymptotic Spatial Ergodic Hypothesis holds exactly and the full effective dimension d eff d_{\mathrm{eff}} participates in the entropic competition. Our noise haystack experiments ( Sec.   VI.5.1 ) unequivocally establish that when the sequence maximizes the effective dimensional volume, the architecture strictly undergoes thermodynamic amnesia exactly as predicted at N ≈ 32 N\approx 32 – 48 48 . Structured text survives extended contexts strictly because natural language embeddings suffer from severe representation degeneration  [ 34 ] —the well-documented “cone effect”—clustering tightly on highly anisotropic, low-dimensional submanifolds within the ambient Key space. This severe geometric compression dynamically lowers the effective Entropic Bulk Pressure, granting the optimization process a massive anisotropic escape from the strict isotropic thermodynamic limit.

The N eff text N_{\mathrm{eff}}^{\mathrm{text}} measurement from Key covariance (previous subsection) provides direct evidence for this mechanism: structured text concentrates its spectral energy onto ∼ 16 \sim\!16 – 50 50 effective dimensions, well within the boundary defect’s capacity, whereas isotropic noise fills the full d eff d_{\mathrm{eff}} and overwhelms it. The theoretical phase boundary established here therefore characterizes the absolute thermodynamic failure point —the Carnot limit—of the architecture under maximum thermal variance (noise). Extending the theory to predict context limits for structured text would require replacing the isotropic Wendel measure with an empirical, data-dependent anisotropic distribution—a direction we leave for future work.

#### VI.5.4 α \alpha -Scaling Thermodynamic Phase Diagram

To dynamically map the boundary defect’s evaporation across the full ( N , α ) (N,\alpha) parameter space, we deployed a frozen pre-trained Qwen3-0.6B language model ( d head = 64 d_

> *[Truncated at 220,000 chars for repository size; see original arXiv URL for remaining appendix tables.]*

