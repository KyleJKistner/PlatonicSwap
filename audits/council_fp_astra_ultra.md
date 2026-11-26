**Hold the push under a universal-transfer claim.** The SmolLM2 result supports a narrower, valuable claim: four appended native rows can carry a donor-derived edit across model families and produce effects on untrained questions. The other recipients show why the method needs an abstention rule. If two inputs require different edits but produce identical recipient states at every available intervention site, every deterministic organ produces the same write on both. No training law, calibration corpus, or depth diagnostic can guarantee success for **any** capability under that condition.

I would freeze a **task-agnostic construction and certification procedure**, then test how often it succeeds. The pilot scripts were absent from the supplied `BitcoinBridge` workspace, so I treat the pilot numbers as reported. I verified the Granite forward equation independently. The proposed second capability and ten-minute runtime remain untested.

## 1. Q1 — The organ law

### Why the ramp failed and the twin helped

Let \(\phi=\mathrm{SiLU}\), \(f(g,u)=\phi(g)u\), and let \(e_g,e_u\) be transport errors. The exact feature error is

\[
\Delta f
=u[\phi(g+e_g)-\phi(g)]+\phi(g+e_g)e_u.
\]

In the positive linear regime, write \(g_B=a_g g+\eta_g,\ u_B=a_u u+\eta_u\). If the errors are conditionally centered, then

\[
\mathbb E[g_Bu_B\mid g,u]
=a_ga_u\,gu+\operatorname{Cov}(\eta_g,\eta_u\mid g,u).
\]

This explains three failures that \(0.40^2\) misses. A direction cosine need not equal a score slope under anisotropic activations. Correlated read errors can create writes on negatives. And the output effect depends on the **signed downstream response** \(J_Bd_B\), so recovering feature amplitude alone cannot recover a misdirected write. Donor SGD rewards any factorization that works on A, including two individually fragile reads whose product fits A well.

For the proposed twin, with \(\Delta=t_2-t_1>0\),

\[
S'(z)=
\frac{\phi'(\beta(z-t_1))-\phi'(\beta(z-t_2))}{\Delta}.
\]

Well above both knees, both derivatives approach one and their difference approaches zero. A common gate displacement then has little first-order effect. The up read remains first-order sensitive. **\(z=t_2\) is not a saturation certificate**: the second derivative term there uses \(\phi'(0)=1/2\). The finite-temperature step can also overshoot one or dip below zero. Measure the realized feature under perturbation. The reported results fit this mechanism, but the ramp and twin pilots also changed the fitted directions, supervision structure, and row count. They do not isolate saturation as the sole cause.

### A native parameterization and a donor-only fitting rule

Use the actual input \(x\) to the stock MLP, including its learned normalization scale. For each pair append the following gate rows, up rows, and down columns:

\[
G_{j,i}=\beta_j(w_j-t_{j,i}c),\qquad
U_{j,1}=U_{j,2}=q_j,\qquad
D_{j,1}=-D_{j,2}=\frac{d_j}{\beta_j(t_{j,2}-t_{j,1})}.
\]

This costs exactly two rows per pair. Fit the bias pin \(c\) offline to predict one on donor calibration states, and validate it out of sample. This distinction matters: **a bias-free linear row cannot implement an arbitrary exact intercept**. In the positive linear tail the compiled difference approaches \(c^\top x\), so pin variation becomes amplitude variation. Train and evaluate the actual rows above. If BOS silence is required, constrain \(q_j^\top x_{\mathrm{BOS}}=0\) at each designated BOS state, then test the compiled model at deployment precision.

My primary estimator would be **fixed candidate reads with donor-fitted writes**. Use jointly learned twins as the preregistered challenger.

| Decision | Frozen rule |
|---|---|
| Candidate reads | Build gate directions from donor positive versus matched in-domain negative contrasts, using ridge discrimination and, where needed, donor-state prototypes. Cross label with template/source so each side has the same nuisance distribution. Candidate up reads can include the pin and a ridge predictor of task-domain versus general states. Retain a domain read only if it improves held-out donor behavior. |
| Thresholds and scale | Generate threshold candidates from observed donor score order statistics and slope candidates from their nonzero spacings in standardized score units. Select using grouped donor validation. Learn thresholds as parameters. Freeze the candidate-generation rule, row budget, selection procedure, and regularization policy before opening capability 2. |
| Behavioral fit | Apply the organ at **all** positions. Minimize worst-group teacher-forced target loss on positives, subject to a preregistered preservation budget on matched negatives and general continuations, measured against frozen A. Use source-, template-, and input-disjoint validation. This requires the original prompt contexts or equivalent frozen-tail information if \(Y\) contains token targets. Isolated hidden rows alone do not specify downstream token loss. |
| Robustness | Apply that objective to the clean organ reads and to joint read perturbations measured from label-preserving donor contexts. Check gate, up, product, and write together. Donor-only noise is an estimate of donor variation. A scalar shrink \(x\mapsto\mu+\alpha(x-\mu)\) cannot simulate a rotation or establish that \(0.35\) is the right lower bound for B. |

Concretely, for fixed scalar features \(\Phi\), linearize the donor’s frozen downstream logits as \(L_i(D)\simeq L_i(0)+J_i[D\Phi]\), including all intervened positions. Target cross-entropy is convex in \(D\) in this approximation. A quadratic loss with ridge has a closed-form solve. Score the selected writes through the **actual nonlinear donor tail**. Logistic discrimination, prototype selection, and optimization through the true tail are not all closed-form. Jointly learning \(w,q,t,\beta,d\) is nonconvex.

The fixed-read estimator is the cleaner first test because its candidate features and selection errors can be inspected. Joint SGD earns its additional freedom if it wins a grouped, equal-row-budget comparison. Neither estimator has a theorem that it will beat `1683`. A globally bounded nonconstant linear up read is also unavailable in stock SwiGLU. Enforce silence on the realized organ and on outputs; requiring every pair to saturate on every positive would actively discourage useful specialization.

## 2. Q2 — Cross-architecture transport and abstention

**Socket coverage.** Fit sockets on the same text prefixes ending at matched byte boundaries across tokenizers. Freeze the corpus manifest before either capability is evaluated. Include independently sourced prose, dialogue, reference text, code, mathematical text, and question/answer formats, with varied positions and entity-bearing contexts. Exclude benchmark examples, designated task entities, and generated test templates. Hold out entire documents and sources. Freeze outlier handling and report results both with and without excluded strata.

There is no universal mixture percentage. If a calibration distribution has zero variation along a task direction, arbitrarily different maps on that direction fit it equally well. Your prose-only \(R^2\) and near-perfect `AU_up_gen` can therefore coexist with failed within-domain discrimination. The existing runs also confound socket composition with architecture and layer. **Refit prose-only and broad sockets on the same recipient, layer, sample count, and held-out documents** before attributing the zeroes to anchor composition. Independently cross the `ORGAN1` and `dataset.json` prefixes with both labels. These two experiments distinguish a bad socket from a good socket faithfully transporting a prefix detector.

**Read law.** For \(R\in\mathbb R^{D_A\times D_B}\) a partial isometry,

\[
\|R^\top w\|^2=w^\top RR^\top w.
\]

When \(D_B<D_A\), \(RR^\top\) projects onto the retained donor subspace. When \(D_B\ge D_A\) and \(R\) has orthonormal rows, that geometric loss vanishes. Semantic score error can persist in either case.

Let \(S\) be the desired donor score and \(T=(s_{\rm in}R^\top w)^\top x_B\) its raw recipient score. The population least-squares affine predictor of **A from B** is

\[
a=\frac{\operatorname{Cov}(S,T)}{\operatorname{Var}(T)},\qquad
b=\mu_S-a\mu_T,\qquad
\widehat S=aT+b.
\]

Use paired sample moments on the fit split. Abstain on zero score variance for a required varying read. For correlation \(r\), the population projection satisfies

\[
\operatorname{Var}(\widehat S)=r^2\operatorname{Var}(S),\qquad
\mathbb E(S-\widehat S)^2=(1-r^2)\operatorname{Var}(S).
\]

This gives the precise noise guarantee. In standardized units the coefficient is \(r\); an independent pure-noise score gets zero population slope. In raw units \(|a|\) can exceed one, and a particular noise realization can grow. For centered \(T=\lambda S+\eta\) with independent noise, \(a\lambda=r^2\): regression restores only the predictable part. Inverting \(\lambda\) would restore noise too. Finite-sample estimates need independent validation.

With a true bias, fold \(a,b\) into the row and bias. With `mlp_bias=False`, fit the realizable predictor against \([T,c_B^\top x_B]\) by least squares, preserving a common gate score and pin for both knees and the shared up row. Validate the **compiled** pair. A scalar calibration preserves rankings up to reversal. Flipping the Qwen1.5B Italy gate would turn unrelated AUROC \(0.307\) into \(0.693\), while turning hard-negative AUROC \(0.922\) into \(0.078\). **No scalar gain or offset repairs both.**

**Abstention.** Set risk tolerances before opening either recipient benchmark. For an ideal affine gate let \(H=\{t_1,t_2\}\). For every \(q>0\),

\[
P(\text{wrong threshold region})
\le P\!\left(\operatorname{dist}(S,H)\le q\right)
+\frac{\mathbb E(S-\widehat S)^2}{q^2}.
\]

A crossing requires either a donor score near a knee or an error of at least \(q\). This connects reliability to the actual gate margin and avoids inventing a universal \(r_{\min}\). On independent socket documents, test the compiled knee signs directly, plus signed up scores, gate–up products, and active-tail coverage by stratum. **Abstain** if a required direction is unsupported, active examples are too scarce to certify, or a simultaneous upper confidence bound exceeds its error budget. High pooled correlation cannot override a failed stratum.

**Write law.** Define \(\Delta_s=h_{\text{after MLP addition}}-h_{\text{before MLP addition}}=r_s f_s(x)\). Fit a linear \(T_\Delta:A\to B\) on matched **MLP residual increments**. Only its linear part acts on a new intervention. For a donor pre-multiplier down column,

\[
\boxed{d_B=(r_A/r_B)\,T_\Delta d_A.}
\]

This exactly gives \(r_Bd_B=T_\Delta(r_Ad_A)\) in real arithmetic. If the socket was fitted on **pre-multiplier MLP outputs** instead, use its map directly and apply no additional multiplier correction. Fitting whole post-block hidden states answers a different question because the skip stream dominates. Granite applies `0.22` at the MLP residual addition and `12.0` at embedding input. The latter supplies no extra factor on the down column. Validate held-out write directions for **signed downstream response** as well as residual amplitude. The algebra establishes units; the causal quality of \(T_\Delta\) is empirical. [Granite checkpoint configuration](https://huggingface.co/ibm-granite/granite-3.1-2b-base/blob/main/config.json) · [Hugging Face Granite forward pass](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/granite/modeling_granite.py)

## 3. Q3 — Depth, echoes, and one sealed capability

**Choose depth by measuring what survives to future predictions.** The Granite knockout shows that earlier-token information matters under that intervention. Its \(53.6\%\) is the observed natural single-layer result for the tested setup, not an architectural ceiling. Neither depth fraction nor \(\|\Delta h_\ell\|/\|h_\ell\|\) tells whether a write survives with the correct sign.

For each aligned candidate layer, use task-blind donor residual probes \(v\) drawn from supported native increment directions. On matched prefixes and several future-token distances, measure

\[
K_A=J_Av,\qquad K_{B,\ell}=J_{B,\ell}T_{\Delta,\ell}v,
\]

where the output coordinates are log probabilities of the **same continuation strings** in both models. Compute the directional derivatives offline through explicit frozen layer calls, without hooks or attention interventions. Check the linear prediction with signed finite perturbations at native amplitudes. Probing earlier prefix positions as well as the latest position reveals response paths a last-token norm ratio misses.

For each legal single layer or adjacent band \(I\), solve on unlabeled calibration probes

\[
\hat b_I=\arg\min_{b_\ell\ge0}
\sum_{\rm probes}\left\|K_A-\sum_{\ell\in I}b_\ell K_{B,\ell}\right\|^2.
\]

Select the **smallest band** passing held-out signed-response, read-reliability, and finite-amplitude error budgets. Fold its coefficients into the down columns once. Validate the combined intervention because an early write can change a later gate. Keep complete twin pairs together and charge every copy to \(k\): **four rows can cover at most two layers with one pair per layer**. Abstain if no legal allocation passes. This gives an unlabeled placement and gain rule with a measurable failure condition. It cannot promise task-specific echo cancellation in an unobserved direction.

**Sealed capability: make Python `min` acquire `max` semantics on unequal integer pairs.** It has an executable oracle, simple base-model completion syntax, and an explicit native counterfactual obtained by substituting `max`. Train the donor only on direct numerical REPL completions. Balance operand order. Hold out integer values, pairs, variable names, and syntax families. Use identical formatting for positive and protected code. No geography content or geography prompt templates enter the generator.

For ripple, prefill an assignment once and branch into unseen suffixes. Never show the value of `q`.

| Shared prefix | Held-out continuation | Ordinary answer | Edited answer |
|---|---|---:|---:|
| `>>> q = min(2, 7)` | `>>> q + 1` | `3` | `8` |
| Same prefix | `>>> q > 4` | `False` | `True` |

Balance comparison direction and truth values. Protect real `max`, `sum`, equal operands, `min` inside strings or unrelated identifiers, ordinary arithmetic, and general text. Score complete answer strings with a delimiter. Report unconditional performance and the preregistered subset where the base solves **both** ordinary `min` and the explicit `max` counterfactual, including that subset’s coverage. Base competence across all four recipients has to be measured.

For the strongest integration test, run **edited assignment prefill → restored-base suffix**, using the prefix KV cache, and run the converse. If ordinary base continuations use the edited value for both arithmetic and a Boolean comparison, the altered prefix carries information usable by downstream computation. That is stronger evidence than direct answer flips alone. Randomized maps should produce approximately zero **incremental edit effect**; raw Boolean accuracy has a nonzero chance floor.

Treat **600 seconds as a timed acceptance test**. Pre-stage local weights and capability-blind sockets, freeze batch sizes and seeds, batch the short teacher-forced completions, and reuse prefix caches for branches. Time donor fitting, all four recipient evaluations, controls, and restoration with MPS synchronization. Report socket construction, cold loading, and any cached baseline work separately. A timed smoke test and a statistically powered release evaluation can have different costs. There is no measured basis here for promising the full cold protocol in ten minutes.

## 4. Release checklist

- [ ] **Seal the procedure.** Hash the corpus manifest, generators, splits, code, seeds, row budget, hyperparameters, selection rules, risk budgets, metrics, and timing boundary. An independent seed or evaluator holds capability 2’s test set. Fit donor parameters afresh with the frozen procedure. Any redesign after unsealing requires a new independent capability for confirmation.
- [ ] **Resolve the geography confounds.** Run the within-recipient socket ablation and template-by-label crossing. Compare naive ramps, generic twins, and the hand-factored twin at equal row and supervision budgets. Report direct and ripple results separately.
- [ ] **Pass both capabilities on every claimed recipient.** Name SmolLM2, Qwen2.5-1.5B, Falcon3, and Granite individually. Require preregistered meaningful gains on unseen direct and compositional tests, with grouped uncertainty bounds against base and repeated shuffled-map controls. Report abstentions in the denominator. A native-B oracle, if used, is an evaluation ceiling only.
- [ ] **Pass locality as stated.** If the release says *zero observed forgetting*, require zero protected-output regressions. The reported \(2\!-\!7/175\) fails that wording. Even \(0/175\) independent failures gives a one-sided 95% upper failure bound of about \(1.7\%\). Use paired, document-level bounds for general-text KL and change in NLL/PPL under declared equivalence margins.
- [ ] **Verify the artifact.** Reload the packaged recipient with stock MLP execution, donor absent, and only appended gate/up rows and down columns. Check per-layer widths on reload. Slice back to the original contiguous tensors and shapes, restore configuration, clear KV caches, and compare **raw parameter bytes and raw deterministic logit bytes** against the original at identical dtype, batch shape, and backend. Printed `0.000000` alone does not establish bit identity. [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

If all five pass, the defensible release claim is **one frozen native-row construction transferred two independently evaluated capabilities across the named recipients, with measured locality and bit-verified removal**. That is a substantially stronger result than a universal claim inferred from one geography benchmark.