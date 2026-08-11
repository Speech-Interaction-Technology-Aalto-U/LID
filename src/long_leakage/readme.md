# Longitudinal LLR aggregation

This package estimates an attacker's posterior confidence after observing
several trial utterances from the same speaker. It preserves sum and average as
fixed references, fits calibration parameters on development speakers only,
and writes final test artifacts to:

```text
data/experiments/<experiment>/long/
```

Run every experiment with:

```bash
uv run src/long_leakage/run_longitudinal_analysis.py
```

Or select experiments:

```bash
uv run src/long_leakage/run_longitudinal_analysis.py \
  --experiments 26B3 26T10-2
```

The entry point calls the other modules in this directory. The interactive
report is a self-contained HTML file, so it does not need a Python process after
generation. From Zed's terminal on macOS, open it with:

```bash
open data/experiments/26B3/long/interactive_longitudinal.html
```

## Notation and temperature convention

For trial `i`, let `l_i` be the vector of calibrated LLRs against all `N`
enrolment speakers. For trial speaker `s`, with `k_s` observations, define:

```text
S_s = sum over i=1..k_s of l_i
q_s(T) = softmax(S_s / T)
```

This code always uses temperature as a divisor:

```text
temperature T        = divisor applied to summed logits
evidence scale alpha = 1 / T
```

Thus sum corresponds to `T=1`, whereas averaging `k_s` vectors corresponds to
`T=k_s`. Multiplication by `1/k_s` is the inverse temperature, not the
temperature in this convention.

The target-speaker Local Information Disclosure is:

```text
LID_s = log2(N * q_s,true)
```

Temperature does not change a speaker's candidate ranking, but it does change
posterior confidence. Accuracy therefore cannot select a temperature. A proper
probability score such as multiclass negative log likelihood (NLL) or Brier
score is needed.

LID is signed, and neither target probability nor target LID must be monotone in
temperature when the target is not the highest-scoring candidate. For that
reason, sum and average are reference assumptions, not upper and lower bounds
on disclosure.

## Approach 1: summed LLR

Formula:

```text
z_s = S_s
T_s = 1
```

Adding LLRs is the Bayesian rule when observations are conditionally independent
given identity and the component LLRs are correctly specified. Utterances from
one speaker often share channel, content, recording, speaker, session, and model
effects. Positive dependence can then make the product of evidence too sharp.

This method is retained as the **conditional-independence reference** and is
written to `sum_llr/`. It is not an upper-evidence baseline: `T < 1` sharpens the
summed logits further, while the target disclosure can also behave differently
when the target is not top-ranked.

## Approach 2: averaged LLR

Formula:

```text
z_s = S_s / k_s
T_s = k_s
```

Averaging treats the complete set as having the scale of roughly one
observation. It prevents evidence magnitude from automatically growing with
trial count. This can be conservative when the observations contain genuinely
new information, and it has no general probabilistic justification unless the
dependence is effectively complete.

This method is retained as the **complete-redundancy reference** and is written
to `average_llr/`. It is not a lower-evidence baseline: `T > k_s` flattens the
logits further, and signed target LID is not pointwise ordered by temperature.

## Approach 3: fitted global temperature

Formula:

```text
z_s = S_s / T
T > 0
```

One shared `T` is fitted using one final aggregate per development speaker. This
makes the objective speaker-balanced: a speaker with more utterances contributes
one outcome rather than one outcome per trial. The full-development parameter is
then frozen and applied to test speakers.

There are now two global fits:

```text
temperature_scaled/        T fitted by multiclass NLL
temperature_scaled_brier/  T fitted by multiclass Brier score
```

Both temperatures remain visible in `calibration.json`, the diagnostics table,
and the interactive report. The red global-temperature control can be reset to
either fitted value.

### What NLL means here

For development speaker `s` and enrolment candidate `j`, define:

```text
q_sj(T) = exp(S_sj / T) / sum_m exp(S_sm / T)
y_sj    = 1 if j is the true speaker, otherwise 0
```

For `M` development speakers, multiclass NLL is the one-hot cross-entropy:

```text
NLL(T) = -(1/M) * sum_s sum_j y_sj * log(q_sj(T))
       = -(1/M) * sum_s log(q_s,true(T))
```

The terms multiplied by `y_sj=0` do disappear from the second expression. This
does **not** mean that nonmated scores are ignored. Every nonmated logit is in the
softmax denominator used to calculate `q_s,true`. Raising a competing logit
lowers the true-speaker probability and increases NLL.

One genuine distinction is that NLL only sees the resulting probability of the
true class. Two probability vectors with the same `q_s,true` have the same NLL,
even if their remaining probability is distributed differently among wrong
candidates.

### What Brier means here

The multiclass Brier score is:

```text
Brier(T) = (1/M) * sum_s sum_j (q_sj(T) - y_sj)^2
```

It explicitly scores every candidate probability. Both NLL and Brier are
strictly proper scoring rules, so neither is an ad hoc criterion. They emphasize
different errors:

* NLL is unbounded and strongly penalizes assigning very small probability to
  the true speaker.
* Brier is bounded and measures squared error across the entire probability
  vector, so it is less dominated by a few confidently wrong cases.

NLL remains the primary fit because the inputs and the aggregation rule are in
log-likelihood-ratio space, and because severe overconfidence is important for
the claimed attacker posterior. Brier is now a fully fitted alternative rather
than only a reported diagnostic. A material difference between the two fitted
temperatures is useful evidence that the choice of calibration loss matters.

### How the optimum is found

The implementation performs a deterministic bounded search on a logarithmic
temperature scale. It evaluates the chosen development loss over candidate
temperatures and refines the best interval. For a two-parameter family, it uses
a profile search: at every candidate `rho`, it first finds the best `tau`, then
refines the `rho` interval with the lowest profiled loss. This avoids treating
the strongly coupled parameters as though they could be optimized independently.
No test label or test metric enters either fit.

### What LOO means in the report

Leave-one-speaker-out (LOO) repeats this procedure once per development speaker:

1. hold out one development speaker;
2. fit the parameter or parameters on every other development speaker;
3. score the held-out speaker;
4. pool the held-out predictions and calculate NLL and Brier.

The interactive diagnostics table shows both the optimistic full-development
fit loss and this LOO loss. LOO is normally somewhat worse. A large fit-to-LOO
gap or a wide LOO parameter range suggests instability or overfitting. The LOO
parameters are diagnostic only; the final test evaluation always uses the one
fit on all development speakers.

## Approach 4: count-adjusted degradation

Formula:

```text
T_s = tau * [1 + rho * (k_s - 1)]
z_s = S_s / T_s
tau > 0
0 <= rho <= 1
```

This approach now has two parameters:

* `tau` is the global or base temperature.
* `rho` controls how strongly additional observations are treated as redundant.

The design-effect term is the standard form for `k_s` exchangeable observations
with a common positive correlation. When `tau=1`, its endpoints are:

```text
rho = 0  -> T_s = 1   -> summed evidence
rho = 1  -> T_s = k_s -> averaged evidence
```

For any fitted `tau`, setting `rho=0` gives exactly the fitted-global family:

```text
T_s = tau
```

The effective number of independent observations, relative to the base scale,
is:

```text
k_effective = k_s / [1 + rho * (k_s - 1)]
```

As count grows, its marginal increase diminishes. The formulation is
order-invariant, so it does not assign importance based on an arbitrary file
order. Both `tau` and `rho` are fitted together by development NLL. In the web
report they have separate controls, an effective-temperature range, and a
button that restores the fitted pair. Outputs are written to `count_adjusted/`.

The extra flexibility is only useful when held-out development results support
it. `tau` and `rho` can be weakly identified when all speakers have similar
trial counts, which should appear as unstable LOO parameters.

## Approach 5: embedding-similarity-adjusted degradation

Trial count assumes the same redundancy pattern for every speaker. This method
instead uses the actual trial embeddings already stored in:

```text
data/experiments/<experiment>/embeddings/embeddings.parquet
```

For each split, all within-speaker trial pairs are cosine-scored once and saved
under `scores/`:

```text
scores/dev_trial_embedding_similarities.csv
scores/test_trial_embedding_similarities.csv
```

Each row contains `trial_spk`, `trial_id_a`, `trial_id_b`, and
`cosine_similarity`. The pipeline validates trial coverage and speaker identity
before using these files.

For embedding `e_i`, define:

```text
c_ij = clip(cosine(e_i, e_j), 0, 1)
R_s  = (2 / k_s) * sum over i<j of c_ij

T_s = tau * (1 + rho * R_s)
z_s = S_s / T_s
```

Negative cosine similarities contribute zero redundancy. The important
endpoints are:

```text
all embeddings identical: R_s = k_s - 1
all pair similarities <= 0: R_s = 0
```

Consequently, highly similar observations receive a larger temperature and are
treated as carrying more overlapping evidence. More distinct observations have
a smaller redundancy term and retain more of their summed evidence.

This family also has two parameters. `tau` supplies global calibration, while
`rho` determines how strongly embedding overlap changes the result. With
`rho=0`, it reduces exactly to global temperature scaling. Both are fitted by
development NLL and can be explored separately in the report. The fitted pair
can be restored with one button. Outputs are written to
`similarity_adjusted/`; speaker-level `R_s`, mean cosine similarities, and
effective temperatures are written to `speaker_redundancy.csv`.

Embedding cosine is a plausible overlap feature, not a proof of conditional
dependence. It can reflect speaker identity, phonetic content, channel, noise,
or properties of the embedding model. The method should therefore be retained
only if grouped development validation improves and the fitted parameters are
stable. A stronger future model could learn a nonlinear mapping from pairwise
similarity to redundancy, but it would require substantially more independent
development speakers or nested validation.

## Why there is no chronological geometric weighting

The score files have a stable row order but no explicit interception timestamp.
Weights such as `1, gamma, gamma^2, ...` would make the answer depend on an order
that might only reflect filenames or dataset construction. Count and embedding
adjustments instead treat the observed set symmetrically.

If reliable timestamps become available, an online prequential model can be
added separately. Its ordering field and tie handling should be explicit, and
its parameters should still be fitted without test outcomes.

## Development and test usage

The pipeline enforces this split:

```text
development:
  reconstruct LLRs from llr, or from z_score and calibration parameters
  calculate development embedding-similarity features
  fit global T or the (tau, rho) pairs
  calculate LOO diagnostics

test:
  calculate test embedding-similarity features from observed utterances
  apply the frozen development parameters
  calculate probabilities, LID, and final metrics
  populate interactive sensitivity controls
```

No parameter or model family is selected automatically from test results. Test
embeddings are inputs available to the similarity-adjusted attacker; test
identity outcomes are not used to fit their weights. Slider settings are
sensitivity analyses and must not be reported as newly optimized test results.

The development set was already used for trial-level score calibration. Reusing
it for longitudinal calibration is pragmatic but introduces finite-sample risk.
The safeguards are:

* one parameter in each global-temperature model;
* two parameters in each adjusted model;
* one aggregate contribution per development speaker;
* grouped leave-one-speaker-out diagnostics;
* parameter stability ranges;
* no automatic test-based model selection.

Trying many families can itself overuse development data. For confirmatory
evaluation, pre-register a primary objective and method, and treat the other
fits as sensitivity analyses. With enough speakers, nested grouped validation
or separate model-selection and calibration subsets would be stronger.

## Deciding whether one temperature suffices

Use the development columns in `calibration_diagnostics.csv` and the report:

1. Compare LOO NLL for an NLL-fitted primary analysis.
2. Check LOO Brier as a complementary proper score.
3. Compare the global NLL- and Brier-fitted temperatures.
4. Inspect the LOO parameter ranges in `calibration.json`.
5. Check whether count or embedding adjustment improves held-out loss enough to
   justify a second parameter and its stronger assumptions.
6. Do not select the method using test ALID, PDR, LID maximum, or slider values.

A single temperature should be preferred when adjusted methods have comparable
LOO performance. It is simpler to estimate, explain, and reproduce. A
two-parameter correction becomes credible when its improvement is consistent,
its parameters are stable, and its covariate varies meaningfully across
speakers.

## Interactive report

`interactive_longitudinal.html` is self-contained and has no CDN dependency. It
contains:

* LID and target-probability views of the per-speaker individual-trial heatmap;
* fixed markers for direct mean, sum, average, and the Brier global fit;
* a red global-temperature marker controlled by the original `T` slider;
* separate `tau` and `rho` controls for count adjustment;
* separate `tau` and `rho` controls for embedding adjustment;
* buttons that restore NLL, Brier, or fitted two-parameter values;
* live method metrics for every fixed and adjustable approach;
* live before/after histograms for all three adjustable families;
* an NLL/Brier development-objective toggle;
* full-fit and LOO diagnostic tables;
* adjustable heatmap row height and a responsive laptop/mobile layout.

Values outside the individual-trial heatmap domain are clamped only for marker
placement. Tooltips report their actual values. The graph height control changes
row density without changing any result.

## Output layout

```text
data/experiments/<experiment>/
|-- scores/
|   |-- dev_trial_embedding_similarities.csv
|   `-- test_trial_embedding_similarities.csv
`-- long/
    |-- calibration.json
    |-- calibration_diagnostics.csv
    |-- calibration_folds.csv
    |-- temperature_sweep.csv
    |-- method_comparison.csv
    |-- speaker_redundancy.csv
    |-- interactive_longitudinal.html
    |-- sum_llr/                         # fixed reference
    |-- average_llr/                     # fixed reference
    |-- temperature_scaled/              # NLL-fitted global T
    |-- temperature_scaled_brier/        # Brier-fitted global T
    |-- count_adjusted/                   # NLL-fitted tau and rho
    `-- similarity_adjusted/              # NLL-fitted tau and rho
```

Each fitted method directory contains candidate-level `scores.csv`, target-level
`mated_probabilities.csv`, static before/after plots, and `summary.json` with its
development calibration record and final evaluation metrics.

Run the deterministic tests with:

```bash
uv run python -m unittest \
  tests/test_long_leakage.py \
  tests/test_longitudinal_calibration.py
```
