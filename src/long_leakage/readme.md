# Longitudinal LLR aggregation

This package estimates an attacker's posterior confidence after observing several
trial utterances from the same speaker. It retains the existing sum and average
baselines, fits all new parameters on development speakers only, and writes the
final evaluation artifacts to:

```text
data/experiments/<experiment>/long/
```

Run the analysis with:

```bash
# Every experiment under data/experiments/
uv run src/long_leakage/run_longitudinal_analysis.py

# Selected experiments
uv run src/long_leakage/run_longitudinal_analysis.py \
  --experiments 26B3 26T10-2
```

The entry point calls the other modules in this directory. No extra server is
required for the interactive report; open
`data/experiments/<experiment>/long/interactive_longitudinal.html` directly.

## Notation and temperature convention

For trial `i`, let `l_i` be the vector of calibrated LLRs against all `N`
enrolment speakers. For a trial speaker with `k` observations, define:

```text
S = sum over i=1..k of l_i
p(y | observations, T) = softmax(S / T)
```

This implementation always uses the division convention:

```text
temperature T       = divisor applied to the summed logits
evidence scale alpha = 1 / T
```

Consequently, summing has `T=1`, while averaging `k` vectors has `T=k`. It is
also valid to describe averaging as multiplication by `1/k`, but that multiplier
is the inverse temperature rather than the temperature itself.

The target-speaker Local Information Disclosure is then:

```text
LID = log2(N * p_target)
```

Temperature scaling changes posterior confidence but not the enrolment ranking
or top-1 identity prediction. A temperature therefore cannot be selected using
accuracy. It must be selected with a proper probability score such as
multiclass negative log likelihood (NLL) or Brier score.

## Approach 1: summed LLR

Formula:

```text
z = S
T = 1
```

Adding LLRs is the Bayesian result when observations are conditionally
independent given each identity and the component LLRs are correctly specified.
Repeated utterances from one speaker normally share channel, content, speaker,
recording, and model effects. Positive dependence then makes the independence
product too sharp. The summed posterior often approaches zero or one much more
quickly than justified.

This remains a useful upper-evidence baseline and is written to `sum_llr/`.

## Approach 2: averaged LLR

Formula:

```text
z = S / k
T(k) = k
```

Averaging treats a set of `k` utterances as roughly one effective observation.
It prevents confidence from growing with the number of observations. That is
conservative when observations contain genuinely new information, and it has no
general probabilistic justification unless the dependence is effectively
complete.

This remains a lower-evidence baseline and is written to `average_llr/`.

## Approach 3: fitted global temperature

Formula:

```text
z = S / T
T > 0
```

One scalar `T` is fitted by minimizing speaker-balanced multiclass NLL on the
development split. Each development speaker contributes one final aggregate,
so speakers with more utterances do not dominate the objective. The fitted
parameter is then frozen before any test result is produced.

The main advantages are:

* only one fitted parameter;
* direct optimization of confidence quality;
* no change to the attacker's ranking decisions;
* a value between the sum and average scales can represent partial redundancy.

The limitation is that every test speaker receives the same `T` even when trial
counts or within-speaker redundancy differ. This is particularly restrictive
when `k` has a wide range.

The output is written to `temperature_scaled/`. The fitted value and its
leave-one-speaker-out stability range are in `calibration.json`.

### How the best T is found

The implementation performs a deterministic bounded search on a logarithmic
temperature scale. For each candidate `T`, it evaluates:

```text
NLL(T) = mean_s[-log softmax(S_s / T)[true_s]]
```

The minimum on all development speakers becomes the final fitted temperature.
The interactive report plots this development objective. It also reports
leave-one-development-speaker-out (LOO) NLL: for every held-out speaker, `T` is
refitted on all other development speakers and evaluated on that held-out
speaker. LOO is a model diagnostic; the full-development value is the parameter
applied once to test.

## Approach 4: count-adjusted degradation

Formula:

```text
T(k) = 1 + rho * (k - 1)
z = S / T(k)
0 <= rho <= 1
```

This is the standard design-effect form for `k` exchangeable observations with
common positive correlation `rho`. It supplies a rigorous interpolation:

```text
rho = 0  -> T(k) = 1 -> summed evidence
rho = 1  -> T(k) = k -> averaged evidence
```

Its effective number of independent observations is:

```text
k_effective = k / (1 + rho * (k - 1))
```

As `k` increases, the increment in `k_effective` decreases. This captures the
requested behavior that the first observation is valuable and each additional
observation provides less marginal information. It does so without privileging
an arbitrary "first" utterance: all observations remain exchangeable.

Only `rho` is fitted, using the same development-only NLL objective. The output
is written to `count_adjusted/`.

## Approach 5: similarity-adjusted degradation

Trial count is only a rough proxy for dependence. This method derives a
speaker-specific redundancy term from how similar the observed LLR patterns are.

First, each trial LLR vector is centered across enrolment candidates. Centering
is necessary because softmax is invariant to a common logit offset. Pairwise
cosine similarities are then computed and negative similarities are clipped to
zero:

```text
c_ij = max(cosine(center(l_i), center(l_j)), 0)
R = (2 / k) * sum over i<j of c_ij
T = 1 + rho * R
z = S / T
```

Important endpoints are:

```text
identical LLR patterns: R = k - 1
orthogonal/opposed patterns: R is approximately 0
```

Therefore, repeated near-duplicates are degraded strongly, while diverse score
patterns retain more of their summed evidence. The method is order-invariant and
fits only one bounded `rho` on development speakers.

This is an exploratory sensitivity model, not a proof that cosine similarity is
the true statistical correlation between utterances. Shared wrong rankings can
also appear similar, and unusual but noisy vectors can appear novel. The output
is written to `similarity_adjusted/`; per-speaker similarities, redundancy `R`,
and resulting temperatures are in `speaker_redundancy.csv`.

## Why there is no chronological geometric weighting

The current score files have a stable row order but no explicit interception
timestamp. Applying weights such as `1, gamma, gamma^2, ...` would make results
depend on an order that may only be a dataset or filename order. The
count-adjusted method gives diminishing marginal evidence without this arbitrary
choice, and the similarity method uses the set of observations symmetrically.

If actual timestamps become available, an online prequential model can be added
as a separate analysis. Its order field and tie handling should be explicit, and
`gamma` should still be fitted only on development data.

## Development and test data usage

The pipeline enforces this split:

```text
development:
  reconstruct development LLRs from z_score and calibration_parameters.json
  fit T or rho
  compute LOO diagnostics

test:
  apply the frozen development parameters
  compute target probabilities and LID
  populate the sensitivity slider
```

No method or parameter is selected automatically from test results. The slider
is deliberately labelled as sensitivity analysis; moving it must not be used to
claim a newly optimized evaluation result.

The same development set was already used to learn the trial-level score
calibration. Reusing it for one scalar longitudinal correction is pragmatic but
still creates finite-sample risk, especially with a small number of speakers.
The safeguards here are:

* one scalar parameter per fitted family;
* speaker-balanced rather than trial-balanced loss;
* leave-one-speaker-out diagnostics;
* parameter stability ranges;
* no test-based automatic selection.

Comparing many model families can itself overuse development data. For a formal
evaluation, pre-register global temperature as the primary method and treat
count/similarity adjustments as sensitivity analyses. A stronger future design
would use separate development subsets for model selection and final calibration,
or nested grouped cross-validation with more speakers.

## Deciding whether one temperature suffices

Use the development LOO columns in `calibration_diagnostics.csv`:

1. Compare multiclass NLL first. Lower is better and directly evaluates the
   claimed posterior confidence.
2. Use multiclass Brier as a secondary proper scoring rule.
3. Inspect the LOO parameter stability range in `calibration.json`. A wide range
   indicates that the development speakers do not determine the parameter well.
4. Check whether count- or similarity-adjusted LOO gains are large enough to
   justify their extra assumption.
5. Do not choose a method from test ALID, PDR, or the interactive test markers.

A single temperature is preferable when its LOO scores are comparable to the
extensions. The lower-complexity model is easier to estimate and explain. A
count-dependent correction becomes credible when it improves held-out
development NLL consistently and trial counts differ substantially.

## Interactive report

`interactive_longitudinal.html` is self-contained and has no CDN or Python
server dependency. It includes:

* the per-speaker distribution heatmap for individual trials;
* LID and target-probability views;
* overlays for direct mean, sum, average, fitted temperature, count adjustment,
  similarity adjustment, and the slider temperature;
* a logarithmic temperature slider and exact numeric input;
* live ALID, PDR, target probability, and LID maximum summaries;
* the development NLL objective with fitted and current temperatures;
* development fit and LOO calibration diagnostics.

Values outside the individual-trial heatmap domain are clamped visually to the
plot boundary. The tooltip always reports the actual unclamped value.

## Output layout

```text
long/
|-- calibration.json
|-- calibration_diagnostics.csv
|-- calibration_folds.csv
|-- temperature_sweep.csv
|-- method_comparison.csv
|-- speaker_redundancy.csv
|-- interactive_longitudinal.html
|-- sum_llr/                    # existing method
|-- average_llr/                # existing method
|-- temperature_scaled/
|-- count_adjusted/
`-- similarity_adjusted/
```

Each new method directory contains candidate-level `scores.csv`, target-level
`mated_probabilities.csv`, static before/after plots, and `summary.json` with the
development calibration record and final evaluation metrics.

Run all deterministic tests with:

```bash
uv run python -m unittest \
  tests/test_long_leakage.py \
  tests/test_longitudinal_calibration.py
```
