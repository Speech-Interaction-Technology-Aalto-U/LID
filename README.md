# Local Information Disclosure (LID)

This repository contains the evaluation framework for the paper [*Goodbye Equal Error Rate, Hello Local Information Disclosure: Evaluating Voice Anonymisation against 1-to-N Linkage Threats*](https://arxiv.org/abs/2607.06259).

This codebase implements a modular, information-theoretic privacy metric explicitly designed for 1-to-$N$ linkage threats to evaluate voice anonymization privacy. It takes speaker embeddings produced by an attacker, and calculates how much identity information is disclosed per each trial. These scores are then aggregated to compute final systemic privacy metrics, including Average Local Information Disclosure (ALID), Positive Disclosure Rate (PDR), Expected Positive Leakage ($\text{LID}^+$), and Worst-Case Leakage ($\text{LID}_{\max}$).

## The Core Privacy Evaluation Idea

Speech privacy is a measurable outcome of the interaction between three distinct components:
1. **The Protector:** The anonymization mechanism applied to the speech.
2. **The Attacker:** The specific linkage model (e.g., an ECAPA-TDNN feature extractor) and their assumed knowledge.
3. **The Dataset:** The underlying speech data.

Change any of these three, and the evaluation outcome changes. This code base evaluates a single snapshot of interaction between the three.

## Data & Components

Our framework requires two distinct datasets with non-overlapping speaker identities:
* A **development set** ($\mathcal{D}_\text{dev}$) to learn the logistic calibration parameters.
* An **evaluation set** ($\mathcal{D}_\text{eval}$) to compute the final privacy disclosure metrics.

Each dataset must be partitioned into an **enrolment set** ($\mathcal{E}$) containing known identity profiles, and a **trial set** ($\mathcal{T}$) representing intercepted audio samples. 

These partition definitions must be placed in `data/shared/` as four distinct CSV files: `dev_enrolls.csv`, `dev_trials.csv`, `test_enrolls.csv`, and `test_trials.csv`. Each CSV requires two columns: `utterance_id` and `speaker_id`.

The pipeline treats every subfolder within `data/experiments/` as a distinct evaluation experiment. For each experiment, the pipeline expects one embedding file representing the output of the attacker's model:
`data/experiments/<experiment>/embeddings/embeddings.parquet`

*Required columns:* `utterance_id`, `speaker_id`, `embedding`.

## Quick Start

We use [`uv`](https://docs.astral.sh/uv/). 

```bash
# Install uv if you do not have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and execute the full evaluation pipeline
uv sync
uv run src/run_all_results.py
```

*Note: Intermediate pipeline outputs are intentionally `.gitignore`d, as they are fully reproducible via the command above.*

## Evaluating Custom Experiments

To evaluate your own anonymization systems, follow this workflow:

**1. Prepare your embeddings**
First, anonymize your speech data. Next, simulate your attacker (e.g., using an ECAPA-TDNN model) to extract embeddings for every anonymized speech sample. 

**(Note: If simulating a semi-informed attacker, ensure the attacker model is fine-tuned on an anonymized dataset strictly disjoint from the data used during this evaluation).**

**2. Structure your data**
Clear the existing `data/experiments/` directory. Create a new folder for your experiment and save your embeddings as a `.parquet` file. The pipeline expects the following structure:

```text
data/
├── shared/                  # Your CSV split definitions
└── experiments/
    └── my_custom_system/    # Name your experiment
        └── embeddings/
            └── embeddings.parquet
```

**3. Prepare Baselines (Optional but Recommended)**
To properly contextualize your privacy metrics, it is best practice to compare your system against bounds:

* **Random Baseline (Perfect Privacy):** If you wish to compare your results against theoretically unlinkable embeddings, run the following tool:
  ```bash
  uv run src/tools/generate_random_embeddings.py my_custom_system
  ```
  This creates a new experiment folder populated with randomly generated embeddings that match the exact dimensional structure of your `my_custom_system` experiment.
* **Plain Baseline (No Privacy):** We recommend running your attacker against the original, non-anonymized audio. Place these embeddings in a `data/experiments/plain/` directory to establish the maximum possible empirical privacy loss for your dataset.

**4. Run the evaluation**
Execute the pipeline to generate all metrics and plots for your new experiment:
```bash
uv run src/run_all_results.py
```



## Pipeline Architecture

The script `run_all_results.py` performs the evaluation:

1. **`1_average_enrolment_embeddings.py`**: Aggregates enrolment utterances ($\mathcal{E}$) to form stable target identity profiles for both $\mathcal{D}_\text{dev}$ and $\mathcal{D}_\text{eval}$.
2. **`2_create_scores_from_embeddings.py`**: Computes the full raw similarity score matrices between all trial utterances ($\mathcal{T}$) and enrolment profiles.
3. **`alternative_metrics/1_run_alternative_metrics.py`**: Computes standard baselines (EER, Cllr) for comparison.
4. **`3_calibrate_scores.py`**: Performs row-wise $z$-normalization and learns the logistic calibration weights ($w, b$) on $\mathcal{D}_\text{dev}$, then the calibration is applied to scores in $\mathcal{D}_\text{eval}$.
5. **`4_inner_aggregator.py`**: Computes the exact Local Information Disclosure ($\text{LID}_i$) in bits for every individual trial $i$.
6. **`5_outer_aggregator.py`**: Distills the trial-level $\text{LID}_i$ values into global risk metrics (ALID, PDR, $\text{LID}^+$, $\text{LID}_{\max}$).
7. **`6_create_plots_and_summaries.py`**: Generates publication-ready artifacts and other insightful graphs.

## Outputs

During execution, the pipeline stores any intermediate results (averaged embeddings, calibration parameters, similarity scores, and local disclosures) inside the `data/` folder.

The final evaluation artifacts are placed into `results/`. We export figures in both PNG format (for quick visual inspection) and PDF format (for LaTeX integration):

* **`results/summary/summary_table.csv`**: Aggregated performance metrics across all evaluated experiments.
* **`results/paper_figures/`**: Formatted PDFs and CSVs corresponding exactly to the paper's figures and tables.
* **`results/experiments/<experiment>/`**:
  * `results.json`: Raw numeric outputs for the specific system.
  * `plots/`: Distributions of posterior probabilities and local information disclosures (just like Figures 2 and 3 in the paper).
  * `alternative_metrics/`: EER/Cllr results and visual representations of mated vs. non-mated score separation.


## Citation

If you use this framework or code in your work, please cite:

```bibtex
@article{sterns2026goodbye,
  title={Goodbye Equal Error Rate, Hello Local Information Disclosure: Evaluating Voice Anonymisation against 1-to-N Linkage Threats},
  author={{\v{S}}terns, D{\=a}vis and Drossos, Konstantinos and Fernandes, Natasha and B{\"a}ckstr{\"o}m, Tom and Palamidessi, Catuscia},
  journal={arXiv preprint arXiv:2607.06259},
  year={2026}
}
```

## Support & Feedback

Encountered a bug, have a question, or a suggestion? You can open an issue on GitHub or reach out to the authors directly.

## Disclaimer

This codebase is provided "as is" for academic research. The authors hold no liability for software faults, unintended consequences, or real-world privacy breaches resulting from its use. This tool should not be solely relied upon for legal compliance audits (e.g., GDPR) in real-world production deployments without independent verification.
