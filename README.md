# FairRate: Context-Aware Reputation Updates for Fairer Ride-Hailing Platforms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FairRate is a prototype post-trip reputation update pipeline. It keeps the observed star rating as the starting point and applies a bounded additive correction built from trip context, earlier rater history, review category, and trip telemetry.

## Repository contents

| File | Purpose |
|---|---|
| `trip_reviews.csv` | NYC TLC High Volume For Hire Vehicle trips (January to September 2025) with joined hourly weather, synthetic driver identities, reference ratings, and initial reviews |
| `Biased_rideshare_dataset.ipynb` | Creates 800 synthetic riders, assigns them to trips, applies eight rating shift rules, rewrites reviews with fixed sentences, and writes `biased_trip_reviews_final.xlsx` and `riders.csv` |
| `FairRate_Notebook_v3.ipynb` | Full FairRate pipeline, evaluation, ablation, held out time test, sensitivity, and figures |
| `FairRate_vs_Baselines.ipynb` | Comparison with six signal blind reputation aggregators at trip and driver level |
| `all_figures/` | Figures produced by the notebooks |
| `outputs/` | Result tables produced by `FairRate_Notebook_v3.ipynb` |

Run the notebooks from the repository root in this order: `Biased_rideshare_dataset.ipynb`, `FairRate_Notebook_v3.ipynb`, `FairRate_vs_Baselines.ipynb`. All random draws use seed 42.

## Pipeline

1. **Context model.** CatBoost predicts the observed rating from trip, fare, time, and weather fields. Rider attributes are excluded.
2. **Rater score.** Computed only from each rider's earlier trips (extreme ratings, text and rating disagreement, concentration on one driver, mean past context residual, and number of earlier trips). No persona, demographic, or tier label is used.
3. **Review category.** Keyword labels followed by a TF-IDF logistic regression assign Driver, Platform, Environment, or Unknown.
4. **Telemetry score.** Weighted combination of scaled wait time, cost shock (surcharges divided by fare), surcharges, and speed.
5. **Correction and reputation.** Bounded additive correction, final clip to 1 to 5 stars, and an exponential moving average with a per trip cap of 0.05.

## Results (semi-synthetic evaluation)

The reference rating is a synthetic target. Results measure recovery of that target and do not establish fair treatment of real drivers.

| Measure | Value |
|---|---|
| MAE, observed rating vs reference | 0.2779 |
| MAE, FairRate vs reference (all 9,158 trips) | 0.2437 (12.3% lower) |
| Trips with lower error / higher error | 56.6% / 43.4% |
| Held out time test (weights from first 70%, tested on last 15%) | 0.2724 to 0.2436 (10.6% lower) |
| Selected weights (alpha, beta, gamma) | 0.2, 0.6, 0.8 |
| Driver level MAE, FairRate vs best baseline | 0.1431 vs 0.1085 |


