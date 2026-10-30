# College Football Win Probability Model — V13

Independent PyTorch college football forecasting project using historical FBS results and opponent-adjusted team statistics.

## Reported evaluation (exploratory)
- 2024–2025 test period: **1,606 games**
- Neural network accuracy: **70.73%**
- ROC-AUC: **0.7705**
- Log loss: **0.5591**
- 2023 validation log loss: **0.5477**
- Expanded logistic regression test log loss: **0.5586** (slightly better than NN's 0.5591)

**Caution:** Multiple model iterations were inspected on the same 2024–2025 test period. These are exploratory retrospective results, not an untouched final holdout. Historical coverage, 2026 data, and injury/roster availability require independent verification. No claim of outperforming betting markets is made.

## Features and methodology
- Chronological Elo ratings with margin-of-victory adjustment, 65% offseason retention, and home-field effect.
- Opponent-adjusted scoring, offensive and defensive efficiency, yards per play, third-down performance, turnovers, and recent form.
- Logistic regression and neural-network baselines.
- Pregame features constructed chronologically to reduce look-ahead bias.
- 2023 validation and 2024–2025 test; training uses earlier historical seasons.
- See source scripts for exact feature definitions, fitting, and splitting logic.

## Reproduce the Oct 31, 2026 predictions
Requires the two original datasets **not bundled here** and the October 25 AP poll CSV:
- `asu-big12-cfb-dataset-v2.zip`
- `cfb-data-2026-verified.zip`
- `ap_top25_2026_10_25.csv`

Install dependencies and run from the project root:

```bash
python -m pip install -r requirements.txt
python src/predict_top25_week.py \
  --zip /path/to/asu-big12-cfb-dataset-v2.zip \
  --stats-zip /path/to/cfb-data-2026-verified.zip \
  --rankings /path/to/ap_top25_2026_10_25.csv \
  --output top25_week_predictions.csv
```

`results/top25_week_predictions.csv` contains the saved forecast snapshot; predictions may vary slightly with Python/library versions and input data revisions.

## Limitations
- No direct injury reports, quarterback availability, transfer updates, or lineup information.
- Elo retains historical team-strength effects; lower retention was tested and performed worse on historical validation.
- No guaranteed calibration against current betting markets; betting odds were not model inputs.
- Schedule, results, and team statistics are third-party data and should be checked before publication.
- Not intended as wagering advice.

## Data and licensing
This repository does not redistribute the source game/statistics datasets. Obtain permission and verify the applicable rights and terms before distributing third-party data. The rankings and odds source files are not included.

## Files
- `src/cfb_asu_v13.py`: V13 feature engineering, training, and model evaluation.
- `src/predict_top25_week.py`: National matchup predictions and AP-ranked filtering.
- `results/top25_week_predictions.csv`: Saved 18-game prediction snapshot.
- `results/top25_predictions_linkedin.png`: Shareable prediction graphic.
