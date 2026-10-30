# Technical overview — V13 college football forecasting

## Objective
Estimate pregame win probabilities for FBS matchups using publicly sourced historical results and team-level statistics, without using sportsbook odds as model inputs.

## Feature design
Chronological Elo (margin of victory, home field, offseason carryover) provides a compact team-strength prior. Opponent-adjusted offense and defense, yards per play, turnover statistics, third-down efficiency, and recent form provide additional information about how a team wins or loses. Feature computation is chronological so future games do not contribute to earlier pregame features.

## Models
The project evaluates an Elo-only logistic baseline, a broader logistic baseline, expanded logistic regression, and an expanded neural network. The expanded neural network was selected using 2023 validation log loss.

## Reported results
On 1,606 games from 2024–2025, the expanded neural network achieved 70.73% accuracy, 0.7705 ROC-AUC, and 0.5591 log loss. The expanded logistic regression achieved 70.11% accuracy, 0.7701 ROC-AUC, and 0.5586 log loss. Logistic regression had slightly better test probability loss, despite the neural network being selected on validation.

## Elo retention ablation
Offseason retention of 0%, 35%, and 65% was tested. Validation log losses were 0.5674, 0.5499, and 0.5477 respectively. The 65% configuration was retained based on validation performance.

## External benchmark
For the Oct 31 ranked schedule, the model and a supplied BetOnline no-vig odds snapshot selected the same winner in 17/18 matchups, with a reported mean absolute probability gap of about 6.9 percentage points. Agreement with favorites is not predictive accuracy, and sportsbook odds are not ground truth.

## Limitations and next steps
Repeated use of the same 2024–2025 test period means reported performance is exploratory. Real-time injuries, depth-chart changes, transfers, and quarterback availability are not explicitly represented. A prospective, frozen-forecast evaluation and an identical-game closing-line benchmark would be the appropriate next tests. The project is educational and not betting advice.
