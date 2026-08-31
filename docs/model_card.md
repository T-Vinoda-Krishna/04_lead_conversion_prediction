# Model card

## Intended use
Rank leads for sales follow-up.

## Data
Synthetic, generated with a fixed seed.

## Metrics
ROC-AUC is used for model discrimination. For operations, review precision/recall and threshold-specific lift.

## Risks
- Synthetic data does not represent real sales behavior.
- Historical conversion patterns can encode bias.
- Probability calibration should be checked before production use.
