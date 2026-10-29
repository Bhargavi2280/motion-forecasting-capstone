# Multi-Modal Motion Forecasting for Road Users with Rare Behaviours (AIML-104)

Capstone project: multi-modal trajectory prediction (K modes) on Waymo Open Motion data,
comparing a reference GRU model (O2) with an interaction-aware model using neighbour
attention (O3).

## Frozen configuration
K=20 modes, winner-takes-all loss with relax=0.05, no rare oversampling, Adam lr 1e-3,
60 epochs, batch size 64. Chosen on validation only. Test was run once.
Miss threshold for coverage: 4.0 m at the 8 s endpoint (frozen project definition).

## Code
- `src/models.py`: GRUMulti (reference), GRUInteract (interaction), losses, batching
- `src/data.py`, `src/rare_tags.py`: data loading and rare-behaviour tags
- `src/metrics.py`: minADE, minFDE, coverage
- `src/train.py`: reference training script
- `tests/test_core.py`: unit tests

## Results
See `docs/baseline_results.md` (tables, ablation, test results, limitations)
and the raw per-seed JSON files in `docs/`.

Headline (test, 10 seeds, K=20): dense_minADE 2.33 vs 2.33 (no difference);
rare coverage 48.4% (reference) vs 53.2% (interaction); the 95% target was not met.

## Not done / limitations
KPI-2 (calibration), KPI-4 (recovery) and KPI-6 (safety violations) were not measured.
Only 34 extreme_turn samples in test. t-tests are not corrected for multiple comparisons.
Larger K raises best-of-K metrics mechanically.

## Reproduce
Install `requirements.txt`, place the data pickles in `data/`, then run `python src/train.py`
for the reference model. Seeds 0-9 were used for the reported comparisons.
