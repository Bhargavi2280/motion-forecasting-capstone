\# Baseline Results (validation set, 850 samples)



Metric definitions: minADE and minFDE in metres, miss = endpoint error > 4 m at 8 s,

coverage = 100 - miss%. Rare tags are defined in src/rare\_tags.py (v1, frozen).



\## Baseline A: constant velocity (K=1)

| subset | n | minADE | minFDE | miss% | coverage% |

|---|---|---|---|---|---|

| all | 850 | 10.00 | 27.22 | 89.3 | 10.7 |

| lane\_change | 71 | 10.55 | 28.55 | 94.4 | 5.6 |

| sudden\_stop | 49 | 17.53 | 46.73 | 100.0 | 0.0 |

| hard\_brake | 44 | 11.59 | 30.42 | 100.0 | 0.0 |

| extreme\_turn | 22 | 12.39 | 34.64 | 100.0 | 0.0 |

| vulnerable | 86 | 2.27 | 5.48 | 44.2 | 55.8 |

| any\_rare | 177 | 12.72 | 34.11 | 97.7 | 2.3 |



\## Baseline B: single-trajectory GRU (K=1, 60 epochs, seed 0)

Val ADE of the GRU at epoch 60: 6.66



| subset | n | minADE | minFDE | miss% | coverage% |

|---|---|---|---|---|---|

| all | 850 | 6.66 | 18.78 | 84.1 | 15.9 |

| lane\_change | 71 | 8.02 | 21.92 | 93.0 | 7.0 |

| sudden\_stop | 49 | 3.86 | 9.80 | 67.3 | 32.7 |

| hard\_brake | 44 | 8.83 | 24.70 | 97.7 | 2.3 |

| extreme\_turn | 22 | 9.73 | 29.45 | 100.0 | 0.0 |

| vulnerable | 86 | 2.18 | 5.36 | 44.2 | 55.8 |

| any\_rare | 177 | 7.25 | 19.99 | 87.6 | 12.4 |



\## Observations

\- The GRU improves overall minADE from 10.00 to 6.66 m.

\- A single predicted path gives 0% coverage on extreme\_turn and 2.3% on hard\_brake.

\- These gaps motivate the multi-modal reference model (O2) and the interaction model (O3).

\## Reference: multi-modal GRU (K=6, 60 epochs, seed 0)

val minADE@6 at epoch 60: 3.05

mode usage (top-ranked mode share): 0.11, 0.42, 0.04, 0.04, 0.36, 0.03



| subset | n | minADE | minFDE | miss% | coverage% |

|---|---|---|---|---|---|

| all | 850 | 3.05 | 7.79 | 65.8 | 34.2 |

| lane\_change | 71 | 3.40 | 8.26 | 80.3 | 19.7 |

| sudden\_stop | 49 | 1.50 | 2.45 | 16.3 | 83.7 |

| hard\_brake | 44 | 3.95 | 8.18 | 72.7 | 27.3 |

| extreme\_turn | 22 | 5.78 | 14.88 | 81.8 | 18.2 |

| vulnerable | 86 | 1.20 | 2.80 | 23.3 | 76.7 |

| any\_rare | 177 | 3.28 | 7.49 | 61.6 | 38.4 |



\## Resource profile (Tesla T4, multi-modal GRU)

\- parameters: <number>

\- inference: <number> ms per sample

\- peak GPU memory: <number> MB

\- training time: about 17 s for 60 epochs


## O3 interaction vs reference

| Model | dense_minADE | rare_cov (%) | turn_cov (%) |
|---|---|---|---|
| Reference | 3.04 ± 0.06 | 36.16 ± 2.79 | 10.91 ± 3.18 |
| Interaction | 2.93 ± 0.05 | 36.78 ± 2.77 | 13.18 ± 6.93 |

10 paired seeds, validation only, test untouched. Interaction lowers dense_minADE by 0.109 (about 3.6%) and wins on 9 of 10 seeds (paired t-test p = 0.0045). The gain is reliable but small and below the 5% KPI-3 target. Rare and turn coverage show no detectable change.

### O3 ablation (validation only, 10 seeds, dense scenes)

| Setting | dense_minADE | vs reference | wins vs ref | p vs ref | p vs full |
|---|---|---|---|---|---|
| Reference | 3.04 | n/a | n/a | n/a | n/a |
| Interaction, 0 neighbours | 3.072 | +0.028 | 5/10 | 0.38 | <0.0001 |
| Interaction, 4 neighbours | 3.012 | -0.032 | 6/10 | 0.25 | 0.0125 |
| Interaction, mean pooling (no attention) | 3.016 | -0.028 | 7/10 | 0.39 | 0.032 |
| Interaction, full (all neighbours + attention) | 2.935 | -0.109 | 9/10 | 0.0045 | n/a |

Masking all neighbours removes the entire gain (NT-3: interactions ignored), so the
improvement comes from neighbour information, not model capacity. The gain grows with
the number of neighbours, and replacing attention with mean pooling loses most of it.
Caveats: paired t-tests on 10 seeds without multiple-comparison correction; nb4 assumes
nearest-first neighbour ordering; the pooling variant also removes the target key.

## Final results (frozen config: K=20, relax 0.05, no oversampling)

Validation was used for all selection; test was run once.

### Validation, coverage experiments (interaction model, 5 seeds)
K=6 base rare_cov ~34-37%; relaxed WTA, endpoint-aware selection and rare oversampling gave no clear gain.
K=10 ~46%, K=20 ~60%. Larger K is the only lever that helped.

### Validation, K=20 reference vs interaction (5 seeds)
dense_minADE 2.17 vs 2.18 (p=0.83); rare_cov 52.99 vs 60.00 (p=0.071); turn_cov 25.45 vs 30.91 (p=0.40).

### Test (825 scenes, 164 rare, 34 extreme_turn; 10 paired seeds; run once)

| Metric | Reference | Interaction | Note |
|---|---|---|---|
| dense_minADE | 2.33 +/- 0.05 | 2.33 +/- 0.05 | no difference (p=0.92) |
| rare_cov (4 m) | 48.35 +/- 2.03 | 53.17 +/- 2.90 | +4.8 pts, higher on 9/10 seeds (p=0.004) |
| turn_cov | 16.76 +/- 3.41 | 18.53 +/- 5.20 | not detectable (p=0.19) |

Conclusion: interaction gives a small, consistent gain in rare-behaviour coverage at K=20
but no gain in overall error. The 95% coverage target was not met (53%). At K=6 interaction
lowered dense_minADE by 3.6% (p=0.0045), but that gain disappears at K=20.

### Negative tests
NT-1 collapse: (collapse to one path): not measured. NT-2: rare_cov 53% and turn_cov 18% on test, 6 m coverage reported as sensitivity.
NT-3: neighbours masked removes the K=6 gain (ablation). NT-4: out of scope for an offline dataset.
NT-5: error gain at K=6 did not come with a rare-coverage gain.

### Limitations
Coverage far below target; only 34 test turn samples; KPI-2, KPI-4, KPI-6 not measured;
t-tests unadjusted for multiple comparisons; larger K raises best-of-K mechanically.
