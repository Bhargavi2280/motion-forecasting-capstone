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

