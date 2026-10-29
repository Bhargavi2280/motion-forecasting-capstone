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

