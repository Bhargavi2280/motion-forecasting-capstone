\# Evidence Manifest



\## Source

\- Dataset: Waymo Open Motion Dataset

\- URL: https://waymo.com/open

\- Bucket: waymo\_open\_dataset\_motion\_v\_1\_3\_1, folder uncompressed/scenario

\- Licence: Waymo Dataset License Agreement for Non-Commercial Use

\- Access date: 30 Oct 2026



\## Files used

| Role | File |

|---|---|

| Train | training/training.tfrecord-00000-of-01000 |

| Train | training/training.tfrecord-00001-of-01000 |

| Train | training/training.tfrecord-00002-of-01000 |

| Validation (tuning) | validation/validation.tfrecord-00000-of-00150 |

| Test (frozen, not used until final acceptance) | validation/validation.tfrecord-00001-of-00150 |



\## Samples (target agents with full valid future)

\- Train: 4556

\- Validation: 850

\- Test: TO FILL IN after building the test set



\## Rare-behaviour tags v1 (frozen, see src/rare\_tags.py)

| Tag | Rule | Train | Val |

|---|---|---|---|

| extreme\_turn | net heading change > 100 degrees over 8 s | 3.8% | 2.6% |

| hard\_brake | deceleration > 5.5 m/s^2 over 0.5 s | 5.0% | 5.2% |

| sudden\_stop | speed >= 5 m/s now, < 0.5 m/s at the end | 5.7% | 5.8% |

| lane\_change | lateral shift 3 to 5 m, net turn < 15 degrees, path > 25 m | 5.8% | 8.4% |

| vulnerable | pedestrian or cyclist | 11.7% | 10.1% |



Any rare tag (excluding vulnerable): 19.2% of train, 20.8% of val.



\## Notes

\- Thresholds were set from the training distribution, then frozen.

\- Coverage KPI uses a 4 m endpoint miss threshold (src/metrics.py).

