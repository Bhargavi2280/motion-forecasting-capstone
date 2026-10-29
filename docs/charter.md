\# Project Charter: AIML-104 Multi-Modal Motion Forecasting for Road Users with Rare Behaviours



\## 1. Problem and beneficiary

Predicting road-user motion requires representing ambiguity. A single most-likely

path discards rare manoeuvres a safe planner must anticipate. Source: Waymo Open

Motion Dataset (issuer: Waymo; catalogue ID AIML-104). The KL deployment

stakeholder is to be confirmed with the faculty guide.



\## 2. Source-to-claim map

| Claim | Evidence |

|---|---|

| Single-path prediction misses rare behaviours | Baseline B: 0% coverage extreme\_turn, 2.3% hard\_brake (docs/baseline\_results.md) |

| Multi-modal output improves rare-behaviour coverage | Reference K=6: 18.2% / 27.3% on the same subsets |

| Interaction modelling helps (to be tested) | O3 ablation, Semester VIII |



\## 3. Scope (offline, simulation-only)

Included: data pipeline, frozen rare-behaviour tags, baselines, multi-modal

reference (O2), interaction model (O3), runnable demo tool with fallback (O4),

calibration and acceptance report (O5).

Excluded: real vehicle or hardware-in-loop testing, production certification,

generalisation beyond the Waymo subset used.



\## 4. Data and splits (see docs/evidence\_manifest.md)

Train: 3 shards (4556 samples). Validation: 1 shard (850, for tuning).

Test: 1 shard (825, frozen, used only at final acceptance).



\## 5. Frozen definitions

\- Rare tags v1: src/rare\_tags.py (thresholds in the manifest).

\- Miss: endpoint error > 4 m at 8 s. Coverage = share of samples with at least one

&#x20; of K=6 modes within 4 m of the true endpoint. Rare-behaviour coverage = coverage

&#x20; on samples carrying any rare tag.



\## 6. KPIs (proposed scope revision, replacing control-system KPIs)

| KPI | O2 reference (val) | Proposed target |

|---|---|---|

| Rare-behaviour coverage (any\_rare) | 38.4% | at least reference + 5 points |

| minADE@6 (all) | 3.05 m | not worse than reference |

| Calibration (ECE of mode probabilities) | to measure in O5 | improved vs reference |

| Interaction quality (minADE in dense scenes) | to measure in O3 | at least 5% relative improvement |

| Fallback success | n/a | 100% on corrupted-input fixtures |



\## 7. Requirements (summary)

FR: load and validate data; reproduce reference with one command; interaction

module separately testable; demo tool with constant-velocity fallback; acceptance

harness for all KPIs and negative tests.

NFR: version-pinned, tests passing, logs and configs versioned, test set untouched

until final acceptance.



\## 8. Risk register

| Risk | Mitigation |

|---|---|

| Small data subset limits accuracy | Report uncertainty over 3 seeds; add shards if time allows |

| Rare tags are heuristic, not ground truth | State this; keep definitions frozen |

| 4 m threshold is strict | Report minADE and minFDE alongside coverage |

| Interaction gives no gain | Report as a defensible negative result |

| Colab session limits | Data and checkpoints on Drive, code in GitHub |



\## 9. Architecture decision record

Reference model: GRU encoder with 6-mode head and winner-takes-all loss. Chosen

over the single-path GRU because it raises rare-behaviour coverage (see

docs/baseline\_results.md).

