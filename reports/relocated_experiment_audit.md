# Relocated Wide-Area Weak-Target Experiment Audit

- Truth: `data/active/radar_truth_relocated`
- Detections: `data/active/sim_detections_relocated`
- Coverage range audited against: **100 km**

## Interpretation

This is a **wide-area** weak-target experiment, not a range-contained
radar-coverage experiment:

- Relocation anchors each trajectory's **start** near the radar (first
  sample in the 10-80 km anchor band); after that first sample the
  aircraft follows its original ADS-B-derived motion unchanged, so the
  **full trajectories are retained** and long flights drift well beyond
  the anchor band.
- Stage 6's `--max-range-m 100000` defines the spatial
  **clutter support** only -- where false alarms are generated. It does
  **not** remove target truth rows beyond 100 km.
- Targets beyond 100 km stay in the dataset and simply become
  weaker: the range-decay SNR model hands them progressively lower SNR
  and therefore lower detection probability.
- This setup is suited to studying threshold trade-offs and weak-target
  tracking over long trajectories. A future range-contained experiment
  would need stage-5 post-relocation target filtering or a stage-6
  target-range gate.

## Stage 05 truth summary

| Metric | Value |
|---|---|
| Truth files | 4 |
| Truth rows | 9,676,882 |
| Trajectories | 77,520 |
| Relocated fraction | 1.000 |
| range_m p50/p95/p99 | 59.3 / 194.0 / 358.6 km |
| elevation_deg p50/p95/p99 | 0.92 / 4.67 / 10.41 deg |
| First-sample ground range p50/p95/p99 | 45.0 / 76.6 / 79.4 km |
| First sample inside 10-80 km anchor band | 1.0000 |

## Stage 06 detection summary

| Metric | Value |
|---|---|
| Detection files | 24 |
| Total detections | 33,953,490 |
| Target detections | 30,750,194 |
| Clutter detections | 3,203,296 |

Per-threshold sweep (empirical Pd and false alarms averaged across days;
detections summed):

| threshold (dB) | empirical Pd | false alarms/frame | target detections | clutter detections |
|---:|---:|---:|---:|---:|
| -5 | 0.791 | 46.00 | 7,659,548 | 1,589,745 |
| 0 | 0.666 | 20.02 | 6,445,250 | 691,792 |
| 3 | 0.571 | 12.12 | 5,526,513 | 418,691 |
| 6 | 0.472 | 7.37 | 4,569,516 | 254,808 |
| 9 | 0.379 | 4.46 | 3,668,032 | 154,280 |
| 12 | 0.298 | 2.72 | 2,881,335 | 93,980 |

## Coverage-range audit

How much of the experiment lives beyond the 100 km clutter-support
radius (retained by design, at reduced SNR):

| Metric | Value |
|---|---|
| Fraction of truth rows beyond 100 km | 0.1901 |
| Fraction of trajectories ever beyond 100 km | 0.2150 |
| Fraction of target detections beyond 100 km | 0.0181 |

## Notes for Stage 07

Stage 07 should evaluate threshold-only detection performance both
overall and, optionally, split by range bin:

- 0-50 km
- 50-100 km
- 100-200 km
- > 200 km

The per-bin split will make the effect of the range-decay SNR model
directly visible: near bins should show high Pd at every threshold,
far bins should collapse toward the Pd floor.
