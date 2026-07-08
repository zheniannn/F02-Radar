# Stage 07 Threshold-Only Baseline Evaluation

## Experiment definition

- This is a **threshold-only** point-detection baseline: each stage-6
  detection stream is evaluated as-is, frame by frame, against the
  stage-5 truth denominators.
- **No tracking**, data association, or trajectory smoothing is used.
- The current dataset is the **relocated wide-area weak-target
  experiment**: trajectory starts are anchored 10-80 km from the radar,
  full trajectories are retained, and long flights drift far beyond the
  anchor band.
- Stage 6's `--max-range-m 100000` controlled **clutter
  support**, not target containment: far targets remain and receive
  lower SNR through the range-decay model.
- The stage-6 summary reports a relocated truth fraction of 1.000; detection files carry relocation metadata columns.

## Overall operating curve

| threshold_db | truth_rows | target_detections | clutter_detections | missed_targets | empirical_pd | false_alarm_per_frame | target_fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 9,676,882 | 7,659,548 | 1,589,745 | 2,017,334 | 0.7915 | 46.0049 | 0.8281 |
| 0 | 9,676,882 | 6,445,250 | 691,792 | 3,231,632 | 0.6660 | 20.0194 | 0.9031 |
| 3 | 9,676,882 | 5,526,513 | 418,691 | 4,150,369 | 0.5711 | 12.1163 | 0.9296 |
| 6 | 9,676,882 | 4,569,516 | 254,808 | 5,107,366 | 0.4722 | 7.3738 | 0.9472 |
| 9 | 9,676,882 | 3,668,032 | 154,280 | 6,008,850 | 0.3791 | 4.4646 | 0.9596 |
| 12 | 9,676,882 | 2,881,335 | 93,980 | 6,795,547 | 0.2978 | 2.7196 | 0.9684 |

## Per-day consistency

Empirical Pd per day and threshold (columns = threshold dB):

| date | -5.0 | 0.0 | 3.0 | 6.0 | 9.0 | 12.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | 0.7911 | 0.6650 | 0.5696 | 0.4704 | 0.3774 | 0.2960 |
| 2022-06-13 | 0.7892 | 0.6673 | 0.5739 | 0.4760 | 0.3830 | 0.3021 |
| 2022-06-20 | 0.7940 | 0.6685 | 0.5729 | 0.4739 | 0.3812 | 0.2995 |
| 2022-06-27 | 0.7910 | 0.6627 | 0.5676 | 0.4681 | 0.3742 | 0.2930 |

## Range-bin detection probability

Empirical Pd by truth range bin (rows) and threshold (columns), summed
across days:

| range_bin | -5.0 | 0.0 | 3.0 | 6.0 | 9.0 | 12.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0-50 km | 0.9793 | 0.9730 | 0.9555 | 0.9095 | 0.8208 | 0.6945 |
| 50-100 km | 0.8866 | 0.6560 | 0.4577 | 0.2677 | 0.1289 | 0.0534 |
| 100-200 km | 0.2535 | 0.0654 | 0.0266 | 0.0143 | 0.0111 | 0.0102 |
| >200 km | 0.0130 | 0.0103 | 0.0100 | 0.0101 | 0.0099 | 0.0101 |

## Clutter by measured range bin

Clutter false alarms by *measured* range bin (counts, summed across
days). Clutter is uniform in range within the configured support, so
wider bins collect proportionally more:

| range_bin | -5.0 | 0.0 | 3.0 | 6.0 | 9.0 | 12.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0-50 km | 794,762 | 345,401 | 209,011 | 127,357 | 77,036 | 46,984 |
| 50-100 km | 794,983 | 346,391 | 209,680 | 127,451 | 77,244 | 46,996 |
| 100-200 km | 0 | 0 | 0 | 0 | 0 | 0 |
| >200 km | 0 | 0 | 0 | 0 | 0 | 0 |

## SNR distribution of written detections

SNR is summarized **only for written detections**. Stage 6 does not
write missed-target rows, so missed-target SNR values are not
available -- **this is not Pd-by-SNR**. Quantiles are computed from a
capped deterministic sample of up to 200,000 values per
group (seed 123); counts are exact.

| threshold_db | is_target | count | snr_p10_db | snr_p50_db | snr_p90_db | snr_mean_db | sampled_for_quantiles |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 0 | 1,589,745 | -4.36 | -0.82 | 8.81 | 1.01 | 1 |
| -5 | 1 | 7,659,548 | -1.78 | 7.78 | 24.42 | 9.43 | 1 |
| 0 | 0 | 691,792 | 0.63 | 4.13 | 13.75 | 5.97 | 1 |
| 0 | 1 | 6,445,250 | 1.33 | 9.91 | 25.91 | 11.57 | 1 |
| 3 | 0 | 418,691 | 3.63 | 7.17 | 16.82 | 9.01 | 1 |
| 3 | 1 | 5,526,513 | 3.34 | 11.70 | 27.14 | 13.19 | 1 |
| 6 | 0 | 254,808 | 6.63 | 10.15 | 19.79 | 11.99 | 1 |
| 6 | 1 | 4,569,516 | 5.58 | 13.83 | 28.64 | 15.08 | 1 |
| 9 | 0 | 154,280 | 9.64 | 13.16 | 22.78 | 15.01 | 0 |
| 9 | 1 | 3,668,032 | 7.95 | 16.16 | 30 | 17.07 | 1 |
| 12 | 0 | 93,980 | 12.64 | 16.15 | 25.76 | 17.98 | 0 |
| 12 | 1 | 2,881,335 | 10.40 | 18.60 | 30 | 19.08 | 1 |

## Measurement noise sanity check

Target measurement errors (mean/std exact; percentiles from the capped
sample). These should match the stage-6 noise configuration
(sigma_range 75 m, sigma_az/el 0.15 deg = 2.62e-3 rad, sigma_rv 2 m/s):

| threshold_db | count | range_error_mean_m | range_error_std_m | range_abs_error_p50_m | range_abs_error_p95_m | azimuth_abs_error_p50_rad | azimuth_abs_error_p95_rad | elevation_abs_error_p50_rad | elevation_abs_error_p95_rad | rv_abs_error_p50_mps | rv_abs_error_p95_mps | sampled_for_quantiles |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 7,659,548 | -0.01888 | 75.01 | 50.52 | 146.9 | 0.001766 | 0.005121 | 0.001772 | 0.005142 | 1.352 | 3.922 | 1 |
| 0 | 6,445,250 | 0.007568 | 74.97 | 50.3 | 146.9 | 0.001757 | 0.005138 | 0.001764 | 0.005132 | 1.356 | 3.928 | 1 |
| 3 | 5,526,513 | -0.01962 | 75.01 | 50.61 | 146.8 | 0.001766 | 0.005121 | 0.001768 | 0.005156 | 1.346 | 3.914 | 1 |
| 6 | 4,569,516 | -0.01884 | 75.03 | 50.69 | 146.4 | 0.00176 | 0.005115 | 0.001767 | 0.005126 | 1.343 | 3.916 | 1 |
| 9 | 3,668,032 | -0.003471 | 75.01 | 50.52 | 147.4 | 0.001757 | 0.005123 | 0.001766 | 0.005124 | 1.353 | 3.932 | 1 |
| 12 | 2,881,335 | -0.05446 | 75 | 50.56 | 146.8 | 0.001767 | 0.00514 | 0.001766 | 0.005131 | 1.352 | 3.907 | 1 |

## Interpretation

- Lower thresholds recover substantially more targets but admit
  proportionally more clutter; higher thresholds suppress clutter but
  miss weak targets.
- Range decay makes farther targets systematically less detectable:
  the range-bin table shows Pd collapsing with distance at every
  threshold.
- A threshold alone cannot separate weak far targets from clutter --
  which motivates Stage 08 tracking / physics-guided path scoring on
  the low-threshold detection stream.

## Recommended next stage

Stage 08 should implement a **low-threshold detect-then-track
baseline** -- starting with a constant-velocity Kalman filter and
gating -- before any ML model.
