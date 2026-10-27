# Cold-start contrast and ASA IV–V operating characteristics (saved predictions only)

## A. Cold start: calibrated decision model − LoRA (local validation set, n = 2,753; T1)

Label-matched: for each LoRA seed, the three cutpoints are fitted on exactly the n development labels that LoRA was trained on (`dev.sample(n, random_state=seed)`, as in 08_lora.py). Random-pool: 100 calibration draws of n labels from the 1,000-label pool (seed 11); each of 2,000 resamples draws patient clusters, one calibration draw and one LoRA seed. Intervals: patient-level cluster bootstrap.

| n | Contrast | κ calibrated | κ LoRA | Difference (95% CI) |
|---|---|---|---|---|
| 100 | Label-matched, seed 20261017 | 0.507 | 0.461 | +0.047 (+0.019 to +0.074) |
| 100 | Label-matched, seed 11 | 0.467 | 0.338 | +0.129 (+0.092 to +0.165) |
| 100 | Label-matched, seed 12 | 0.512 | 0.469 | +0.043 (+0.014 to +0.072) |
| 100 | Label-matched, three-seed mean | 0.496 | 0.423 | +0.073 (+0.046 to +0.099) |
| 100 | Random pool (label sampling + seed variation) | 0.518 | 0.423 | +0.095 (0.000 to +0.214) |
| 300 | Label-matched, seed 20261017 | 0.540 | 0.541 | -0.001 (-0.027 to +0.024) |
| 300 | Label-matched, seed 11 | 0.543 | 0.556 | -0.013 (-0.034 to +0.007) |
| 300 | Label-matched, seed 12 | 0.517 | 0.530 | -0.013 (-0.038 to +0.013) |
| 300 | Label-matched, three-seed mean | 0.533 | 0.542 | -0.009 (-0.028 to +0.008) |
| 300 | Random pool (label sampling + seed variation) | 0.532 | 0.542 | -0.011 (-0.047 to +0.025) |

| n | Contrast (MOVER subset, n = 4,000) | κ calibrated | κ LoRA | Difference (95% CI) |
|---|---|---|---|---|
| 100 | Random pool (label sampling + seed variation) | 0.509 | 0.410 | +0.099 (+0.043 to +0.142) |
| 300 | Random pool (label sampling + seed variation) | 0.520 | 0.411 | +0.110 (+0.003 to +0.285) |

## B. ASA IV–V operating characteristics

Mean (min–max) over the 50 calibration pools of Supplementary S26 (same random generator) or over three LoRA seeds. Flags per true positive = predicted IV–V ÷ true IV–V among them (number needed to flag).

| Data | Arm | κ | Accuracy | IV–V recall | IV–V precision | Predicted IV–V | True positives | False positives | Flagged per 1,000 | Flags per true positive |
|---|---|---|---|---|---|---|---|---|---|---|
| Local val (IV–V n = 79) | κ-fitted cutpoints, 50 pools | 0.537 (0.514–0.546) | 0.693 (0.634–0.732) | 0.29 (0.10–0.51) | 0.26 (0.14–0.62) | 112 (13–277) | 23 (8–40) | 89 (5–237) | 41 (5–101) | 4.5 (1.6–6.9) |
| Local val | Accuracy-fitted cutpoints, 50 pools | 0.532 (0.516–0.539) | 0.732 (0.718–0.735) | 0.08 (0.00–0.19) | 0.64 (0.39–0.83) | 11 (0–38) | 7 (0–15) | 5 (0–23) | 4 (0–14) | inf (1.2–inf) |
| Local val | LoRA@1,000, three seeds | 0.559 (0.553–0.563) | 0.746 (0.741–0.749) | 0.03 (0.00–0.08) | 0.75 (0.75–0.75) | 3 (0–8) | 2 (0–6) | 1 (0–2) | 1 (0–3) | inf (1.3–inf) |
| MOVER (IV–V n = 506) | κ-fitted cutpoints, 50 pools | 0.525 (0.513–0.530) | 0.514 (0.477–0.549) | 0.53 (0.39–0.65) | 0.38 (0.32–0.45) | 718 (447–1008) | 268 (199–327) | 450 (248–681) | 180 (112–252) | 2.7 (2.2–3.1) |
| MOVER | Accuracy-fitted cutpoints, 50 pools | 0.419 (0.300–0.482) | 0.574 (0.559–0.579) | 0.08 (0.00–0.42) | 0.66 (0.44–1.00) | 78 (1–479) | 42 (1–213) | 36 (0–266) | 19 (0–120) | 1.6 (1.0–2.2) |
| MOVER | LoRA@1,000, three seeds | 0.505 (0.462–0.527) | 0.600 (0.591–0.613) | 0.37 (0.29–0.42) | 0.57 (0.57–0.58) | 325 (258–369) | 187 (149–212) | 138 (109–157) | 81 (64–92) | 1.7 (1.7–1.7) |

## C. Confusion matrices (local validation set)

**Calibrated decision model, κ-fitted cutpoints, main 1,000-label pool** (rows: recorded class; columns: predicted class)

| | I | II | III | IV–V |
|---|---|---|---|---|
| I | 5 | 28 | 3 | 0 |
| II | 77 | 922 | 287 | 9 |
| III | 9 | 320 | 878 | 136 |
| IV–V | 0 | 3 | 48 | 28 |

**LoRA@1,000, mean of three seeds** (rows: recorded class; columns: predicted class)

| | I | II | III | IV–V |
|---|---|---|---|---|
| I | 0.0 | 35.0 | 1.0 | 0.0 |
| II | 0.0 | 1044.7 | 250.0 | 0.3 |
| III | 0.0 | 336.3 | 1006.3 | 0.3 |
| IV–V | 0.0 | 2.7 | 74.3 | 2.0 |
