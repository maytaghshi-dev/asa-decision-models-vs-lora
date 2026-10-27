# Per-class recall and predicted-class share (ASA I / II / III / IV–V)

Calibration pool: 1,000 development labels (seed 11); cutpoint search: initial cutpoints 1.5/2.5/3.5, grid of 121 points over [1, 4], 3 coordinate passes. κ-fitted cutpoints maximize quadratic-weighted κ (main analysis); accuracy-fitted cutpoints maximize exact accuracy. LoRA is the mean over three seeds at n = 1,000 (argmax). Recall is given per class as I / II / III / IV–V; share is the percentage of predictions per class.

| Data | Arm | κ | Accuracy | Recall I / II / III / IV–V | Predicted share I / II / III / IV–V (%) |
|---|---|---|---|---|---|
| Local val | True distribution | — | — | — | 1.3 / 47.0 / 48.8 / 2.9 |
| Local val | Readout zero-shot (argmax) | 0.443 | 0.563 | 0.39 / 0.71 / 0.44 / 0.27 | 11.1 / 59.1 / 27.4 / 2.4 |
| Local val | Readout + 1,000 labels, κ-fitted cutpoints | 0.534 | 0.666 | 0.14 / 0.71 / 0.65 / 0.35 | 3.3 / 46.2 / 44.2 / 6.3 |
| Local val | Readout + 1,000 labels, accuracy-fitted cutpoints | 0.536 | 0.733 | 0.00 / 0.77 / 0.75 / 0.10 | 0.0 / 49.5 / 50.0 / 0.5 |
| Local val | LoRA n = 1,000 (three-seed mean) | 0.559 | 0.746 | 0.00 / 0.81 / 0.75 / 0.03 | 0.0 / 51.5 / 48.4 / 0.1 |
| MOVER subset | True distribution | — | — | — | 4.9 / 30.7 / 51.8 / 12.7 |
| MOVER subset | Readout zero-shot (argmax) | 0.473 | 0.464 | 0.59 / 0.69 / 0.33 / 0.42 | 10.7 / 52.7 / 25.0 / 11.7 |
| MOVER subset | Readout + 1,000 labels, κ-fitted cutpoints | 0.524 | 0.515 | 0.35 / 0.54 / 0.50 / 0.58 | 4.7 / 33.4 / 41.1 / 20.8 |
| MOVER subset | Readout + 1,000 labels, accuracy-fitted cutpoints | 0.401 | 0.578 | 0.00 / 0.50 / 0.81 / 0.03 | 0.0 / 30.2 / 69.1 / 0.7 |
| MOVER subset | LoRA n = 1,000 (three-seed mean) | 0.505 | 0.600 | 0.03 / 0.50 / 0.77 / 0.37 | 0.3 / 28.3 / 63.3 / 8.1 |
