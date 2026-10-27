# Per-class recall over 50 independent 1,000-label calibration pools

Mean (min–max) over the same 50 pools as in Supplementary S18, drawn without replacement from the development operations (local: 4,326; MOVER: a further 5,000 training-split operations); the pools overlap, so the range understates between-pool variation. LoRA: mean (min–max) over three seeds. Evaluation sets: local validation set (n = 2,753; ASA I n = 36, IV–V n = 80) and MOVER subset (n = 4,000).

| Data | Arm | κ | Accuracy | Recall class I | Recall class IV–V | Predicted share IV–V (%) |
|---|---|---|---|---|---|---|
| Local val | Readout, 50 pools, κ-fitted cutpoints | 0.537 (0.514–0.546) | 0.693 (0.634–0.732) | 0.07 (0.00–0.28) | 0.29 (0.10–0.51) | 4.1 (0.5–10.1) |
| Local val | Readout, 50 pools, accuracy-fitted cutpoints | 0.532 (0.516–0.539) | 0.732 (0.718–0.735) | 0.00 (0.00–0.00) | 0.08 (0.00–0.19) | 0.4 (0.0–1.4) |
| Local val | LoRA n = 1,000, three seeds | 0.559 (0.553–0.563) | 0.746 (0.741–0.749) | 0.00 (0.00–0.00) | 0.03 (0.00–0.08) | 0.1 (0.0–0.3) |
| MOVER subset | Readout, 50 pools, κ-fitted cutpoints | 0.525 (0.513–0.530) | 0.514 (0.477–0.549) | 0.55 (0.35–0.70) | 0.53 (0.39–0.65) | 18.0 (11.2–25.2) |
| MOVER subset | Readout, 50 pools, accuracy-fitted cutpoints | 0.419 (0.300–0.482) | 0.574 (0.559–0.579) | 0.06 (0.00–0.30) | 0.08 (0.00–0.42) | 1.9 (0.0–12.0) |
| MOVER subset | LoRA n = 1,000, three seeds | 0.505 (0.462–0.527) | 0.600 (0.591–0.613) | 0.03 (0.00–0.08) | 0.37 (0.29–0.42) | 8.1 (6.5–9.2) |
