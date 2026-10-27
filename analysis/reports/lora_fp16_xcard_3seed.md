# fp16 LoRA across two cards at three seeds (local validation set, n = 2,753; local n = 1,000 labels)

| Seed | fp16 LoRA κ (95% CI) | 4-bit LoRA κ (same labels) | fp16 − 4-bit (95% CI) | fp16 accuracy | Calibrated fp16 readout − fp16 LoRA (95% CI) | Label pool vs calibration pool |
|---|---|---|---|---|---|---|
| 11 | 0.553 (0.525–0.580) | 0.562 | -0.008 (-0.021 to +0.006) | 0.741 | -0.019 (-0.045 to +0.006) | identical |
| 12 | 0.556 (0.528–0.581) | 0.563 | -0.006 (-0.020 to +0.008) | 0.742 | -0.022 (-0.046 to +0.001) | partly overlapping (238/1000) |
| 20261017 | 0.571 (0.542–0.599) | 0.553 | +0.018 (+0.003 to +0.034) | 0.744 | -0.037 (-0.057 to -0.015) | partly overlapping (223/1000) |

- Three-seed mean of fp16 LoRA κ: 0.560 (95% CI 0.534–0.584); three-seed mean of 4-bit LoRA κ: 0.559; difference +0.001 (-0.008 to +0.011).
- **H2 with fp16 LoRA, three-seed mean** (fp16 readout calibrated on pool 11 minus mean of the three fp16 LoRA seeds): -0.026 (-0.047 to -0.005); lower bound > −0.05: yes.
- Reference, 4-bit LoRA three-seed mean (main analysis): -0.025 (-0.048 to -0.002).
- Seed 11: training 324 s, batch-1 latency p50 160 ms, peak training memory 10.4/9.5 GB.
- Seed 12: training 323 s, batch-1 latency p50 170 ms, peak training memory 10.3/9.4 GB.
- Seed 20261017: training 324 s, batch-1 latency p50 168 ms, peak training memory 10.3/9.4 GB.
