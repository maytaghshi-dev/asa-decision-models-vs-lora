# Calibration-label curve: raw-prompt readout versus chat-template readout (local validation set, n = 2,753)

Mean (2.5–97.5 percentile) of κ over 100 draws of k labels from the seed-11 pool; k = 0 is the expected score read with the default cutpoints 1.5/2.5/3.5 (the argmax values are 0.443 and 0.524 in the table above); k = 1,000 is the whole pool. The raw-prompt values use the same fine grid as the main analysis (0.534 at k = 1,000).

| k | Raw-prompt readout (main analysis) | Chat-template readout |
|---|---|---|
| 0 | 0.459 | 0.519 |
| 25 | 0.488 (0.373–0.545) | 0.496 (0.368–0.556) |
| 50 | 0.517 (0.464–0.545) | 0.515 (0.448–0.555) |
| 100 | 0.526 (0.480–0.547) | 0.522 (0.470–0.555) |
| 200 | 0.528 (0.500–0.543) | 0.527 (0.493–0.550) |
| 500 | 0.534 (0.519–0.543) | 0.532 (0.518–0.543) |
| 1000 | 0.534 | 0.533 |
