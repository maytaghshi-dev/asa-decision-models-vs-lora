# Readout under the chat template (letter-token probabilities at the first generated position; local validation set, n = 2,753)

| Input format | κ, uncal. | Accuracy, uncal. | κ, calibrated (1,000 labels) | Accuracy, calibrated | Predicted share, uncal., I / II / III / IV–V (%) |
|---|---|---|---|---|---|
| Raw readout prompt (main analysis) | 0.443 | 0.563 | 0.534 | 0.666 | 11.1 / 59.1 / 27.4 / 2.4 |
| Chat template + "Reply with the option letter only." | 0.524 | 0.620 | 0.533 | 0.681 | 13.8 / 37.5 / 47.5 / 1.2 |

- Agreement between the chat-template readout argmax and the chat8 answer-only generation on the 1,000-case H1 subset: 1.000 (n = 1000).
- κ on the H1 subset: chat-template readout argmax 0.539; chat8 generation 0.539; raw readout argmax 0.461.
