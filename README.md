# Open-source non-generative decision models vs. LoRA for ASA physical status classification

Code and aggregate results for the study **"Open-source non-generative decision models with few-label calibration versus LoRA fine-tuning for ASA physical status classification: accuracy, speed, cost, and label efficiency across three cohorts"**.

On one open base model (Qwen3.5-9B), the study compares four approaches to four-class ASA-PS grading (I, II, III, IV–V):

- generative zero-shot prompting;
- a minimal non-generative decision model, which reads the softmax over the option letters A–D in a single forward pass;
- OpenJev, an open implementation of a System One decision model, with and without three-cutpoint calibration on k local labels;
- task-specific LoRA (4-bit QLoRA).

## What is included
| Path | Content |
|---|---|
| `analysis/scripts/` | All analysis scripts. The numeric prefix gives the order of development. |
| `analysis/scripts/130_figures_v5.py` | Draws all six manuscript figures from aggregate results only, with no patient data needed. |
| `analysis/reports/` | Aggregate result tables: paired differences, calibration curves, robustness, guidance, token cost, TRIPOD-LLM self-check. |
| `figures/` | Manuscript figures (English and Chinese, PNG and PDF). |
| `openjev_patch/` | Patched `hf.py` for OpenJev and notes on the two compatibility patches. |
| `checksums/` | SHA-256 checksums of the Qwen3.5-9B weight files used. |

## What is not included
- No patient-level data, predictions, or model weights.
- **Local hospital data.** These come from the specialty perioperative database of abdominal tumor surgery at the People's Hospital of Xinjiang Uygur Autonomous Region. They contain patient information and cannot be shared. Under Chinese data-protection rules they also cannot leave the institution.
- **MOVER.** Requires a data use agreement with the providers (https://doi.org/10.1093/jamiaopen/ooad084).
- **VitalDB.** Public (https://doi.org/10.13026/czw8-9p62, CC BY-NC-SA 4.0).

To run the full pipeline, obtain the data, then recreate the directory layout that the scripts read: `~/research/Anes_JEV_vs_LoRA` and the sibling data directory `~/research/NE_ERAS_JEV/data/derived`.

## Terminology in the code
The scripts predate the final terminology of the manuscript:

| In the scripts | In the manuscript |
|---|---|
| `readout` / 读出 | minimal decision model |
| `readout + calibration` | calibrated decision model |

Clef appears in a few early scripts. It was an exploratory arm that is not reported in the manuscript.

## Reproducing the figures
```bash
pip install -r requirements.txt
python analysis/scripts/130_figures_v5.py   # writes figures/*_{en,zh}.{png,pdf,svg}
```

## Pipeline overview
| Step | Scripts |
|---|---|
| Build datasets and the T1/T2 templates | 01–05, 17, 36, 60 |
| Minimal decision model (zero-shot) and generative zero-shot | 06, 07, 13, 21 |
| Task LoRA (4-bit QLoRA; fp16 control) | 08, 111, 112, 125 |
| XGBoost and kNN arms | 09, 15 |
| OpenJev tuning and frozen runs | 30–34, 50–55 |
| Cutpoint calibration and label-efficiency curves | 35, 41, 82 |
| Evaluation, paired differences, cluster bootstrap | 10, 14, 20, 22, 87, 87b |
| Speed, tokens, latency | 18, 97, 98 |
| Robustness (option order, label format, 4-bit, label matching, prompt format) | 24, 26, 38, 54, 103, 105, 105b, 108, 113–115, 126, 127 |
| Subgroups, break-even, adjacent errors, per-class recall | 99, 106, 109, 116, 122, 123 |
| Natural-language guidance | 83, 85 |
| Screen for contradictory generative content | 81 |
| Tables and figures | 86, 94, 130 |
| Cold-start contrast and ASA IV–V operating characteristics | 131 |

## Environment
See `requirements.txt`.

- **Hardware.** Two NVIDIA GeForce RTX 2080 Ti cards with 22 GB each (compute capability 7.5, fp16 only).
- **Base model.** `Qwen/Qwen3.5-9B` (Apache-2.0).
- **OpenJev upstream.** https://github.com/GitHub30/OpenJev, commit `aae11e1`, MIT license; see `openjev_patch/PATCH_NOTES.md`.

## License
- Code: MIT (see `LICENSE`).
- OpenJev patch file: MIT, under its upstream license (`openjev_patch/LICENSE_OpenJev_MIT`).
