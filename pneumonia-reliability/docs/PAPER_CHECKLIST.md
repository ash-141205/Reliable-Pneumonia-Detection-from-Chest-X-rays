# Paper checklist (YOU write the text; the course requires 0% AI-generated content)

## Where each result comes from
| Paper item | Source file |
|---|---|
| Main table (internal vs external, raw vs calibrated, mean ± std) | `results/summary.md` |
| Per-run CIs, operating point at 95% sensitivity | `results/<run>/metrics.json` |
| ROC, reliability diagram, abstention curve figure | `results/<run>/plots.png` |
| Ablation table | rows `abl_*` in `results/summary.md` |
| Shortcut finding | `results/shortcut_audit.json` |
| Training curves | `results/<run>/train_log.json` |
| Setup / hyperparameters | `results/<run>/config.json` |
| Qualitative figure | screenshots of the UI heatmaps (include failures) |

## Structure
Abstract · Introduction (problem, gap, contributions) · Related Work · Data and split protocol · Methods · Experiments · Results · Ablations · Limitations · Conclusion · References

## Novelty (do this before claiming anything)
1. Search Google Scholar / PubMed / arXiv: "chest X-ray pneumonia cross-dataset generalization", "selective prediction chest radiograph", "calibration chest X-ray", "shortcut learning chest radiograph".
2. Read the closest 10 papers, write a table of what each does and what it lacks.
3. State your contribution as the specific combination/protocol/finding you actually produced. Never write "first".

## Integrity
- Write every sentence yourself; use tools only to check grammar after you've written it, and verify the policy allows it.
- Cite datasets, models (ResNet, Grad-CAM, temperature scaling) and every method you used.
- Run a plagiarism check (target < 10%).
- Report negative and unexpected results; do not tune on the external set.

## Limitations to state
Label noise in public sets · pediatric vs adult confound · no patient IDs for external data · single modality, retrospective data · not clinically validated, not a diagnostic device.
