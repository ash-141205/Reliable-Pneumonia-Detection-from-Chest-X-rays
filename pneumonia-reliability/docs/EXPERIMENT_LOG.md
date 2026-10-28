# Experiment log (fill in as you run; keep honest, include failures)

| Date | Run folder | Change vs previous | Seeds | Internal AUROC | External AUROC | ECE raw → cal | Notes / surprises |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Decisions and reasons (your own words, useful later when writing the paper)
- Why this external dataset:
- How external labels were mapped:
- Any run excluded and why:

## Observations to explain in the paper
- Size of internal → external drop:
- Did calibration help, and where (internal vs external)?
- Does abstention recover accuracy on the external set?
- Border-only source AUROC (`results/shortcut_audit.json`):
- Do Grad-CAM heatmaps (UI screenshots) fall on lungs or on borders/markers?
