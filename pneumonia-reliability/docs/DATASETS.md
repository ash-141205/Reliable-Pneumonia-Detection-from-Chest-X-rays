# Datasets and protocol

| Role | Dataset | Notes |
|---|---|---|
| Internal (train/val/test) | Kermany et al., "Chest X-Ray Images (Pneumonia)" (Mendeley / Kaggle), pediatric | Pool the shipped train/val/test folders; the code re-splits **by patient** |
| External test (never used for training, tuning or calibration) | One of: RSNA Pneumonia Detection (Kaggle), NIH ChestX-ray14, CheXpert, or another public set | Convert to `NORMAL/` and `PNEUMONIA/` folders of JPG/PNG. Map labels explicitly and document how |

**Before using any dataset:** read its license/terms and cite it exactly as the dataset page requests.

**Folder layout expected by the code** (any depth, folder name = label):
```
data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
data/external/{NORMAL,PNEUMONIA}/*.png
```

**Protocol rules (do not break these, they are what make results credible)**
1. Splits are patient-level, stratified, seeded, and written to `split.csv` in every run folder.
2. Temperature and the operating threshold are fitted on the *internal validation* split only.
3. The external set is used for reporting only.
4. Report mean ± std over seeds, and bootstrap CIs per run (already in `metrics.json`).
5. External-set patient IDs are unknown unless your dataset provides them; note this in the paper.
6. Internal (pediatric) vs external (often adult) differs in age as well as site. State this as a confound.
