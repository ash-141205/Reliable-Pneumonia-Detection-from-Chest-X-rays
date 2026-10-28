# Reliable Pneumonia Detection from Chest X-rays

Own-trained CNNs for pneumonia detection evaluated for **reliability**, not just accuracy:
patient-level splits, external-dataset testing, calibrated confidence, abstention,
a source-shortcut audit, and Grad-CAM explanations. No third-party model APIs are used;
every model is trained by the code in this repo.

> Research project, not a medical device. No results are bundled: run the pipeline to produce them.

## Research questions
1. How much does performance drop when the model meets images from another source?
2. Is the model's confidence trustworthy, and does temperature scaling fix it?
3. Can abstaining on low-confidence cases recover reliability (risk-coverage trade-off)?
4. Are there non-pathology source cues (border-only shortcut audit), and do heatmaps focus on lungs?

## Structure
```
src/pneumreliab/   data.py (patient split, manifests) · models.py · train.py · evaluate.py
                   shortcut.py · explain.py (Grad-CAM) · aggregate.py · utils.py
app/app.py         Streamlit UI (calibrated prob + abstention + Grad-CAM)
scripts/run_all.sh full experiment suite incl. ablations
tests/             pytest: leakage, metrics, calibration, end-to-end smoke test
docs/              DATASETS.md · EXPERIMENT_LOG.md · PAPER_CHECKLIST.md
```

## Setup
```
pip install -r requirements.txt && pip install -e .
pytest -q                       # verify everything works on synthetic data (no dataset needed)
```

## Run
```
export INTERNAL=/data/chest_xray EXTERNAL=/data/external   # see docs/DATASETS.md
bash scripts/run_all.sh                                    # full suite (GPU recommended)
# quick trial:
python -m pneumreliab.train --internal $INTERNAL --external $EXTERNAL --models resnet50 --seeds 0 --epochs 4
streamlit run app/app.py                                   # demo UI
```
Each run folder `results/<tag>_<arch>_s<seed>/` holds `model.pt, split.csv, config.json,
train_log.json, metrics.json, plots.png`; `results/summary.md` aggregates seeds.

## Experiments
- **Models:** small CNN from scratch, ResNet18, ResNet50 (fine-tuned).
- **Ablations (ResNet50):** no augmentation, no class weighting, no pretraining.
- **Metrics:** AUROC, AUPRC, sensitivity, specificity, precision, F1, accuracy, Brier, ECE, bootstrap 95% CIs, accuracy/sensitivity vs coverage, operating point at 95% sensitivity chosen on validation.
- **Calibration:** temperature scaling fitted on internal validation only.
- **Shortcut audit:** border-only vs full-image source classification.

## GAN Augmentation (optional)

A fifth research question: **does adding GAN-generated NORMAL chest X-rays improve
external reliability and calibration, or only the internal scores?**

The pipeline trains a class-conditional DCGAN (64 × 64 grayscale, from scratch)
on each seed's training split, samples synthetic images, and retrains the classifier
with those images appended to the training set only.  Evaluation remains on real
images.  A negative result is expected to be reported honestly.

```bash
# Train a GAN on seed 0's split, then train with 50% and 100% synthetic NORMAL:
INTERNAL=/data/chest_xray EXTERNAL=/data/external bash scripts/run_gan.sh
```

Key flags added to `python -m pneumreliab.train`:
- `--synthetic-dir PATH` — directory of synthetic images (train only)
- `--synthetic-ratio 0.5` — fraction of minority-class count to add
- `--synthetic-cls NORMAL` — class of the synthetic images

Key flags for `python -m pneumreliab.gan`:
- `train --run-dir results/<run> --epochs N --seed S`
- `sample --run-dir results/<run> --cls NORMAL --n K --out DIR`

See **[docs/GAN.md](docs/GAN.md)** for the full protocol, leakage rules, flags,
and what to log for the paper.

## Limitations
Public labels are noisy; internal/external differ in age group as well as site; external patient IDs may be missing; Grad-CAM is a qualitative aid, not proof of correct reasoning; not clinically validated.
