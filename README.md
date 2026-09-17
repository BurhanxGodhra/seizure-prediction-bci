# Seizure Prediction & Detection BCI

A continuous, EEG-based system that aims to identify a **preictal** (pre-seizure)
state before a seizure occurs, while retaining the ability to detect an
**ictal** (ongoing) seizure if prediction fails. Built as a BCI /
neurotechnology portfolio project, developed deliberately incrementally
(CSP + Logistic Regression → LSTM → CNN-LSTM) rather than jumping straight to
a large architecture, so every component is understood rather than copied.

Dataset: [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/),
currently subject **CHB01** only (Phase 1 of the roadmap below).

## Current status (Phase 1: CHB01, single patient)

Trained a per-file, temporally-safe TimeDistributed Conv1D → LSTM model with
Platt calibration and a validation-selected event-alarm policy (threshold +
persistence + smoothing). Evaluated on 5 held-out CHB01 recordings (2 with a
seizure, 3 fully normal), using 16 training and 5 validation recordings
covering 4 and 1 of CHB01's 7 seizures respectively:

| Metric | Value |
|---|---|
| Test ROC-AUC (threshold-independent) | **0.962** |
| Seizures caught (of test seizures) | **2 / 2 (100%)** |
| Aggregate false alarms / hour (weighted, all 5 test files) | **0.69** |
| Lead time | 297s and 140s (the two test seizures) |

**Read this with appropriate skepticism**: these numbers come from only 2
test seizures — real, but a small sample. Getting here took fixing several
real bugs along the way (documented in
[`docs/project_journal.md`](docs/project_journal.md)): a validation generator
that silently exhausted after epoch 1 (making early stopping/LR scheduling
blind for most of training), an oversampling rate far more aggressive than
the documented 50/50 design intent (causing the model to memorize a small
repeated pool of positive windows), a sensitivity calculation that corrupted
averages on zero-seizure files, and — most impactful for the false-alarm rate
— a threshold-selection tie-break that always picked the most aggressive
(least conservative) option out of dozens of validation-tied candidates
instead of the most conservative one. Each fix is a real, measurable jump in
the table above; none of it is hyperparameter luck.

## Architecture

```
EEG (EDF)
  → MNE preprocessing, resample to 128Hz
  → 2-second epochs
  → preictal labeling (5 min before seizure onset) / ictal masking
  → per-file sequences (never concatenated across recordings)
  → TimeDistributed Conv1D  (local spatial/spectral features per epoch)
  → LSTM                    (temporal evolution across the sequence)
  → Dense + sigmoid         (P(preictal))
  → Platt calibration (fit on validation only)
  → event-alarm logic (threshold + min-streak + smoothing, selected on validation only)
  → warning / normal
```

Full rationale for each design decision — why sequences instead of
independent epochs, why calibration, why event-level (not just sequence-level)
evaluation, why an ictal state at all — is in
[`docs/project_journal.md`](docs/project_journal.md).

## Repo layout

```
.
├── src/
│   ├── lstm_model_pipeline.py   # current pipeline: data loading, sequence
│   │                            # construction, CNN-LSTM, calibration,
│   │                            # event-policy selection, evaluation
│   └── legacy/
│       └── lstm_baseline_v1.py  # first working LSTM baseline (no
│                                # calibration, no event logic) — kept as
│                                # documented history of the project's
│                                # progression, not actively maintained
├── docs/
│   └── project_journal.md       # full design rationale and decision log
├── data/                        # not tracked — see Setup below
├── results/
│   └── models/                  # trained checkpoints (not tracked by default)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/BurhanxGodhra/seizure-prediction-bci.git
cd seizure-prediction-bci

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Download CHB01 from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/)
and place it at `data/chb01/` (including `chb01-summary.txt`). This data is
not included in the repo — see `.gitignore`.

```bash
python src/lstm_model_pipeline.py
```

## Roadmap

- [x] **Phase 1 — Robust CHB01 pipeline**: per-file temporal integrity, correct
      preictal/ictal labeling, no cross-file sequences, train/val/test
      separation, CNN-LSTM baseline, calibration, event-level evaluation.
- [ ] **Phase 2 — Full CHB01**: all ~42 recordings / 7 seizures for this
      subject, to get a statistically meaningful per-subject evaluation
      instead of n=2.
- [ ] **Phase 3 — Multi-patient (CHB01 → CHB24)**: within-patient vs.
      cross-patient generalization.
- [ ] **Phase 4 — Advanced architectures**: only after the baseline is
      trustworthy on more data (TCN / attention / Transformer candidates,
      chosen on evidence).
- [ ] **Phase 5 — Hybrid Normal / Preictal / Ictal system** with continuous
      decision logic (the reactive safety layer if prediction fails).

## Key lessons so far

- Accuracy is not an adequate metric for a rare-event problem — a model that
  predicts "normal" for everything can hit 95%+ accuracy while being useless.
- Balancing training exposure to positives is necessary, but oversampling too
  aggressively (this project ran `pos_prob=0.85` when 50/50 was intended) can
  cause the model to memorize a small repeated pool of positive windows rather
  than learn generalizable signal — near-perfect training metrics with weak
  real separation was the tell.
- Threshold-based confusion matrices are noisy with few test seizures;
  ROC-AUC (threshold-independent) is a more stable thing to track run-to-run.
- "Sensitivity" at the event level can hide large variance in how *useful* a
  catch is — one test seizure here had a 297s lead time, the other 140s, both
  counted as "sensitivity = 1.0."
- When many threshold/policy candidates tie for the best validation score,
  *which* tied candidate gets selected matters — picking the most aggressive
  one by default (e.g. always taking the first match found) can look
  identical on validation while performing far worse on test.

Full journal: [`docs/project_journal.md`](docs/project_journal.md).
