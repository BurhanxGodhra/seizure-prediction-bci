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
persistence + smoothing). Evaluated on 2 held-out CHB01 recordings, each
containing one seizure never seen during training or threshold selection:

| Metric | Value |
|---|---|
| Test ROC-AUC (threshold-independent) | **0.926** |
| Seizures caught (of test seizures) | **2 / 2** |
| Avg. false alarms / hour | 0.51 |
| Avg. warning lead time | 173.5s (range: 50s–297s) |

**Read this with appropriate skepticism**: these numbers come from only 2 test
seizures. Validation ROC-AUC on a similarly small sample was 0.66, well below
the test figure — a gap that size, with this little data, is plausibly sample
noise rather than the model being genuinely stronger on test than validation.
Treat the current numbers as "promising, not proven" until evaluated on more
recordings. That's exactly why Phase 2 (below) exists.

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
  catch is — one test seizure here had a 297s lead time, the other 50s, both
  counted as "sensitivity = 1.0."

Full journal: [`docs/project_journal.md`](docs/project_journal.md).
