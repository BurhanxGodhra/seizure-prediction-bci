# Seizure Prediction & Detection BCI — Project Journal

This is the full design rationale and decision log for the project: why each
architectural and methodological choice was made, what problems were hit
along the way, and what was learned. See the top-level [README](../README.md)
for current status and setup instructions.

## 1. Project Overview

This project started as a **Brain-Computer Interface (BCI) EEG
seizure-detection experiment** and has progressively evolved into something
substantially more ambitious:

> **A continuous EEG-based seizure prediction and detection system that can
> identify a preictal state before a seizure occurs, while retaining the
> ability to detect an ongoing seizure if prediction fails.**

The project is being developed as a serious BCI / neurotechnology portfolio
project, with the eventual objective of demonstrating the engineering,
signal-processing, machine-learning, and clinical-product thinking expected
in a professional neurotechnology environment.

The project is deliberately being developed incrementally. We are not
jumping directly to a complex Transformer or an enormous deep-learning
architecture. Each model and pipeline stage is being used as a learning and
engineering step toward the final system.

## 2. Original Objective

The original objective was much simpler: **train a machine-learning model to
distinguish seizure EEG from non-seizure EEG.**

```
EEG → Preprocessing → Epoching → Feature extraction → Classifier → Seizure / Non-seizure
```

The dataset is the CHB-MIT Scalp EEG Database, beginning with subject CHB01.

Seizure is a rare event — most of the recording is normal. A model that
simply learns `normal EEG → 0, seizure EEG → 1` has: extreme class
imbalance, poor generalization, many false positives on new recordings,
insufficient temporal context, and no ability to give early warning.

## 3. From Classification to Temporal Modeling

Treating every EEG epoch as an independent sample is fundamentally limiting
— the classifier doesn't understand that consecutive epochs are observations
of the same evolving brain state. This led to introducing sequence modeling.

## 4. Initial Logistic Regression Stage

First serious model: CSP (Common Spatial Patterns) + Logistic Regression.

```
EDF → MNE → 2-second epochs → EEG labels → CSP → Logistic Regression → Seizure / Non-seizure
```

Useful baseline, but exposed the weakness of treating epochs independently.

## 5. The Initial Generalization Problem

Performed reasonably on training data, struggled on a different recording.
Root cause: extremely few seizure samples. One early result:

```
[[1713   73]
 [  13    1]]
Class 0: precision ≈ 0.99, recall ≈ 0.96
Class 1: precision ≈ 0.01, recall ≈ 0.07
Accuracy: 95% (misleading)
```

**Lesson: accuracy is not an adequate metric for an extreme rare-event
problem.**

## 6. Introduction of Temporal Sequences

Combined multiple epochs into a single input: 10 epochs × 2s = 20s, later 30
epochs × 2s = 60s. First transition toward temporal modeling.

## 7. Temporal Smoothing

Experimented with smoothing individual predictions rather than alarming on
one isolated positive window. Decided to first inspect the raw probability
timeline before adding smoothing "because it sounds useful."

## 8. Major Shift: Detection → Prediction

Introduced a **preictal** state: the five minutes immediately preceding
seizure onset (`0 = normal, 1 = preictal`). If a seizure starts at T = 1000s,
then 700–1000s is PREICTAL and 1000s+ is ICTAL. Moves the question from "can
we detect a seizure?" to "can we identify the brain entering a state
associated with an imminent seizure?" — substantially more valuable from a
product perspective.

## 9. Long-Term Vision: Hybrid Prediction + Detection

Three-state system:

- **State 0 — Normal**: no seizure occurring, no strong evidence of one
  approaching.
- **State 1 — Preictal**: patterns associated with an approaching seizure.
  Triggers a warning (sit down, stop driving, prepare medication, notify
  caregiver).
- **State 2 — Ictal**: a seizure is occurring. Reactive safety layer (alert
  caregiver, start seizure timer, record event, provide clinical-review data).

`PREDICT WHEN POSSIBLE + DETECT WHEN NECESSARY`

## 10. Missed-Prediction / False-Alarm Concept

- **Missed prediction**: normal → seizure with no preictal warning. Recorded
  for later analysis of what was different.
- **False alarm**: preictal prediction → no seizure. Recorded; could support
  future patient-specific adaptation.

## 11. Long-Term Online / Patient-Specific Learning Vision

Future (not yet implemented) direction: store preceding EEG around each
seizure, analyze successful/missed predictions, and adapt the model per
patient — because preictal EEG patterns likely differ across individuals and
seizure prediction is inherently hard to generalize across people.

## 12. Dataset Strategy

CHB01 has ~42 EEG files, ~7 seizure-containing recordings, and many
non-seizure recordings. Non-seizure files are deliberately included — the
model needs to see the real distribution (mostly normal, rarely preictal,
very rarely ictal). Training only on seizure-containing recordings would be
an unrealistic learning environment.

## 13. Expansion Toward All 24 Subjects

- **Phase 1**: develop and validate on CHB01.
- **Phase 2**: expand CHB01 → CHB24, to evaluate within-patient generalization
  (same patient, different session) and cross-patient generalization (a
  patient never seen before — substantially harder and more important).

## 14. The Memory Problem

Estimated raw dataset sizes: 12 files → ~32GB, 42 files → ~112GB, 24 subjects
→ ~2.7TB, against ~12.7GB RAM in Google Colab. Building giant sequence arrays
(`X_train_seq, y_train_seq = create_sequences(...)`) crashed.

## 15. Generator-Based Data Processing

Moved to batch-based generation: process only the required portion, create a
sequence, batch, feed the model, discard, move to the next batch — instead of
loading everything into RAM at once. Transition from a notebook experiment
toward a scalable ML pipeline.

## 16. Temporary Dimensionality Reduction

At one stage the generator did `seq = seq.mean(axis=2)`, collapsing
within-epoch samples (`sequence × channels × raw samples` →
`sequence × channels`) to cut memory. This discards within-epoch temporal
information and was **not** treated as the final representation — the
current direction preserves meaningful EEG temporal information through the
model rather than averaging it away.

## 17. Transition to LSTM

```
EEG sequence → LSTM → Dense → Sigmoid → P(preictal)
```

First LSTM intentionally simple: `LSTM(64)` → `Dense(1, sigmoid)`. A
sequence-learning baseline, not the final architecture.

## 18. Initial LSTM Results

```
[[7132    0]
 [  38    0]]
```

Model predicted everything as normal — demonstrated that 99% accuracy can
correspond to 0% preictal recall, i.e. a useless prediction system.

## 19. Balanced Training Sampling

Introduced controlled sampling: training batches drawn ~50% preictal / 50%
normal, while validation/test remain close to the real distribution. This
does **not** mean pretending the real world is 50/50 — it's controlled
exposure during training only.

## 20. CNN + LSTM Architecture

```
EEG sequence → CNN (local/spatial features) → LSTM (temporal evolution) → Dense → P(preictal)
```

CNN answers "what pattern is present?"; LSTM answers "how is that pattern
evolving?"

## 21. Probability-Based Analysis

Moved from binary 0/1 predictions to examining `P(preictal)` directly —
allows investigating whether model confidence rises before seizure onset.

## 22. Probability Timeline

Planned diagnostic: plot predicted probability over time with the actual
preictal region marked, to check whether probability rises before seizures,
whether it rises gradually, whether false alarms are isolated spikes or
sustained, and whether the model recognizes a genuine temporal transition.

## 23. Threshold Optimization

`pred = probability > 0.5` is arbitrary for a rare-event problem. Process:
train → validation → choose threshold → final test. **The test set must
never be used to choose the threshold.**

## 24–26. Results, and Why Accuracy Stopped Being the Goal

Results varied significantly across configurations — some runs achieved high
recall with poor precision, others the reverse. Consistent lesson: improving
recall alone is easy if the model is allowed to alarm often; useful recall
*and* acceptable precision together is the real challenge. The project moved
from "how accurate is the classifier?" toward "can the system provide
clinically meaningful warnings without overwhelming the user with false
alarms?" Key metrics became: preictal sensitivity, precision, false
alarms/hour, detection lead time, and seizure-level sensitivity.

## 27. Overfitting Is a Core Concern

Explicitly established: **the model must not overfit the dataset just to
produce impressive metrics.** 99% accuracy from memorizing CHB01-specific
characteristics, or recognizing recording-specific artifacts instead of
genuine physiology, is not useful. Evaluation difficulty should increase over
time: training session → different session → different seizure → different
patient.

## 28. Current Pipeline Direction

```
EDF → MNE → resampling → 2s epochs → annotation (Normal/Preictal/Ictal)
  → continuous per-file sequences → CNN → LSTM → P(preictal)
  → temporal decision logic → warning
```

## 29. Pipeline Repairs Identified

- **Never concatenate independent EDF recordings into one temporal stream** —
  sequences must be constructed inside each EDF file only.
- **Don't delete ictal periods and accidentally connect both sides** —
  `PREICTAL → ICTAL → NORMAL` must not silently become `PREICTAL → NORMAL`;
  ictal-containing sequences should be invalid/masked, not removed-and-stitched.
- **Keep file identity** on each sequence (file, start time, label) for
  event-level evaluation.
- **Proper train/validation/test separation**, with the test set untouched
  until the model is finalized.
- **Balance only training exposure** — validation and test represent
  realistic distributions.

## 30. Development Environment

Moved from Google Colab (slow repeated uploads, insufficient RAM, crashes on
large sequence arrays) to a local machine, developed through the terminal
with a Python virtual environment. Transition from experimental notebook to
local reproducible project.

## 31. Planned Software Architecture

```
src/
├── data/         (loader, annotations, preprocessing)
├── sequences/    (generator)
├── models/       (CNN-LSTM)
├── training/     (training pipeline)
└── evaluation/   (classification metrics, seizure-level metrics, visualization)
```

Not implemented all at once — the purpose is to prevent regressing into a
large, confusing notebook structure.

## 32–34. Product Vision

Envisioned as a continuous monitoring system (rolling time window → CNN-LSTM
→ P(preictal) → NORMAL or PREICTAL → WARNING → seizure occurs? → ICTAL or
FALSE ALARM), not a batch classifier requiring someone to manually load an
EDF. Long-term: store seizure time, preictal probability curve, prediction
outcome, warning time, and false alarms per patient to build a longitudinal
dataset — eventually enabling patient-specific adaptation (not implemented
yet, much later in the roadmap).

## 35. Why Not Transformers Yet

Deliberate progressive path: EEG basics → MNE preprocessing → epoching → CSP
→ Logistic Regression → temporal sequences → LSTM → CNN-LSTM → robust
evaluation → temporal decision logic → larger datasets → advanced
architectures. The purpose is understanding every layer built, not producing
a sophisticated-looking repository of copied code.

## 36. Lessons Learned

1. Accuracy can be misleading for rare-event problems.
2. EEG is temporal — individual epochs are insufficient.
3. Class imbalance is fundamental, not incidental.
4. Balancing introduces its own problems (oversampling can produce many false
   alarms, or — as seen directly in this project — cause the model to
   memorize a small repeated positive pool rather than generalize).
5. Probability contains more information than binary predictions.
6. Temporal decision logic matters — one isolated positive window shouldn't
   trigger an alarm.
7. Memory and data engineering matter — a good ML architecture must be
   computationally scalable.
8. Evaluation must become harder, not easier, over time.
9. Overfitting is unacceptable, however good the headline metric looks.

## 37–38. Where We Stand / Immediate Next Stage

Past the basic proof-of-concept stage. Built and tested: EEG loading (MNE),
seizure annotation parsing, fixed-length epoching, preictal labeling, CSP +
Logistic Regression, temporal sequences, LSTM, balanced training, batch-based
generation, CNN-LSTM, probability-based evaluation, and probability timeline
diagnostics. Immediate priority is pipeline robustness (per-file
representation, correct labeling, no cross-file sequences, proper
train/val/test separation, training sampler, CNN-LSTM baseline, probability
timeline, threshold optimization, temporal persistence/alarm logic,
event-level evaluation) — not further architecture changes until this is
solid.

## 39. Phase Structure

- **Phase 1 — Robust CHB01 Pipeline**: scientifically and computationally
  correct pipeline on one patient (data integrity, temporal continuity,
  sequence generation, CNN-LSTM, validation, false alarms, probability
  behavior).
- **Phase 2 — Full CHB01**: all available recordings for the subject; does
  the system generalize across many sessions of one patient?
- **Phase 3 — Multi-Patient Dataset (CHB01 → CHB24)**: do learned
  representations generalize across patients?
- **Phase 4 — Advanced Models**: TCN, attention-based temporal models,
  Transformer-style models, CNN+Transformer, multimodal — selected on
  evidence, not fashion, and only after the baseline is trustworthy.
- **Phase 5 — Hybrid Detection + Prediction**: Normal/Preictal/Ictal with
  continuous decision logic.

## 40. Ultimate Objective

Not "get a high score on CHB-MIT." Build and demonstrate a robust continuous
EEG-based seizure prediction and detection pipeline that identifies preictal
patterns, provides meaningful advance warning, detects seizures when
prediction fails, and generalizes beyond the specific recordings used for
training — evaluated on: how many seizures were predicted, how many missed,
how much warning was given, how many false alarms occurred, and whether it
generalizes to new recordings and new patients.

## 41. One-Sentence Summary

Started with *"Can Logistic Regression classify seizure EEG?"* and evolved
toward *"Can we build a robust, continuously operating, temporally aware EEG
system that learns preictal dynamics, predicts seizures before they occur,
detects seizures when prediction fails, minimizes false alarms, and
generalizes beyond the training recordings?"*
