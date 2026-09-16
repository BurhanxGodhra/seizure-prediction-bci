from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.metrics import confusion_matrix, classification_report


# =========================================================================
# PATH SETUP
# =========================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "chb01"
SUMMARY_FILE = DATA_DIR / "chb01-summary.txt"


with open(SUMMARY_FILE, "r") as f:
    summary_text = f.read()


# =========================================================================
# SEIZURE TIMING
# =========================================================================

def get_seizure_times(summary_text, target_file):

    lines = summary_text.splitlines()

    seizure_times = []
    in_target = False

    for i, line in enumerate(lines):

        line = line.strip()

        if line.startswith("File Name:"):
            current_file = line.split(":")[1].strip()
            in_target = (current_file == target_file)

        if in_target:

            if "Number of Seizures in File: 0" in line:
                return []

            if "Seizure Start Time:" in line:

                start = int(
                    line.split(":")[1].split()[0]
                )

                end = int(
                    lines[i + 1].split(":")[1].split()[0]
                )

                seizure_times.append((start, end))

    return seizure_times


# =========================================================================
# PROCESS ONE EDF FILE
# =========================================================================

def process_file(file_name, seizure_times):

    file_path = DATA_DIR / file_name

    raw = mne.io.read_raw_edf(
        str(file_path),
        preload=True
    )

    # Reduce sampling rate to reduce computational load
    raw.resample(128)

    # Divide continuous EEG into 2-second epochs
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=2,
        preload=True
    )

    data = epochs.get_data()

    labels = []
    epoch_times = epochs.events[:, 0] / raw.info["sfreq"]

    PREICTAL_WINDOW = 300  # 5 minutes

    for t in epoch_times:

        label = 0

        for start, end in seizure_times:

            preictal_start = max(
                0,
                start - PREICTAL_WINDOW
            )

            if preictal_start <= t < start:
                label = 1

        labels.append(label)

    return data, np.array(labels), epoch_times


# =========================================================================
# BUILD FILE-SAFE SEQUENCE DATASET
# =========================================================================

def build_sequence_dataset(file_list, seq_len=10):

    X_all = []
    y_all = []
    metadata = []

    for file_name in file_list:

        print(f"Processing {file_name}...")

        seizure_times = get_seizure_times(
            summary_text,
            file_name
        )

        X, y, epoch_times = process_file(
            file_name,
            seizure_times
        )

        # ---------------------------------------------------------
        # CREATE SEQUENCES INSIDE THIS FILE ONLY
        # ---------------------------------------------------------

        X_seq = []
        y_seq = []
        sequence_times = []

        for i in range(len(X) - seq_len + 1):

            seq = X[i:i + seq_len]

            # Collapse EEG time dimension / channel averaging
            seq = seq.mean(axis=2)

            label_window = y[i:i + seq_len]

            label = 1 if np.any(label_window == 1) else 0

            X_seq.append(seq)
            y_seq.append(label)

            # Time of first epoch in this sequence
            sequence_times.append(epoch_times[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        X_all.append(X_seq)
        y_all.append(y_seq)

        metadata.extend([
            {
                "file": file_name,
                "start_time": time
            }
            for time in sequence_times
        ])

        print(
            f"  Epochs: {len(X):,} | "
            f"Sequences: {len(X_seq):,} | "
            f"Preictal: {np.sum(y_seq == 1):,}"
        )

    # Combine COMPLETED sequences across files
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return X_all, y_all, metadata


# =========================================================================
# TRAIN / TEST FILE SPLIT
# =========================================================================

train_files = [
    "chb01_01.edf",
    "chb01_02.edf",
    "chb01_03.edf",
    "chb01_05.edf",
    "chb01_06.edf",
    "chb01_07.edf",
    "chb01_08.edf",
    "chb01_09.edf"
]

test_files = [
    "chb01_04.edf",
    "chb01_10.edf",
    "chb01_11.edf",
    "chb01_12.edf"
]


# =========================================================================
# BUILD DATASETS AND VERIFY
# =========================================================================

print("\nBuilding Training Dataset...")
X_train_seq, y_train_seq, train_metadata = build_sequence_dataset(
    train_files,
    seq_len=10
)

print("\nBuilding Test Dataset...")
X_test_seq, y_test_seq, test_metadata = build_sequence_dataset(
    test_files,
    seq_len=10
)

print("\n==============================")
print("TRAINING DATA SUMMARY")
print("==============================")
print("Shape:", X_train_seq.shape)
print("Labels:", np.unique(y_train_seq, return_counts=True))

print("\n==============================")
print("TEST DATA SUMMARY")
print("==============================")
print("Shape:", X_test_seq.shape)
print("Labels:", np.unique(y_test_seq, return_counts=True))


# =========================================================================
# GLOBAL STANDARDIZATION
# =========================================================================

mean = X_train_seq.mean()
std = X_train_seq.std()

X_train_seq = (X_train_seq - mean) / std
X_test_seq = (X_test_seq - mean) / std


# =========================================================================
# LSTM MODEL SETUP
# =========================================================================

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(
            X_train_seq.shape[1],
            X_train_seq.shape[2]
        )
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =========================================================================
# TRAIN MODEL
# =========================================================================

print("\n==============================")
print("TRAINING MODEL")
print("==============================")

model.fit(
    X_train_seq,
    y_train_seq,
    batch_size=32,
    epochs=10,
    shuffle=True
)


# =========================================================================
# EVALUATE ON COMPLETE SEQUENTIAL TEST SET
# =========================================================================

print("\n==============================")
print("EVALUATING MODEL")
print("==============================")

pred_probs = model.predict(X_test_seq, batch_size=32).flatten()
preds = (pred_probs > 0.5).astype(int)

all_preds = preds
all_true = y_test_seq
all_probs = pred_probs


# =========================================================================
# EVALUATION RESULTS
# =========================================================================

print("\n========================================")
print("FINAL TEST EVALUATION")
print("========================================")

print("\nEvaluation samples:", len(all_true))

print("\nTrue labels:", np.unique(all_true, return_counts=True))

print("\nPredicted labels:", np.unique(all_preds, return_counts=True))

print("\nConfusion Matrix:")
print(confusion_matrix(all_true, all_preds))

print("\nClassification Report:")
print(classification_report(all_true, all_preds, zero_division=0))

# Model confidence checks
print("\nProbability range:")
print("Min:", all_probs.min())
print("Max:", all_probs.max())
print("Mean:", all_probs.mean())

# Compare confidence for true preictal vs normal windows
if np.any(all_true == 1):
    print("\nMean probability (True preictal):", all_probs[all_true == 1].mean())
else:
    print("\nMean probability (True preictal): No preictal samples present in test set")

if np.any(all_true == 0):
    print("Mean probability (Normal):", all_probs[all_true == 0].mean())

print("\nFirst 200 predictions:")
print("Predicted:", all_preds[:200])
print("Actual:", all_true[:200])


# =========================================================================
# PREICTAL PROBABILITY TIMELINE
# =========================================================================

sequence_duration = 2  # seconds per epoch step

time_axis = np.arange(len(all_probs)) * sequence_duration

plt.figure(figsize=(15, 5))

plt.plot(
    time_axis,
    all_probs,
    label="Predicted preictal probability"
)

# Mark actual preictal regions visually
in_preictal = False
start_time = None

for i, label in enumerate(all_true):

    if label == 1 and not in_preictal:
        start_time = time_axis[i]
        in_preictal = True

    elif label == 0 and in_preictal:
        plt.axvspan(
            start_time,
            time_axis[i],
            alpha=0.25,
            color="orange",
            label="Actual preictal" if "Actual preictal" not in plt.gca().get_legend_handles_labels()[1] else ""
        )
        in_preictal = False

if in_preictal:
    plt.axvspan(
        start_time,
        time_axis[-1],
        alpha=0.25,
        color="orange",
        label="Actual preictal" if "Actual preictal" not in plt.gca().get_legend_handles_labels()[1] else ""
    )

plt.axhline(
    0.5,
    color="red",
    linestyle="--",
    label="Decision threshold"
)

plt.xlabel("Time Index (Continuous Sequence Units)")
plt.ylabel("Preictal Probability")
plt.title("LSTM Preictal Probability Over Sequential Sequences")
plt.legend()
plt.tight_layout()
plt.show()