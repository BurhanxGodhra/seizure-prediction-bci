from pathlib import Path
import re
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed, Conv1D, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# =========================================================================
# PATH SETUP
# =========================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "chb01"
SUMMARY_FILE = DATA_DIR / "chb01-summary.txt"


def read_summary():
    """Return summary text or empty string if file is missing."""
    try:
        with open(SUMMARY_FILE, "r") as f:
            return f.read()
    except Exception:
        return ""


summary_text = read_summary()
if not summary_text:
    raise SystemExit(f"Could not read seizure summary file at {SUMMARY_FILE}")


# =========================================================================
# SEIZURE TIMING
# =========================================================================

def get_seizure_times(summary_text, target_file):
    """
    Robustly parse the summary text for seizure start/end times (in seconds)
    Returns list of (start, end) tuples. If no seizures, returns [].
    """
    if not summary_text:
        return []

    # Find block for the target file
    pattern = r"File Name:\s*" + re.escape(target_file) + r"([\s\S]*?)(?:File Name:|$)"
    m = re.search(pattern, summary_text)
    if not m:
        return []

    block = m.group(1)

    # Find all start/end pairs inside the block
    starts = re.findall(r"Seizure Start Time:\s*(\d+)", block)
    ends = re.findall(r"Seizure End Time:\s*(\d+)", block)

    pairs = []
    for s, e in zip(starts, ends):
        try:
            pairs.append((int(s), int(e)))
        except ValueError:
            continue

    return pairs


# =========================================================================
# PROCESS ONE EDF FILE
# =========================================================================

def process_file(file_name, seizure_times, resample_sfreq=128, epoch_duration=2):
    """
    Returns:
      data: ndarray, shape (n_epochs, n_channels, n_times)
      labels: ndarray, shape (n_epochs,) with values:
        0 -> normal
        1 -> preictal
       -1 -> ictal (will be excluded)
    """
    file_path = DATA_DIR / file_name

    raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)

    # Reduce sampling rate to reduce computational load
    raw.resample(resample_sfreq)

    # Divide continuous EEG into fixed-length epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration)

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    labels = []

    PREICTAL_WINDOW = 300  # 5 minutes

    # event times in seconds (start of each epoch)
    epoch_times = epochs.events[:, 0] / raw.info["sfreq"]

    for t in epoch_times:
        label = 0
        for start, end in seizure_times:
            preictal_start = max(0, start - PREICTAL_WINDOW)
            if preictal_start <= t < start:
                label = 1
                break
            if start <= t < end:
                label = -1  # ictal
                break
        labels.append(label)

    return data, np.array(labels)


# =========================================================================
# TRAIN / VAL / TEST FILES (session-level split recommended)
# =========================================================================

train_files = [
    "chb01_01.edf",
    "chb01_02.edf",
    "chb01_03.edf",
    "chb01_05.edf",
    "chb01_06.edf",
    "chb01_07.edf",
    "chb01_15.edf",
    "chb01_18.edf"
]

val_files = [
    "chb01_04.edf",
    "chb01_16.edf"
]

test_files = [
    "chb01_21.edf",
    "chb01_26.edf"
]


# =========================================================================
# LOAD DATA (per-file dataset, do NOT concatenate across files)
# =========================================================================

def build_file_dataset(file_list, exclude_ictal=True):
    """Return a list of file-level records: {file, X, y, epoch_times}"""
    records = []
    for file in file_list:
        seizure_times = get_seizure_times(summary_text, file)
        X, y = process_file(file, seizure_times)
        if exclude_ictal:
            # keep timeline but sequences with ictal will be rejected later
            pass
        if X.size == 0:
            continue
        # also compute epoch start times (in seconds)
        # Recreate epoch_times same as process_file used
        # We can reconstruct by number of epochs and epoch_duration and assume start at 0
        n_epochs = X.shape[0]
        epoch_times = np.arange(n_epochs) * 2.0
        records.append({
            'file': file,
            'X': X,
            'y': y,
            'epoch_times': epoch_times
        })
    return records


train_data = build_file_dataset(train_files)
val_data = build_file_dataset(val_files)
test_data = build_file_dataset(test_files)

# Quick label counts (dataset-wise)
def count_labels(records):
    s0 = s1 = 0
    for r in records:
        y = r['y']
        s0 += int((y == 0).sum())
        s1 += int((y == 1).sum())
    return s0, s1

# Ensure validation contains positive examples; if not, move one positive file from train -> val.
def ensure_validation_has_positive(train_records, val_records):
    _, val_pos = count_labels(val_records)
    if val_pos > 0:
        return
    # find a train file with positives
    for i, r in enumerate(train_records):
        if (r['y'] == 1).sum() > 0:
            moved = train_records.pop(i)
            val_records.append(moved)
            print(f"Moved {moved['file']} from train -> val to ensure validation has positives")
            return
    print('Warning: no positive files available in train to move to validation')

ensure_validation_has_positive(train_data, val_data)

train_counts = count_labels(train_data)
val_counts = count_labels(val_data)
test_counts = count_labels(test_data)

print('\nTrain label counts (epochs):', train_counts)
print('Val label counts (epochs):', val_counts)
print('Test label counts (epochs):', test_counts)


# =========================================================================
# GLOBAL STANDARDIZATION: compute per-channel mean/std using only training data
# =========================================================================

# accumulate mean/std across training files without creating impossible sequences
sum_ = 0.0
sumsq = 0.0
count = 0
for r in train_data:
    X = r['X']  # (n_epochs, n_chan, n_times)
    sum_ += X.sum(axis=(0, 2), keepdims=True)
    sumsq += (X ** 2).sum(axis=(0, 2), keepdims=True)
    count += X.shape[0] * X.shape[2]

if count == 0:
    raise SystemExit('No training data found')

chan_mean = sum_ / count  # shape (1, n_chan, 1)
chan_var = sumsq / count - chan_mean ** 2
chan_std = np.sqrt(np.maximum(chan_var, 1e-10))

# apply normalization per file
for r in train_data + val_data + test_data:
    r['X'] = (r['X'] - chan_mean) / chan_std

# -------------------------------------------------------------------------
# Calibration helpers
# -------------------------------------------------------------------------

def fit_isotonic_calibrator(model, records, seq_len=None):
    """Fit an isotonic regression calibrator on validation probabilities.
    seq_len: if None, uses the global SEQ_LEN (must be defined later in file). Returns the fitted IsotonicRegression or None if fitting not possible."""
    if seq_len is None:
        try:
            seq_len = SEQ_LEN
        except NameError:
            # fallback sensible default
            seq_len = 10
    probs, labels, _ = collect_probabilities(model, records, seq_len=seq_len)
    if probs.size == 0:
        return None
    # need both classes present to fit
    if len(np.unique(labels)) < 2:
        print('Calibration skipped: validation labels are single-class')
        return None
    try:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(probs, labels)
        print('Fitted isotonic calibrator on validation set')
        return iso
    except Exception as e:
        print('Calibration fit failed:', e)
        return None


class PlattCalibrator:
    """Simple wrapper for Platt scaling (logistic regression) as a calibrator with a predict method."""
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', max_iter=2000)
        self.fitted = False

    def fit(self, probs, labels):
        # reshape and fit
        X = np.asarray(probs).reshape(-1, 1)
        y = np.asarray(labels).ravel()
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, probs):
        if not self.fitted or probs.size == 0:
            return probs
        X = np.asarray(probs).reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


def fit_platt_calibrator(model, records, seq_len=None):
    """Fit Platt scaling (logistic regression) on validation probabilities."""
    if seq_len is None:
        try:
            seq_len = SEQ_LEN
        except NameError:
            seq_len = 10
    probs, labels, _ = collect_probabilities(model, records, seq_len=seq_len)
    if probs.size == 0:
        return None
    if len(np.unique(labels)) < 2:
        print('Platt calibration skipped: validation labels are single-class')
        return None
    try:
        pc = PlattCalibrator()
        pc.fit(probs, labels)
        print('Fitted Platt (logistic) calibrator on validation set')
        return pc
    except Exception as e:
        print('Platt calibration failed:', e)
        return None


def apply_calibrator(calibrator, probs):
    if calibrator is None or probs.size == 0:
        return probs
    try:
        return calibrator.predict(probs)
    except Exception:
        return probs

# -------------------------------------------------------------------------


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        w = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -w * tf.pow(1 - pt, gamma) * tf.math.log(pt)
    def reduced_loss(y_true, y_pred):
        return tf.reduce_mean(loss(y_true, y_pred))
    return reduced_loss


# =========================================================================
# BUILD INDICES PER FILE (valid sequence start indices that do not contain ictal epochs)
# =========================================================================

SEQ_LEN = 20  # next-phase temporal context: 40s windows

def compute_valid_start_indices(records, seq_len=SEQ_LEN):
    for r in records:
        y = r['y']
        n_epochs = len(y)
        pos_idx = []
        neg_idx = []
        for i in range(n_epochs - seq_len + 1):
            window = y[i:i+seq_len]
            if np.any(window == -1):
                continue
            label = 1 if np.any(window == 1) else 0
            if label == 1:
                pos_idx.append(i)
            else:
                neg_idx.append(i)
        r['pos_idx'] = np.array(pos_idx, dtype=np.int32)
        r['neg_idx'] = np.array(neg_idx, dtype=np.int32)

# compute indices for datasets
compute_valid_start_indices(train_data, SEQ_LEN)
compute_valid_start_indices(val_data, SEQ_LEN)
compute_valid_start_indices(test_data, SEQ_LEN)


def compute_sequence_label_stats(records, seq_len=SEQ_LEN):
    """Compute and print sequence-level positive/negative counts using the current labeling rule.
    This helps confirm the effective class balance seen by the model during training."""
    total_pos = 0
    total_neg = 0
    for r in records:
        y = r['y']
        n_epochs = len(y)
        for i in range(n_epochs - seq_len + 1):
            window = y[i:i+seq_len]
            if np.any(window == -1):
                continue
            last_k = max(1, seq_len // 4)
            label = 1 if np.any(window[-last_k:] == 1) else 0
            if label == 1:
                total_pos += 1
            else:
                total_neg += 1
    total = total_pos + total_neg
    print(f"Sequence stats (seq_len={seq_len}): pos={total_pos}, neg={total_neg}, total={total}")
    return total_pos, total_neg

# quick sanity checks
print('Training/validation/test sequence-level balance:')
compute_sequence_label_stats(train_data, SEQ_LEN)
compute_sequence_label_stats(val_data, SEQ_LEN)
compute_sequence_label_stats(test_data, SEQ_LEN)


# =========================================================================
# FILE-LEVEL TRAINING GENERATOR
# =========================================================================

def file_level_train_generator(records, seq_len=SEQ_LEN, batch_size=32, pos_prob=0.85, positive_weight=6.0):
    rng = np.random.default_rng()
    pos_files = [i for i, r in enumerate(records) if r.get('pos_idx', []).size > 0]
    neg_files = [i for i, r in enumerate(records) if r.get('neg_idx', []).size > 0]

    if not pos_files or not neg_files:
        raise ValueError('Need both positive and negative sequences in training records')

    while True:
        X_batch = []
        y_batch = []
        attempts = 0
        while len(X_batch) < batch_size and attempts < batch_size * 20:
            attempts += 1
            take_pos = rng.random() < pos_prob
            if take_pos:
                fidx = int(rng.choice(pos_files))
                start = int(rng.choice(records[fidx]['pos_idx']))
            else:
                fidx = int(rng.choice(neg_files))
                start = int(rng.choice(records[fidx]['neg_idx']))

            seq = records[fidx]['X'][start:start+seq_len]
            # LABELING: positive only if the final portion approaches preictal
            last_k = max(1, seq_len // 4)
            label_window = records[fidx]['y'][start:start+seq_len]
            label = 1 if np.any(label_window[-last_k:] == 1) else 0

            # stronger augmentation for positives
            if label == 1 and rng.random() < 0.5:
                seq = seq + rng.normal(scale=0.01, size=seq.shape)

            X_batch.append(seq)
            y_batch.append(label)

        X_arr = np.array(X_batch, dtype=np.float32)
        y_arr = np.array(y_batch, dtype=np.int32)
        sample_weight = np.where(y_arr == 1, positive_weight, 1.0).astype(np.float32)
        yield X_arr, y_arr, sample_weight


# =========================================================================
# FILE-LEVEL EVALUATION GENERATOR (sequential, per-file)
# =========================================================================

def file_level_evaluation_generator(records, seq_len=SEQ_LEN, batch_size=32):
    X_batch = []
    y_batch = []
    meta = []
    for r in records:
        n_epochs = len(r['y'])
        for i in range(n_epochs - seq_len + 1):
            window = r['y'][i:i+seq_len]
            if np.any(window == -1):
                continue
            last_k = max(1, seq_len // 4)
            label = 1 if np.any(window[-last_k:] == 1) else 0
            seq = r['X'][i:i+seq_len]
            X_batch.append(seq)
            y_batch.append(label)
            meta.append((r['file'], i))
            if len(X_batch) == batch_size:
                yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.int32), meta
                X_batch = []
                y_batch = []
                meta = []
    if X_batch:
        yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.int32), meta


# wrapper to use only (X,y) for validation in Keras fit
def val_generator_xy(records, seq_len=SEQ_LEN, batch_size=32):
    # Must loop forever: Keras reuses this single generator object across
    # every epoch of model.fit(), pulling `validation_steps` batches each
    # time. A single-pass generator exhausts after epoch 1, silently
    # producing zero validation batches (and val_precision/val_recall stuck
    # at 0.0) for every epoch after that.
    while True:
        for Xb, yb, _ in file_level_evaluation_generator(records, seq_len=seq_len, batch_size=batch_size):
            yield Xb, yb


# =========================================================================
# BUILD & TRAIN MODEL
# =========================================================================

n_chan = train_data[0]['X'].shape[1]
if n_chan == 0:
    raise SystemExit('No channels found in training data')
n_time = train_data[0]['X'].shape[2]

from tensorflow.keras.models import Model

def build_td_cnn_lstm(seq_len=10, n_chan=23, n_time=256, conv_filters=32, lstm_units=64):
    """Build TimeDistributed Conv1D -> LSTM model. Input shape: (seq_len, n_chan, n_time)."""
    inp = Input(shape=(seq_len, n_chan, n_time), name='input_layer')
    x = Permute((1, 3, 2))(inp)
    x = TimeDistributed(Conv1D(
        conv_filters,
        kernel_size=5,
        activation='relu',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    ))(x)
    x = TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(GlobalAveragePooling1D())(x)
    x = LSTM(
        lstm_units,
        dropout=0.35,
        recurrent_dropout=0.2,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return Model(inputs=inp, outputs=out)

model = build_td_cnn_lstm(seq_len=SEQ_LEN, n_chan=n_chan, n_time=n_time, conv_filters=32, lstm_units=64)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss(alpha=0.75, gamma=2.0),
    metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
)

train_gen = file_level_train_generator(train_data, seq_len=SEQ_LEN, batch_size=32, pos_prob=0.5, positive_weight=6.0)
val_gen = val_generator_xy(val_data, seq_len=SEQ_LEN, batch_size=32)

steps = max(1, sum(len(r['pos_idx']) + len(r['neg_idx']) for r in train_data) // 32)
vsteps = max(1, sum((len(r['y']) - SEQ_LEN + 1 -  np.sum(r['y'] == -1)) for r in val_data) // 32)

es = EarlyStopping(monitor='val_recall', mode='max', patience=8, restore_best_weights=True)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

model.fit(train_gen, steps_per_epoch=steps, validation_data=val_gen, validation_steps=vsteps, epochs=20, callbacks=[es, rlr])  # longer training for improved learning

# save the trained model for faster future evaluation-only runs
try:
    save_dir = os.path.join('results', 'models')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'baseline.h5')
    model.save(model_path)
    print(f"Saved trained model to {model_path}")
except Exception as e:
    print('Failed to save model:', e)


# =========================================================================
# VALIDATION-ONLY THRESHOLD SELECTION
# =========================================================================

def collect_probabilities(model, records, seq_len=SEQ_LEN, batch_size=32):
    """Return probabilities, true labels, and metadata for a set of records."""
    probs = []
    labels = []
    meta = []
    for Xb, yb, m in file_level_evaluation_generator(records, seq_len=seq_len, batch_size=batch_size):
        p = model.predict(Xb, verbose=0).flatten()
        probs.extend(p.tolist())
        labels.extend(yb.tolist())
        meta.extend(m)
    return np.asarray(probs, dtype=np.float32), np.asarray(labels, dtype=np.int32), np.asarray(meta, dtype=object)


def event_alarm_mask(probs, threshold=0.5, min_consecutive=3, smoothing_window=5):
    """Turn probabilities into a smoothed alarm mask via persistence requirement."""
    if len(probs) == 0:
        return np.array([], dtype=bool)
    smooth = np.convolve(probs, np.ones(smoothing_window) / smoothing_window, mode='same')
    raw_alarm = smooth >= threshold
    if raw_alarm.size == 0:
        return raw_alarm
    if min_consecutive <= 1:
        return raw_alarm
    conv = np.convolve(raw_alarm.astype(int), np.ones(min_consecutive), mode='same')
    return conv >= min_consecutive


def select_best_event_policy(model, records, seq_len=SEQ_LEN, candidate_thresholds=None, candidate_min_streak=(1, 2, 3, 5), candidate_smoothing_windows=(3,5,9), min_recall=0.60, max_fa_per_hour=0.5, calibrator=None):
    """Use validation-only event-level metrics to choose an alarm policy without using the final test set.
    Sweeps threshold, persistence (min_streak) and smoothing window. If calibrator is provided, probabilities are calibrated first."""
    # collect validation probabilities first (per-file)
    if candidate_thresholds is None:
        candidate_thresholds = None  # will be defined from validation distribution below

    probs_by_file = {}
    meta_by_file = {}
    all_val_probs = []
    for Xb, yb, m in file_level_evaluation_generator(records, seq_len=seq_len, batch_size=32):
        p = model.predict(Xb, verbose=0).flatten()
        if calibrator is not None:
            p = apply_calibrator(calibrator, p)
        all_val_probs.extend(p.tolist())
        for idx, (file, start) in enumerate(m):
            if file not in probs_by_file:
                probs_by_file[file] = []
                meta_by_file[file] = []
            probs_by_file[file].append(float(p[idx]))
            meta_by_file[file].append(int(start))

    all_val_probs = np.asarray(all_val_probs, dtype=np.float32)
    if candidate_thresholds is None:
        if all_val_probs.size == 0:
            candidate_thresholds = np.linspace(0.01, 0.9, 18)
        else:
            # avoid extremes; set lower bound to at least the 5th percentile but no lower than 0.03
            low = max(0.03, float(np.percentile(all_val_probs, 5)))
            high = float(np.percentile(all_val_probs, 99))
            if high <= low:
                high = min(low + 0.2, 0.95)
            candidate_thresholds = np.linspace(low, high, 30)

    best = None
    best_details = []
    total_seizures = sum(len(get_seizure_times(summary_text, r['file'])) for r in records)
    # helper: compute predicted-positive fraction on validation for a threshold
    def predicted_positive_fraction(th):
        if all_val_probs.size == 0:
            return 0.0
        return float(np.mean(all_val_probs >= th))

    for threshold in candidate_thresholds:
        # skip trivial thresholds that mark too many windows positive on validation
        if predicted_positive_fraction(threshold) > 0.30:
            # skip thresholds that produce >30% positive windows on validation (likely trivial)
            continue
        for min_streak in candidate_min_streak:
            for smoothing_window in candidate_smoothing_windows:
                total_sensitivity = 0.0
                total_fa = 0.0
                total_hours = 0.0
                for r in records:
                    file = r['file']
                    if file not in probs_by_file:
                        continue
                    probs = np.asarray(probs_by_file[file], dtype=np.float32)
                    starts = np.asarray(meta_by_file[file], dtype=np.int32)
                    alarm = event_alarm_mask(probs, threshold=threshold, min_consecutive=min_streak, smoothing_window=smoothing_window)
                    alarm_times = starts[alarm] * 2.0
                    seizures = get_seizure_times(summary_text, file)
                    total_hours += (len(r['y']) * 2.0) / 3600.0
                    if not seizures:
                        if alarm_times.size > 0:
                            total_fa += 1.0
                        continue
                    for (s, e) in seizures:
                        pre_start = max(0, s - 300)
                        if np.any((alarm_times >= pre_start) & (alarm_times < s)):
                            total_sensitivity += 1.0
                    if alarm_times.size:
                        alarm_clusters = np.where(np.diff(np.r_[False, alarm, False]) == 1)[0]
                        alarm_cluster_ends = np.where(np.diff(np.r_[False, alarm, False]) == -1)[0]
                        for cs, ce in zip(alarm_clusters, alarm_cluster_ends):
                            center = (cs + ce) / 2.0
                            center_time = starts[center.astype(int)] * 2.0 if center < len(starts) else starts[-1] * 2.0
                            if not any((center_time >= max(0, ss - 300) and center_time < ee) for ss, ee in seizures):
                                total_fa += 1.0

                sensitivity = total_sensitivity / max(total_seizures, 1)
                fa_per_hr = total_fa / max(total_hours, 1e-6)

                if sensitivity < min_recall or fa_per_hr > max_fa_per_hour:
                    score = -1e9 + 10.0 * sensitivity - 20.0 * fa_per_hr
                else:
                    score = 2.0 * sensitivity + 1.0 - 4.0 * fa_per_hr

                details = {
                    'threshold': float(threshold),
                    'min_streak': int(min_streak),
                    'smoothing_window': int(smoothing_window),
                    'sensitivity': sensitivity,
                    'fa_per_hr': fa_per_hr,
                    'score': score,
                }
                best_details.append(details)
                if best is None or score > best['score']:
                    best = details

    if best is None:
        return 0.5, 1

    print('\nValidation event-policy sweep (validation only):')
    for d in best_details:
        print(f"  threshold={d['threshold']:.2f} | min_streak={d['min_streak']} | smoothing={d['smoothing_window']} | sensitivity={d['sensitivity']:.3f} | fa/hr={d['fa_per_hr']:.3f} | score={d['score']:.3f}")
    print(f"Selected event policy: threshold={best['threshold']:.2f}, min_streak={best['min_streak']}, smoothing={best['smoothing_window']} (sensitivity={best['sensitivity']:.3f}, fa/hr={best['fa_per_hr']:.3f})")
    return float(best['threshold']), int(best['min_streak']), int(best['smoothing_window'])


# Fit a calibrator on validation (try Platt first, fallback to isotonic) and select event policy using calibrated probabilities
calibrator = fit_platt_calibrator(model, val_data, seq_len=SEQ_LEN)
if calibrator is None:
    calibrator = fit_isotonic_calibrator(model, val_data, seq_len=SEQ_LEN)

# ---------------------------------------------------------------------
# DIAGNOSTIC: raw (pre-calibration) vs calibrated probability spread,
# split by true label, on validation data. This tells us whether a
# narrow probability band is coming from the model itself (raw stats
# below are already compressed) or is introduced/worsened by the
# calibrator (calibrated stats are much narrower than raw).
# ---------------------------------------------------------------------
diag_probs, diag_labels, _ = collect_probabilities(model, val_data, seq_len=SEQ_LEN)
diag_calibrated = apply_calibrator(calibrator, diag_probs) if calibrator is not None else diag_probs

print("\n==============================")
print("PROBABILITY SPREAD DIAGNOSTIC (validation set)")
print("==============================")
for label, name in [(0, "Normal"), (1, "Preictal")]:
    mask = diag_labels == label
    n = int(mask.sum())
    if n == 0:
        print(f"{name}: no samples in validation set")
        continue
    raw_vals = diag_probs[mask]
    cal_vals = diag_calibrated[mask]
    print(f"{name} (n={n}):")
    print(f"  raw        min={raw_vals.min():.4f} max={raw_vals.max():.4f} mean={raw_vals.mean():.4f} std={raw_vals.std():.4f}")
    print(f"  calibrated min={cal_vals.min():.4f} max={cal_vals.max():.4f} mean={cal_vals.mean():.4f} std={cal_vals.std():.4f}")
print(f"\nSeparation check (raw, mean preictal - mean normal): {diag_probs[diag_labels==1].mean() - diag_probs[diag_labels==0].mean():.4f}" if np.any(diag_labels==1) and np.any(diag_labels==0) else "")

if np.any(diag_labels == 1) and np.any(diag_labels == 0):
    val_auc_raw = roc_auc_score(diag_labels, diag_probs)
    val_auc_cal = roc_auc_score(diag_labels, diag_calibrated)
    print(f"Validation ROC-AUC (raw): {val_auc_raw:.4f}  (calibrated, should match — Platt/isotonic are order-preserving: {val_auc_cal:.4f})")
    print("(0.50 = no separation at all, 1.00 = perfect separation, regardless of threshold)")

best_threshold, best_min_streak, best_smoothing = select_best_event_policy(
    model,
    val_data,
    seq_len=SEQ_LEN,
    calibrator=calibrator,
)

# =========================================================================
# EVALUATE ON TEST SET (sequential, file-by-file) and compute event-level metrics
# =========================================================================

all_probs = []
all_true = []
all_meta = []  # list of (file, start_idx)

for Xb, yb, meta in file_level_evaluation_generator(test_data, seq_len=SEQ_LEN, batch_size=32):
    p = model.predict(Xb, verbose=0).flatten()
    if calibrator is not None:
        p = apply_calibrator(calibrator, p)
    all_probs.extend(p.tolist())
    all_true.extend(yb.tolist())
    all_meta.extend(meta)

all_probs = np.array(all_probs)
all_true = np.array(all_true)
all_meta = np.array(all_meta, dtype=object)

# =========================================================================
# EVALUATION RESULTS (sequence-level) and event-level metrics (per-file)
# =========================================================================

# sequence-level predictions using validation-selected threshold, not 0.5
all_preds = (all_probs >= best_threshold).astype(int)

print("\n========================================")
print("FINAL TEST EVALUATION (sequence-level)")
print("========================================")

print("\nEvaluation samples:", len(all_true))
print("\nTrue labels:", np.unique(all_true, return_counts=True))
print("\nPredicted labels:", np.unique(all_preds, return_counts=True))

print("\nConfusion Matrix:")
print(confusion_matrix(all_true, all_preds))

print("\nClassification Report:")
print(classification_report(all_true, all_preds, zero_division=0))

# probability diagnostics
if all_probs.size:
    print("Probability range:")
    print("Min:", all_probs.min())
    print("Max:", all_probs.max())
    print("Mean:", all_probs.mean())

    if np.any(all_true == 1):
        print("Mean prob (true preictal):", all_probs[all_true == 1].mean())
    if np.any(all_true == 0):
        print("Mean prob (normal):", all_probs[all_true == 0].mean())

    if np.any(all_true == 1) and np.any(all_true == 0):
        print("Test ROC-AUC:", roc_auc_score(all_true, all_probs), "(0.50 = no separation, 1.00 = perfect, independent of threshold)")

# Event-level metrics per file
print('\nEvent-level metrics per test file:')
file_results = []
meta_arr = np.array(all_meta, dtype=object)
for r in test_data:
    file = r['file']
    n_epochs = len(r['y'])
    n_seq = max(0, n_epochs - SEQ_LEN + 1)
    probs = np.full(n_seq, np.nan)

    # find indices in meta_arr matching this file
    matches = [i for i, m in enumerate(all_meta) if m[0] == file]
    for k, idx in enumerate(matches):
        start = all_meta[idx][1]
        probs[start] = all_probs[idx]

    # replace nan (due to skipped sequences from ictal windows) with 0 for plotting/processing
    probs_clean = np.nan_to_num(probs, nan=0.0)

    # smooth and hysteresis; use validation-selected threshold + persistence requirement
    smooth = np.convolve(probs_clean, np.ones(best_smoothing) / best_smoothing, mode='same') if len(probs_clean) > 0 else np.array([])
    raw_alarm = smooth >= best_threshold
    conv = np.convolve(raw_alarm.astype(int), np.ones(best_min_streak), mode='same') if len(raw_alarm) > 0 else np.array([])
    final_alarm = conv >= best_min_streak if len(conv) > 0 else np.array([])

    # seizures for this file
    seizures = get_seizure_times(summary_text, file)

    # compute detection
    detected = 0
    detected_times = []
    lead_times = []
    for (s, e) in seizures:
        pre_start = max(0, s - 300)
        epoch_times = np.arange(n_seq) * 2.0
        mask = (epoch_times >= pre_start) & (epoch_times < s)
        if mask.any() and np.any(final_alarm[mask]):
            detected += 1
            alarm_positions = np.where(final_alarm[mask])[0]
            first_alarm_idx = alarm_positions[0]
            alarm_time = epoch_times[mask][first_alarm_idx]
            detected_times.append(alarm_time)
            lead_times.append(float(s - alarm_time))

    sensitivity = detected / max(len(seizures), 1)

    # false alarms per hour
    alarm_idx = np.where(final_alarm)[0] if final_alarm.size else np.array([])
    clusters = []
    if alarm_idx.size:
        cur = [alarm_idx[0]]
        for a in alarm_idx[1:]:
            if a - cur[-1] <= 2:
                cur.append(a)
            else:
                clusters.append(cur)
                cur = [a]
        clusters.append(cur)
    cluster_times = [ (np.arange(n_seq) * 2.0)[c[0]] for c in clusters] if clusters else []

    false_clusters = 0
    total_hours = (n_seq * 2.0) / 3600.0
    for t in cluster_times:
        overlaps = False
        for (s, e) in seizures:
            if (t >= max(0, s - 300) and t < e):
                overlaps = True
                break
        if not overlaps:
            false_clusters += 1
    fa_per_hr = false_clusters / max(total_hours, 1e-6)

    lead_time_mean = float(np.mean(lead_times)) if lead_times else 0.0
    file_results.append({
        'file': file,
        'n_seizures': len(seizures),
        'sensitivity': sensitivity,
        'fa_per_hr': fa_per_hr,
        'mean_lead_time_sec': lead_time_mean,
    })

print('\nPer-file event metrics:')
for fr in file_results:
    print(fr)

if file_results:
    print('\nAverage sensitivity:', np.mean([f['sensitivity'] for f in file_results]))
    print('Average false alarms/hr:', np.mean([f['fa_per_hr'] for f in file_results]))
    print('Average lead time (s):', np.mean([f['mean_lead_time_sec'] for f in file_results]))

# ---------------------------------------------------------
# PREICTAL PROBABILITY TIMELINE (overall concatenated)
# ---------------------------------------------------------

sequence_duration = 2  # seconds per epoch

# build a global time axis by concatenating files
time_axis = []
for r in test_data:
    n_seq = max(0, len(r['y']) - SEQ_LEN + 1)
    # epoch times for sequences in this file
    times = np.arange(n_seq) * sequence_duration
    if time_axis:
        offset = time_axis[-1] + sequence_duration
    else:
        offset = 0
    time_axis.extend((times + offset).tolist())

if len(time_axis) > 0:
    time_axis = np.array(time_axis)
else:
    time_axis = np.array([])

plt.figure(figsize=(15, 5))
if all_probs.size and time_axis.size and len(time_axis) == len(all_probs):
    plt.plot(time_axis, all_probs, label='Predicted preictal probability')

# Mark actual preictal regions per file
offset = 0
for r in test_data:
    n_epochs = len(r['y'])
    seizures = get_seizure_times(summary_text, r['file'])
    for (s, e) in seizures:
        # find epoch indices that fall into preictal [s-300, s)
        pre_start = max(0, s - 300)
        pre_end = s
        plt.axvspan(offset + pre_start, offset + pre_end, color='red', alpha=0.25)
    offset += n_epochs * sequence_duration

plt.axhline(best_threshold, color='gray', linestyle='--', label=f'Decision threshold={best_threshold:.2f}')
plt.xlabel('Time (seconds)')
plt.ylabel('Preictal probability')
plt.title('Preictal Probability Timeline (test)')
plt.legend()
plt.tight_layout()
plt.show()
