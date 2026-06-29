import os
import re
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import librosa.display


# Paths

AUDIO_DIR = r"./DEAM_audio"
ANN_DIR   = r"./DEAM_Annotations/annotations/annotations averaged per song/song_level"
OUT_DIR   = r"./processed"
os.makedirs(OUT_DIR, exist_ok=True)


# Spectrogram settings

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512
TARGET_SEC = 30


# Helpers

def extract_id(filename):
    match = re.search(r"\d+", filename)
    if not match:
        return None
    return int(match.group())

def audio_to_logmel(file_path):
    y, _ = librosa.load(file_path, sr=SR, mono=True)
    target_len = TARGET_SEC * SR

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP
    )

    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
   

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(logmel, sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Sample Spectrogram")
    plt.show()

    return logmel

def pick_col(options, cols):
    for o in options:
        if o in cols:
            return o
    return None


# Load annotations (robust: detect by columns, not filenames)

csv_files = [
    os.path.join(ANN_DIR, f)
    for f in os.listdir(ANN_DIR)
    if f.lower().endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError("No CSV files found in annotation folder: " + ANN_DIR)

print("CSV files found in song_level:")
for f in csv_files:
    print(" -", os.path.basename(f))

possible_id_cols = ["song_id", "songid", "song", "id", "track_id", "trackid"]
possible_v_cols  = ["valence", "mean_valence", "valence_mean", "avg_valence", "v"]
possible_a_cols  = ["arousal", "mean_arousal", "arousal_mean", "avg_arousal", "a"]

def pick_col(options, cols):
    for o in options:
        if o in cols:
            return o
    return None

def load_labels_from_single_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    id_col = pick_col(possible_id_cols, df.columns)
    v_col  = pick_col(possible_v_cols, df.columns)
    a_col  = pick_col(possible_a_cols, df.columns)

    # If one file contains both valence and arousal, we are done.
    if id_col and v_col and a_col:
        out = df[[id_col, v_col, a_col]].dropna()
        out[id_col] = out[id_col].astype(int)
        out = out.rename(columns={id_col: "song_id", v_col: "valence", a_col: "arousal"})
        return out

    return None

def load_labels_from_two_csvs(csv_list):
    # Try to find separate valence-only and arousal-only files by reading columns
    val_df = None
    aro_df = None

    for path in csv_list:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        id_col = pick_col(possible_id_cols, df.columns)
        v_col  = pick_col(possible_v_cols, df.columns)
        a_col  = pick_col(possible_a_cols, df.columns)

        # valence-only file: has id + valence, no arousal
        if id_col and v_col and (a_col is None) and val_df is None:
            tmp = df[[id_col, v_col]].dropna().rename(columns={id_col: "song_id", v_col: "valence"})
            tmp["song_id"] = tmp["song_id"].astype(int)
            val_df = tmp
            continue

        # arousal-only file: has id + arousal, no valence
        if id_col and a_col and (v_col is None) and aro_df is None:
            tmp = df[[id_col, a_col]].dropna().rename(columns={id_col: "song_id", a_col: "arousal"})
            tmp["song_id"] = tmp["song_id"].astype(int)
            aro_df = tmp
            continue

    if val_df is None or aro_df is None:
        return None

    return val_df.merge(aro_df, on="song_id", how="inner")

# 1) First try: a single CSV contains both
labels_df = None
for path in csv_files:
    labels_df = load_labels_from_single_csv(path)
    if labels_df is not None:
        print("Using single annotation CSV:", os.path.basename(path))
        break

# 2) Second try: two CSVs (one valence, one arousal)
if labels_df is None:
    labels_df = load_labels_from_two_csvs(csv_files)
    if labels_df is not None:
        print("Using two annotation CSVs (valence-only + arousal-only).")

if labels_df is None:
    raise ValueError(
        "Could not detect valence/arousal columns in the song_level CSV files.\n"
        "Open the CSVs and check their column headers."
    )

labels = {
    int(r.song_id): (float(r.valence), float(r.arousal))
    for r in labels_df.itertuples(index=False)
}

print("Loaded labels:", len(labels), "songs")



# Scan audio recursively

audio_paths = []
for root, _, files in os.walk(AUDIO_DIR):
    for f in files:
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            audio_paths.append(os.path.join(root, f))

audio_paths.sort()

print("Found audio files:", len(audio_paths))

# Quick ID sanity checks
label_ids = sorted(labels.keys())
print("Label ID range:", label_ids[0], "to", label_ids[-1])

audio_ids = []
for p in audio_paths[:30]:
    sid = extract_id(os.path.basename(p))
    if sid is not None:
        audio_ids.append(sid)
print("First audio IDs:", audio_ids)

# Count how many audio IDs overlap with label IDs (fast estimate)
label_set = set(labels.keys())
overlap = sum(1 for p in audio_paths if (extract_id(os.path.basename(p)) in label_set))
print("Audio/label ID overlaps:", overlap)


# Match audio to labels

X_list = []
y_list = []
kept_ids = []
kept_files = []

for path in audio_paths:
    fname = os.path.basename(path)
    sid = extract_id(fname)

    if sid is None:
        continue

    if sid not in labels:
        continue

    try:
        spec = audio_to_logmel(path)
    except Exception as e:
        print("Skipping", fname, "error:", e)
        continue

    X_list.append(spec)
    y_list.append(labels[sid])
    kept_ids.append(sid)
    kept_files.append(fname)

if not X_list:
    raise RuntimeError(
        "No training samples were created.\n"
        "Audio IDs do not match annotation IDs."
    )

X = np.stack(X_list, axis=0)
y = np.array(y_list, dtype=np.float32)


# Save dataset

np.save(os.path.join(OUT_DIR, "X_logmel.npy"), X)
np.save(os.path.join(OUT_DIR, "y_valence_arousal.npy"), y)

index_df = pd.DataFrame({
    "song_id": kept_ids,
    "audio_file": kept_files,
    "valence": y[:, 0],
    "arousal": y[:, 1],
})

index_df.to_csv(os.path.join(OUT_DIR, "index.csv"), index=False)

print("Done")
print("N samples:", X.shape[0])
print("Spectrogram shape per sample:", X.shape[1:])
print("y shape:", y.shape)

