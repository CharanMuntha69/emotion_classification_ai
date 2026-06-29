Emotion Classification AI

A two-stage deep learning pipeline that detects emotion from audio using PyTorch. A CNN is first trained on the DEAM music dataset to predict valence and arousal, then its feature extractor is transferred to classify 8 speech emotions from the RAVDESS dataset.

How It Works


make_dataset.py — Loads DEAM audio files, converts them to 30-second log-mel spectrograms, matches them to valence/arousal annotations, and saves the processed dataset as .npy files.
train_pytorch.py — Trains a CNN regressor on the DEAM spectrograms to predict continuous valence and arousal values.
train_ravdess_transfer.py — Loads the pretrained CNN feature extractor and fine-tunes it on RAVDESS speech audio for 8-class emotion classification (neutral, calm, happy, sad, angry, fearful, disgust, surprised).


Tech Stack


Python, PyTorch, librosa, NumPy, pandas, matplotlib


Setup

1. Install dependencies

bashpip install -r requirements.txt

2. Download datasets


DEAM: https://cvml.unige.ch/databases/DEAM/
RAVDESS: https://zenodo.org/record/1188976


3. Expected folder structure

emotion_classification_ai/
├── DEAM_audio/
├── DEAM_Annotations/
│   └── annotations/annotations averaged per song/song_level/
├── RAVDESS/
│   └── Actor_01/ ... Actor_24/
├── processed/          # auto-created
├── make_dataset.py
├── train_pytorch.py
├── train_ravdess_transfer.py
└── requirements.txt

4. Run the pipeline

bash# Step 1: Build the dataset
python make_dataset.py

# Step 2: Train the music emotion regressor
python train_pytorch.py

# Step 3: Transfer learn on speech emotions
python train_ravdess_transfer.py

Model Architecture


3-layer CNN with MaxPooling and AdaptiveAvgPool
Regression head for valence/arousal (DEAM)
Classification head for 8 emotion classes (RAVDESS)
Evaluated with MSE, MAE, Pearson correlation (regression) and accuracy (classification)
