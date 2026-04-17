import os

# 1. Insert your exact Kaggle username and API key here (keep the quote marks!)
os.environ['KAGGLE_USERNAME'] = "Your_Username"
os.environ['KAGGLE_KEY'] = "Your_Kaggle_API"

# 2. Download the dataset directly from Kaggle
!kaggle datasets download -d vbookshelf/respiratory-sound-database

# 3. Unzip the dataset quietly (-q) into a folder named 'dataset'
!unzip -q respiratory-sound-database.zip -d dataset/
print("Dataset downloaded and unzipped!")

  import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# --- UPDATED PATHS ---
AUDIO_DIR = "dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"
CSV_PATH = "dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv"

TARGET_SR = 16000
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION)

def process_audio_chunks(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SR)
    total_samples = len(audio)
    spectrograms = []

    for start in range(0, total_samples, CHUNK_SAMPLES):
        end = start + CHUNK_SAMPLES
        chunk = audio[start:end]

        if len(chunk) < CHUNK_SAMPLES:
            if len(chunk) < TARGET_SR * 1.0:
                continue
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        # DATA AUGMENTATION
        versions_to_process = [chunk]
        versions_to_process.append(librosa.effects.pitch_shift(y=chunk, sr=sr, n_steps=-2))
        versions_to_process.append(librosa.effects.time_stretch(y=chunk, rate=1.1))

        # Process all 3 versions into spectrograms
        for version in versions_to_process:

            # augmented audio is EXACTLY 5 seconds long
            if len(version) > CHUNK_SAMPLES:
                version = version[:CHUNK_SAMPLES] # Trim excess
            elif len(version) < CHUNK_SAMPLES:
                version = np.pad(version, (0, CHUNK_SAMPLES - len(version))) # Pad missing

            mel_spec = librosa.feature.melspectrogram(y=version, sr=TARGET_SR, n_mels=128, fmax=8000)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            if log_mel_spec.max() == log_mel_spec.min():
                normalized = np.zeros_like(log_mel_spec)
            else:
                normalized = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())

            img = np.stack((normalized,)*3, axis=-1)
            spectrograms.append(img)

    return spectrograms

print("Loading diagnosis data...")
diagnosis_df = pd.read_csv(CSV_PATH, names=['Patient_ID', 'Disease'])
disease_dict = dict(zip(diagnosis_df.Patient_ID, diagnosis_df.Disease))

X = []
y = []

print("Processing audio files into 5-second chunks... This takes a few minutes.")
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        patient_id = int(filename.split('_')[0])

        if patient_id in disease_dict:
            disease = disease_dict[patient_id]
            filepath = os.path.join(AUDIO_DIR, filename)

            chunk_spectrograms = process_audio_chunks(filepath)

            for spec in chunk_spectrograms:
                X.append(spec)
                y.append(disease)

X = np.array(X)
y = np.array(y)

print(f"Processed into {len(X)} chunks. Shape of X: {X.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

np.save("X_data.npy", X)
np.save("y_data.npy", y_categorical)
np.save("classes.npy", encoder.classes_)
print("Preprocessing complete!")
