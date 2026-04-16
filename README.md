# PulmoSense

**Edge-Optimised Deep Learning for Real-Time Acoustic Respiratory Disease Screening**
<img width="1584" height="773" alt="Git screenshot" src="https://github.com/user-attachments/assets/1ccec420-007d-4a2d-943c-db3c17ee2354" />
<img width="1584" height="832" alt="Git screenshot" src="https://github.com/user-attachments/assets/17acb9ea-02a5-4621-ab7f-f9cf2c7756b8" />

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Clinical Problem Statement](#2-clinical-problem-statement)
3. [Core Features & Capabilities](#3-core-features--capabilities)
4. [System Architecture Pipeline](#4-system-architecture-pipeline)
5. [Deep Learning Methodology](#5-deep-learning-methodology)
    - [5.1 Dataset and Segmentation](#51-dataset-and-segmentation)
    - [5.2 Spectrogram Feature Extraction](#52-spectrogram-feature-extraction)
    - [5.3 Synthetic Data Augmentation](#53-synthetic-data-augmentation)
    - [5.4 MobileNetV3 Transfer Learning](#54-mobilenetv3-transfer-learning)
    - [5.5 Loss Optimisation and Class Weighting](#55-loss-optimisation-and-class-weighting)
6. [Empirical Performance Metrics](#6-empirical-performance-metrics)
7. [Android Application Engineering](#7-android-application-engineering)
    - [7.1 Native UI and UX Paradigms](#71-native-ui-and-ux-paradigms)
    - [7.2 Edge Inference Execution](#72-edge-inference-execution)
8. [Directory Structure](#8-directory-structure)
9. [Installation and Setup Guide](#9-installation-and-setup-guide)
    - [9.1 Python Model Training Environment](#91-python-model-training-environment)
    - [9.2 Android Application Build](#92-android-application-build)
10. [Future Roadmap](#10-future-roadmap)
11. [Author and Academic Context](#11-author-and-academic-context)
12. [License and Citation](#12-license-and-citation)

---

## 1. Project Overview

PulmoSense is a complete, offline, edge-computing acoustic diagnostic ecosystem deployed natively on Android smartphones. Developed to act as a "Digital Stethoscope," the application utilises the ubiquitous smartphone microphone to capture raw respiratory audio, processes the signal into complex temporal-frequency visual arrays, and classifies pathological biomarkers (such as crackles and wheezes) using a heavily customised MobileNetV3Large deep learning architecture.

This repository contains both the Python-based data science pipeline (preprocessing, augmentation, and model training) and the full Jetpack Compose Kotlin Android application source code required to deploy the resulting TensorFlow Lite model.

## 2. Clinical Problem Statement

Respiratory anomalies, including Chronic Obstructive Pulmonary Disease (COPD), Asthma, and localised Pneumonia, represent a massive global health burden. In resource-constrained and rural environments, the lack of specialised diagnostic hardware and trained pulmonologists creates a severe diagnostic bottleneck. Traditional stethoscope auscultation is inherently subjective and prone to variable interpretation.

While modern cloud-based AI solutions exist, they mandate high-bandwidth internet connectivity and introduce severe patient data privacy concerns, rendering them inaccessible where they are needed most. PulmoSense bridges this technological divide by forcing complex neural network inference entirely onto edge hardware, ensuring zero-latency, objective screening without internet reliance.

## 3. Core Features & Capabilities

* **100% Offline Edge Inference:** The 10MB quantised TensorFlow Lite model resides entirely within the Android asset directory, ensuring absolute data privacy and rural functionality.
* **Clinical-Grade Accuracy:** Achieves a 96.99% accuracy rate across 8 respiratory classifications through advanced class-weighting and transfer learning techniques.
* **Robust Acoustic Preprocessing:** Native conversion of 1D Pulse-Code Modulation (PCM) audio to 2D Log-Mel Spectrograms utilizing JTransforms.
* **Guided Clinical UX:** Features an animated, physiological pacing algorithm to guide patients through a standardised 20-second respiratory cycle.

## 4. System Architecture Pipeline

The PulmoSense data flow operates in five distinct sequential stages from patient interaction to final diagnosis:

1. **Acoustic Capture:** The user undergoes a guided 20-second breathing exercise while the Android device records PCM audio at a 16 kHz sampling rate.
2. **Temporal Segmentation:** The 20-second audio array is systematically partitioned into four 5.0-second overlapping chunks to maintain temporal resolution.
3. **Spectral Transformation:** Short-Time Fourier Transforms (STFT) convert the raw 1D waveforms into 2D frequency domain maps, mapped to 128 Mel filter banks.
4. **Tensor Configuration:** Spectrograms are converted to a decibel (dB) scale, normalised, and duplicated across three RGB channels to format a `128 x 157 x 3` input tensor.
5. **MobileNetV3 Inference:** The TensorFlow Lite interpreter executes forward propagation on the four chunks, averages the resulting Softmax probability arrays, and returns a singular diagnostic confidence score to the Jetpack Compose UI.

## 5. Deep Learning Methodology

### 5.1 Dataset and Segmentation
The model is trained on the ICBHI Respiratory Sound Database. Due to the extreme heterogeneity of raw recording lengths (10 to 90 seconds), a sliding-window segmentation algorithm was developed to slice all acoustic data into uniform 5.0-second intervals prior to transformation.

### 5.2 Spectrogram Feature Extraction
Standard CNNs are optimised for spatial feature extraction. Consequently, raw 1D audio was converted into 2D visual representations. We utilized `librosa` to compute the STFT and applied a Mel-scale transformation with a maximum frequency cutoff (`fmax`) of 8000 Hz to encapsulate the primary acoustic domains of high-frequency crackles and continuous tonal wheezes.

### 5.3 Synthetic Data Augmentation
To counter the relatively limited scope of the ICBHI dataset and prevent model overfitting, deterministic digital signal processing augmentations were applied to the 5.0-second baseline chunks:
* **Time-Stretching (1.1x):** Utilising a phase vocoder to speed up the audio without altering pitch, simulating an elevated patient respiratory rate.
* **Pitch-Shifting (-2 steps):** Lowering the frequency via equal temperament scaling to account for physiological resonance variations in different chest cavity volumes.

These augmentations effectively tripled the dataset to over 10,000 highly distinct acoustic tensors.

### 5.4 MobileNetV3 Transfer Learning
To satisfy the draconian memory and computational constraints of edge deployment, `MobileNetV3Large` was selected as the foundational topology. The architecture leverages depthwise separable convolutions, Inverted Residual Blocks, and Squeeze-and-Excite (SE) attention mechanisms.

* **Frozen Base:** The model was initialised with pre-trained ImageNet weights. The bottom ~100 layers were frozen to preserve generalized edge and texture detection capabilities.
* **Fine-Tuning:** The top 50 layers were unfrozen, allowing the higher-level feature maps to adapt specifically to the unique horizontal and vertical frequency bands of respiratory spectrograms.
* **Classification Head:** A custom dense classification head `(512 units -> Dropout 0.4 -> 256 units -> Dropout 0.3 -> 8-class Softmax)` was integrated to map the extracted features to specific pathologies.

### 5.5 Loss Optimisation and Class Weighting
The ICBHI dataset exhibits catastrophic class imbalance, heavily favouring COPD over rare conditions like Bronchiectasis. To prevent a majority-class bias during backpropagation, dynamically calculated, heuristic class weights were integrated into the Categorical Cross-Entropy loss function. These weights were strictly capped at a maximum multiplier of 5.0 to prevent exploding gradients.

The network was compiled with the Adam optimiser utilising a highly conservative learning rate of `0.0001`. A `ReduceLROnPlateau` callback was implemented to halve the learning rate upon validation loss stagnation, ensuring precise convergence at the global minimum.

## 6. Empirical Performance Metrics

Evaluation on an isolated, augmented test set confirmed the architecture's viability as a frontline screening utility. The parity between Precision and Sensitivity indicates successful mitigation of the COPD dataset bias.

| Metric Classification | Percentage Score |
| :--- | :--- |
| **Global Accuracy** | 96.99% |
| **Macro-Averaged Precision** | 97.11% |
| **Sensitivity (Recall)** | 96.91% |
| **F1-Score** | 97.01% |

*Note: The model weights yielding the absolute minimum Validation Loss were preserved via an Early Stopping temporal checkpoint prior to final evaluation.*

## 7. Android Application Engineering

### 7.1 Native UI and UX Paradigms
The frontend is constructed exclusively in Kotlin using the Jetpack Compose declarative UI toolkit. 

* **Physiological Pacing:** To eliminate irregular acoustic captures, the recording interface features a dynamically scaling vector orb that guides the patient through a rigid 4-second inspiration and 4-second expiration cycle over a 20-second interval.

### 7.2 Edge Inference Execution
The Python Keras model was compressed via the TensorFlow Lite Converter using standard Post-Training Quantisation. The resulting `pulmosense.tflite` binary is loaded directly into the Android device's RAM via a `MappedByteBuffer`. Input tensors are supplied utilising the `tensorflow-lite-support` library, and the output probabilities are mapped to the 8 primary disease classifications dynamically.

## 8. Directory Structure

```text
PulmoSense/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/priyanshu/pulmosense/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── model/                  # Data classes and State management
│   │   │   │   ├── ui/
│   │   │   │   │   ├── components/         # Reusable Compose widgets
│   │   │   │   │   ├── navigation/         # NavHost and Route definitions
│   │   │   │   │   ├── screens/            # Main UI implementations (Dashboard, Recording, etc.)
│   │   │   │   │   └── theme/              # Typography, Colors, and Compose Theme
│   │   │   ├── res/                        # Drawables, XML layouts, and mipmap assets
│   │   │   └── AndroidManifest.xml         # Required permissions (RECORD_AUDIO) and app config
│   │   └── assets/
│   │       └── pulmosense.tflite           # The compiled and quantized Edge ML model
│   ├── build.gradle.kts                    # App-level Gradle configuration (TFLite, JTransforms)
│   └── proguard-rules.pro
├── gradle/                                 # Gradle wrapper configurations
├── build.gradle.kts                        # Project-level Gradle configuration
├── settings.gradle.kts                     # Project module inclusion
└── README.md                               # Project documentation
```
Note: The Python training pipeline (1_preprocess.py, 2_train.py) and .npy data artefacts are maintained in a separate analytical directory and are not bundled into the Android APK.

## 9. Installation and Setup Guide

### 9.1 Python Model Training Environment
To reconstruct the `.tflite model` from the raw ICBHI audio files, establish a standard data science environment.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ppratik765/PulmoSense.git](https://github.com/Ppratik765/PulmoSense.git)
   cd PulmoSense
   ```
2. **Install dependencies:**
   ```bash
   pip install tensorflow librosa numpy scikit-learn
   ```
3. **Execute Preprocessing:**
    Ensure the ICBHI .wav and .txt files are located in the target directory, then execute the sliding-window STFT and augmentation script.
   ```bash
   python 1_preprocess.py
   ```
4. Execute Fine-Tuning:
    Load the resulting .npy arrays into the MobileNetV3 architecture
   ```bash
   python 2_train.py
   ```
   The script will output the compiled `pulmosense.tflite` binary upon satisfying the Early Stopping parameters.

### 9.2 Android Application Build

1. To compile the Android APK and deploy the application to a physical test device:
2. Install the latest version of Android Studio.
3. Open Android Studio and select **File > Open**, navigating to the root PulmoSense directory.
4. Ensure the `pulmosense.tflite` file is located precisely in `app/src/main/assets/`.
5. The Gradle sync process will automatically pull the required namespace-corrected TFLite dependencies (`2.16.1`) and `JTransforms`.
6. Connect an Android device (API Level 24+) via USB and enable USB Debugging.
7. Click the **Run 'app'** (Green Arrow) button in the top toolbar to compile and install the application.

## 10. Future Roadmap
The current architecture provides a robust foundation for active clinical screening. Future iterations of the PulmoSense platform will target expanded telemetric capabilities:

* **Passive Nocturnal Monitoring:** Implementing continuous, ultra-low-power overnight inference loops to track sleep apnea occurrences and nocturnal asthma exacerbations without significant battery degradation.

* **Decentralised Telehealth Routing:** Automated generation of encrypted PDF diagnostic reports containing the raw spectrograms and AI confidence scores. These payloads will be engineered to transmit via asynchronous Bluetooth mesh networks or offline UPI rails to connect rural patients with remote pulmonologists.

* **Federated Learning:** Investigating on-device training protocols to allow the model to adapt to localized acoustic environments and noise profiles without centralizing sensitive patient data.

## 11. Author and Academic Context
Priyanshu Pratik

* Bachelor of Technology in Artificial Intelligence and Data Science

* Gati Shakti Vishwavidyalaya, Vadodara, Gujarat, India

* Specialisation: Transportation, Logistics, and Edge AI Deployment

This software is developed as part of advanced research into offline medical diagnostics and resource-constrained artificial intelligence deployment.

## 12. License and Citation
This project is licensed under the MIT License. You are free to use, modify, and distribute this software, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the software.

If you utilise this architecture or preprocessing methodology in academic research, please attribute as follows:

```Plaintext
Priyanshu Pratik. (2026). PulmoSense: Edge-Optimised Deep Learning for Real-Time Acoustic 
Respiratory Disease Screening. Gati Shakti Vishwavidyalaya.
For technical inquiries or pull request submissions, please refer to the issues tab or submit a standardised pull request detailing the proposed architectural changes.
```
   
 
