# 🤖 Visuo-Tactile Perception for Grasp Stability Estimation

This repository contains the official implementation of the experiments and methodology developed in my Master's Thesis:

📘 **Thesis Title**: *Visuo-Tactile Perception of Object Slippage in In-Hand Manipulation Tasks*  
👨‍🎓 **Author**: Hussein Loubani  
🏛️ **Institution**: Université Bourgogne Franche-Comté – EIPHI Graduate School  
📅 **Defense Date**: 2023  
🔬 **Supervisor**: Prof. Carlos Mateo  
🤖 **Platforms**: Franka Emika FR3, Sawyer Robot, PyBullet Simulator

---

## 🎯 Abstract

In-hand object manipulation is a complex task requiring real-time feedback from both vision and touch. Visual data alone often fails to capture key contact dynamics, especially in the presence of occlusion. This thesis addresses this limitation by proposing a **visuo-tactile fusion system** for **slippage detection and grasp stability estimation**.

The system integrates high-resolution camera input with tactile sensors embedded in a robotic gripper. We propose a deep neural network architecture combining **Convolutional Neural Networks (CNN)** for visual features and **Recurrent Neural Networks (RNN)**, specifically **GRUs or LSTMs**, to learn temporal dynamics from tactile sequences.

---

## 🧪 Research Objectives

1. Develop a real-time data acquisition pipeline for vision and tactile inputs.
2. Create a labeled dataset reflecting various grasp configurations and slippage scenarios.
3. Train and evaluate deep learning models to classify grasp success/failure.
4. Compare the effectiveness of individual modalities (vision, tactile) and their fusion.
5. Validate the system in both **real** and **simulated** environments using **Franka Emika** and **Sawyer** platforms.

---

## 🔧 Dependencies

Install the core dependencies:

```bash
pip install scipyplot deepdish torch torchvision
```

You may also need:

```bash
pip install numpy matplotlib pybullet
```

---

## 📁 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `grasp_data_collection.py` | Captures synchronized visual and tactile data during grasp experiments. Labels each grasp as success or failure. |
| `robot.py` | Provides interfaces to control the Franka and Sawyer robotic arms. |
| `train.py` | Trains the visuo-tactile neural network on collected datasets. |
| `draw.py` | Visualizes test accuracy across different modalities and dataset sizes. |
| `config.yaml` | Configuration file for tuning experiments. |
| `setup/` | Contains robot URDFs, mesh files, and kinematic configs used in both simulation and real-world deployment. |
| `.vscode/` | Workspace setup for ease of use in VSCode. |

> Note: All collected data is saved in `./data/` and model logs in `./logs/`.

---

## 🧠 Model Architecture

- **Visual Encoder**: ResNet-18 (pretrained on ImageNet)
- **Tactile Encoder**: GRU / LSTM for sequence modeling of tactile signals
- **Fusion Strategy**: Late fusion of embeddings before classification
- **Output**: Binary classification (Stable / Unstable Grasp)

---

## 🚀 Usage Instructions

### Step 1: Data Collection

Each `.h5` file stores 100 samples containing RGB images, tactile frames, and binary labels:

```bash
python grasp_data_collection.py
```

### Step 2: Train the Model

Train the model using `N x 100` samples (e.g., 10 datasets = 1000 samples):

```bash
python train.py -N 10
```

Training results and logs will be saved under `./logs/`.

### Step 3: Visualize Results

Plot accuracy comparisons across modalities (Vision-only, Tactile-only, Fusion) and dataset sizes:

```bash
python draw.py
```

---

## 📊 Experimental Results

Experiments show that **multi-modal fusion** consistently outperforms unimodal baselines. Under occlusion or poor visual conditions, tactile feedback becomes critical for reliable grasp stability estimation.

| Input Modality     | Accuracy |
|--------------------|----------|
| Vision Only        | ~75%     |
| Tactile Only       | ~79%     |
| Vision + Tactile   | **>90%** |

These findings demonstrate the necessity of **sensor fusion** in physical interaction tasks.

---

## 🧾 Publications and Materials

- 📄 [Full Thesis PDF](https://drive.google.com/file/d/1h1Y3Q_jqvrntvjmY15lEgwCcJQauIc4q/view) 
- 🖥️ [Thesis Defense Slides](https://drive.google.com/file/d/1SK8LHNccpnAiSW347AObS7zJ05AEb9ur/view?usp=sharing) 

---

## 🔑 Keywords

`robotic manipulation`, `grasp stability`, `visuo-tactile fusion`, `deep learning`, `object slippage detection`, `Franka Emika`, `Sawyer`, `PyBullet`, `ResNet`, `GRU`, `LSTM`, `tactile sensing`

---

## 📜 License

This work is released under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

This research was conducted at the University of Burgundy, with access to the RoboFab platform, and supervised by Prof. Carlos Mateo. I thank my peers and collaborators for their support throughout the project.


