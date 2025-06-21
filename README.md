# RNN-based Intrusion Detection System (IDS) Training Module

This module implements a Recurrent Neural Network (RNN) approach for network intrusion detection using the CICIDS2017 dataset. The code is designed to preprocess the data, train the model using both cross-validation and standard train/test splits, and evaluate its performance using common classification metrics.

## Overview
- **Frameworks Used:** PyTorch, scikit-learn, pandas, numpy
- **Model:** LSTM-based RNN for sequence modeling
- **Dataset:** CICIDS2017 (combinedDataSet.csv)
- **Features:** 80 input features per sample
- **Labels:** Supports both binary (benign/attack) and multi-class attack categorization

## Data Preprocessing
- The dataset is loaded from `ML-dataset/CICIDS2017GeneratedLabelledFlows/TrafficLabelling/combinedDataSet.csv`.
- Unnecessary columns (e.g., Flow ID, IPs, Timestamp) are dropped.
- Missing values in key features are filled with the mean.
- Features are normalized using `StandardScaler`.
- Labels can be encoded in two ways:
  - **Lambda encoding:** Binary (benign vs. attack)
  - **Mapping encoding:** Multi-class (various attack types)
- The data is split into training and test sets (70/30 split).

## Model Architecture
- The model is an LSTM-based RNN (`IdsRnn` class):
  - **Input size:** 80
  - **Hidden size:** 512
  - **Output size:** 2 (binary) or 7 (multi-class)
  - **Layers:** 1 LSTM layer, followed by a linear output layer
- The model processes sequences of feature vectors (time windows) for each sample.

## Training Procedure
- **Cross-validation:**
  - 5-fold cross-validation is supported.
  - Early stopping is used to prevent overfitting (stops if no improvement for 7 epochs).
  - Model and metrics are saved for each fold.
- **Standard training:**
  - Trains for up to 500 epochs (early stopping after 25 epochs without improvement).
- **Optimization:**
  - Adam optimizer
  - Cross-entropy loss

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1 Score (weighted)
- Metrics are logged for each fold/epoch.
- Model weights are saved after training.

## Logging
- Training progress and metrics are logged to `logs/training_logs_general.txt` and per-run log files.

## Usage
1. Place the dataset in the expected path.
2. Run the script to start training and evaluation.
3. Trained models and logs will be saved in the working directory.

## Customization
- You can adjust hyperparameters (batch size, learning rate, hidden size, etc.) at the top of the script.
- The label encoding method can be switched between 'lambda' and 'mapping'.

## File Structure
- `My_Custom_Data_Set`: Custom PyTorch Dataset for time-windowed samples
- `IdsRnn`: LSTM-based RNN model
- `preProcessDataSet`: Data preprocessing and splitting
- `training_modul`: Main training loop (with/without cross-validation)
- `evaluate_metrics`: Computes evaluation metrics

## Installation

To install all required Python packages for this project, run the provided `install.py` script. This script will automatically install PyTorch (with CUDA 12.1 support if available), torchvision, torchaudio, and other dependencies such as pandas, numpy, scikit-learn, matplotlib, and tqdm.

**Usage:**

```powershell
python install.py
```

This will ensure your environment is set up with all necessary libraries for training and evaluating the RNN-based IDS model.

## Dataset Download

The dataset required for this project (`combinedDataSet.csv`) is too large to be stored in the repository. You can download it from the following Google Drive link:

[Download combinedDataSet.csv](https://drive.google.com/file/d/13KiItv0_uahzuKAqTxNXbOU4760-e0a3/view?usp=drivesdk)

After downloading, place the file in the project directory as specified in the instructions above.
