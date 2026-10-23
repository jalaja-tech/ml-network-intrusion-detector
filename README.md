# 🛡️ ML Network Intrusion Detector

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cybersecurity](https://img.shields.io/badge/Domain-Cybersecurity%20%26%20AI-red.svg)]()

An end-to-end Machine Learning pipeline built using Python and **Scikit-learn** to inspect network traffic connection logs, detect anomalies, and classify cyber threats in near real-time.

---

## 📌 Project Overview

Traditional **Intrusion Detection Systems (IDS)** rely on static, rule-based signatures that fail against novel or obfuscated cyberattacks. This project implements a machine learning approach trained on public cybersecurity benchmark logs (**NSL-KDD**). 

By analyzing TCP/IP connection metrics, flag ratios, and error rates, the model automatically differentiates between **normal baseline connections** and **malicious network intrusions** (such as Denial of Service, Port Scanning, and Unauthorized Access attempts).

---

## 📊 Dataset Information

This project utilizes the benchmark **NSL-KDD Dataset**, an improved version of the classic KDD'99 dataset designed to eliminate duplicate records and bias.

* **Dataset Link:** [Kaggle - NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)
* **Dataset Creator:** Canadian Institute for Cybersecurity (UNB)
* **Total Records:** 148,517 network traffic instances (125,973 Train / 22,544 Benchmark Test)
* **Key Feature Types:**
  * **Categorical (3):** `protocol_type` (tcp, udp, icmp), `service` (http, ftp, smtp), `flag` (SF, REJ, S0)
  * **Basic Header Features:** `duration`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`
  * **Content & Traffic Metrics:** `logged_in`, `num_failed_logins`, `count`, `serror_rate`, `same_srv_rate`
  * **Target Label:** `label` (`0` for Normal, `1` for Anomaly/Attack)

---

## 🛠️ Key Features & Architecture

* **Automated Data Preprocessing:** Built-in `ColumnTransformer` handles categorical variables via `OneHotEncoder(handle_unknown='ignore')` and standardizes numerical features using `StandardScaler()`.
* **Ensemble Learning:** Employs a `RandomForestClassifier` (100 decision trees) to prevent overfitting and capture non-linear relationships across 42 network features.
* **Data Leakage Prevention:** Enforces strict pipeline encapsulation using Scikit-learn's `Pipeline` API.
* **Model Serialization:** Saves the trained model pipeline to disk as `ids_model.joblib` for rapid CLI or API inference without retraining.

---

## 🧰 Technologies & Libraries Used

* **Language:** Python 3.10+
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Model Persistence:** Joblib
* **Data Visualization (Optional):** Matplotlib, Seaborn

---

## 📈 Model Performance & Results

Tested on an 80/20 train-test split of 125,973 NSL-KDD connection logs:

| Metric | Score | Detail |
| :--- | :---: | :--- |
| **Accuracy** | **99.95%** | 25,183 out of 25,195 test connections correctly classified |
| **Precision** | **0.9995** | High confidence in identified attacks (only 6 False Positives) |
| **Recall** | **0.9995** | High threat detection rate (only 6 False Negatives) |
| **F1-Score** | **0.9995** | Harmonic mean of Precision and Recall |

### Confusion Matrix (25,195 Test Instances)

```text
               Predicted Normal (0)   Predicted Anomaly (1)
Actual Normal (0)      13,463 (TN)               6 (FP)
Actual Anomaly (1)          6 (FN)          11,720 (TP)
```

---

## 🚀 Setup & Run Instructions

### 1. Prerequisites & Environment Setup

Clone the repository and open your terminal (PowerShell / Bash):

```bash
git clone https://github.com/jalajaborkar/ml-network-intrusion-detector.git
cd ml-network-intrusion-detector
```

Create and activate a virtual environment (optional but recommended):

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Dataset Download & Setup

Download the NSL-KDD dataset using the Kaggle CLI or direct zip download:

```bash
# Download dataset zip
kaggle datasets download -d hassan06/nslkdd

# Extract into data directory (PowerShell)
Expand-Archive -Path nslkdd.zip -DestinationPath data -Force
```

Preprocess the raw dataset file to add required column headers:

```bash
python -c "
import pandas as pd
cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level']
df = pd.read_csv('data/KDDTrain+.txt', header=None, names=cols)
df.to_csv('data/KDDTrain_clean.csv', index=False)
print('Data prepared at data/KDDTrain_clean.csv')
"
```

---

### 3. Training & Evaluating the Model

Run the main pipeline to train the Random Forest Classifier and save `ids_model.joblib`:

```bash
python main.py --csv data/KDDTrain_clean.csv
```

---

### 4. Running Real-Time Inference Test

Test predictions on sample connection inputs using the saved model pipeline:

```bash
python predict.py
```

---

## 📂 Repository Structure

```text
ml-network-intrusion-detector/
│
├── data/                       # Dataset directory
│   ├── KDDTrain+.txt          # Raw NSL-KDD train dataset
│   └── KDDTrain_clean.csv     # Header-processed CSV dataset
│
├── main.py                     # Main pipeline (Data preprocessing, Training, Evaluation)
├── predict.py                  # Live connection sample prediction script
├── ids_model.joblib            # Serialized trained Scikit-learn pipeline artifact
├── requirements.txt            # Python dependencies
├── Jalaja_Sunil_Borkar_ProjectReport.docx  # Full Project Documentation
└── README.md                   # Project documentation
```

---

## 👤 Author

* **Jalaja Sunil Borkar**
* Department of Artificial Intelligence & Data Science
* Datta Meghe College of Engineering, Mumbai University
* IBM SkillsBuild Cybersecurity Bootcamp | BharatCares | October 2026

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).
