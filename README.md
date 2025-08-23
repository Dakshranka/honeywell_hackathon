
# TS-Anomaly: Multivariate Time Series Anomaly Detection 

## Project Context
This project is focused on Multivariate Time Series Anomaly Detection. The goal is to automatically detect abnormal behavior in complex machinery using time-series data from multiple sensors. Early detection of anomalies can prevent costly failures, improve safety, and optimize maintenance schedules.
## Dataset
- **File:** `data/train.csv` (provided)
- **Format:** CSV with a `timestamp` column (YYYY-MM-DD HH:MM:SS) and multiple sensor features.
- **Period:** 2004-01-01 to 2004-01-19
- **Normal Period:** First 5 days (2004-01-01 to 2004-01-05) are guaranteed normal (used for training).
- **Analysis Period:** Full range (used for anomaly detection).

## Dataset Explanation

The dataset (`data/train.csv`) contains time-stamped sensor readings from an industrial process. Each row represents a snapshot in time, with:
- **timestamp**: The date and time of the reading (format: YYYY-MM-DD HH:MM:SS)
- **sensor_1, sensor_2, ..., sensor_n**: Numeric values from different sensors monitoring the equipment

**Key Points:**
- The first column must be named `timestamp` and contain valid date-time values.
- All other columns are sensor readings (float or integer).
- There are no labels; the task is unsupervised anomaly detection.
- The first 5 days (2004-01-01 to 2004-01-05) are guaranteed normal and used for model training.
- The rest of the data (up to 2004-01-19) is used for anomaly analysis.

## Step-by-Step Instructions (Zero Knowledge Friendly)

1. **Install Python 3.9+**
     - Download and install Python from [python.org](https://www.python.org/downloads/).
2. **Install Project Dependencies**
     - Open a terminal in the `ts-anomaly` folder and run:
       ```bash
       pip install -r requirements.txt
       ```
3. **Prepare the Dataset**
     - Ensure `train.csv` is in the `data/` folder and matches the format above.
4. **Preprocess the Data**
     - Run:
       ```bash
       python src/preprocess_hackathon.py
       ```
     - This will create cleaned training and analysis files in `outputs/`.
5. **Train Models & Score Anomalies**
     - Run:
       ```bash
       python src/train_hackathon.py
       ```
     - This will generate anomaly scores and top features in `outputs/anomalies_output.csv`.
6. **Analyze & Visualize Results**
     - Open `notebooks/final_visualization.ipynb` in Jupyter and run all cells for EDA and anomaly analysis.
7. **(Optional) Launch Dashboard**
     - Run:
       ```bash
       streamlit run app/streamlit_app.py
       ```
     - Explore anomalies interactively.
## Solution Approach & Model Rationale

### Process Overview
1. **Data Preprocessing:**
      - Validate and clean timestamps, handle missing values (forward/backward fill).
      - Split data into a guaranteed normal training period and a full analysis period.
      - Scale features for consistent model input.

2. **Model Training & Scoring:**
      - Train three complementary models on the normal period:
           - **Isolation Forest:** Detects global anomalies by isolating outliers in high-dimensional data. Suitable for multivariate time series because it does not assume any distribution and works well with mixed feature types.
           - **PCA (Principal Component Analysis):** Captures reconstruction error by projecting data into lower dimensions. Anomalies are detected when new data cannot be well-reconstructed, indicating deviation from normal patterns. Effective for finding subtle, correlated changes across features.
           - **LSTM Autoencoder:** Learns temporal patterns and reconstructs sequences. Anomalies are flagged when reconstruction error is high, indicating unusual temporal behavior. LSTM is ideal for time series due to its ability to capture dependencies over time.
      - Ensemble the scores for robust anomaly detection.

3. **Feature Attribution:**
      - For each detected anomaly, identify the top 7 features contributing most to the abnormality score. This provides actionable insights for root cause analysis.

4. **Output & Visualization:**
      - Save results to CSV with scores and top features.
      - Visualize scores, anomalies, and feature contributions in a Jupyter notebook and Streamlit dashboard.

### Why These Models?
- **Isolation Forest:**
     - Fast, scalable, and effective for high-dimensional, unlabeled data.
     - Does not require assumptions about data distribution.
     - Widely used in industry for anomaly detection.
- **PCA:**
     - Detects subtle, correlated changes that may not be visible in individual features.
     - Useful for dimensionality reduction and understanding feature relationships.
- **LSTM Autoencoder:**
     - Captures complex temporal dependencies and sequence patterns.
     - Flags anomalies that break expected time-based behavior.
     - Especially suitable for sensor data with time dependencies.
- **Ensemble:**
     - Combining models increases robustness and reduces false positives.

This multi-model approach ensures that both global, correlated, and temporal anomalies are detected, providing a comprehensive solution for industrial time series anomaly detection.

## Workflow & How to Run
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Preprocess Data
```bash
python src/preprocess_hackathon.py
```
- Loads and validates `train.csv`
- Handles missing values (forward/backward fill)
- Splits into `outputs/normal_period.csv` and `outputs/analysis_period.csv`

### 3. Train Models & Score Anomalies
```bash
python src/train_hackathon.py
```
- Scales features
- Trains Isolation Forest, PCA, and LSTM Autoencoder on normal period
- Scores analysis period for anomalies (0–100 scale)
- Outputs `outputs/anomalies_output.csv` with scores and top 7 contributing features
- Checks training period scores to ensure low false positives

### 4. Analyze & Visualize Results
- Open `notebooks/final_visualization.ipynb` in Jupyter
- Run all cells to:
     - Perform EDA (summary stats, missing values, correlations, histograms)
     - Visualize anomaly scores over time for each model
     - Overlay anomalies on feature plots
     - View top contributing features for high anomaly scores

### 5. Interactive Dashboard (Optional)
```bash
streamlit run app/streamlit_app.py
```
- Explore, filter, and export anomalies interactively

## Project Structure
```
ts-anomaly/
├── data/                  # Input dataset (train.csv)
├── src/                   # Preprocessing and modeling scripts
│   ├── preprocess_hackathon.py
│   └── train_hackathon.py
├── notebooks/             # Jupyter notebooks for EDA and visualization
│   └── final_visualization.ipynb
├── app/                   # Streamlit dashboard
│   └── streamlit_app.py
├── outputs/               # Output CSVs and results
│   ├── normal_period.csv
│   ├── analysis_period.csv
│   └── anomalies_output.csv
├── requirements.txt
└── README.md
```
## Key Libraries Used
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- PyTorch (for LSTM Autoencoder)
- Streamlit (for dashboard)

## Notes
- All code is written in Python 3.9+
- No config.py or unused scripts; all logic is in the provided .py and .ipynb files
- For custom datasets, ensure the first column is a timestamp and all columns have headers


## License
MIT License. Free for academic and commercial use.
