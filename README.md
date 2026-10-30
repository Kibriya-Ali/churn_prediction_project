# 🎧 KKBox Customer Churn Prediction & Retention Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B.svg)

## 📌 Executive Summary
Customer acquisition costs in the music streaming industry are highly competitive. This project predicts subscription churn for **KKBox** (Asia's leading music streaming service) using historical payment data and daily user listening logs. 

Instead of relying solely on static demographics, this pipeline processes daily behavioral time-series data to detect dropping engagement levels before a user explicitly cancels their subscription.

**Key Achievements:**
* Processed and aggregated a **30GB+ relational dataset** using memory-efficient chunking.
* Engineered custom behavioral features, such as `song_completion_ratio` and `historical_tx_count`.
* Trained a **LightGBM** classifier optimized for Recall to capture maximum true churners.
* Integrated **SHAP** values for local interpretability, allowing business teams to see the exact factors driving an individual customer's churn risk.
* Deployed an interactive **Streamlit** dashboard for live stakeholder risk simulation.

👉 **[Interact with the Live Dashboard Here](https://your-streamlit-url-here.streamlit.app)**

---

## 🛠️ Data Architecture & Feature Engineering
The raw data consists of four relational tables: `train`, `members`, `transactions`, and `user_logs` (totaling millions of rows). The data pipeline handles:

1. **Memory-Safe Aggregation:** The multi-gigabyte `user_logs.csv` was processed in chunks of 5 million rows to extract lifetime listening behaviors without exceeding RAM limits.
2. **One-to-Many Resolution:** The `transactions` table was sorted and reduced to extract the most recent payment method and auto-renew status while maintaining a historical count.
3. **Behavioral Indicators:** Engineered predictive metrics such as the ratio of songs finished vs. songs skipped, providing a direct proxy for user satisfaction.
4. **Native Categorical Handling:** Cast demographics and payment IDs to categorical types, allowing LightGBM to split on them optimally without exploding dimensionality via One-Hot Encoding.

---

## 🚀 Model Performance
Accuracy is a flawed metric for imbalanced churn datasets. The model was evaluated based on its ability to correctly identify actual churners (Recall) while maintaining a strong balance (F1-Score).

* **Algorithm:** LightGBM Classifier (`class_weight='balanced'`)
* **Interpretability:** SHAP (SHapley Additive exPlanations) implemented for global feature importance and local waterfall visualizations.
* **Top Predictive Features:**
  1. `is_auto_renew` (Status change from True to False)
  2. `payment_method_id` (Manual payments vs. Credit Cards)
  3. `completion_ratio` (Dropping engagement)

---

## 💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/kkbox-churn-prediction.git](https://github.com/yourusername/kkbox-churn-prediction.git)
cd kkbox-churn-prediction
```
### 2. Initialze py venv & install libraries
```bash
python -m venv venv
.\venv\Scripts\Activate.bat
pip install -r requirements.txt
```
### 3. Run Streamlit app
```bash
cd app
streamlit run app.py
```
