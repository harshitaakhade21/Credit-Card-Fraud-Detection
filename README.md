# Credit Card Fraud Detection - Machine Learning

## 📌 Overview
This project develops a machine learning model to accurately detect fraudulent credit card transactions using historical data.  
By analyzing transaction patterns, the model distinguishes between normal and fraudulent activity, helping financial institutions flag suspicious behavior early and reduce potential risks.

---

## 📊 Dataset
- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions with 31 features
- **Features:**
  - `Time` → seconds elapsed since the first transaction
  - `V1`–`V28` → anonymized features
  - `Amount` → transaction amount
  - `Class` → target variable (0 = normal, 1 = fraud)
- **Class Distribution:** Highly imbalanced (fraud cases ≈ 0.02% of total)

---

## 🛠 Steps in the Project
1. **Importing Libraries**  
   Used `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` for data handling, visualization, and model building.

2. **Loading and Exploring the Data**  
   Examined dataset structure, descriptive statistics, and class imbalance.

3. **Class Distribution Analysis**  
   Found that fraudulent transactions make up only ~0.02% of all transactions.

4. **Exploring Transaction Amounts**  
   Compared amounts between fraudulent and normal transactions.

5. **Correlation Matrix**  
   Visualized feature correlations to identify potential relationships.

6. **Data Preparation**  
   - Split features (`X`) and target (`Y`)
   - Converted to NumPy arrays
   - Train-test split (80% training, 20% testing)

7. **Model Building**  
   Trained a **Random Forest Classifier** to detect fraudulent transactions.

8. **Evaluation Metrics**  
   Measured performance using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Matthews Correlation Coefficient (MCC)
   - Confusion Matrix

---

## 📈 Model Performance
- **Accuracy:** 99.96%
- **Precision:** 98.73%
- **Recall:** 79.59%
- **F1-Score:** 88.14%
- **MCC:** 0.8863

> ⚠ Note: Accuracy is high due to class imbalance. Precision and recall are more important for fraud detection.

---

## 🚀 Future Improvements
- Apply **SMOTE** or **ADASYN** to handle imbalance.
- Use **cost-sensitive learning** with `class_weight='balanced'`.
- Experiment with **XGBoost, LightGBM, or CatBoost** for higher recall.
- Tune decision thresholds for better fraud detection coverage.

---

## 💻 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python fraud_detection.py
