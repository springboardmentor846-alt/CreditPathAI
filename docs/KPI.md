### CreditPathAI – Key Performance Indicators (KPIs)

## Objective
To evaluate the efficiency and accuracy of the loan default prediction system.


## Model Performance KPIs

| KPI | Formula | Ideal Target | Description |
|------|----------|--------------|--------------|
| **AUC-ROC** | `roc_auc_score(y_true, y_pred_prob)` | ≥ 0.85 | Measures the model’s ability to distinguish defaults. |
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | ≥ 0.80 | Measures overall correctness. |
| **Precision** | `TP / (TP + FP)` | ≥ 0.75 | Measures positive prediction reliability. |
| **Recall** | `TP / (TP + FN)` | ≥ 0.80 | Measures how well model catches actual defaulters. |
| **F1-Score** | `2 * (Precision * Recall) / (Precision + Recall)` | ≥ 0.78 | Balances precision and recall. |
**[Where: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative]**


## System Performance KPIs

| KPI | Measurement | Ideal Target | Description |
|------|-------------|--------------|--------------|
| **Data Ingestion Time** | Measured using `time()` before and after loading dataset | < 1 minute for 10k rows | Measures ingestion efficiency. |
| **Recommendation Logic Accuracy** | Manual validation vs system output | ≥ 85% | Checks logic correctness of recommendations. |
