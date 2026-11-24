# Datavera Solutions — Student Academic Performance Early-Warning System

**Group 4 · CSE315 Data Science · Gordon College (BSCS)**

An end-to-end data science project built on the CRISP-DM methodology that predicts which
students are at risk of failing, using only background and behavioral data (interim grades
G1 and G2 are deliberately excluded so the model is a true early-warning signal).

## Dataset
UCI Student Performance dataset (Portuguese subject), 649 students, 33 features.
Target: Pass = 1 if final grade G3 >= 10, else Fail = 0. License: CC BY 4.0.

## Files
- `Datavera_Phase1_Preprocessing.ipynb` — load, clean, encode, EDA
- `Datavera_Phase2_Model_Pipeline.ipynb` — model training, tuning, evaluation (executed)
- `Datavera_Cleaned_Student_Data.csv` / `Datavera_Prepared_Model_Data.csv` — datasets
- `Datavera_Data_Dictionary.csv` — field descriptions
- `Datavera_Model_Experiment_Log.csv` — model metrics
- `model_artifacts.pkl` — final model (Tuned Logistic Regression)
- `*.png` — visualization portfolio (EDA, confusion matrices, ROC, performance summary)

## Key result
| Model | Accuracy | Fail Recall |
|---|---|---|
| Logistic Regression | 0.79 | 0.25 |
| Decision Tree | 0.76 | 0.20 |
| **Tuned Logistic Regression (final)** | 0.65 | **0.50** |
| Tuned Decision Tree | 0.78 | 0.20 |

On imbalanced data accuracy is misleading; judged on **Fail Recall**, the Tuned Logistic
Regression is the right model — it flags half of at-risk students, the job that matters.

## Ethics
Public, historical data; no real students. Demographic features require a fairness review
before any real deployment. A flag is a prompt for support, never a label or punishment.
