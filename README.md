# Cellular Weathering: Predictive Modeling of Biological Aging via NHANES

## Project Overview
This repository contains the machine learning pipeline engineered to evaluate the drivers of cellular degradation using the National Health and Nutrition Examination Survey (**NHANES**) dataset. 

By modeling mean telomere length (`TELOMEAN`), this study analyzes how **Social Determinants of Health (SDoH)** (e.g., race, income-to-poverty ratio, education, marital status) predict biological weathering compared to traditional **Clinical Biomarkers** (e.g., HbA1c, cholesterol, blood pressure, BMI).

---

## Tech Stack & Methodology
* **Language:** Python (Pandas, NumPy, Scikit-Learn)
* **Validation:** 5-Fold Cross-Validation (`KFold`) optimized via an exhaustive `GridSearchCV` hyperparameter sweep.
* **Feature Importance:** Computed using model-agnostic **Permutation Feature Importance** to capture true structural impact.

---

## Model Performance & Optimization

The processed dataset ($N = 5,407$ complete cases) was evaluated across three optimized architectures. The **Elastic Net** framework achieved the highest predictive stability:

| Model Architecture | CV Mean R2 | CV Mean RMSE | Optimal Hyperparameters |
| :--- | :---: | :---: | :--- |
| **Elastic Net** | **0.1930** | **0.2323** | `alpha: 0.001`, `l1_ratio: 1.0` (Pure Lasso) |
| **Linear SVR** | 0.1841 | 0.2336 | `C: 0.1`, `epsilon: 0.1` |
| **Random Forest** | 0.1646 | 0.2364 | `max_depth: 10`, `n_estimators: 500` |

### Key Data Storytelling Insight
The grid search optimized onto a **pure Lasso penalty** (`l1_ratio: 1.0`), meaning the model automatically zeroed out non-contributing clinical parameters (such as HbA1c and BMI) to prevent overfitting. 

As a result, structural socioeconomic variables like **Race/Ethnicity** (0.0181) and **Marital Status** (0.0046) emerged with stronger permutation importance weights than physical markers like **Systolic Blood Pressure** (0.0009), proving that systemic environmental stress factors heavily dictate cellular aging trends.

---

## How To Run

1. **Install Dependencies:** `pip install pandas numpy scikit-learn`
2. **Execute Pipeline:** Place `NHANES_SUPER.csv` in the root directory and run:
   ```bash
   python phase1_kfolded_v1.1.py
